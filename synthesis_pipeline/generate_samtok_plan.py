"""Sample multi-mask SAMTok rows and generate per-region edit instructions.

This is a pilot sampler, not a production eligibility filter. It samples an
equal number of two-mask GRES and VER rows so that every selected source can be
reused twice and the requested edit-case count is exact. Within each subset the
rows are stratified by minimum region area to retain both tiny/hard and larger
targets. Qwen3-VL sees the clean source plus an annotation-safe target panel
and writes one localized edit instruction for every original mask.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from synthesis_pipeline.prepare_samtok_data import (
    decode_rle,
    load_jsonl,
    qwen_canvas_size,
    resize_mask,
)
from synthesis_pipeline.visual_prompt_utils import (
    instruction_target_panel,
    isolated_target_image,
)
import utils.vlm_utils as vlm


TASK_TYPES = ("add", "remove", "replace", "attribute")
PLANNING_VISUAL_INPUT_VERSION = "clean_crop_cutout_binary_mask_grounded_v3"
FORBIDDEN_OUTPUT_PATTERN = re.compile(
    r"\b(marked|mask|overlay|annotations?|highlighted|bbox|coordinates?|target region|"
    r"source crop|(?:full|clean) source image|source image|image [123]|"
    r"(?:from|in) the image)\w*\b|"
    r"\b(?:localization|mask|target|left|middle|right|top|bottom|upper|lower|first|"
    r"second|third|three-part)\s+panels?\b|\bpanels?\s+(?:view|image)\b",
    flags=re.IGNORECASE,
)
TYPE_GUIDANCE = {
    "add": (
        "Use the masked content only as the placement anchor. Directly name the exact "
        "physical item to add and its short spatial relation to that anchor; an "
        "unspecified generic item is invalid."
    ),
    "remove": (
        "Remove only a complete, independently removable physical object contained by "
        "the mask. A body fragment, surface patch, or attached section is incompatible. Never ask to "
        "remove a rider, held item, supporting person, or other content outside it. If "
        "removing only the masked pixels would leave an implausible dependent object, "
        "mark this mask incompatible with remove."
    ),
    "replace": (
        "Replace only a complete physical object with a different identity, category, "
        "or model. A color, material, pattern, text, or styling change to the same kind "
        "of object is an attribute edit and is invalid for replace. If the mask is a "
        "fragment or omits a dependency, mark it incompatible."
    ),
    "attribute": (
        "Change one conspicuous visual attribute of only the content contained by the "
        "mask. If the requested local attribute cannot be changed coherently within "
        "that extent, mark it incompatible. Directly specify the desired color, "
        "material, pattern, or other value; 'different color' is invalid."
    ),
}

PROMPT = """You are designing one difficult, localized image-editing training example.

IMAGE 1 contains ONLY the original photographic pixels inside one mask, magnified on a gray checkerboard. IMAGE 2 is a three-part verification panel: LEFT is an unmodified clean context crop; MIDDLE repeats the isolated mask pixels; RIGHT is the aligned black/white binary shape. IMAGE 3 is the clean full source for context. Only IMAGE 1 and IMAGE 2 MIDDLE define the editable content. Objects visible only in the context crop or full source are outside the mask.
Required edit type: <<<TASK_TYPE>>>

Task-specific rule:
<<<TYPE_GUIDANCE>>>

Requirements:
- First inventory only what is actually visible in IMAGE 1. Do not name a rider, board, held item, supporting person, nearby object, or context object unless its own photographic pixels are visibly present in IMAGE 1. Treat connected pixels as one object when they form its parts (for example, a differently colored handle is still part of its tool).
- Never infer color, material, texture, identity, or object name from the black/white mask. Every descriptive attribute in the output must be visibly confirmed in the clean source or clean crop.
- The final text must NOT mention the panel, crop, mask, annotation, bbox, coordinates, image number, or target region.
- `masked_content` must describe the full connected photographic extent in IMAGE 1 from end to end, not only its most salient tip or subpart. Set `edit_unit_status` to `complete_object` for a whole independent object, `complete_part` for a coherent local part of a larger object, or `incomplete` for a fragment/mixed extent. Remove and replace require `complete_object`; add and attribute may use `complete_part`.
- Inspect the context for an object physically carried or supported BY the masked content but absent from IMAGE 1. Record it in `outside_dependencies`, or write `none`. A floor, table, road, shelf, or other surface that supports the target is not a dependency; neither is an independent object merely touching or near it. For remove/replace, a true outside dependency that would visibly float or become impossible makes the mask incompatible.
- `refer_object` must name the masked content itself in at most 10 words, with only enough visible context to distinguish its instance. The instruction must clearly edit that same named subject, not a context object or one of its unmasked possessions.
- For remove, replace, and attribute, every explicitly changed object or part must lie inside the mask. Do not expand the request to unmasked attached, held, worn, supported, or nearby content.
- For remove or replace, a person without their vehicle, a mount without its rider, or an object without a visibly attached/held dependent part is incompatible. Set `mask_compatibility` to `incompatible` when this edit cannot be completed coherently within the exact extent. Do not repair incompatibility by changing a different object or expanding the scope.
- When compatible, write a direct instruction of 4-22 words. State the action, exact target, and requested result. Omit reconstruction recipes, preservation clauses, explanations, and lists of things that should stay unchanged.
- Keep `new_instruction` to 3-16 words and make it perform the same edit.
- For replace, directly name the concrete replacement; phrases such as a different/another/similar object, model, person, or one are invalid.
- The edit must remain visible at full-image scale. For add, invent a suitable item from this scene rather than following any example catalog.
- Use ASCII English.

Return exactly one JSON object and no markdown. Use exactly these keys: `masked_content`, `edit_unit_status`, `outside_dependencies`, `refer_object`, `mask_compatibility`, `compatibility_reason`, `editing_instruction`, `new_instruction`. Fill every value with your actual decision; never copy field descriptions or requirement wording.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--positive-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument(
        "--candidate-cases",
        type=int,
        default=None,
        help=(
            "Number of region candidates to plan before mask-compatibility filtering. "
            "Defaults to three times --num-cases; use a larger multiple of 20 if needed."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--vlm",
        choices=["qwen8b-vllm", "qwen4b-vllm", "qwen8b", "qwen4b"],
        default="qwen8b-vllm",
    )
    parser.add_argument("--vlm-model-id", default=None)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--vlm-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def normalized_area(rle: dict[str, Any]) -> float:
    value = dict(rle)
    if isinstance(value["counts"], str):
        value["counts"] = value["counts"].encode("ascii")
    height, width = [int(number) for number in value["size"]]
    return float(mask_utils.area(value)) / float(height * width)


def sample_rows(
    rows: list[dict[str, Any]], num_cases: int, seed: int
) -> list[dict[str, Any]]:
    if num_cases % 4:
        raise ValueError("--num-cases must be divisible by four for exact type balance")
    if num_cases % 2:
        raise ValueError("--num-cases must be even for two-mask source reuse")
    source_count = num_cases // 2
    if source_count % 2:
        raise ValueError("The source count must divide equally between GRES and VER")
    per_subset = source_count // 2
    if per_subset % 5:
        raise ValueError("Sources per subset must be divisible by five area strata")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for subset in ("gres", "ver"):
        candidates = []
        for row in rows:
            if row["source_subset"] != subset or int(row["num_masks"]) != 2:
                continue
            areas = [normalized_area(mask) for mask in row["masks"]]
            candidates.append({**row, "mask_area_fractions": areas, "min_mask_area": min(areas)})
        candidates.sort(key=lambda row: (row["min_mask_area"], row["parquet_row_index"]))
        per_stratum = per_subset // 5
        for stratum in range(5):
            start = round(len(candidates) * stratum / 5)
            end = round(len(candidates) * (stratum + 1) / 5)
            bucket = candidates[start:end]
            if len(bucket) < per_stratum:
                raise ValueError(f"Not enough {subset} rows in area stratum {stratum}")
            chosen = rng.sample(bucket, per_stratum)
            for row in chosen:
                row["area_stratum"] = stratum
            selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def image_bytes_from_cell(value: Any) -> bytes:
    images = value.as_py()
    if not isinstance(images, list) or not images or not images[0].get("bytes"):
        raise ValueError("Parquet image cell does not contain embedded bytes")
    return bytes(images[0]["bytes"])


def build_prompt(row: dict[str, Any], task_type: str) -> str:
    return (
        PROMPT.replace("TASK_TYPE", task_type)
        .replace("TYPE_GUIDANCE", TYPE_GUIDANCE[task_type])
    )


def instruction_messages(
    source: Image.Image,
    target_panel: Image.Image,
    row: dict[str, Any],
    task_type: str,
    previous_response: str | None = None,
    target_cutout: Image.Image | None = None,
) -> list[dict[str, Any]]:
    prompt = build_prompt(row, task_type)
    if previous_response:
        prompt += (
            "\n\nYour previous response was invalid because it leaked annotation "
            "language, was too long, omitted required fields, or used the wrong edit "
            "action. Re-inventory the full connected extent of IMAGE 1 pixels and make "
            "refer_object name that same instance in at most 10 words. Ensure at least "
            "one concrete subject noun from refer_object also appears in the full "
            "instruction. If "
            "the required edit cannot stay within the mask, return "
            "mask_compatibility=incompatible instead of choosing another object or "
            "expanding the scope. Do not mark a mask incompatible merely because the "
            "previous wording was invalid. For replace, directly choose a concrete "
            "new category, identity, or named model; never say only different/another "
            "person, object, model, or one, and never use a mere color/material/text "
            "change. Otherwise keep the full instruction to 4-22 words "
            "and omit preservation clauses and explanations. Both instruction fields "
            "must perform the required task type. Do not refer to any panel, crop, "
            "mask, annotation, image number, or target region. "
            "Previous response:\n" + previous_response[:1600]
        )
    visual_content = []
    if target_cutout is not None:
        visual_content.append({"type": "image", "image": target_cutout})
    visual_content.extend(
        [
            {"type": "image", "image": target_panel},
            {"type": "image", "image": source},
            {"type": "text", "text": prompt},
        ]
    )
    return [
        {
            "role": "user",
            "content": visual_content,
        }
    ]


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip())
    decoder = json.JSONDecoder()
    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


GENERIC_INSTRUCTION_PATTERN = re.compile(
    r"\b(?:semantically plausible|visible) accessor(?:y|ies)\b|"
    r"\baccessory or detail\b|\bplausible object or detail\b|"
    r"\bplausible object from (?:the )?scene context\b|"
    r"\babsent from (?:the )?target\b|\bnot a second copy\b|"
    r"\binvent the addition\b|\bfixed or suggested catalog\b|"
    r"\bscene-(?:appropriate|plausible) (?:addition|object|item|detail)\b|"
    r"\bsmall,? clearly visible addition\b|"
    r"\b(?:different|another|similar) (?:model|person|object|item|one)\b|"
    r"\b(?:different|another|new) colou?r\b|"
    r"\bnew one\b|\bsame (?:object|item|one)\b",
    flags=re.IGNORECASE,
)
PRESERVATION_BOILERPLATE_PATTERN = re.compile(
    r"\b(?:preserve|keep|leave)\s+(?:all|every|the other|other|unrelated|surrounding)\b|"
    r"\bdo not (?:change|alter|remove|modify)\b|"
    r"\bwhile (?:preserving|keeping|leaving)\b|"
    r"\bappear as if (?:it|they|he|she) (?:was|were) never there\b",
    flags=re.IGNORECASE,
)
PARTIAL_EDIT_UNIT_PATTERN = re.compile(
    r"\b(?:section|portion|fragment|patch|corner)\b|"
    r"\b(?:body|door|front|quarter|rear|side) panel\b|"
    r"\b(?:window|door) frame\b|"
    r"^(?:a |an |the )?(?:person|woman|man|boy|girl)'s "
    r"(?:head|neck|arm|hand|leg|foot|torso)\b",
    flags=re.IGNORECASE,
)
FALSE_INCOMPATIBILITY_REASON_PATTERN = re.compile(
    r"\b(?:self-contained|standalone)\b|"
    r"\bno (?:visible )?(?:outside|external|missing) dependenc(?:y|ies)\b|"
    r"\bnot physically (?:carrying|holding|supporting|attached)\b",
    flags=re.IGNORECASE,
)
TYPE_ACTION_PATTERNS = {
    "add": re.compile(
        r"\b(add|attach|place|put|give|tie|hang|mount|stick|install)\b",
        re.IGNORECASE,
    ),
    "remove": re.compile(r"\b(remove|erase|delete)\b", re.IGNORECASE),
    "replace": re.compile(r"\b(replace|substitut\w*|swap|exchange)\b", re.IGNORECASE),
    "attribute": re.compile(
        r"\b(change|make|recolor|turn|paint|dye|brighten|darken|alter)\b",
        re.IGNORECASE,
    ),
}
LEADING_CROSS_TYPE_ACTION = re.compile(
    r"^(?:please\s+)?(?:add|attach|place|put|give|tie|hang|mount|stick|install|"
    r"remove|erase|delete|replace|substitut\w*|swap|exchange)\b",
    flags=re.IGNORECASE,
)

PANEL_LOCATION_PATTERN = re.compile(
    r"\s+(?:in|on|from)\s+the\s+"
    r"(?:(?:top|bottom|upper|lower|left|right|center|middle|central)\s+){0,3}"
    r"panel\b",
    flags=re.IGNORECASE,
)
SUBJECT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "black",
    "blue",
    "brown",
    "green",
    "gray",
    "grey",
    "in",
    "left",
    "near",
    "of",
    "on",
    "orange",
    "pink",
    "purple",
    "red",
    "right",
    "the",
    "white",
    "with",
    "yellow",
}
REPLACEMENT_GENERIC_WORDS = SUBJECT_STOPWORDS | {
    "another",
    "clean",
    "colored",
    "colour",
    "different",
    "golden",
    "identical",
    "large",
    "material",
    "metal",
    "metallic",
    "modern",
    "new",
    "one",
    "ornate",
    "same",
    "scene",
    "similar",
    "small",
    "style",
    "styled",
    "tall",
    "texture",
    "textured",
}
HUMAN_IDENTITY_TERMS = {
    "adult",
    "baby",
    "boy",
    "child",
    "elderly",
    "girl",
    "man",
    "person",
    "teenager",
    "woman",
    "young",
}


def remove_panel_location(text: str) -> str:
    """Drop an accidental localization-panel phrase without changing semantics."""
    return re.sub(r"\s{2,}", " ", PANEL_LOCATION_PATTERN.sub("", text)).strip()


def ascii_text(value: Any) -> str:
    """Normalize harmless smart punctuation emitted by the VLM."""
    return str(value).translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    ).strip()


def normalize_generated(
    value: dict[str, Any] | None, task_type: str
) -> dict[str, str] | None:
    if not value:
        return None
    normalized: dict[str, str] = {}
    raw_status = value.get("edit_unit_status", "")
    if isinstance(raw_status, bool):
        value["edit_unit_status"] = "complete_object" if raw_status else "incomplete"
    elif str(raw_status).strip().lower() in {
        "isolated",
        "self-contained",
        "self contained",
        "coherent",
        "complete",
    }:
        value["edit_unit_status"] = "complete_object"
    raw_compatibility = value.get("mask_compatibility", "")
    if isinstance(raw_compatibility, bool):
        value["mask_compatibility"] = (
            "compatible" if raw_compatibility else "incompatible"
        )
    raw_dependencies = value.get("outside_dependencies", "")
    if isinstance(raw_dependencies, list):
        value["outside_dependencies"] = (
            ", ".join(ascii_text(item) for item in raw_dependencies) or "none"
        )
    for key in ("masked_content", "edit_unit_status", "outside_dependencies", "refer_object", "mask_compatibility", "compatibility_reason"):
        text = remove_panel_location(ascii_text(value.get(key, "")))
        if key == "compatibility_reason" and not text:
            normalized[key] = ""
            continue
        if not text or not text.isascii():
            return None
        if key in {"masked_content", "refer_object"} and FORBIDDEN_OUTPUT_PATTERN.search(text):
            return None
        normalized[key] = text

    compatibility = normalized["mask_compatibility"].lower()
    edit_unit_status = normalized["edit_unit_status"].lower()
    if edit_unit_status not in {"complete_object", "complete_part", "incomplete"}:
        return None
    normalized["edit_unit_status"] = edit_unit_status
    if compatibility not in {"compatible", "incompatible"}:
        return None
    normalized["mask_compatibility"] = compatibility
    outside_dependencies = normalized["outside_dependencies"].strip().lower()
    has_outside_dependencies = outside_dependencies not in {
        "none",
        "no",
        "nothing",
        "n/a",
        "null",
    }
    if compatibility == "incompatible":
        if not normalized["compatibility_reason"]:
            return None
        if FALSE_INCOMPATIBILITY_REASON_PATTERN.search(
            normalized["compatibility_reason"]
        ):
            return None
        normalized["editing_instruction"] = ""
        normalized["new_instruction"] = ""
        return normalized
    if task_type in {"remove", "replace"} and edit_unit_status != "complete_object":
        return None
    if task_type in {"remove", "replace"} and PARTIAL_EDIT_UNIT_PATTERN.search(
        normalized["masked_content"]
    ):
        return None
    if task_type in {"remove", "replace"} and has_outside_dependencies:
        return None
    if not 2 <= len(normalized["refer_object"].split()) <= 10:
        return None
    masked_tokens = {
        token
        for token in re.findall(r"[a-z]+", normalized["masked_content"].lower())
        if token not in SUBJECT_STOPWORDS and len(token) >= 3
    }
    refer_tokens = {
        token
        for token in re.findall(r"[a-z]+", normalized["refer_object"].lower())
        if token not in SUBJECT_STOPWORDS and len(token) >= 3
    }
    if not masked_tokens or not refer_tokens or not (masked_tokens & refer_tokens):
        return None

    for key in ("editing_instruction", "new_instruction"):
        text = remove_panel_location(ascii_text(value.get(key, "")))
        if not text or not text.isascii() or FORBIDDEN_OUTPUT_PATTERN.search(text):
            return None
        normalized[key] = text
    full_words = len(normalized["editing_instruction"].split())
    regional_words = len(normalized["new_instruction"].split())
    if not 4 <= full_words <= 22 or not 3 <= regional_words <= 16:
        return None
    if any(
        GENERIC_INSTRUCTION_PATTERN.search(normalized[key])
        for key in ("editing_instruction", "new_instruction")
    ):
        return None
    if PRESERVATION_BOILERPLATE_PATTERN.search(normalized["editing_instruction"]):
        return None
    instruction_tokens = set(
        re.findall(r"[a-z]+", normalized["editing_instruction"].lower())
    )
    if not refer_tokens or not (refer_tokens & instruction_tokens):
        return None
    if task_type == "replace" and not re.search(
        r"\breplace\b.+\bwith\b", normalized["editing_instruction"], re.IGNORECASE
    ):
        return None
    if task_type == "replace":
        lowered_instruction = normalized["editing_instruction"].lower()
        if re.search(r"\breplace\b.+\bto\b", lowered_instruction):
            return None
        article_markers = list(
            re.finditer(r"\bwith\s+(?:a|an|the)\s+", lowered_instruction)
        )
        if article_markers:
            replacement_text = lowered_instruction[article_markers[-1].end() :]
        else:
            replacement_text = lowered_instruction.rsplit(" with ", 1)[-1]
        replacement_tokens = {
            token
            for token in re.findall(r"[a-z]+", replacement_text)
            if len(token) >= 3
        }
        novel_replacement_tokens = (
            replacement_tokens - refer_tokens - REPLACEMENT_GENERIC_WORDS
        )
        if (
            not replacement_tokens
            or "one" in re.findall(r"[a-z]+", replacement_text)
            or not novel_replacement_tokens
        ):
            return None
        refer_human_identity = refer_tokens & HUMAN_IDENTITY_TERMS
        replacement_human_identity = replacement_tokens & HUMAN_IDENTITY_TERMS
        if (
            refer_human_identity
            and replacement_human_identity
            and not (replacement_human_identity - refer_human_identity)
        ):
            # Replacing a person with the same person-category plus different
            # clothes or a held item is an attribute edit, not identity change.
            return None
    if not TYPE_ACTION_PATTERNS[task_type].search(normalized["editing_instruction"]):
        return None
    if not TYPE_ACTION_PATTERNS[task_type].search(normalized["new_instruction"]):
        return None
    if task_type == "attribute" and any(
        LEADING_CROSS_TYPE_ACTION.search(normalized[key])
        for key in ("editing_instruction", "new_instruction")
    ):
        return None
    return normalized


def deterministic_fallback(raw: str, task_type: str) -> dict[str, str] | None:
    """Rewrite a specific regional instruction when the model copies a rule.

    This fallback never invents a new edit. It retains the model's concrete
    target expression and short regional instruction without adding scope.
    """
    value = parse_json_object(raw)
    if not value:
        return None
    masked_content = remove_panel_location(
        str(value.get("masked_content", value.get("refer_object", ""))).strip()
    )
    refer_object = remove_panel_location(str(value.get("refer_object", "")).strip())
    edit_unit_status = str(value.get("edit_unit_status", "complete")).strip().lower()
    outside_dependencies = remove_panel_location(
        str(value.get("outside_dependencies", "none")).strip()
    )
    compatibility = str(value.get("mask_compatibility", "compatible")).strip().lower()
    compatibility_reason = remove_panel_location(
        str(value.get("compatibility_reason", "")).strip()
    )
    if compatibility == "incompatible":
        return normalize_generated(
            {
                "masked_content": masked_content,
                "edit_unit_status": edit_unit_status,
                "outside_dependencies": outside_dependencies,
                "refer_object": refer_object,
                "mask_compatibility": compatibility,
                "compatibility_reason": compatibility_reason,
                "editing_instruction": "",
                "new_instruction": "",
            },
            task_type,
        )
    editing_instruction = remove_panel_location(
        str(value.get("editing_instruction", "")).strip()
    )
    new_instruction = remove_panel_location(
        str(value.get("new_instruction", "")).strip()
    ).rstrip(".")
    if (
        not masked_content
        or not refer_object
        or not new_instruction
        or not refer_object.isascii()
        or not new_instruction.isascii()
        or FORBIDDEN_OUTPUT_PATTERN.search(refer_object)
        or FORBIDDEN_OUTPUT_PATTERN.search(new_instruction)
    ):
        return None
    if (
        not editing_instruction
        or not TYPE_ACTION_PATTERNS[task_type].search(editing_instruction)
        or GENERIC_INSTRUCTION_PATTERN.search(editing_instruction)
        or FORBIDDEN_OUTPUT_PATTERN.search(editing_instruction)
    ):
        editing_instruction = f"{new_instruction}."

    if task_type == "attribute" and any(
        LEADING_CROSS_TYPE_ACTION.search(text)
        for text in (editing_instruction, new_instruction)
    ):
        return None

    if not TYPE_ACTION_PATTERNS[task_type].search(new_instruction):
        if task_type == "add":
            new_instruction = f"Add {new_instruction}"
        elif task_type == "remove":
            new_instruction = f"Remove {refer_object} completely"
        elif task_type == "replace":
            new_instruction = f"Replace the target with {new_instruction}"
        else:
            new_instruction = f"Change the target so it is {new_instruction}"

    return normalize_generated(
        {
            "masked_content": masked_content,
            "edit_unit_status": edit_unit_status,
            "outside_dependencies": outside_dependencies,
            "refer_object": refer_object,
            "mask_compatibility": "compatible",
            "compatibility_reason": "",
            "editing_instruction": editing_instruction,
            "new_instruction": new_instruction,
        },
        task_type,
    )


def _take_area_stratified(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Take compatible rows round-robin across the five area strata."""
    queues: dict[int, list[dict[str, Any]]] = {index: [] for index in range(5)}
    for row in sorted(rows, key=lambda value: int(value["parquet_row_index"])):
        queues[int(row["area_stratum"])].append(row)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        progressed = False
        for stratum in range(5):
            if queues[stratum] and len(chosen) < count:
                chosen.append(queues[stratum].pop(0))
                progressed = True
        if not progressed:
            break
    if len(chosen) != count:
        raise RuntimeError(
            f"Only {len(chosen)} compatible rows are available; {count} required"
        )
    return chosen


def select_compatible_source_rows(
    candidate_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    num_cases: int,
) -> list[dict[str, Any]]:
    """Keep whole two-mask sources while preserving exact type/subset balance."""
    tasks_by_row: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_row.setdefault(int(task["row"]["parquet_row_index"]), []).append(task)

    compatible_rows = []
    for row in candidate_rows:
        row_tasks = sorted(
            tasks_by_row[int(row["parquet_row_index"])],
            key=lambda value: int(value["mask_index"]),
        )
        if len(row_tasks) != 2:
            raise AssertionError("Candidate source does not have exactly two planned masks")
        row["task_pair"] = tuple(task["task_type"] for task in row_tasks)
        row["mask_compatibilities"] = [
            task["generated"]["mask_compatibility"] for task in row_tasks
        ]
        if all(value == "compatible" for value in row["mask_compatibilities"]):
            compatible_rows.append(row)

    sources_per_subset = num_cases // 4
    add_remove_gres = (sources_per_subset + 1) // 2
    requested = {
        ("gres", ("add", "remove")): add_remove_gres,
        ("gres", ("replace", "attribute")): sources_per_subset - add_remove_gres,
        ("ver", ("add", "remove")): sources_per_subset - add_remove_gres,
        ("ver", ("replace", "attribute")): add_remove_gres,
    }
    chosen: list[dict[str, Any]] = []
    shortages = []
    for key, count in requested.items():
        subset, task_pair = key
        bucket = [
            row
            for row in compatible_rows
            if row["source_subset"] == subset and row["task_pair"] == task_pair
        ]
        if len(bucket) < count:
            shortages.append(
                {
                    "source_subset": subset,
                    "task_pair": list(task_pair),
                    "required": count,
                    "available": len(bucket),
                }
            )
            continue
        chosen.extend(_take_area_stratified(bucket, count))
    if shortages:
        raise RuntimeError(
            "Candidate over-sampling did not provide enough mask-compatible source "
            f"rows: {json.dumps(shortages, ensure_ascii=False)}. Increase "
            "--candidate-cases."
        )
    chosen.sort(key=lambda row: (row["source_subset"], row["parquet_row_index"]))
    return chosen


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.output_dir / "instruction_target_panels"
    source_dir = args.output_dir / "instruction_sources"
    panel_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    positive_rows = load_jsonl(args.positive_index)
    candidate_cases = args.candidate_cases or args.num_cases * 3
    if candidate_cases < args.num_cases:
        raise ValueError("--candidate-cases must be at least --num-cases")
    candidates = sample_rows(positive_rows, candidate_cases, args.seed)
    candidates.sort(key=lambda row: (row["source_subset"], row["parquet_row_index"]))
    write_jsonl(args.output_dir / "candidate_source_rows.jsonl", candidates)

    read_started = time.perf_counter()
    image_column = pq.read_table(args.parquet, columns=["images"])["images"]
    image_read_seconds = time.perf_counter() - read_started

    tasks: list[dict[str, Any]] = []
    for source_position, row in enumerate(tqdm(candidates, desc="instruction assets")):
        row_index = int(row["parquet_row_index"])
        embedded = image_bytes_from_cell(image_column[row_index])
        with Image.open(io.BytesIO(embedded)) as handle:
            original = handle.convert("RGB")
        canvas_size = qwen_canvas_size(*original.size)
        source = original.resize(canvas_size, Image.Resampling.LANCZOS)
        source_name = f"source_{row['source_subset']}_r{row_index}.jpg"
        source.save(source_dir / source_name, quality=92)
        for mask_index, raw_rle in enumerate(row["masks"]):
            task_type = TASK_TYPES[(source_position * 2 + mask_index) % len(TASK_TYPES)]
            mask = resize_mask(decode_rle(raw_rle), canvas_size)
            target_panel = instruction_target_panel(source, mask)
            target_cutout = isolated_target_image(source, mask)
            panel_name = f"{row['source_subset']}_r{row_index}_m{mask_index}_{task_type}.jpg"
            target_panel.save(panel_dir / panel_name, quality=95)
            tasks.append(
                {
                    "row": row,
                    "mask_index": mask_index,
                    "task_type": task_type,
                    "source": source.copy(),
                    "target_panel": target_panel,
                    "target_cutout": target_cutout,
                    "panel_name": panel_name,
                }
            )

    candidate_type_counts = Counter(task["task_type"] for task in tasks)
    expected_candidate_per_type = candidate_cases // len(TASK_TYPES)
    if any(
        candidate_type_counts[name] != expected_candidate_per_type
        for name in TASK_TYPES
    ):
        raise AssertionError(
            f"Candidate task type assignment is not balanced: {candidate_type_counts}"
        )

    vlm.configure_backend(
        name=args.vlm,
        model_id=args.vlm_model_id,
        device=args.vlm_device,
        dtype=args.vlm_dtype,
    )
    load_started = time.perf_counter()
    backend = vlm.get_backend()
    backend_load_seconds = time.perf_counter() - load_started
    inference_seconds = 0.0
    raw_rows = []
    pending = list(tasks)
    for attempt in range(3):
        if not pending:
            break
        retry_pending = []
        ranges = range(0, len(pending), args.batch_size)
        for start in tqdm(ranges, desc=f"instruction VLM attempt {attempt + 1}"):
            batch = pending[start : start + args.batch_size]
            inference_started = time.perf_counter()
            outputs = backend.chat_batch(
                [
                    instruction_messages(
                        task["source"],
                        task["target_panel"],
                        task["row"],
                        task["task_type"],
                        task.get("previous_response"),
                        task["target_cutout"],
                    )
                    for task in batch
                ],
                max_new_tokens=args.max_new_tokens,
            )
            inference_seconds += time.perf_counter() - inference_started
            for task, raw in zip(batch, outputs):
                generated = normalize_generated(parse_json_object(raw), task["task_type"])
                raw_record = {
                    "parquet_row_index": task["row"]["parquet_row_index"],
                    "mask_index": task["mask_index"],
                    "task_type": task["task_type"],
                    "target_panel": task["panel_name"],
                    "attempt": attempt + 1,
                    "parsed": generated,
                    "raw_response": raw,
                }
                raw_rows.append(raw_record)
                if generated is None:
                    task["previous_response"] = raw
                    retry_pending.append(task)
                else:
                    task["generated"] = generated
        pending = retry_pending
    if pending:
        fallback_pending = []
        for task in pending:
            generated = deterministic_fallback(
                str(task.get("previous_response", "")), task["task_type"]
            )
            raw_rows.append(
                {
                    "parquet_row_index": task["row"]["parquet_row_index"],
                    "mask_index": task["mask_index"],
                    "task_type": task["task_type"],
                    "target_panel": task["panel_name"],
                    "attempt": "deterministic_fallback",
                    "parsed": generated,
                    "raw_response": task.get("previous_response"),
                }
            )
            if generated is None:
                fallback_pending.append(task)
            else:
                task["generated"] = generated
        pending = fallback_pending
    vlm.shutdown_backend()
    write_jsonl(args.output_dir / "instruction_responses.jsonl", raw_rows)
    validation_failure_count = len(pending)
    if pending:
        failures = [
            {
                "parquet_row_index": task["row"]["parquet_row_index"],
                "mask_index": task["mask_index"],
                "task_type": task["task_type"],
                "raw_response": task.get("previous_response"),
            }
            for task in pending
        ]
        write_jsonl(args.output_dir / "instruction_failures.jsonl", failures)
        # These are candidate-level failures, not batch-level failures. Mark
        # them incompatible so over-sampling can still fill the requested
        # balanced output without admitting an unvalidated instruction.
        for task in pending:
            task["generated"] = {
                "masked_content": "unresolved candidate",
                "edit_unit_status": "incomplete",
                "outside_dependencies": "unknown",
                "refer_object": "unresolved candidate",
                "mask_compatibility": "incompatible",
                "compatibility_reason": "instruction response failed validation",
                "editing_instruction": "",
                "new_instruction": "",
            }
        pending = []
    else:
        # A successful rerun must not leave an obsolete failure manifest from
        # an earlier attempt in the same output directory.
        (args.output_dir / "instruction_failures.jsonl").unlink(missing_ok=True)

    compatible_candidate_cases = sum(
        task["generated"]["mask_compatibility"] == "compatible" for task in tasks
    )
    incompatible_by_type = Counter(
        task["task_type"]
        for task in tasks
        if task["generated"]["mask_compatibility"] == "incompatible"
    )
    selected = select_compatible_source_rows(candidates, tasks, args.num_cases)
    selected_ids = {int(row["parquet_row_index"]) for row in selected}
    tasks = [
        task
        for task in tasks
        if int(task["row"]["parquet_row_index"]) in selected_ids
    ]
    write_jsonl(args.output_dir / "sampled_source_rows.jsonl", selected)
    type_counts = Counter(task["task_type"] for task in tasks)
    expected_per_type = args.num_cases // len(TASK_TYPES)
    if len(tasks) != args.num_cases or any(
        type_counts[name] != expected_per_type for name in TASK_TYPES
    ):
        raise AssertionError(f"Selected task types are not balanced: {type_counts}")

    tasks_by_row: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_row.setdefault(int(task["row"]["parquet_row_index"]), []).append(task)
    plan = []
    for row in selected:
        row_index = int(row["parquet_row_index"])
        edits = []
        for task in sorted(tasks_by_row[row_index], key=lambda value: value["mask_index"]):
            edits.append(
                {
                    "mask_index": task["mask_index"],
                    "name": f"auto_{task['task_type']}",
                    "task_type": task["task_type"],
                    "planning_visual_input": PLANNING_VISUAL_INPUT_VERSION,
                    **task["generated"],
                }
            )
        plan.append(
            {
                "parquet_row_index": row_index,
                "name": "auto",
                "area_stratum": row["area_stratum"],
                "mask_area_fractions": row["mask_area_fractions"],
                "edits": edits,
            }
        )
    plan_path = args.output_dir / "generated_plan.jsonl"
    write_jsonl(plan_path, plan)
    elapsed = time.perf_counter() - started
    summary = {
        "seed": args.seed,
        "source_rows": len(selected),
        "cases": len(tasks),
        "candidate_source_rows": len(candidates),
        "candidate_cases": candidate_cases,
        "compatible_candidate_cases": compatible_candidate_cases,
        "validation_failed_candidate_cases": validation_failure_count,
        "incompatible_candidate_counts_by_type": dict(
            sorted(incompatible_by_type.items())
        ),
        "source_subset_counts": dict(Counter(row["source_subset"] for row in selected)),
        "task_type_counts": dict(sorted(type_counts.items())),
        "candidate_task_type_counts": dict(sorted(candidate_type_counts.items())),
        "area_stratum_counts": dict(sorted(Counter(row["area_stratum"] for row in selected).items())),
        "image_column_read_seconds": round(image_read_seconds, 3),
        "backend_load_seconds": round(backend_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "inference_cases_per_minute": round(candidate_cases / inference_seconds * 60.0, 3),
        "wall_seconds": round(elapsed, 3),
        "plan_jsonl": str(plan_path),
    }
    (args.output_dir / "instruction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
