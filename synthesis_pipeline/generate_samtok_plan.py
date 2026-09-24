"""Sample multi-mask SAMTok rows and generate per-region edit instructions.

This is a pilot sampler, not a production eligibility filter. It samples an
equal number of two-mask GRES and VER rows so that every selected source can be
reused twice and the requested edit-case count is exact. Within each subset the
rows are stratified by minimum region area to retain both tiny/hard and larger
targets. Qwen3-VL sees the clean source plus one outlined photographic crop
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

import cv2
import numpy as np
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
    instruction_target_crop,
)
import utils.vlm_utils as vlm


TASK_TYPES = ("add", "remove", "replace", "attribute")
PLANNING_VISUAL_INPUT_VERSION = "full_source_bound_region_v13_scope"
LARGE_HOLE_MIN_PIXELS = 128
LARGE_HOLE_MIN_FRACTION = 0.03
FORBIDDEN_OUTPUT_PATTERN = re.compile(
    r"\b(mask|overlay|annotations?|highlighted|bbox|coordinates?|target region|"
    r"source crop|(?:full|clean) source image|source image|image [123]|"
    r"(?:from|in) the image)\w*\b|"
    r"\b(?:localization|mask|target|left|middle|right|top|bottom|upper|lower|first|"
    r"second|third|three-part)\s+panels?\b|\bpanels?\s+(?:view|image)\b|%",
    flags=re.IGNORECASE,
)
TYPE_GUIDANCE = {
    "add": (
        "Use the outlined content as the placement anchor. Name one concrete, "
        "scene-appropriate new item that is absent from the clean source, and say "
        "where to add it relative to the uniquely identified anchor. The added "
        "item should be visible at full-image scale. A direction for the new item "
        "cannot stand in for the anchor's own locator relative to the scene. "
        "Use an existing visible surface or attachment point on this anchor; "
        "state the contact relation. Do not introduce a new hand, holder, support "
        "or unrelated foreground object. Keep the addition smaller than its anchor. "
        "Verify that the named attachment surface actually exists and is visible, "
        "belongs to this instance, and lies inside the outline. Do not infer "
        "an occluded body part or garment feature from its category. Locate a "
        "body part by its visible action or relation, not anatomical left/right. "
        "Do not add another copy of "
        "the anchor. If no coherent addition can be anchored here, mark incompatible."
    ),
    "remove": (
        "Identify one independently removable physical object. If the current "
        "outline contains a coherent part but omits a clearly visible attached "
        "structural component of that SAME object, list it in structural_parts "
        "and use complete refinement; downstream segmentation must confirm the "
        "whole unit before editing. Mark the current extent complete_part, not "
        "complete_object. Do not use this exception for an arbitrary fragment, "
        "an independent held object, or a different instance. If removing the "
        "object would leave an independent dependent object implausibly suspended, "
        "mark incompatible. "
        "Before accepting, inspect every held, worn or carried item and every "
        "inner mask hole. Items outside the selected pixels are not removed by "
        "this operation. Also inspect contact with other people: do not approve "
        "an edit that would leave their limb abruptly truncated or unsupported. "
        "Trace thin protrusions to their ends. Being physically associated with "
        "an object does not put them inside its outline. An independent accessory "
        "that must disappear together but lies outside makes this task incompatible; "
        "do not assert full coverage or disguise it as a structural component. "
        "Ask simply to remove the uniquely identified object, including its "
        "structural component when needed for clarity. Do not "
        "describe reconstruction or authorize independent neighboring changes."
    ),
    "replace": (
        "The outline must contain one complete physical object with no missing "
        "dependent object outside it, including in an inner outlined hole. "
        "Inspect thin structural extensions of that SAME object, not just its "
        "main mass. If the outline omits such attached components, list them "
        "separately in structural_parts, mark complete_part and use complete "
        "refinement; downstream segmentation must confirm their attachment. "
        "This exception does not include independent supported/held objects "
        "or another instance. Otherwise mark incompatible. The instruction must explicitly say "
        "Replace [uniquely identified old object] with [concrete new object]; "
        "never stop after naming the old object. Choose a photorealistic, "
        "scene-plausible replacement of roughly compatible apparent size that can "
        "physically occupy the same support and setting. A change "
        "only to its color, material, pattern, text, or styling is insufficient. "
        "Do not request changes outside the outline."
        " Do not invent an interaction with another object or a new supporting "
        "entity; choose something feasible in the existing space. Inspect the "
        "old object's actual contacts before choosing the replacement. If an "
        "independent held or carried item is outside the outline, do not claim "
        "there are no dependencies merely because it will be protected. This "
        "pipeline cannot repair that external interaction; mark incompatible."
    ),
    "attribute": (
        "Change one conspicuous visual property of the outlined content while "
        "retaining its identity and shape. Name the exact desired property value, "
        "not merely 'different'. The change must remain visible at full-image "
        "scale and fit entirely within the outline; otherwise mark incompatible."
        " Choose a clearly contrasting change, not 'slightly' brighter/darker. "
        "An object carried by a person is not part of the person's mask. "
        "Prefer a visible surface property, not a new pose or shape."
        " Identify the material or visible surface to change. If the selected "
        "extent mixes body parts or materials, choose one clearly identifiable "
        "surface within it and request surface refinement. Do not recolor a "
        "whole person, head-and-shoulder silhouette, or a framed reflective object "
        "as one solid shape. Preserve meaningful surface structure; an optical "
        "property applies to the optical surface, not its decorative surround."
        " Name the smallest complete visible part actually selected, not its "
        "larger owner or the whole repeated group. A narrow strip is not every "
        "surface of the structure. If the extent contains different functional "
        "components, choose one material and use surface refinement; do not "
        "treat an object's front silhouette as one paintable sheet."
    ),
}

PROMPT = """Design one localized image-editing training example.

IMAGE 1 is the clean, unmarked full source. IMAGE 2 is a magnified crop of its original photographic pixels with surrounding context retained. The thin black/white outline in IMAGE 2 marks the exact existing mask; only pixels INSIDE it are the selected content. Inner outlined holes are outside the mask. The label and outline are annotations, not colors or objects in the scene. Use the full source to understand other instances and spatial relations.
<<<GEOMETRY>>>
<<<REFERENCE>>>
Trace the WHOLE selected extent, including portions touching the photo boundary. Do not select the narrow object alongside a large outlined surface or an object in a hole. Geometry describes membership, not object identity. If uncertain which side is selected, mark incompatible.
Ground the edit before writing it: inspect the boundary and holes, identify only the included object or part, then check its real contacts and visible placement surfaces. The dataset reference may name a larger owner or group; it does not override the exact outline. Do not choose an appealing instruction first and rationalize the coverage afterward.
Required edit type: <<<TASK_TYPE>>>

Rules for this edit type:
<<<TYPE_GUIDANCE>>>

Output requirements:
- `masked_content`: inventory the full photographic extent inside the outline, not just its most salient tip. Do not include adjacent or attached content outside it. Derive color and identity from photographic pixels, never from the outline or label.
- If the outlined pixels really contain multiple independent objects, mark incompatible rather than treating their group as one editable instance.
- `edit_unit_status`: `complete_object` for a whole independent object, `complete_part` for a coherent local part, or `incomplete` for a fragment or mixed extent.
- `outside_dependencies`: name any visible object supported or carried BY the selected content but outside the outline, or write `none`. A surface supporting the selected content is not its dependency. An object held by a DIFFERENT person is independent, not a dependency, and must not enter the target description.
- `refer_object`: use 3-14 words naming the selected object/surface and the shortest sufficient full-image locator. Mentally hide the outline and search the WHOLE image for other objects matching this phrase. If more than one matches, include an ordinal within a clearly named row/group or a stable visible neighbor relation. A broad left/right/middle label is insufficient when that area contains several similar instances. Do not use the proposed new appearance to locate an existing instance. For a surface task name the surface and its owner/location, not a whole person. Repeat this short reference in editing_instruction.
- `mask_compatibility`: `compatible` only when this edit can be done coherently for this exact content and extent; otherwise `incompatible`. In `compatibility_reason`, briefly state the actual visible boundary/part or contact that makes the task feasible, or the specific missing coverage/dependency that makes it infeasible. Do not merely assert that the mask is suitable or complete.
- If compatible, `editing_instruction` must be one direct 4-24-word command containing the short refer_object and the result. No second paraphrase is needed. Avoid explanations, background recipes, and default-preservation clauses.
- Instruction text must stand alone on IMAGE 1: no mask, contour, label, crop, image number, panel, bbox, or coordinates. Use ASCII English.
- Use left/right for full-image instance locations, not anatomical left/right body parts. Prefer the part's visible action or neighboring landmark; if that still leaves two plausible targets, choose another visible edit location or mark incompatible. Do not invent an unseen part to satisfy the requested type.
- `segmentation_target`: 1-6 words naming the EXISTING object/surface in IMAGE 1. For a new addition, this is the existing anchor; for substitution, it is the OLD object. NEVER put a proposed new object here. No positions or scene descriptions. Downstream segmentation runs on the source photograph.
- `mask_refinement`: <<<MASK_POLICY>>> Refinement cannot authorize changing an independent neighboring object.
- `protected_objects`: at most four plain category names of 1-3 words, for visible neighboring instances. No positions, colors, actions or long descriptions. These are segmentation queries, not instructions.
<<<STRUCTURAL_PARTS>>>

Return exactly one JSON object with these keys and no markdown: `masked_content`, `edit_unit_status`, `outside_dependencies`, `refer_object`, `mask_compatibility`, `compatibility_reason`, `editing_instruction`, `segmentation_target`, `mask_refinement`, `protected_objects`<<<PARTS_KEY>>>. For an incompatible case, leave editing_instruction empty. Keep masked_content a short factual inventory, not a narrative.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--positive-index", type=Path, required=True)
    parser.add_argument(
        "--exclude-annotations", type=Path, action="append", default=[],
        help="Exclude source rows already used by these annotation JSONL files (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument(
        "--candidate-cases",
        type=int,
        default=None,
        help=(
            "Number of region candidates to plan before mask-compatibility filtering. "
            "Defaults to five times --num-cases; use a larger multiple of 20 if needed."
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


def largest_internal_hole(mask: np.ndarray) -> tuple[int, float]:
    """Measure enclosed uneditable islands within a target mask."""
    binary = mask.astype(bool)
    if not binary.any():
        return 0, 0.0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~binary).astype(np.uint8), connectivity=8
    )
    exterior = (
        set(labels[0, :])
        | set(labels[-1, :])
        | set(labels[:, 0])
        | set(labels[:, -1])
    )
    largest = max(
        (
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, len(stats))
            if index not in exterior
        ),
        default=0,
    )
    return largest, largest / int(binary.sum())


def has_central_panel_seam(source: Image.Image) -> bool:
    """Conservatively identify side-by-side montages with a black divider."""
    pixels = np.asarray(source.convert("RGB"), dtype=np.uint8)
    dark_fraction = (pixels.max(axis=2) < 48).mean(axis=0)
    width = pixels.shape[1]
    center_band = dark_fraction[int(width * 0.4) : int(width * 0.6)]
    run = 0
    for fraction in center_band:
        run = run + 1 if fraction >= 0.9 else 0
        if run >= 2:
            return True
    return False


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
        PROMPT.replace("<<<TASK_TYPE>>>", task_type)
        .replace("<<<TYPE_GUIDANCE>>>", TYPE_GUIDANCE[task_type])
        .replace("<<<GEOMETRY>>>", row.get("mask_geometry_hint", ""))
        .replace("<<<REFERENCE>>>", source_reference_hint(row))
        .replace('<<<MASK_POLICY>>>', {
            'attribute':'Use `original` for the same extent or `surface` for a smaller coherent material surface inside it. `complete` is not allowed for this task.',
            'add':'Use `original`; this is the placement anchor, not a new object mask.',
            'remove':'Use `complete` to verify the full physical removal unit including its structural parts. Unresolved segmentation blocks generation.',
            'replace':'Use `complete` to verify the full old object footprint. Unresolved segmentation blocks generation.',
        }[task_type])
        .replace('<<<STRUCTURAL_PARTS>>>', '- `structural_parts`: separately localized short noun phrases for visible structural components belonging to the same old object but outside the current outline; at most three, or an empty list. Independent objects are never structural parts. Inspect narrow extensions and interior holes before claiming the outline is complete.' if task_type in {'remove','replace'} else '')
        .replace('<<<PARTS_KEY>>>', ', `structural_parts`' if task_type in {'remove','replace'} else '')
    )


def source_reference_hint(row: dict[str, Any]) -> str:
    """Recover dataset referring text without SAMTok tokens or guessed mask labels."""
    from synthesis_pipeline.reference_binding import bind_reference, binding_prompt
    binding=row.get('reference_binding')
    if binding is None and 'mask_index' in row and ('num_masks' in row or 'masks' in row):
        binding=bind_reference(row.get('answer',''),int(row['mask_index']),int(row.get('num_masks',len(row.get('masks',[])))))
    if binding:
        prompt=binding_prompt(binding)
        if prompt:
            return prompt
    problem = str(row.get('problem', '')).replace('<image>', '').strip()
    answer = str(row.get('answer', '')).strip()
    parsed = None
    try:
        parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', answer))
    except (ValueError, TypeError):
        pass
    if isinstance(parsed, list):
        answer = '; '.join(str(x.get('label', '')) for x in parsed if isinstance(x, dict))
    answer = re.sub(r'<\|[^>]+\|>', '', answer)
    text = ' '.join((problem + ' ' + answer).split())
    if not text:
        return ''
    return (
        'Dataset referring context (may describe several regions, not all selected): '
        + text[:1800]
        + '\nUse this only as an identity cross-check against the actual outlined pixels. '
        'It does not authorize editing the other listed regions. If annotation and visible '
        'mask extent conflict, do not invent a match: mark incompatible.'
    )


def mask_geometry_hint(mask: np.ndarray) -> str:
    """Give measured membership evidence, never a guessed semantic target."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("Expected a nonempty 2D mask")
    h, w = binary.shape
    ys, xs = np.nonzero(binary)
    cells = []
    for yi, yn in enumerate(("top", "middle", "bottom")):
        for xi, xn in enumerate(("left", "center", "right")):
            cell = binary[round(h * yi / 3):round(h * (yi + 1) / 3),
                          round(w * xi / 3):round(w * (xi + 1) / 3)]
            if cell.size and cell.mean() >= .05:
                cells.append(f"{yn}-{xn} {cell.mean():.0%}")
    edges = [name for name, arr in (("left", binary[:, 0]), ("right", binary[:, -1]),
                                   ("top", binary[0]), ("bottom", binary[-1])) if arr.any()]
    return (
        f"Measured target geometry in IMAGE 1: selected pixels cover {binary.mean():.1%} "
        f"of the FULL photo. Bounding extent: x={xs.min()/w:.0%}..{(xs.max()+1)/w:.0%}, "
        f"y={ys.min()/h:.0%}..{(ys.max()+1)/h:.0%} (left/top = 0%). "
        f"Photo edges touched: {', '.join(edges) or 'none'}. "
        "Selected fraction within occupied cells of the full-photo 3x3 grid: "
        + "; ".join(cells) + ". These coordinates are input evidence only, never instruction text."
    )


def instruction_messages(
    source: Image.Image,
    target_crop: Image.Image,
    row: dict[str, Any],
    task_type: str,
    previous_response: str | None = None,
    validation_feedback: str | None = None,
) -> list[dict[str, Any]]:
    prompt = build_prompt(row, task_type)
    if previous_response is not None:
        prompt += (
            "\n\nPrevious INVALID response (not source evidence; correct it rather than treating it as fact):\n"
            + previous_response[:5000]
            + "\nYour previous JSON failed validation. "
            + (f"Specific issue: {validation_feedback} " if validation_feedback else "")
            + "Do not repeat the same JSON. Reinspect IMAGE 2: inventory "
            "only the photographic content inside its outline. Give refer_object a "
            "short, unique locator from IMAGE 1, then retain its distinguishing "
            "position and landmark in editing_instruction. Keep the full "
            "instruction to 4-24 words; no separate paraphrase is needed. Follow the stated edit-type rule; "
            "do not choose a different object. If this exact edit is genuinely "
            "infeasible, mark incompatible, but do not reject it merely because "
            "your earlier wording was invalid. Do not mention input annotations."
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source},
                {"type": "image", "image": target_crop},
                {"type": "text", "text": prompt},
            ],
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
    r"\b(?:new|different|another|similar) (?:model|person|object|item|thing|one)\b|"
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
    r"^(?:a |an |the )?\w+'s "
    r"(?:(?:left|right|upper|lower)\s+)*"
    r"(?:back|head|neck|arm|hand|leg|foot|tail|paw|wing|torso)\b|"
    r"^(?:a |an |the )?(?:back|head|wing|paw|tail|torso|body)\s+of\s+"
    r"(?:a |an |the )?\w+\b",
    flags=re.IGNORECASE,
)
INDEPENDENT_OUTSIDE_PATTERN = re.compile(
    r"\b(?:held|carried|worn|supported)\s+by\s+"
    r"(?:(?:a|an|the)\s+)?(?:other|another|different)\s+"
    r"(?:person|man|woman|boy|girl|individual)\b",
    flags=re.IGNORECASE,
)
MULTI_INSTANCE_PATTERN = re.compile(
    r"^(?:a |an |the )?(?:two|three|four|several|multiple|a pair of)\b",
    flags=re.IGNORECASE,
)
GROUP_TARGET_PATTERN = re.compile(
    r"\band\s+(?:(?:a|an|the)\s+)?(?:calf|cub|foal|rider|passenger|"
    r"child|baby|man|woman|person|boy|girl|dog|horse|giraffe|zebra|"
    r"elephant)\b",
    flags=re.IGNORECASE,
)
NON_PHOTOREALISTIC_REPLACEMENT = re.compile(
    r"\b(?:cartoon|anime|animated|illustrated|comic)\b",
    flags=re.IGNORECASE,
)
HUMAN_TO_TOY_PATTERN = re.compile(
    r"\b(?:toy|plush|figurine|doll)\b",
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
LOCATOR_PATTERN = re.compile(
    r"\b(?:left(?:most)?|right(?:most)?|center|central|middle|upper|lower|"
    r"top|bottom|front|rear|back|foreground|background|nearest|closest|"
    r"farther|farthest|beside|near|next to|above|below|beneath|behind|"
    r"between|adjacent|opposite|first|last|handstand|upside[ -]down)\b",
    flags=re.IGNORECASE,
)
STRONG_LOCATOR_PATTERN = re.compile(
    r"\b(?:left(?:most)?|right(?:most)?|center|central|middle|upper|lower|"
    r"top|bottom|front|rear|back|nearest|closest|farther|farthest|beside|near|next to|"
    r"above|below|beneath|under|behind|between|adjacent|opposite|first|last|"
    r"handstand|upside[ -]down)\b",
    flags=re.IGNORECASE,
)
RELATION_ANCHOR_PATTERN = re.compile(
    r"\b(?:left|right)\s+of\b|"
    r"\b(?:near|beside|behind|above|below|beneath|under|opposite)\b|"
    r"\b(?:next|adjacent)\s+to\b",
    flags=re.IGNORECASE,
)
REPLACEMENT_GENERIC_WORDS = SUBJECT_STOPWORDS | {
    "another",
    "clean",
    "colored",
    "colour",
    "different",
    "golden",
    "identical",
    "large",
    "leather",
    "material",
    "metal",
    "metallic",
    "modern",
    "patterned",
    "plastic",
    "wood",
    "wooden",
    "glass",
    "steel",
    "brass",
    "bronze",
    "ceramic",
    "stone",
    "fabric",
    "silver",
    "gold",
    "automated",
    "automatic",
    "foreground",
    "background",
    "upper",
    "lower",
    "center",
    "central",
    "middle",
    "leftmost",
    "rightmost",
    "new",
    "one",
    "object",
    "item",
    "thing",
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
    "young",
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


def contains_exact_reference(instruction: str, refer_object: str) -> bool:
    """Require the full discriminative target phrase, not a shared head noun."""
    instruction_words = re.findall(r"[a-z0-9]+", instruction.lower())
    reference_words = re.findall(r"[a-z0-9]+", refer_object.lower())
    return bool(reference_words) and any(
        instruction_words[start : start + len(reference_words)] == reference_words
        for start in range(len(instruction_words) - len(reference_words) + 1)
    )


def contains_distinctive_reference(instruction: str, refer_object: str) -> bool:
    """Allow a small grammatical omission, never a lost locator or vague noun."""
    if contains_exact_reference(instruction, refer_object):
        return True
    reference_words_all = re.findall(r"[a-z0-9]+", refer_object.lower())
    instruction_words_all = re.findall(r"[a-z0-9]+", instruction.lower())
    # A bare direction word may describe the *new item's placement* rather than
    # the intended source instance. Keep the anchor noun of each relation and
    # require it to occur after the target phrase in the full instruction.
    first_relation = RELATION_ANCHOR_PATTERN.search(refer_object)
    target_prefix = refer_object[: first_relation.start()] if first_relation else refer_object
    target_terms = [
        token for token in re.findall(r"[a-z0-9]+", target_prefix.lower())
        if token not in SUBJECT_STOPWORDS and token not in {"the", "a", "an"}
    ]
    target_position = next(
        (index for index, token in enumerate(instruction_words_all)
         if target_terms and token == target_terms[0]),
        -1,
    )
    for relation in RELATION_ANCHOR_PATTERN.finditer(refer_object):
        following = refer_object[relation.end() :]
        phrase = re.split(
            r"[,.;]|\b(?:with|on|in|at|by|near|beside|behind|above|below|"
            r"beneath|between|opposite|next|adjacent|left|right|standing|"
            r"sitting|holding|wearing)\b",
            following,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        anchor_terms = [
            token for token in re.findall(r"[a-z0-9]+", phrase.lower())
            if token not in {"the", "a", "an"}
        ]
        if not anchor_terms or target_position < 0:
            return False
        if anchor_terms[-1] not in instruction_words_all[target_position + 1 :]:
            return False
    horizontal = list(
        re.finditer(r"\b(?:left(?:most)?|right(?:most)?|center|central|middle)\b", refer_object, re.IGNORECASE)
    )
    locators = horizontal or list(STRONG_LOCATOR_PATTERN.finditer(refer_object))
    if not any(
        re.search(re.escape(locator.group(0)), instruction, re.IGNORECASE)
        for locator in locators
    ):
        return False
    filler = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with"}
    reference_terms = [word for word in reference_words_all if word not in filler]
    instruction_terms = Counter(instruction_words_all)
    overlap = sum((Counter(reference_terms) & instruction_terms).values())
    return bool(reference_terms) and overlap / len(reference_terms) >= 0.5


def protected_dependency_conflicts(value: dict[str, Any], task_type: str) -> list[str]:
    """Catch an explicit contradiction, not infer ownership from proximity.

    The current contract forbids outside dependencies for both removal and
    replacement. A protected held object does not become dependency-free merely
    because a proposed replacement might be able to hold it. Attribute edits
    retain the owner and therefore do not trigger this guard.
    """
    if task_type not in {'remove', 'replace'} or value.get('edit_unit_status') != 'complete_object':
        return []
    inventory = str(value.get('masked_content', '')).lower()
    clauses = []
    for match in re.finditer(r'\b(?:holding|carrying|wearing|supporting)\s+([^.;]+)', inventory):
        # Do not absorb the description of a separate neighboring instance.
        clause = re.split(r'\b(?:beside|near|next to|behind|in front of|while|who)\b', match.group(1))[0]
        clauses.append(set(re.findall(r'[a-z]+', clause)))
    conflicts = []
    for query in value.get('protected_objects', []) or []:
        if not isinstance(query, str):
            continue
        tokens = set(re.findall(r'[a-z]+', query.lower())) - SUBJECT_STOPWORDS
        if tokens and any(tokens <= clause for clause in clauses):
            conflicts.append(query)
    return conflicts


def ambiguous_anatomical_side(value: dict[str, Any]) -> bool:
    """Require observable references instead of unchecked anatomical laterality.

    This is a wording policy, not a claim to infer body pose from text. Scene
    locators such as 'the man on the left' remain valid.
    """
    return bool(re.search(
        r'\b(?:left|right)[ -]+(?:wrist|hand|arm|elbow|shoulder|knee|ankle|foot|leg|ear|eye)\b',
        str(value.get('editing_instruction', '')), re.I))


def validation_feedback(value: dict[str, Any] | None, task_type: str) -> str:
    """Give the same VLM a concrete correction instead of a generic retry."""
    if not value:
        return "Return one complete JSON object with the requested keys."
    value=dict(value)
    value.setdefault('new_instruction',value.get('editing_instruction',''))
    for key in (
        "masked_content", "edit_unit_status", "outside_dependencies",
        "refer_object", "mask_compatibility",
    ):
        field = ascii_text(value.get(key, ""))
        if not field:
            return f"Fill the missing {key} field."
        if not field.isascii():
            return f"Rewrite {key} using ASCII English only."
        if key in {"masked_content", "refer_object"} and has_annotation_language(field):
            return f"Remove annotation or panel language from {key}; use only scene facts."
    status = str(value["edit_unit_status"]).lower().strip()
    dependencies = str(value["outside_dependencies"]).lower().strip()
    compatibility = str(value["mask_compatibility"]).lower().strip()
    if compatibility == "incompatible":
        if not ascii_text(value.get("compatibility_reason", "")):
            return "Explain why this exact mask cannot support the edit in compatibility_reason."
        return "Ensure the incompatibility reason describes a real mask or scene limitation."
    conflicts = protected_dependency_conflicts(value, task_type)
    if conflicts:
        return ("Your masked_content says the old target holds/carries/wears "
                + ', '.join(conflicts) + ", but protected_objects freezes those items. "
                "Recheck the images and mask holes: do not leave unsupported belongings, "
                "do not silently remove the protection to expand the mask. If they lie "
                "outside this edit unit, mark incompatible and explain the dependency.")
    if ambiguous_anatomical_side(value):
        return ("Do not guess anatomical left/right from screen position. Reinspect "
                "the visible part and locate it using its action or a visible landmark, "
                "while retaining the owner's unique full-image locator. If no unique "
                "visible attachment point exists, mark incompatible.")
    if 'segmentation_target' in value and value.get('mask_refinement') not in allowed_mask_policies(task_type):
        return f"For {task_type}, mask_refinement must be one of {sorted(allowed_mask_policies(task_type))}; choose a concrete coherent surface when the original extent mixes materials."
    if '%' in str(value.get('editing_instruction','')):
        return 'Use a natural spatial description, not numeric coordinates or percentages, in the public instruction.'
    if MULTI_INSTANCE_PATTERN.search(str(value["masked_content"])):
        return "The outlined content contains multiple objects; mark it incompatible rather than editing the group as one instance."
    if task_type in {"remove", "replace"} and GROUP_TARGET_PATTERN.search(
        " ".join(str(value.get(key, "")) for key in ("refer_object", "editing_instruction"))
    ):
        return "The instruction names another dependent instance outside the mask; target only the complete outlined object or mark incompatible."
    if task_type in {"remove", "replace"} and (
        status != "complete_object"
        or dependencies not in {"none", "no", "nothing", "n/a", "null"}
    ):
        return "This edit needs a complete standalone object with no outside dependency; mark genuinely incomplete masks incompatible."
    if task_type in {"remove", "replace"} and PARTIAL_EDIT_UNIT_PATTERN.search(str(value["masked_content"])):
        return "The outlined content is only a body or object part; mark this edit incompatible instead of replacing a whole identity."
    refer = remove_panel_location(ascii_text(value["refer_object"]))
    refer_words = len(refer.split())
    if not 2 <= refer_words <= 14:
        return f"refer_object has {refer_words} words; shorten it to 2-14 words while keeping a unique full-image locator."
    if not STRONG_LOCATOR_PATTERN.search(refer):
        return "refer_object lacks a decisive full-image position or landmark; include one and carry it into the full instruction."
    for key in ("editing_instruction", "new_instruction"):
        field = ascii_text(value.get(key, ""))
        if not field:
            return f"Fill {key} with the required edit and result."
        if not field.isascii():
            return f"Rewrite {key} using ASCII English only."
        if has_annotation_language(field):
            return f"Remove annotation or panel language from {key}; use clean-source directions only."
    full = ascii_text(value["editing_instruction"])
    regional = ascii_text(value["new_instruction"])
    if not 4 <= len(full.split()) <= 24:
        return "Keep editing_instruction to 4-24 words without losing the target locator."
    if not 3 <= len(regional.split()) <= 24:
        return "Keep the instruction to at most 24 words while retaining the same edit."
    if not contains_distinctive_reference(full, refer):
        if task_type == "add" and RELATION_ANCHOR_PATTERN.search(refer):
            return (
                f"In editing_instruction, repeat the anchor's refer_object phrase "
                f"'{refer}' immediately after naming the new item; a direction "
                "for placing the new item does not locate the anchor."
            )
        horizontal = re.search(
            r"\b(?:left(?:most)?|right(?:most)?|center|central|middle)\b",
            refer, re.IGNORECASE,
        )
        locator = horizontal or STRONG_LOCATOR_PATTERN.search(refer)
        if locator and not re.search(re.escape(locator.group(0)), full, re.IGNORECASE):
            return (
                f"Keep refer_object concise and put its '{locator.group(0)}' "
                "locator in the target phrase of editing_instruction; do not "
                "expand refer_object instead."
            )
        return "Keep the same masked object and its distinguishing landmark in editing_instruction; do not target a nearby object."
    if task_type == "add" and adds_same_category(full, refer):
        return "The requested new item repeats the already-masked target category; choose a different small addition anchored to this instance."
    if task_type == "attribute" and adds_unmasked_wearable(full, str(value["masked_content"])):
        return "The requested wearable is absent from the masked content; change an existing visible property instead."
    if task_type == 'attribute' and unsafe_surface_attribute(full):
        return 'Name the actual material surface to edit; do not paint a mixed human silhouette or erase a decorative surround. Use surface refinement for a coherent subpart.'
    if task_type == "replace" and not re.search(r"\breplace\b.+\bwith\b", full, re.IGNORECASE):
        return "Name a concrete new object after 'with' in both replacement instructions."
    if task_type == "replace":
        refer_tokens = set(re.findall(r"[a-z]+", refer.lower()))
        replacement_text = replacement_text_from_instruction(full)
        replacement_tokens = set(re.findall(r"[a-z]+", replacement_text))
        if not (replacement_tokens - refer_tokens - REPLACEMENT_GENERIC_WORDS):
            return "The instruction names only the old target; put a concrete new object after the final 'with'."
        source_phrase = re.split(r'\bwith\b', full.lower())[-2] if ' with ' in full.lower() else refer
        if same_replacement_category(refer, replacement_text) or same_replacement_category(source_phrase, replacement_text):
            return "The proposed replacement keeps the same object category and changes only styling; choose a genuinely different category or identity."
        if NON_PHOTOREALISTIC_REPLACEMENT.search(replacement_text):
            return "Choose a photorealistic replacement that belongs naturally in this scene."
        if (
            set(re.findall(r"[a-z]+", refer.lower())) & HUMAN_IDENTITY_TERMS
            and HUMAN_TO_TOY_PATTERN.search(replacement_text)
        ):
            return "Replacing a full-size person with a toy is implausible here; choose a scene-plausible, similarly sized object."
    if GENERIC_INSTRUCTION_PATTERN.search(full) or GENERIC_INSTRUCTION_PATTERN.search(regional):
        return "Name the concrete visual result, not a generic item or unspecified change."
    if PRESERVATION_BOILERPLATE_PATTERN.search(full):
        return "Delete default-preservation clauses and keep only the requested edit."
    return "Follow the task-specific action and name a concrete visible result for the same masked target."


def _singular_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def adds_same_category(instruction: str, refer_object: str) -> bool:
    """Reject a second copy of the masked object in an add instruction."""
    action = re.search(
        r"\b(?:add|attach|place|put|give|tie|hang|mount|stick|install)\b",
        instruction, re.IGNORECASE,
    )
    if not action:
        return False
    tail = instruction[action.end() :]
    item_phrase = re.split(
        r"\b(?:to|on|near|under|above|beside|behind|between|at|in|against|"
        r"around|next to)\b",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    ignored = REPLACEMENT_GENERIC_WORDS | {
        "a", "an", "the", "birdlike", "glowing", "hexagonal", "perched",
        "hanging", "resting", "attached", "positioned", "patterned",
        "striped", "spotted", "tiny", "vintage", "wooden",
    }
    item_terms = [
        _singular_token(token)
        for token in re.findall(r"[a-z]+", item_phrase.lower())
        if token not in ignored
    ]
    refer_terms = {
        _singular_token(token)
        for token in re.findall(r"[a-z]+", refer_object.lower())
        if token not in ignored
    }
    return bool(item_terms) and item_terms[-1] in refer_terms


def adds_unmasked_wearable(instruction: str, masked_content: str) -> bool:
    """An attribute instruction cannot introduce an absent worn object."""
    match = re.search(r"\bwear\s+(?:a|an|the)\s+([^.,;]+)", instruction, re.IGNORECASE)
    if not match:
        return False
    phrase = re.split(
        r"\b(?:on|with|near|beside|around|under|over|while)\b",
        match.group(1), maxsplit=1, flags=re.IGNORECASE,
    )[0]
    words = re.findall(r"[a-z]+", phrase.lower())
    if not words:
        return False
    wearable = _singular_token(words[-1])
    masked_words = {
        _singular_token(token)
        for token in re.findall(r"[a-z]+", masked_content.lower())
    }
    return wearable not in masked_words


def same_replacement_category(refer_object: str, replacement_text: str) -> bool:
    """Catch a material/style variant of the same head object category."""
    old_segment = re.split(
        r"\b(?:with|on|near|beside|behind|between|in|at|under|above|by|of|holding|carrying|wearing|standing|sitting|facing|located|positioned)\b",
        refer_object,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    old_terms = [
        _singular_token(token)
        for token in re.findall(r"[a-z]+", old_segment.lower())
        if token not in REPLACEMENT_GENERIC_WORDS
    ]
    new_segment = re.split(
        r"\b(?:with|on|near|beside|behind|in|at|under|above|by|of|holding|carrying|wearing|standing|sitting|facing|located|positioned)\b",
        replacement_text, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    new_terms = [
        _singular_token(token)
        for token in re.findall(r"[a-z]+", new_segment.lower())
        if token not in REPLACEMENT_GENERIC_WORDS
    ]
    aliases={'twig':'branch','bough':'branch','stick':'branch'}
    old_terms=[aliases.get(x,x) for x in old_terms]
    new_terms=[aliases.get(x,x) for x in new_terms]
    return bool(old_terms and new_terms) and old_terms[-1] in {new_terms[0], new_terms[-1]}


def replacement_text_from_instruction(instruction: str) -> str:
    lowered = instruction.lower()
    article_markers = list(re.finditer(r"\bwith\s+(?:a|an|the)\s+", lowered))
    if article_markers:
        return lowered[article_markers[-1].end() :]
    return lowered.rsplit(" with ", 1)[-1]


def normalize_generated(
    value: dict[str, Any] | None, task_type: str
) -> dict[str, str] | None:
    if not value:
        return None
    value=dict(value)
    if 'new_instruction' not in value:
        value['new_instruction']=value.get('editing_instruction','')
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
        if key in {"masked_content", "refer_object"} and has_annotation_language(text):
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
    if INDEPENDENT_OUTSIDE_PATTERN.search(normalized["outside_dependencies"]):
        normalized["outside_dependencies"] = "none"
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
    if MULTI_INSTANCE_PATTERN.search(normalized["masked_content"]):
        return None
    if task_type in {"remove", "replace"} and GROUP_TARGET_PATTERN.search(
        " ".join(
            str(value.get(key, ""))
            for key in ("refer_object", "editing_instruction", "new_instruction")
        )
    ):
        return None
    pending_part_completion = (
        task_type in {'remove','replace'} and edit_unit_status == 'complete_part'
        and value.get('mask_refinement') == 'complete'
        and isinstance(value.get('structural_parts'),list)
        and 0 < len(value['structural_parts']) <= 3
        and bool(value.get('segmentation_target'))
    )
    if task_type in {"remove", "replace"} and edit_unit_status != "complete_object" and not pending_part_completion:
        return None
    if task_type in {"remove", "replace"} and not pending_part_completion and PARTIAL_EDIT_UNIT_PATTERN.search(
        normalized["masked_content"]
    ):
        return None
    if task_type in {"remove", "replace"} and has_outside_dependencies:
        return None
    if protected_dependency_conflicts(value, task_type):
        return None
    if ambiguous_anatomical_side(value):
        return None
    if not 2 <= len(normalized["refer_object"].split()) <= 14:
        return None
    if not STRONG_LOCATOR_PATTERN.search(normalized["refer_object"]):
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
        if not text or not text.isascii() or has_annotation_language(text):
            return None
        normalized[key] = text
    full_words = len(normalized["editing_instruction"].split())
    regional_words = len(normalized["new_instruction"].split())
    if not 4 <= full_words <= 24 or not 3 <= regional_words <= 24:
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
    if not contains_distinctive_reference(
        normalized["editing_instruction"], normalized["refer_object"]
    ):
        return None
    if task_type == "add" and adds_same_category(
        normalized["editing_instruction"], normalized["refer_object"]
    ):
        return None
    if task_type == "attribute" and adds_unmasked_wearable(
        normalized["editing_instruction"], normalized["masked_content"]
    ):
        return None
    if task_type == "replace" and not re.search(
        r"\breplace\b.+\bwith\b", normalized["editing_instruction"], re.IGNORECASE
    ):
        return None
    if task_type == "replace":
        lowered_instruction = normalized["editing_instruction"].lower()
        if re.search(r"\breplace\b.+\bto\b", lowered_instruction):
            return None
        replacement_text = replacement_text_from_instruction(lowered_instruction)
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
        source_phrase = re.split(r'\bwith\b', lowered_instruction)[-2]
        if (same_replacement_category(normalized["refer_object"], replacement_text)
                or same_replacement_category(source_phrase.removeprefix('replace '), replacement_text)):
            return None
        if NON_PHOTOREALISTIC_REPLACEMENT.search(replacement_text):
            return None
        if refer_tokens & HUMAN_IDENTITY_TERMS and HUMAN_TO_TOY_PATTERN.search(
            replacement_text
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
    if task_type == 'attribute' and unsafe_surface_attribute(normalized['editing_instruction']):
        return None
    if task_type == 'replace':
        replacement = replacement_text_from_instruction(normalized['editing_instruction'])
        if refer_tokens & HUMAN_IDENTITY_TERMS and re.match(r'(?:a |an |the )?(?:person|human|figure)\b', replacement):
            return None
    # Legacy responses can omit these fields. New planning uses the same call.
    if 'segmentation_target' in value:
        phrase = ascii_text(value['segmentation_target'])
        policy = str(value.get('mask_refinement', 'original')).strip().lower()
        neighbors = value.get('protected_objects', [])
        if not 1 <= len(phrase.split()) <= 8 or policy not in allowed_mask_policies(task_type):
            return None
        if not isinstance(neighbors,list) or len(neighbors)>4 or any(not isinstance(x,str) or not 1<=len(x.split())<=6 for x in neighbors):
            return None
        if policy == 'surface' and task_type != 'attribute':
            return None
        normalized.update(segmentation_target=phrase,mask_refinement=policy,
                          protected_objects=[ascii_text(x) for x in neighbors])
        if 'structural_parts' in value:
            parts=value['structural_parts']
            if not isinstance(parts,list) or len(parts)>3 or any(not isinstance(x,str) or not 1<=len(x.split())<=6 for x in parts):
                return None
            normalized['structural_parts']=list(dict.fromkeys(ascii_text(x) for x in parts))
    return normalized


def allowed_mask_policies(task_type):
    return {'attribute':{'original','surface'},'add':{'original'},
            'remove':{'complete'},'replace':{'complete'}}[task_type]


def has_annotation_language(text):
    # A physical face covering is not a segmentation annotation.
    cleaned=re.sub(r'\b(?:face|surgical|medical|protective) masks?\b','face covering',text,flags=re.I)
    return bool(FORBIDDEN_OUTPUT_PATTERN.search(cleaned))


def unsafe_surface_attribute(instruction: str) -> bool:
    """Reject broad human/optical fills while allowing named real surfaces."""
    text = instruction.lower()
    color = re.search(r'\b(red|blue|green|yellow|purple|white|black|gold|silver|pink)\b', text)
    subject = re.split(r'\b(?:of|held|carried|worn|near|beside|behind|at)\b',text,maxsplit=1)[0]
    if re.search(r'\bhead\b.*\bshoulder',subject):
        return True
    broad_human = re.search(r'\b(person|man|woman|girl|boy|head|shoulder)\b',subject)
    named_surface = re.search(r'\b(hair|shirt|coat|jacket|vest|dress|skin|hat|trousers|pants|sleeve|fabric)\b',text)
    if color and broad_human and not named_surface:
        return True
    return bool(re.search(r'\bmirror\b',text) and re.search(r'opaque|non.reflective|polished|glossy|frosted',text)
                and not re.search(r'\b(glass|pane|surface|frame|border)\b',text))


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
    crop_dir = args.output_dir / "instruction_target_crops"
    source_dir = args.output_dir / "instruction_sources"
    crop_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    positive_rows = load_jsonl(args.positive_index)
    excluded_ids = {
        int(row["parquet_row_index"])
        for path in args.exclude_annotations
        for row in load_jsonl(path)
    }
    positive_rows = [
        row for row in positive_rows
        if int(row["parquet_row_index"]) not in excluded_ids
    ]
    candidate_cases = args.candidate_cases or args.num_cases * 5
    if candidate_cases < args.num_cases:
        raise ValueError("--candidate-cases must be at least --num-cases")
    candidates = sample_rows(positive_rows, candidate_cases, args.seed)
    candidates.sort(key=lambda row: (row["source_subset"], row["parquet_row_index"]))
    write_jsonl(args.output_dir / "candidate_source_rows.jsonl", candidates)

    read_started = time.perf_counter()
    import pyarrow.parquet as pq
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
        source_panel_seam = has_central_panel_seam(source)
        source_name = f"source_{row['source_subset']}_r{row_index}.jpg"
        source.save(source_dir / source_name, quality=92)
        for mask_index, raw_rle in enumerate(row["masks"]):
            task_type = TASK_TYPES[(source_position * 2 + mask_index) % len(TASK_TYPES)]
            mask = resize_mask(decode_rle(raw_rle), canvas_size)
            hole_pixels, hole_fraction = largest_internal_hole(mask)
            target_crop = instruction_target_crop(source, mask)
            crop_name = f"{row['source_subset']}_r{row_index}_m{mask_index}_{task_type}.png"
            target_crop.save(crop_dir / crop_name)
            tasks.append(
                {
                    "row": {**row, "mask_index":mask_index, "mask_geometry_hint": mask_geometry_hint(mask)},
                    "mask_index": mask_index,
                    "task_type": task_type,
                    "source": source.copy(),
                    "target_crop": target_crop,
                    "crop_name": crop_name,
                    "source_panel_seam": source_panel_seam,
                    "largest_hole_pixels": hole_pixels,
                    "largest_hole_fraction": hole_fraction,
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

    raw_rows = []
    pending = []
    geometry_prefiltered_count = 0
    source_layout_prefiltered_count = 0
    for task in tasks:
        rejection_reason = None
        attempt_name = None
        if task["source_panel_seam"]:
            source_layout_prefiltered_count += 1
            rejection_reason = "source is a side-by-side panel montage with a central seam"
            attempt_name = "source_layout_prefilter"
        elif (
            task["task_type"] in {"remove", "replace"}
            and task["largest_hole_pixels"] >= LARGE_HOLE_MIN_PIXELS
            and task["largest_hole_fraction"] >= LARGE_HOLE_MIN_FRACTION
        ):
            geometry_prefiltered_count += 1
            rejection_reason = (
                "large uneditable island inside target mask: "
                f"{task['largest_hole_fraction']:.1%} of masked area"
            )
            attempt_name = "mask_geometry_prefilter"
        if rejection_reason:
            task["generated"] = {
                "masked_content": "not evaluated",
                "edit_unit_status": "incomplete",
                "outside_dependencies": "unknown",
                "refer_object": "not evaluated",
                "mask_compatibility": "incompatible",
                "compatibility_reason": rejection_reason,
                "editing_instruction": "",
                "new_instruction": "",
            }
            raw_rows.append(
                {
                    "parquet_row_index": task["row"]["parquet_row_index"],
                    "mask_index": task["mask_index"],
                    "task_type": task["task_type"],
                    "target_crop": task["crop_name"],
                    "attempt": attempt_name,
                    "parsed": task["generated"],
                    "raw_response": "",
                }
            )
        else:
            pending.append(task)

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
    vlm_request_count = 0
    for attempt in range(3):
        if not pending:
            break
        retry_pending = []
        ranges = range(0, len(pending), args.batch_size)
        for start in tqdm(ranges, desc=f"instruction VLM attempt {attempt + 1}"):
            batch = pending[start : start + args.batch_size]
            vlm_request_count += len(batch)
            inference_started = time.perf_counter()
            outputs = backend.chat_batch(
                [
                    instruction_messages(
                        task["source"],
                        task["target_crop"],
                        task["row"],
                        task["task_type"],
                        task.get("previous_response"),
                        task.get("validation_feedback"),
                    )
                    for task in batch
                ],
                max_new_tokens=args.max_new_tokens,
            )
            inference_seconds += time.perf_counter() - inference_started
            for task, raw in zip(batch, outputs):
                raw_value = parse_json_object(raw)
                generated = normalize_generated(raw_value, task["task_type"])
                feedback = (
                    validation_feedback(raw_value, task["task_type"])
                    if generated is None else None
                )
                raw_record = {
                    "parquet_row_index": task["row"]["parquet_row_index"],
                    "mask_index": task["mask_index"],
                    "task_type": task["task_type"],
                    "target_crop": task["crop_name"],
                    "attempt": attempt + 1,
                    "parsed": generated,
                    "validation_feedback": feedback,
                    "raw_response": raw,
                }
                raw_rows.append(raw_record)
                if generated is None:
                    task["previous_response"] = raw
                    task["validation_feedback"] = feedback
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
                    "target_crop": task["crop_name"],
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
            from synthesis_pipeline.reference_binding import bind_reference
            edits.append(
                {
                    "mask_index": task["mask_index"],
                    "name": f"auto_{task['task_type']}",
                    "task_type": task["task_type"],
                    "planning_visual_input": PLANNING_VISUAL_INPUT_VERSION,
                    "reference_binding":bind_reference(row['answer'],task['mask_index'],row['num_masks']),
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
        "excluded_source_rows": len(excluded_ids),
        "source_rows": len(selected),
        "cases": len(tasks),
        "candidate_source_rows": len(candidates),
        "candidate_cases": candidate_cases,
        "compatible_candidate_cases": compatible_candidate_cases,
        "validation_failed_candidate_cases": validation_failure_count,
        "geometry_prefiltered_candidate_cases": geometry_prefiltered_count,
        "source_layout_prefiltered_candidate_cases": source_layout_prefiltered_count,
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
        "vlm_request_count": vlm_request_count,
        "inference_cases_per_minute": round(
            (
                candidate_cases
                - geometry_prefiltered_count
                - source_layout_prefiltered_count
            ) / inference_seconds * 60.0,
            3,
        ) if inference_seconds else 0.0,
        "inference_requests_per_minute": round(
            vlm_request_count / inference_seconds * 60.0, 3
        ) if inference_seconds else 0.0,
        "wall_seconds": round(elapsed, 3),
        "plan_jsonl": str(plan_path),
    }
    (args.output_dir / "instruction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
