"""Fact-first Qwen3-VL prefilter for CrispEdit-2M.

The current method and production configuration are documented in
``docs/CRISPEDIT_PREFILTER.md``:

0. text-only instruction slot extraction;
1. source-only factual questionnaire;
2. target-only factual questionnaire, optionally with a localized crop;
3. instruction-blind paired comparison;
4. text-only matching followed by deterministic code predicates;
5. a budgeted, independent focused source/target state review for unresolved rows.

The model never emits the final PASS/FAIL/UNSURE verdict.  All intermediate
evidence and predicate truth values are written to the audit parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import queue
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm import tqdm

from crispedit.legacy.pipeline import parse_instruction
from crispedit.prefilter.policy import (
    COLOR_WORDS,
    EVIDENCE_SCHEMA,
    PREFILTER_METHOD,
    adjudicate_evidence,
    adjudicate_terminal_no_change,
    canonical_edit_type,
    derive_state_text_match,
    deterministic_slot_conflict,
    json_dumps,
    normalize_pair_evidence,
    normalize_single_image_evidence,
    normalize_slots,
    normalize_text_match,
)


DEFAULT_MODEL = os.environ.get(
    "CRISPEDIT_QWEN_MODEL_PATH",
    "/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct",
)


@dataclass
class ShardJob:
    raw_type: str
    input_path: str
    audit_path: str
    manifest_path: str
    num_rows: int
    limit_rows: Optional[int]


def parse_devices(spec: str) -> List[str]:
    spec = (spec or "auto").strip().lower()
    if spec == "cpu":
        return ["cpu"]
    if spec == "auto":
        import torch

        count = torch.cuda.device_count()
        return [f"cuda:{index}" for index in range(count)] if count else ["cpu"]
    devices = []
    for item in (part.strip() for part in spec.split(",")):
        if not item:
            continue
        devices.append(item if item.startswith("cuda:") else f"cuda:{int(item)}")
    return devices or ["cpu"]


def raw_type_from_filename(path: Path) -> str:
    match = re.match(r"(.+)_\d+\.parquet$", path.name)
    return match.group(1) if match else path.stem


def iter_input_shards(
    input_dir: Path, include_types: Optional[Sequence[str]]
) -> Dict[str, List[Path]]:
    include = {item.strip() for item in include_types if item.strip()} if include_types else None
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(input_dir.glob("*.parquet")):
        raw_type = raw_type_from_filename(path)
        if include is not None and raw_type not in include:
            continue
        grouped.setdefault(raw_type, []).append(path)
    return grouped


def count_rows(path: Path, limit_rows: Optional[int]) -> int:
    total = pq.ParquetFile(path).metadata.num_rows
    return min(total, limit_rows) if limit_rows is not None else total


def build_jobs(args: argparse.Namespace) -> List[ShardJob]:
    include_types = args.include_types.split(",") if args.include_types else None
    grouped = iter_input_shards(args.input_dir, include_types)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    args.keep_manifest_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[ShardJob] = []
    for raw_type, paths in sorted(grouped.items()):
        selected = paths[: args.max_shards_per_type] if args.max_shards_per_type is not None else paths
        for input_path in selected:
            rows = count_rows(input_path, args.limit_rows_per_shard)
            if rows:
                jobs.append(
                    ShardJob(
                        raw_type=raw_type,
                        input_path=str(input_path),
                        audit_path=str(args.audit_dir / input_path.name),
                        manifest_path=str(args.keep_manifest_dir / input_path.name),
                        num_rows=rows,
                        limit_rows=args.limit_rows_per_shard,
                    )
                )
    return jobs


def assign_jobs(jobs: List[ShardJob], devices: List[str]) -> List[Tuple[str, List[ShardJob]]]:
    buckets = [{"device": device, "rows": 0, "jobs": []} for device in devices]
    for job in sorted(jobs, key=lambda item: item.num_rows, reverse=True):
        bucket = min(buckets, key=lambda item: item["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += job.num_rows
    return [(bucket["device"], bucket["jobs"]) for bucket in buckets if bucket["jobs"]]


def rows_to_table(rows: List[Dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def decode_image(cell: Dict) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(cell["bytes"]))).convert("RGB")


def extract_json(text: str) -> Dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"No JSON found in model output: {text[:200]!r}")
        return json.loads(match.group(0))


SLOT_PROMPT = """You extract factual slots from one image-editing instruction.
Do not inspect or imagine any image. Split compound requests into atomic subgoals;
for example, changing skin tone and clothing color must be two subgoals.

Use the raw type only as a coarse hint. Mark NO_OP only when the instruction itself
explicitly requests no change. Mark UNJUDGEABLE when no visible fact can verify it.
For add requests, `object_b` must name the most concrete newly introduced visual
entity rather than a broad scene category. For example, "add a garden with raised
beds" should use "raised garden bed" rather than only "garden".

Return JSON only with this schema (do not add a verdict):
{
  "instruction_status": "NORMAL" | "NO_OP" | "UNJUDGEABLE",
  "subgoals": [
    {
      "edit_type": "add" | "remove" | "replace" | "color" | "motion" | "background" | "style",
      "object_a": "object removed/replaced/modified, or empty",
      "object_b": "object added/replacement result, or empty",
      "attribute": "requested TARGET value with direction/color/style/scene (for example 'darker skin tone', not generic 'skin tone'), or empty",
      "part": "body part/component for motion, or empty",
      "count": null or a positive integer,
      "location": "short location clue, or empty"
    }
  ],
  "confidence": 0.0,
  "notes": "short parsing note"
}

Raw type: __RAW_TYPE__
Instruction: __INSTRUCTION__
"""

SINGLE_IMAGE_PROMPT = """You are recording facts visible in ONE image. You cannot
see the other image. Do not infer an edit, a before/after relationship, success, or
failure. The original editing instruction is intentionally hidden.

Answer every requested slot independently. Use normalized [x1,y1,x2,y2] boxes.
For attributes and poses, report the CURRENT visible value/pose, not the requested
value. A second image, when provided, is only a magnified crop from the same image.

Always describe the scene, style, and main subject. Then return one fact for every
question in the supplied questionnaire. Return JSON only, with no verdict:
{
  "scene_description": "one short sentence",
  "scene_label": "one short scene category",
  "style_label": "one short style label",
  "subject_identity": "one short identity",
  "subject_bbox": [0.0,0.0,1.0,1.0] or null,
  "facts": [
    {
      "subgoal_index": 0,
      "role": "OBJECT_A" | "OBJECT_B" | "PART",
      "query": "copied slot query",
      "present": "YES" | "NO" | "UNCLEAR",
      "count": null or a nonnegative integer,
      "bboxes": [[0.0,0.0,1.0,1.0]],
      "attribute_value": "current visible value, or empty",
      "pose": "specific current pose, or empty",
      "confidence": 0.0
    }
  ],
  "crop_observation": "what the optional crop shows, or empty",
  "confidence": 0.0
}

Questionnaire (slot words only, not an instruction):
__QUESTIONNAIRE__
"""

PAIR_PROMPT = """Compare Image A and Image B without knowing any editing request.
List only visible differences, most significant first. Judge whether the main
subject is the same, whether composition/viewpoint is preserved, whether unrelated
regions are preserved, and whether B looks like a controlled edit of A or a global
regeneration. For composition, compare camera/viewpoint and fixed layout anchors;
an added/removed/replaced item itself does not make composition unpreserved. For
unrelated regions, judge only areas OUTSIDE the listed differences; a large edited
region itself is not an unrelated change. A whole-image style/color transform that preserves content and
geometry is CONTROLLED_EDIT; reserve GLOBAL_REGEN for changed identity, content,
composition, or viewpoint. Do not guess the intended edit or emit a verdict.

Return JSON only:
{
  "visible_differences": [
    {"description": "one visible difference", "significance": "LOW" | "MEDIUM" | "HIGH"}
  ],
  "same_subject": "YES" | "NO" | "UNCLEAR",
  "composition_preserved": "YES" | "NO" | "UNCLEAR",
  "unrelated_regions_preserved": "YES" | "NO" | "UNCLEAR",
  "edit_scope": "CONTROLLED_EDIT" | "GLOBAL_REGEN" | "UNCLEAR",
  "confidence": 0.0
}
"""

MATCH_PROMPT = """This is a text-only matching task. Match each atomic subgoal in
an editing instruction against an instruction-blind list of visible image
differences. Do not add visual facts that are absent from the list.

MATCH means clearly supported; PARTIAL means some requested progress is explicitly
mentioned; MISMATCH means a contradictory/wrong change; NOT_MENTIONED means no
listed difference supports it. Return JSON only and do not emit a keep/drop verdict:
{
  "subgoal_matches": [
    {"subgoal_index": 0, "match": "MATCH" | "PARTIAL" | "MISMATCH" | "NOT_MENTIONED", "reason": "short textual reason", "confidence": 0.0}
  ],
  "overall_match": "MATCH" | "PARTIAL" | "MISMATCH" | "NOT_MENTIONED",
  "confidence": 0.0
}

Instruction: __INSTRUCTION__
Atomic slots: __SLOTS__
Blind visible differences: __DIFFERENCES__
"""

REVIEW_STATE_PROMPT = """Record focused facts visible in this ONE image. You
cannot see the other image. Do not infer a before/after relationship and do not
decide whether an edit, request, or subgoal succeeded. Never output a support,
match, keep/drop, or verdict field.

Answer every question independently. Report only the CURRENT state in this image.
For color/background/style report literal visible attributes. For motion report a
specific pose, orientation, angle, or relative position rather than words such as
normal or changed. If `present` is YES, the requested attribute or pose field must
not be empty. If it cannot be seen reliably, use UNCLEAR and confidence 0.0 rather
than copying words from the questionnaire. A second image, when present, is only a
magnified crop from this same image.

Return JSON only:
{
  "scene_description": "one short factual sentence",
  "scene_label": "one short scene category",
  "style_label": "one short current style",
  "subject_identity": "one short identity",
  "facts": [
    {
      "subgoal_index": 0,
      "role": "OBJECT_A" | "OBJECT_B" | "PART",
      "query": "copied slot query",
      "present": "YES" | "NO" | "UNCLEAR",
      "count": null or a nonnegative integer,
      "attribute_value": "literal current attribute, or empty",
      "pose": "literal current pose/orientation/position, or empty",
      "confidence": 0.0
    }
  ],
  "crop_observation": "literal current facts visible in the optional crop, or empty",
  "confidence": 0.0
}

Questionnaire (neutral slot words, not an instruction):
__QUESTIONNAIRE__
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fact-first local MLLM prefilter over CrispEdit parquet shards"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--keep-manifest-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--include-types", type=str, default=None)
    parser.add_argument("--max-shards-per-type", type=int, default=None)
    parser.add_argument("--limit-rows-per-shard", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--compression", type=str, default="zstd")
    parser.add_argument("--progress-mininterval", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--slot-cache-size",
        type=int,
        default=20000,
        help="Per-worker LRU entries for repeated MLLM slot parses; 0 disables caching",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument(
        "--boundary-review-fraction",
        type=float,
        default=0.05,
        help="Deterministic fraction of unresolved rows allowed to use Step 5",
    )
    return parser.parse_args()


def build_slot_prompt(raw_type: object, instruction: object) -> str:
    return SLOT_PROMPT.replace("__RAW_TYPE__", str(raw_type or "")).replace(
        "__INSTRUCTION__", str(instruction or "")
    )


def build_questionnaire(slots: Dict) -> List[Dict]:
    questions: List[Dict] = []
    for subgoal in slots.get("subgoals") or []:
        index = int(subgoal.get("subgoal_index", 0))
        edit_type = canonical_edit_type(subgoal.get("edit_type"))

        def add_question(role: str, query: str) -> None:
            query = str(query or "").strip()
            if query:
                questions.append(
                    {
                        "subgoal_index": index,
                        "edit_type": edit_type,
                        "role": role,
                        "query": query,
                        "location": subgoal.get("location", ""),
                        "requested_attribute_slot": subgoal.get("attribute", ""),
                    }
                )

        if edit_type == "add":
            add_question("OBJECT_B", subgoal.get("object_b"))
        elif edit_type == "remove":
            add_question("OBJECT_A", subgoal.get("object_a"))
        elif edit_type == "replace":
            add_question("OBJECT_A", subgoal.get("object_a"))
            add_question("OBJECT_B", subgoal.get("object_b"))
        elif edit_type == "color":
            add_question("OBJECT_A", subgoal.get("object_a") or "main subject")
        elif edit_type == "motion":
            add_question("PART", subgoal.get("part") or subgoal.get("object_a"))
        elif edit_type == "background":
            add_question("OBJECT_A", subgoal.get("attribute") or "background")
    return questions


def build_single_image_prompt(slots: Dict) -> str:
    # Keep the established base questionnaire unchanged.  Fast paths below are
    # restricted to discrete object presence/count transitions, where these
    # answers can be used safely without changing color/motion/style behavior.
    questionnaire = {
        "subgoals": slots.get("subgoals") or [],
        "questions": build_questionnaire(slots),
    }
    return SINGLE_IMAGE_PROMPT.replace("__QUESTIONNAIRE__", json_dumps(questionnaire))


def _review_observation_focus(subgoal: Dict) -> str:
    edit_type = canonical_edit_type(subgoal.get("edit_type"))
    attribute = str(subgoal.get("attribute") or "").lower()
    if edit_type == "color":
        if "skin" in attribute:
            return (
                "current apparent lightness of untattooed facial/exposed skin on exactly "
                "one seven-level scale: level 1 very dark, 2 dark, 3 medium-dark, "
                "4 medium, 5 medium-light, 6 light, 7 very light; judge visible appearance "
                "under the image lighting and do not infer ethnicity or intrinsic identity"
            )
        if any(word in attribute for word in ("attire", "clothing", "outfit", "shirt", "dress")):
            return "current visible color of the queried attire/clothing"
        return "current literal visible color of the queried object"
    if edit_type == "motion":
        return "current pose, orientation, angle, and relative position of the queried part"
    if edit_type == "background":
        return "current background elements, lighting, weather, and time-of-day appearance"
    if edit_type == "style":
        return "current visual medium and rendering style"
    if edit_type in {"add", "remove", "replace"}:
        return "current presence and exact visible instance count of the queried object"
    return "current literal visible state"


def build_review_questionnaire(slots: Dict) -> List[Dict]:
    """Build a fact-only questionnaire with no requested target value."""

    subgoals = {
        int(item.get("subgoal_index", 0)): item for item in slots.get("subgoals") or []
    }
    questions = []
    for question in build_questionnaire(slots):
        index = int(question.get("subgoal_index", 0))
        subgoal = subgoals.get(index, {})
        questions.append(
            {
                "subgoal_index": index,
                "role": question.get("role"),
                "query": question.get("query"),
                "location": question.get("location", ""),
                "observation_focus": _review_observation_focus(subgoal),
            }
        )
    return questions


def build_match_prompt(instruction: object, slots: Dict, paired: Dict) -> str:
    return (
        MATCH_PROMPT.replace("__INSTRUCTION__", str(instruction or ""))
        .replace("__SLOTS__", json_dumps(slots.get("subgoals") or []))
        .replace("__DIFFERENCES__", json_dumps(paired.get("visible_differences") or []))
    )


def build_review_state_prompt(slots: Dict) -> str:
    questionnaire = {"questions": build_review_questionnaire(slots)}
    return REVIEW_STATE_PROMPT.replace("__QUESTIONNAIRE__", json_dumps(questionnaire))


def crop_normalized_bbox(
    image: Image.Image, bbox: Optional[Sequence[float]], expansion: float = 0.2
) -> Optional[Image.Image]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    width, height = image.size
    dx, dy = (x2 - x1) * expansion, (y2 - y1) * expansion
    left = max(0, min(width - 1, int((x1 - dx) * width)))
    top = max(0, min(height - 1, int((y1 - dy) * height)))
    right = max(left + 1, min(width, int(math.ceil((x2 + dx) * width))))
    bottom = max(top + 1, min(height, int(math.ceil((y2 + dy) * height))))
    if right - left < 2 or bottom - top < 2:
        return None
    return image.crop((left, top, right, bottom))


def _bbox_from_location(location: object) -> Optional[List[float]]:
    text = str(location or "").lower().replace("_", "-")
    if not text:
        return None
    x1, x2 = 0.15, 0.85
    y1, y2 = 0.1, 0.9
    if "left" in text:
        x1, x2 = 0.0, 0.62
    elif "right" in text:
        x1, x2 = 0.38, 1.0
    elif "center" in text or "central" in text or "middle" in text:
        x1, x2 = 0.18, 0.82
    if any(word in text for word in ("top", "upper", "above")):
        y1, y2 = 0.0, 0.62
    elif any(word in text for word in ("bottom", "lower", "below")):
        y1, y2 = 0.38, 1.0
    elif any(word in text for word in ("center", "central", "middle")):
        y1, y2 = 0.12, 0.88
    return [x1, y1, x2, y2]


def select_focus_bbox(slots: Dict, source_evidence: Dict) -> Optional[List[float]]:
    for subgoal in slots.get("subgoals") or []:
        if canonical_edit_type(subgoal.get("edit_type")) not in {"remove", "color", "motion"}:
            continue
        index = subgoal.get("subgoal_index")
        location_bbox = _bbox_from_location(subgoal.get("location"))
        preferred_roles = (
            ("PART", "OBJECT_A")
            if canonical_edit_type(subgoal.get("edit_type")) == "motion"
            else ("OBJECT_A",)
        )
        for role in preferred_roles:
            for fact in source_evidence.get("facts") or []:
                if fact.get("subgoal_index") != index or fact.get("role") != role:
                    continue
                boxes = fact.get("bboxes") or []
                if boxes:
                    box = boxes[0]
                    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                    if location_bbox and area >= 0.65:
                        return location_bbox
                    return box
        if location_bbox:
            return location_bbox
    return None


def select_focus_crop(
    slots: Dict, source_evidence: Dict, image: Image.Image
) -> Optional[Image.Image]:
    return crop_normalized_bbox(image, select_focus_bbox(slots, source_evidence))


def select_target_crop(
    slots: Dict, source_evidence: Dict, target: Image.Image
) -> Optional[Image.Image]:
    return select_focus_crop(slots, source_evidence, target)


def needs_source_guided_target_crop(slots: Dict) -> bool:
    return any(
        canonical_edit_type(subgoal.get("edit_type")) in {"remove", "color", "motion"}
        for subgoal in slots.get("subgoals") or []
    )


def deterministic_parse(instruction: object, raw_type: object) -> Dict:
    edit_type = canonical_edit_type(raw_type)
    if edit_type == "unknown":
        return {"parse_ok": False}
    try:
        result = dict(parse_instruction(str(instruction or ""), edit_type))
    except Exception as exc:
        return {"parse_ok": False, "error": repr(exc)}
    return result


def parse_simple_instruction_slots(instruction: object, raw_type: object) -> Optional[Dict]:
    """Parse only high-precision instruction templates; return None to use MLLM."""

    text = re.sub(r"\s+", " ", str(instruction or "")).strip()
    edit_type = canonical_edit_type(raw_type)
    subgoals: Optional[List[Dict]] = None
    if re.search(
        r"(?i)\b(?:no\s+changes?|without\s+changes?|unchanged|do\s+not|don't|"
        r"cannot|can't|clarify|sorry)\b",
        text,
    ):
        return None

    if edit_type == "remove":
        match = re.fullmatch(r"(?i)(?:please\s+)?(?:remove|erase|delete)\s+(.+?)[.!?]?", text)
        if match:
            object_a = re.sub(r"(?i)^(?:the|a|an)\s+", "", match.group(1)).strip()
            subgoals = [{"edit_type": "remove", "object_a": object_a}]
    elif edit_type == "replace":
        match = re.fullmatch(
            r"(?i)(?:please\s+)?replace\s+(?:the\s+)?(.+?)\s+with\s+(?:the\s+|a\s+|an\s+)?(.+?)[.!?]?",
            text,
        )
        if match:
            object_a = match.group(1).strip()
            location = ""
            packshot = re.fullmatch(r"(?i)(.+?)\s+(in\s+the\s+packshot)", object_a)
            if packshot:
                object_a, location = packshot.group(1).strip(), packshot.group(2).strip()
            subgoals = [{
                "edit_type": "replace",
                "object_a": object_a,
                "object_b": match.group(2).strip(),
                "location": location,
            }]
    elif edit_type == "color":
        match = re.fullmatch(
            r"(?i)(?:turn|change)\s+(?:the\s+)?(.+?)\s+positioned\s+(.+?)\s+into\s+(?:having\s+)?(.+?)[.!?]?",
            text,
        )
        if match:
            object_a = match.group(1).strip()
            location = re.sub(
                r"(?i)^(?:in|on|at)\s+(?:the\s+)?", "", match.group(2)
            ).strip()
            raw_goal = re.sub(
                r"(?i)^(?:a|an|the)\s+", "", match.group(3)
            ).strip()
            goals = re.split(r"(?i)\s+(?:with|and)\s+", raw_goal, maxsplit=1)
            goals = [re.sub(r"(?i)^(?:a|an|the)\s+", "", item).strip() for item in goals]
            clothing_words = {"attire", "clothes", "clothing", "outfit"}
            is_skin_and_clothing = (
                len(goals) == 2
                and "skin tone" in goals[0].lower()
                and COLOR_WORDS.intersection(re.findall(r"[a-z]+", goals[1].lower()))
                and clothing_words.intersection(re.findall(r"[a-z]+", goals[1].lower()))
            )
            if is_skin_and_clothing:
                subgoals = [
                    {
                        "edit_type": "color",
                        "object_a": object_a,
                        "attribute": goals[0],
                        "location": location,
                    },
                    {
                        "edit_type": "color",
                        "object_a": object_a,
                        "attribute": goals[1],
                        "location": location,
                    },
                ]
            else:
                # Keep multi-color compound requests on the MLLM fallback so
                # distinct objects/attributes can be split into subgoals.
                second_goal_colors = (
                    COLOR_WORDS.intersection(re.findall(r"[a-z]+", goals[1].lower()))
                    if len(goals) == 2
                    else set()
                )
                if raw_goal and not second_goal_colors:
                    subgoals = [{
                        "edit_type": "color",
                        "object_a": object_a,
                        "attribute": raw_goal,
                        "location": location,
                    }]
    elif edit_type == "motion":
        compound = re.fullmatch(
            r"(?i)(?:the\s+)?([a-z][a-z -]*?)\s+"
            r"(lowers?|raises?)\s+(?:his|her|their|its)\s+([a-z-]+)\s+and\s+"
            r"(straightens?|bends?)\s+(?:his|her|their|its)\s+([a-z-]+)[.!?]?",
            text,
        )
        if compound:
            subject = compound.group(1).strip()
            verb_states = {
                "lower": "lowered",
                "lowers": "lowered",
                "raise": "raised",
                "raises": "raised",
                "straighten": "straightened",
                "straightens": "straightened",
                "bend": "bent",
                "bends": "bent",
            }
            subgoals = [
                {
                    "edit_type": "motion",
                    "object_a": f"{subject}'s {compound.group(3)}",
                    "attribute": verb_states[compound.group(2).lower()],
                },
                {
                    "edit_type": "motion",
                    "object_a": f"{subject}'s {compound.group(5)}",
                    "attribute": verb_states[compound.group(4).lower()],
                },
            ]
        else:
            simple = re.fullmatch(
                r"(?i)(?:the\s+)?([a-z][a-z -]*?)\s+"
                r"(tilts?|turns?|raises?|lowers?)\s+(?:his|her|their|its)\s+"
                r"([a-z-]+)(.*?)[.!?]?",
                text,
            )
            if simple and " and " not in simple.group(4).lower():
                part = simple.group(3).strip()
                attribute = " ".join(
                    value.strip()
                    for value in (simple.group(2), part, simple.group(4))
                    if value.strip()
                )
                subgoals = [{
                    "edit_type": "motion",
                    "object_a": simple.group(1).strip(),
                    "part": part,
                    "attribute": attribute,
                }]
    elif edit_type == "background":
        match = re.fullmatch(
            r"(?i)(?:please\s+)?(?:change|replace|turn|make)\s+(?:the\s+)?background\s+(?:to|with|into)\s+(.+?)[.!?]?",
            text,
        )
        if match:
            subgoals = [{"edit_type": "background", "attribute": match.group(1).strip()}]
    elif edit_type == "style" and text:
        style_action = re.search(
            r"(?i)\b(?:apply|create|design|draw|generate|give|make|modify|paint|perform|"
            r"recreate|reinterpret|reimagine|render|stylize|transform|turn)\b",
            text,
        )
        if style_action:
            subgoals = [{"edit_type": "style", "attribute": text.rstrip(".!?")}]

    if not subgoals:
        return None
    return normalize_slots(
        {
            "instruction_status": "NORMAL",
            "subgoals": subgoals,
            "confidence": 1.0,
            "notes": "high-precision deterministic template",
        },
        raw_type,
    )


def _state_evidence_confidence(evidence: Dict) -> float:
    values = [float(evidence.get("confidence") or 0.0)]
    values.extend(float(item.get("confidence") or 0.0) for item in evidence.get("facts") or [])
    nonzero = [value for value in values if value > 0]
    return min(nonzero) if nonzero else 0.0


def _review_selected(shard_name: str, row_idx: int, fraction: float) -> bool:
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction <= 0:
        return False
    digest = hashlib.sha1(f"{shard_name}:{row_idx}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value < fraction


class QwenFactPrefilterRunner:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_new_tokens: int,
        confidence_threshold: float,
        boundary_review_fraction: float,
        slot_cache_size: int,
    ):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.boundary_review_fraction = max(0.0, min(1.0, boundary_review_fraction))
        self.slot_cache_size = max(0, int(slot_cache_size))
        self.slot_cache: OrderedDict[Tuple[str, str], Dict] = OrderedDict()
        self.generation_calls = 0
        self.mllm_conversations = 0
        self._torch = torch
        model_kwargs = {"trust_remote_code": True}
        if device == "cpu":
            model_kwargs["dtype"] = torch.float32
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, **model_kwargs
            ).to("cpu").eval()
        else:
            model_kwargs["dtype"] = torch.bfloat16
            model_kwargs["device_map"] = {"": device}
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, **model_kwargs
            ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    @staticmethod
    def _slot_cache_key(record: Dict) -> Tuple[str, str]:
        return (
            canonical_edit_type(record.get("type")),
            re.sub(r"\s+", " ", str(record.get("instruction") or "")).strip(),
        )

    def _cached_slots(self, key: Tuple[str, str]) -> Optional[Dict]:
        cached = self.slot_cache.get(key)
        if cached is not None:
            self.slot_cache.move_to_end(key)
        return cached

    def _cache_slots(self, key: Tuple[str, str], slots: Dict) -> None:
        if self.slot_cache_size <= 0:
            return
        self.slot_cache[key] = slots
        self.slot_cache.move_to_end(key)
        while len(self.slot_cache) > self.slot_cache_size:
            self.slot_cache.popitem(last=False)

    @staticmethod
    def _text_messages(prompts: Sequence[str]) -> List[List[Dict]]:
        return [
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for prompt in prompts
        ]

    @staticmethod
    def _single_image_messages(
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        crops: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> List[List[Dict]]:
        if crops is None:
            crops = [None] * len(images)
        conversations = []
        for image, crop, prompt in zip(images, crops, prompts):
            content: List[Dict] = [
                {"type": "text", "text": "Full single image:"},
                {"type": "image", "image": image},
            ]
            if crop is not None:
                content.extend(
                    [
                        {"type": "text", "text": "Magnified crop from the same image:"},
                        {"type": "image", "image": crop},
                    ]
                )
            content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])
        return conversations

    @staticmethod
    def _pair_messages(
        first_images: Sequence[Image.Image],
        second_images: Sequence[Image.Image],
        prompts: Sequence[str],
        first_label: str = "Image A (source):",
        second_label: str = "Image B (target):",
    ) -> List[List[Dict]]:
        conversations = []
        for first, second, prompt in zip(first_images, second_images, prompts):
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": first_label},
                            {"type": "image", "image": first},
                            {"type": "text", "text": second_label},
                            {"type": "image", "image": second},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
        return conversations

    def _generate_json(self, conversations: Sequence[List[Dict]]) -> List[Dict]:
        if not conversations:
            return []
        self.generation_calls += 1
        self.mllm_conversations += len(conversations)
        try:
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            inputs = inputs.to(self.model.device)
            prompt_length = int(inputs.input_ids.shape[1])
            with self._torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            trimmed = generated_ids[:, prompt_length:]
            texts = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception as exc:
            return [
                {"parse_ok": False, "parsed": {}, "raw_text": "", "error": repr(exc)}
                for _ in conversations
            ]

        results = []
        for text in texts:
            try:
                parsed = extract_json(text)
                if not isinstance(parsed, dict):
                    raise ValueError("model JSON is not an object")
                results.append({"parse_ok": True, "parsed": parsed, "raw_text": text, "error": ""})
            except Exception as exc:
                results.append(
                    {"parse_ok": False, "parsed": {}, "raw_text": text, "error": repr(exc)}
                )
        if len(results) != len(conversations):
            return [
                {
                    "parse_ok": False,
                    "parsed": {},
                    "raw_text": "",
                    "error": "generation batch-size mismatch",
                }
                for _ in conversations
            ]
        return results

    @staticmethod
    def _empty_evidence() -> Tuple[Dict, Dict, Dict]:
        single = normalize_single_image_evidence({})
        paired = normalize_pair_evidence({})
        return single, dict(single), paired

    def _error_result(self, error: str, stage_raw: Optional[Dict] = None) -> Dict:
        return {
            "model_output": {
                "verdict": "ERROR",
                "confidence": 1.0,
                "source_observation": "",
                "target_observation": "",
                "reason": error,
                "change_presence": "UNKNOWN",
                "instruction_achievement": "UNCLEAR",
                "failure_mode": "OTHER",
            },
            "decision": "drop",
            "parse_ok": False,
            "raw_text": json_dumps(stage_raw or {}),
            "evidence": {},
        }

    def infer_batch(self, records: Sequence[Dict], shard_name: str, row_idx_start: int) -> List[Dict]:
        if not records:
            return []
        count = len(records)
        outputs: List[Optional[Dict]] = [None] * count
        sources: List[Optional[Image.Image]] = [None] * count
        targets: List[Optional[Image.Image]] = [None] * count
        raw_stages: List[Dict] = [{} for _ in records]

        valid_indices = []
        for index, record in enumerate(records):
            try:
                sources[index] = decode_image(record["input_img"])
                targets[index] = decode_image(record["output_img"])
                valid_indices.append(index)
            except Exception as exc:
                outputs[index] = self._error_result(f"image_decode: {exc!r}")

        slots_by_index: Dict[int, Dict] = {}
        deterministic_by_index: Dict[int, Dict] = {}
        slot_fallback_groups: Dict[Tuple[str, str], List[int]] = {}
        active_indices = []
        for index in valid_indices:
            deterministic = deterministic_parse(
                records[index].get("instruction", ""), records[index].get("type", "")
            )
            deterministic_by_index[index] = deterministic
            slots = parse_simple_instruction_slots(
                records[index].get("instruction", ""), records[index].get("type", "")
            )
            if slots is None:
                cache_key = self._slot_cache_key(records[index])
                cached = self._cached_slots(cache_key)
                if cached is None:
                    slot_fallback_groups.setdefault(cache_key, []).append(index)
                else:
                    slots_by_index[index] = cached
                    raw_stages[index]["_slot_method"] = "CACHE"
                    if cached.get("instruction_status") == "NORMAL":
                        active_indices.append(index)
                continue
            slots_by_index[index] = slots
            raw_stages[index]["_slot_method"] = "DETERMINISTIC"
            if slots.get("instruction_status") == "NORMAL":
                active_indices.append(index)

        slot_fallback_items = list(slot_fallback_groups.items())
        slot_fallback_indices = [indices[0] for _, indices in slot_fallback_items]
        slot_results = self._generate_json(
            self._text_messages(
                [
                    build_slot_prompt(records[index].get("type", ""), records[index].get("instruction", ""))
                    for index in slot_fallback_indices
                ]
            )
        )
        for (cache_key, indices), stage in zip(slot_fallback_items, slot_results):
            representative = indices[0]
            raw_stages[representative]["step0_slots"] = stage.get("raw_text", "")
            raw_stages[representative]["_slot_method"] = "MLLM"
            if not stage.get("parse_ok"):
                for index in indices:
                    outputs[index] = self._error_result(
                        f"step0_slots: {stage.get('error', 'parse error')}",
                        raw_stages[index],
                    )
                continue
            slots = normalize_slots(
                stage["parsed"], records[representative].get("type", "")
            )
            self._cache_slots(cache_key, slots)
            for position, index in enumerate(indices):
                slots_by_index[index] = slots
                if position:
                    raw_stages[index]["_slot_method"] = "CACHE"
                if slots.get("instruction_status") == "NORMAL":
                    active_indices.append(index)

        source_by_index: Dict[int, Dict] = {}
        target_by_index: Dict[int, Dict] = {}
        paired_by_index: Dict[int, Dict] = {}
        match_by_index: Dict[int, Dict] = {}

        initial_active_indices = list(active_indices)
        direct_target_indices = [
            index
            for index in initial_active_indices
            if not needs_source_guided_target_crop(slots_by_index[index])
        ]
        # Source and non-localized target observations share one generation.
        # Localized target crops must wait for the source observation's bbox.
        base_state_results = self._generate_json(
            self._single_image_messages(
                [sources[index] for index in initial_active_indices],
                [
                    build_single_image_prompt(slots_by_index[index])
                    for index in initial_active_indices
                ],
            )
            + self._single_image_messages(
                [targets[index] for index in direct_target_indices],
                [
                    build_single_image_prompt(slots_by_index[index])
                    for index in direct_target_indices
                ],
            )
        )
        source_count = len(initial_active_indices)
        source_results = base_state_results[:source_count]
        direct_target_results = base_state_results[source_count:]
        next_indices = []
        for index, stage in zip(initial_active_indices, source_results):
            raw_stages[index]["step1_source"] = stage.get("raw_text", "")
            if not stage.get("parse_ok"):
                outputs[index] = self._error_result(
                    f"step1_source: {stage.get('error', 'parse error')}", raw_stages[index]
                )
                continue
            source_by_index[index] = normalize_single_image_evidence(stage["parsed"])
            next_indices.append(index)
        active_indices = next_indices

        target_stages_by_index = dict(zip(direct_target_indices, direct_target_results))
        guided_target_indices = [
            index
            for index in active_indices
            if needs_source_guided_target_crop(slots_by_index[index])
        ]
        guided_target_results = self._generate_json(
            self._single_image_messages(
                [targets[index] for index in guided_target_indices],
                [
                    build_single_image_prompt(slots_by_index[index])
                    for index in guided_target_indices
                ],
                [
                    select_target_crop(
                        slots_by_index[index], source_by_index[index], targets[index]
                    )
                    for index in guided_target_indices
                ],
            )
        )
        target_stages_by_index.update(zip(guided_target_indices, guided_target_results))

        next_indices = []
        for index in active_indices:
            stage = target_stages_by_index[index]
            raw_stages[index]["step2_target"] = stage.get("raw_text", "")
            if not stage.get("parse_ok"):
                outputs[index] = self._error_result(
                    f"step2_target: {stage.get('error', 'parse error')}", raw_stages[index]
                )
                continue
            target_by_index[index] = normalize_single_image_evidence(stage["parsed"])
            next_indices.append(index)
        active_indices = next_indices

        # High-precision no-op states are terminal.  They do not need the
        # expensive paired-comparison and text-matching calls.  Uncertain
        # missing-object cases deliberately continue through the full path.
        next_indices = []
        for index in active_indices:
            early_decision = adjudicate_terminal_no_change(
                slots_by_index[index], source_by_index[index], target_by_index[index]
            )
            if early_decision is None:
                next_indices.append(index)
                continue
            raw_stages[index]["_early_exit"] = "TERMINAL_NO_CHANGE"
            raw_stages[index]["_match_method"] = "NOT_RUN"
            paired = normalize_pair_evidence({})
            match = normalize_text_match(
                {}, len(slots_by_index[index].get("subgoals") or [])
            )
            outputs[index] = self._success_result(
                slots_by_index[index],
                deterministic_by_index[index],
                source_by_index[index],
                target_by_index[index],
                paired,
                match,
                early_decision,
                None,
                raw_stages[index],
                review_selected=False,
            )
        active_indices = next_indices

        pair_results = self._generate_json(
            self._pair_messages(
                [sources[index] for index in active_indices],
                [targets[index] for index in active_indices],
                [PAIR_PROMPT for _ in active_indices],
            )
        )
        next_indices = []
        for index, stage in zip(active_indices, pair_results):
            raw_stages[index]["step3_pair"] = stage.get("raw_text", "")
            if not stage.get("parse_ok"):
                outputs[index] = self._error_result(
                    f"step3_pair: {stage.get('error', 'parse error')}", raw_stages[index]
                )
                continue
            paired_by_index[index] = normalize_pair_evidence(stage["parsed"])
            next_indices.append(index)
        active_indices = next_indices

        # A positive target match can often be derived directly from the two
        # independent factual states.  Only unresolved rows call the text MLLM.
        initial_decisions: Dict[int, Dict] = {}
        match_fallback_indices = []
        for index in active_indices:
            code_match = derive_state_text_match(
                slots_by_index[index], source_by_index[index], target_by_index[index]
            )
            if code_match is None:
                match_fallback_indices.append(index)
                continue
            match_by_index[index] = code_match
            raw_stages[index]["_match_method"] = "CODE_STATES"
            initial_decisions[index] = adjudicate_evidence(
                slots_by_index[index],
                source_by_index[index],
                target_by_index[index],
                paired_by_index[index],
                match_by_index[index],
                self.confidence_threshold,
            )

        match_results = self._generate_json(
            self._text_messages(
                [
                    build_match_prompt(
                        records[index].get("instruction", ""),
                        slots_by_index[index],
                        paired_by_index[index],
                    )
                    for index in match_fallback_indices
                ]
            )
        )
        next_indices = [index for index in active_indices if index not in match_fallback_indices]
        for index, stage in zip(match_fallback_indices, match_results):
            raw_stages[index]["step4_text_match"] = stage.get("raw_text", "")
            raw_stages[index]["_match_method"] = "MLLM"
            if not stage.get("parse_ok"):
                outputs[index] = self._error_result(
                    f"step4_text_match: {stage.get('error', 'parse error')}", raw_stages[index]
                )
                continue
            match_by_index[index] = normalize_text_match(
                stage["parsed"], len(slots_by_index[index].get("subgoals") or [])
            )
            initial_decisions[index] = adjudicate_evidence(
                slots_by_index[index],
                source_by_index[index],
                target_by_index[index],
                paired_by_index[index],
                match_by_index[index],
                self.confidence_threshold,
            )
            next_indices.append(index)
        active_indices = next_indices

        review_indices = [
            index
            for index in active_indices
            if initial_decisions[index].get("review_needed")
            and _review_selected(
                shard_name, row_idx_start + index, self.boundary_review_fraction
            )
        ]
        review_by_index: Dict[int, Dict] = {}
        review_prompts = [build_review_state_prompt(slots_by_index[index]) for index in review_indices]
        review_source_crops = [
            select_focus_crop(slots_by_index[index], source_by_index[index], sources[index])
            for index in review_indices
        ]
        review_target_crops = [
            select_focus_crop(slots_by_index[index], source_by_index[index], targets[index])
            for index in review_indices
        ]
        # Source and target remain separate conversations, but share one
        # batched generate call for better GPU utilization.
        review_count = len(review_indices)
        review_results = self._generate_json(
            self._single_image_messages(
                [sources[index] for index in review_indices],
                review_prompts,
                review_source_crops,
            )
            + self._single_image_messages(
                [targets[index] for index in review_indices],
                review_prompts,
                review_target_crops,
            )
        )
        review_source_results = review_results[:review_count]
        review_target_results = review_results[review_count:]
        for index, target_stage, source_stage in zip(
            review_indices, review_target_results, review_source_results
        ):
            raw_stages[index]["step5_target_state"] = target_stage.get("raw_text", "")
            raw_stages[index]["step5_source_state"] = source_stage.get("raw_text", "")
            if target_stage.get("parse_ok") and source_stage.get("parse_ok"):
                review_target = normalize_single_image_evidence(target_stage["parsed"])
                review_source = normalize_single_image_evidence(source_stage["parsed"])
                review_by_index[index] = {
                    "method": "FOCUSED_INDEPENDENT_SINGLE_IMAGE_STATES",
                    "source": review_source,
                    "target": review_target,
                    "confidence": min(
                        _state_evidence_confidence(review_source),
                        _state_evidence_confidence(review_target),
                    ),
                }

        for index in active_indices:
            review = review_by_index.get(index)
            decision = (
                adjudicate_evidence(
                    slots_by_index[index],
                    source_by_index[index],
                    target_by_index[index],
                    paired_by_index[index],
                    match_by_index[index],
                    self.confidence_threshold,
                    review=review,
                )
                if review is not None
                else initial_decisions[index]
            )
            outputs[index] = self._success_result(
                slots_by_index[index],
                deterministic_by_index[index],
                source_by_index[index],
                target_by_index[index],
                paired_by_index[index],
                match_by_index[index],
                decision,
                review,
                raw_stages[index],
                review_selected=index in review_indices,
            )

        for index in valid_indices:
            if outputs[index] is not None:
                continue
            slots = slots_by_index[index]
            source, target, paired = self._empty_evidence()
            match = normalize_text_match({}, len(slots.get("subgoals") or []))
            decision = adjudicate_evidence(
                slots, source, target, paired, match, self.confidence_threshold
            )
            outputs[index] = self._success_result(
                slots,
                deterministic_by_index[index],
                source,
                target,
                paired,
                match,
                decision,
                None,
                raw_stages[index],
                review_selected=False,
            )
        return [output or self._error_result("internal missing result") for output in outputs]

    @staticmethod
    def _success_result(
        slots: Dict,
        deterministic: Dict,
        source: Dict,
        target: Dict,
        paired: Dict,
        match: Dict,
        decision: Dict,
        review: Optional[Dict],
        raw_stages: Dict,
        review_selected: bool,
    ) -> Dict:
        change = decision.get("change_presence", "UNKNOWN")
        mllm_stages = sorted(
            key for key in raw_stages if key.startswith("step")
        )
        model_output = {
            "verdict": decision["verdict"],
            "confidence": decision.get("confidence", 0.0),
            "source_observation": source.get("scene_description", ""),
            "target_observation": target.get("scene_description", ""),
            "reason": decision.get("reason", ""),
            "change_presence": "CLEAR" if change == "TRUE" else "NONE" if change == "FALSE" else "SUBTLE",
            "instruction_achievement": decision.get("instruction_achievement", "UNCLEAR"),
            "failure_mode": decision.get("failure_mode", "OTHER"),
        }
        return {
            "model_output": model_output,
            "decision": decision["decision"],
            "parse_ok": True,
            "raw_text": json_dumps(raw_stages),
            "evidence": {
                "slots": slots,
                "deterministic_slots": deterministic,
                "slot_conflict": deterministic_slot_conflict(slots, deterministic),
                "source": source,
                "target": target,
                "paired": paired,
                "text_match": match,
                "predicates": decision.get("predicates", {}),
                "reason_codes": decision.get("reason_codes", []),
                "review_needed": bool(decision.get("review_needed", False)),
                "review_reasons": decision.get("review_reasons", []),
                "review_selected": review_selected,
                "review": review,
                "mllm_calls": len(mllm_stages),
                "mllm_stages": mllm_stages,
                "slot_method": raw_stages.get("_slot_method", "UNKNOWN"),
                "match_method": raw_stages.get("_match_method", "NOT_RUN"),
                "early_exit": raw_stages.get("_early_exit", ""),
            },
        }


def _filter_score(decision: str, verdict: str, confidence: float) -> float:
    if decision == "keep":
        return 0.0
    if verdict == "UNSURE":
        return max(0.5, confidence)
    return confidence


def _make_audit_and_manifest_rows(
    record: Dict,
    result: Dict,
    row_idx: int,
    raw_type: str,
    model_name: str,
    run_id: str,
) -> Tuple[Dict, Dict]:
    parsed = result["model_output"]
    evidence = result.get("evidence") or {}
    verdict = parsed["verdict"]
    decision = result.get("decision", "drop")
    confidence = float(parsed.get("confidence", 0.0))
    reason_codes = evidence.get("reason_codes") or []
    filter_reason_codes = "" if decision == "keep" else "|".join(reason_codes or [f"PREFILTER_{verdict}"])
    review_triggered = evidence.get("review") is not None
    if review_triggered:
        review_resolution = f"REVIEWED_{decision.upper()}"
    elif evidence.get("review_needed"):
        review_resolution = "NOT_SELECTED_BUDGET"
    else:
        review_resolution = "NOT_NEEDED"

    audit_row = {
        "row_idx": row_idx,
        "raw_type": record.get("type", raw_type),
        "instruction": record.get("instruction", ""),
        "prefilter_verdict": verdict,
        "prefilter_decision": decision,
        "prefilter_confidence": confidence,
        "prefilter_source_observation": parsed.get("source_observation", ""),
        "prefilter_target_observation": parsed.get("target_observation", ""),
        "prefilter_reason": parsed.get("reason", ""),
        "prefilter_change_presence": parsed.get("change_presence", "SUBTLE"),
        "prefilter_instruction_achievement": parsed.get("instruction_achievement", "UNCLEAR"),
        "prefilter_failure_mode": parsed.get("failure_mode", "OTHER"),
        "prefilter_parse_ok": bool(result.get("parse_ok", False)),
        "prefilter_model_name": model_name,
        "prefilter_method": PREFILTER_METHOD,
        "prefilter_evidence_schema": EVIDENCE_SCHEMA,
        "prefilter_run_id": run_id,
        "prefilter_slots_json": json_dumps(evidence.get("slots") or {}),
        "prefilter_instruction_status": (evidence.get("slots") or {}).get(
            "instruction_status", "UNJUDGEABLE"
        ),
        "prefilter_subgoals_json": json_dumps(
            (evidence.get("slots") or {}).get("subgoals") or []
        ),
        "prefilter_deterministic_slots_json": json_dumps(evidence.get("deterministic_slots") or {}),
        "prefilter_slot_conflict": bool(evidence.get("slot_conflict", False)),
        "prefilter_source_evidence_json": json_dumps(evidence.get("source") or {}),
        "prefilter_target_evidence_json": json_dumps(evidence.get("target") or {}),
        "prefilter_paired_evidence_json": json_dumps(evidence.get("paired") or {}),
        "prefilter_text_match_json": json_dumps(evidence.get("text_match") or {}),
        "prefilter_predicates_json": json_dumps(evidence.get("predicates") or {}),
        "prefilter_decision_reason_codes_json": json_dumps(reason_codes),
        "prefilter_review_triggered": review_triggered,
        "prefilter_review_reasons_json": json_dumps(evidence.get("review_reasons") or []),
        "prefilter_review_method": (
            str((evidence.get("review") or {}).get("method") or "FACT_ONLY_REVIEW")
            if review_triggered
            else ""
        ),
        "prefilter_review_evidence_json": json_dumps(evidence.get("review") or {}),
        "prefilter_review_resolution": review_resolution,
        "prefilter_mllm_calls": int(evidence.get("mllm_calls", 0)),
        "prefilter_mllm_stages_json": json_dumps(evidence.get("mllm_stages") or []),
        "prefilter_slot_method": str(evidence.get("slot_method") or "UNKNOWN"),
        "prefilter_match_method": str(evidence.get("match_method") or "NOT_RUN"),
        "prefilter_early_exit": str(evidence.get("early_exit") or ""),
        "filter_decision": decision,
        "filter_reason_codes": filter_reason_codes,
        "filter_mismatch_score": float(_filter_score(decision, verdict, confidence)),
        "raw_text": result.get("raw_text", ""),
    }
    manifest_columns = (
        "row_idx",
        "prefilter_verdict",
        "prefilter_decision",
        "prefilter_confidence",
        "prefilter_reason",
        "prefilter_change_presence",
        "prefilter_instruction_achievement",
        "prefilter_failure_mode",
        "prefilter_parse_ok",
        "prefilter_model_name",
        "prefilter_method",
        "prefilter_evidence_schema",
        "prefilter_run_id",
        "prefilter_slot_conflict",
        "prefilter_predicates_json",
        "prefilter_review_triggered",
        "prefilter_review_resolution",
        "prefilter_mllm_calls",
        "prefilter_slot_method",
        "prefilter_match_method",
        "prefilter_early_exit",
        "filter_decision",
        "filter_reason_codes",
        "filter_mismatch_score",
    )
    manifest_row = {key: audit_row[key] for key in manifest_columns}
    return audit_row, manifest_row


def process_shard(
    job: ShardJob,
    runner: QwenFactPrefilterRunner,
    args: argparse.Namespace,
    progress_queue,
    worker_idx: int,
    run_id: str,
) -> Dict:
    input_path = Path(job.input_path)
    audit_path = Path(job.audit_path)
    manifest_path = Path(job.manifest_path)
    tmp_audit_path = audit_path.with_suffix(audit_path.suffix + ".tmp")
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    summary = {
        "rows": 0,
        "errors": 0,
        "kept": 0,
        "dropped": 0,
        "verdicts": {},
        "decisions": {},
        "mllm_conversations": 0,
        "generation_calls": 0,
        "deterministic_slots": 0,
        "cached_slots": 0,
        "code_matches": 0,
        "early_exits": 0,
    }
    generation_start = runner.generation_calls
    pf = pq.ParquetFile(input_path)
    audit_writer = None
    manifest_writer = None
    row_idx = 0
    model_name = Path(args.model_path).name
    completed = False
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            records = batch.to_pylist()
            if job.limit_rows is not None:
                records = records[: max(0, job.limit_rows - row_idx)]
            if not records:
                break
            results = runner.infer_batch(records, input_path.name, row_idx)
            audit_rows, manifest_rows = [], []
            for offset, (record, result) in enumerate(zip(records, results)):
                current_idx = row_idx + offset
                if not result.get("parse_ok"):
                    summary["errors"] += 1
                    if args.fail_fast:
                        raise RuntimeError(
                            f"prefilter error at {input_path.name}:{current_idx}: "
                            f"{result['model_output'].get('reason', '')}"
                        )
                audit_row, manifest_row = _make_audit_and_manifest_rows(
                    record, result, current_idx, job.raw_type, model_name, run_id
                )
                verdict, decision = audit_row["prefilter_verdict"], audit_row["filter_decision"]
                summary["verdicts"][verdict] = summary["verdicts"].get(verdict, 0) + 1
                summary["decisions"][decision] = summary["decisions"].get(decision, 0) + 1
                if decision == "keep":
                    summary["kept"] += 1
                else:
                    summary["dropped"] += 1
                summary["mllm_conversations"] += int(audit_row["prefilter_mllm_calls"])
                summary["deterministic_slots"] += int(
                    audit_row["prefilter_slot_method"] == "DETERMINISTIC"
                )
                summary["cached_slots"] += int(
                    audit_row["prefilter_slot_method"] == "CACHE"
                )
                summary["code_matches"] += int(
                    audit_row["prefilter_match_method"] == "CODE_STATES"
                )
                summary["early_exits"] += int(bool(audit_row["prefilter_early_exit"]))
                audit_rows.append(audit_row)
                manifest_rows.append(manifest_row)

            row_idx += len(records)
            summary["rows"] += len(records)
            audit_table, manifest_table = rows_to_table(audit_rows), rows_to_table(manifest_rows)
            if audit_writer is None:
                tmp_audit_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                audit_writer = pq.ParquetWriter(tmp_audit_path, audit_table.schema, compression=args.compression)
                manifest_writer = pq.ParquetWriter(tmp_manifest_path, manifest_table.schema, compression=args.compression)
            audit_writer.write_table(audit_table)
            manifest_writer.write_table(manifest_table)
            progress_queue.put(
                {
                    "kind": "rows",
                    "count": len(records),
                    "worker": worker_idx,
                    "shard": input_path.name,
                    "done_rows": summary["rows"],
                    "total_rows": job.num_rows,
                }
            )
            if job.limit_rows is not None and row_idx >= job.limit_rows:
                break
        completed = True
    finally:
        if audit_writer is not None:
            audit_writer.close()
        if manifest_writer is not None:
            manifest_writer.close()
        if completed:
            if tmp_audit_path.exists():
                tmp_audit_path.replace(audit_path)
            if tmp_manifest_path.exists():
                tmp_manifest_path.replace(manifest_path)
    summary["generation_calls"] = runner.generation_calls - generation_start
    return summary


def _existing_summary(manifest_path: Path) -> Dict:
    summary = {
        "rows": 0,
        "errors": 0,
        "kept": 0,
        "dropped": 0,
        "verdicts": {},
        "decisions": {},
        "mllm_conversations": 0,
        "generation_calls": 0,
        "deterministic_slots": 0,
        "cached_slots": 0,
        "code_matches": 0,
        "early_exits": 0,
    }
    available = set(pq.ParquetFile(manifest_path).schema.names)
    optional = {
        "prefilter_mllm_calls",
        "prefilter_slot_method",
        "prefilter_match_method",
        "prefilter_early_exit",
    }
    table = pq.read_table(
        manifest_path,
        columns=[
            "prefilter_verdict",
            "filter_decision",
            "prefilter_parse_ok",
            *sorted(optional & available),
        ],
    )
    for row in table.to_pylist():
        verdict = str(row.get("prefilter_verdict") or "ERROR")
        decision = str(row.get("filter_decision") or "drop")
        summary["rows"] += 1
        summary["errors"] += 0 if row.get("prefilter_parse_ok") else 1
        summary["kept"] += int(decision == "keep")
        summary["dropped"] += int(decision != "keep")
        summary["verdicts"][verdict] = summary["verdicts"].get(verdict, 0) + 1
        summary["decisions"][decision] = summary["decisions"].get(decision, 0) + 1
        summary["mllm_conversations"] += int(row.get("prefilter_mllm_calls") or 0)
        summary["deterministic_slots"] += int(
            row.get("prefilter_slot_method") == "DETERMINISTIC"
        )
        summary["cached_slots"] += int(row.get("prefilter_slot_method") == "CACHE")
        summary["code_matches"] += int(row.get("prefilter_match_method") == "CODE_STATES")
        summary["early_exits"] += int(bool(row.get("prefilter_early_exit")))
    return summary


def worker_main(
    worker_idx: int,
    device: str,
    jobs: List[ShardJob],
    args_dict: Dict,
    progress_queue,
    run_id: str,
) -> None:
    args = argparse.Namespace(**args_dict)
    try:
        progress_queue.put(
            {"kind": "log", "message": f"worker-{worker_idx} starting on {device} with {len(jobs)} shards"}
        )
        runner = QwenFactPrefilterRunner(
            args.model_path,
            device,
            args.max_new_tokens,
            args.confidence_threshold,
            args.boundary_review_fraction,
            args.slot_cache_size,
        )
        for job in jobs:
            audit_path, manifest_path = Path(job.audit_path), Path(job.manifest_path)
            if audit_path.exists() and manifest_path.exists() and not args.overwrite:
                summary = _existing_summary(manifest_path)
                progress_queue.put({"kind": "rows", "count": summary["rows"]})
            else:
                summary = process_shard(job, runner, args, progress_queue, worker_idx, run_id)
            progress_queue.put(
                {"kind": "shard_done", "worker": worker_idx, "shard": job.input_path, "summary": summary}
            )
    except Exception as exc:
        progress_queue.put(
            {"kind": "worker_error", "worker": worker_idx, "device": device, "error": repr(exc)}
        )
        if args.fail_fast:
            raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_idx, "device": device})


def aggregate_run_summary(shard_summaries: List[Dict]) -> Dict:
    verdicts: Dict[str, int] = {}
    decisions: Dict[str, int] = {}
    rows = errors = kept = dropped = 0
    mllm_conversations = generation_calls = deterministic_slots = cached_slots = 0
    code_matches = early_exits = 0
    for item in shard_summaries:
        summary = item.get("summary") or {}
        rows += int(summary.get("rows", 0))
        errors += int(summary.get("errors", 0))
        kept += int(summary.get("kept", 0))
        dropped += int(summary.get("dropped", 0))
        mllm_conversations += int(summary.get("mllm_conversations", 0))
        generation_calls += int(summary.get("generation_calls", 0))
        deterministic_slots += int(summary.get("deterministic_slots", 0))
        cached_slots += int(summary.get("cached_slots", 0))
        code_matches += int(summary.get("code_matches", 0))
        early_exits += int(summary.get("early_exits", 0))
        for key, value in (summary.get("verdicts") or {}).items():
            verdicts[key] = verdicts.get(key, 0) + int(value)
        for key, value in (summary.get("decisions") or {}).items():
            decisions[key] = decisions.get(key, 0) + int(value)
    return {
        "rows": rows,
        "errors": errors,
        "keep": kept,
        "drop": dropped,
        "drop_rate": round(dropped / max(rows, 1), 6),
        "verdict_counts": verdicts,
        "decision_counts": decisions,
        "mllm_conversations": mllm_conversations,
        "generation_calls": generation_calls,
        "average_mllm_conversations_per_row": round(
            mllm_conversations / max(rows, 1), 6
        ),
        "deterministic_slot_rows": deterministic_slots,
        "cached_slot_rows": cached_slots,
        "code_match_rows": code_matches,
        "early_exit_rows": early_exits,
        "prefilter_method": PREFILTER_METHOD,
        "evidence_schema": EVIDENCE_SCHEMA,
    }


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.audit_dir = args.audit_dir.resolve()
    args.keep_manifest_dir = args.keep_manifest_dir.resolve()
    jobs = build_jobs(args)
    if not jobs:
        print("No shards matched the requested filters.")
        return
    devices = parse_devices(args.devices)
    assignments = assign_jobs(jobs, devices)
    total_rows = sum(job.num_rows for job in jobs)
    run_id = time.strftime("prefilter_%Y%m%d_%H%M%S")
    args_dict = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    run_config = {
        "input_dir": str(args.input_dir),
        "audit_dir": str(args.audit_dir),
        "keep_manifest_dir": str(args.keep_manifest_dir),
        "devices": devices,
        "job_count": len(jobs),
        "total_rows": total_rows,
        "run_id": run_id,
        "prefilter_method": PREFILTER_METHOD,
        "evidence_schema": EVIDENCE_SCHEMA,
        "args": args_dict,
        "jobs": [asdict(job) for job in jobs],
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for worker_idx, (device, worker_jobs) in enumerate(assignments):
        process = ctx.Process(
            target=worker_main,
            args=(worker_idx, device, worker_jobs, args_dict, progress_queue, run_id),
            daemon=False,
        )
        process.start()
        processes.append(process)

    active_workers = len(processes)
    shard_summaries: List[Dict] = []
    pbar = tqdm(
        total=total_rows,
        dynamic_ncols=True,
        mininterval=args.progress_mininterval,
        desc="fact prefilter rows",
    )
    try:
        while active_workers > 0:
            try:
                message = progress_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = message.get("kind")
            if kind == "rows":
                pbar.update(int(message.get("count", 0)))
                if message.get("shard"):
                    pbar.set_postfix_str(
                        f"{message['shard']} {message.get('done_rows')}/{message.get('total_rows')}"
                    )
            elif kind == "log":
                tqdm.write(message.get("message", ""))
            elif kind == "shard_done":
                shard_summaries.append(message)
                summary = message.get("summary") or {}
                tqdm.write(
                    f"done {Path(message['shard']).name}: rows={summary.get('rows', 0)} "
                    f"keep={summary.get('kept', 0)} drop={summary.get('dropped', 0)} "
                    f"errors={summary.get('errors', 0)}"
                )
            elif kind == "worker_error":
                tqdm.write(
                    f"worker-{message.get('worker')} on {message.get('device')} error: "
                    f"{message.get('error')}"
                )
                if args.fail_fast:
                    raise RuntimeError(message.get("error"))
            elif kind == "worker_done":
                active_workers -= 1
    finally:
        pbar.close()
        for process in processes:
            process.join()

    summary = aggregate_run_summary(shard_summaries)
    (args.audit_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
