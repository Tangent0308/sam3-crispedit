"""Sample multi-mask SAMTok rows and generate per-region edit instructions.

This is a pilot sampler, not a production eligibility filter. It samples an
equal number of two-mask GRES and VER rows so that every selected source can be
reused twice and the requested edit-case count is exact. Within each subset the
rows are stratified by minimum region area to retain both tiny/hard and larger
targets. Qwen3-VL sees the clean source and one red-mask overlay and writes one
localized edit instruction for every original mask.
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
    overlay_masks,
    qwen_canvas_size,
    resize_mask,
)
import utils.vlm_utils as vlm


TASK_TYPES = ("add", "remove", "replace", "attribute")
FORBIDDEN_OUTPUT_PATTERN = re.compile(
    r"\b(marked|mask|overlay|highlighted|bbox|coordinates?|target region)\w*\b",
    flags=re.IGNORECASE,
)
TYPE_GUIDANCE = {
    "add": (
        "Add one small, clearly visible and semantically plausible accessory or "
        "detail to the marked target (for example a collar, hat, sticker, ribbon, "
        "or light). Do not add a second copy of the target itself."
    ),
    "remove": (
        "Remove the entire marked target and reconstruct the occluded background "
        "naturally. Do not remove similar non-target instances."
    ),
    "replace": (
        "Replace the entire marked target with exactly one visually distinct but "
        "scene-plausible object of similar pose and scale. The replacement must "
        "change object identity, category, or model; a color or clothing change "
        "alone is an attribute edit and is invalid here."
    ),
    "attribute": (
        "Change one conspicuous local attribute of the marked target, such as its "
        "color, material, or pattern, while preserving identity, geometry, and pose."
    ),
}

PROMPT = """You are designing one difficult, localized image-editing training example.

IMAGE 1 is the clean source. IMAGE 2 is the same image with exactly one target region highlighted in RED.
Original referring question: <<<QUESTION>>>
Original answer: <<<ANSWER>>>
Required edit type: <<<TASK_TYPE>>>

Task-specific rule:
<<<TYPE_GUIDANCE>>>

Requirements:
- Identify only the red-marked instance. The final text must NOT mention a red mask, overlay, bbox, coordinates, or mask index.
- RED in IMAGE 2 is only an annotation color. Never claim that the source object is red unless it is visibly red in IMAGE 1.
- Write a concise visual referring expression that distinguishes this instance from same-category instances using visible relations, position, appearance, or context.
- The edit must be localized, unambiguous, visibly judgeable, and realistic for this scene and target size.
- The requested new object/attribute must not already be present on the target in IMAGE 1.
- For a tiny target, prefer a simple high-contrast change that remains visible after editing.
- Explicitly preserve other same-category instances and unrelated scene content.
- Use ASCII English.

Return exactly one JSON object and no markdown:
{"refer_object":"specific target expression","editing_instruction":"complete full-image instruction","new_instruction":"short instruction for the regional branch"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--positive-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=100)
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
        PROMPT.replace("QUESTION", str(row["problem"])[:1200])
        .replace("ANSWER", str(row["answer"])[:1600])
        .replace("TASK_TYPE", task_type)
        .replace("TYPE_GUIDANCE", TYPE_GUIDANCE[task_type])
    )


def audit_messages(
    source: Image.Image,
    overlay: Image.Image,
    row: dict[str, Any],
    task_type: str,
    previous_response: str | None = None,
) -> list[dict[str, Any]]:
    prompt = build_prompt(row, task_type)
    if previous_response:
        prompt += (
            "\n\nYour previous response was invalid because it leaked mask/overlay "
            "language or copied a generic task rule. Rewrite it with a concrete "
            "visible target expression and scene-specific edit. Never use the words "
            "marked, mask, overlay, highlighted, bbox, coordinate, or target region. "
            "Previous response:\n" + previous_response[:1600]
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source},
                {"type": "image", "image": overlay},
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
    r"\baccessory or detail\b|\bplausible object or detail\b",
    flags=re.IGNORECASE,
)
TYPE_ACTION_PATTERNS = {
    "add": re.compile(r"\b(add|attach|place|put|give|tie|hang)\b", re.IGNORECASE),
    "remove": re.compile(r"\b(remove|erase|delete)\b", re.IGNORECASE),
    "replace": re.compile(r"\b(replace|substitut\w*|swap|exchange)\b", re.IGNORECASE),
    "attribute": re.compile(r"\b(change|make|recolor|turn)\b", re.IGNORECASE),
}


def normalize_generated(
    value: dict[str, Any] | None, task_type: str
) -> dict[str, str] | None:
    if not value:
        return None
    normalized = {}
    for key in ("refer_object", "editing_instruction", "new_instruction"):
        text = str(value.get(key, "")).strip()
        if not text or not text.isascii() or FORBIDDEN_OUTPUT_PATTERN.search(text):
            return None
        normalized[key] = text
    if len(normalized["editing_instruction"].split()) < 7:
        return None
    if GENERIC_INSTRUCTION_PATTERN.search(normalized["editing_instruction"]):
        return None
    if not TYPE_ACTION_PATTERNS[task_type].search(normalized["editing_instruction"]):
        return None
    return normalized


def deterministic_fallback(raw: str, task_type: str) -> dict[str, str] | None:
    """Rewrite a specific regional instruction when the model copies a rule.

    This fallback never invents a new edit. It retains the model's concrete
    target expression and short regional instruction, then adds preservation
    language without any mask/overlay terminology.
    """
    value = parse_json_object(raw)
    if not value:
        return None
    refer_object = str(value.get("refer_object", "")).strip()
    new_instruction = str(value.get("new_instruction", "")).strip().rstrip(".")
    if (
        not refer_object
        or not new_instruction
        or not refer_object.isascii()
        or not new_instruction.isascii()
        or FORBIDDEN_OUTPUT_PATTERN.search(refer_object)
        or FORBIDDEN_OUTPUT_PATTERN.search(new_instruction)
    ):
        return None
    if task_type == "remove":
        editing_instruction = (
            f"Remove only {refer_object} and reconstruct the background naturally; "
            "preserve all other people, objects, and scene content."
        )
    else:
        editing_instruction = (
            f"{new_instruction}. Apply this change only to {refer_object}; preserve "
            "all similar instances and unrelated scene content."
        )
    return normalize_generated(
        {
            "refer_object": refer_object,
            "editing_instruction": editing_instruction,
            "new_instruction": new_instruction,
        },
        task_type,
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "instruction_overlays"
    source_dir = args.output_dir / "instruction_sources"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    positive_rows = load_jsonl(args.positive_index)
    selected = sample_rows(positive_rows, args.num_cases, args.seed)
    selected.sort(key=lambda row: (row["source_subset"], row["parquet_row_index"]))
    write_jsonl(args.output_dir / "sampled_source_rows.jsonl", selected)

    read_started = time.perf_counter()
    image_column = pq.read_table(args.parquet, columns=["images"])["images"]
    image_read_seconds = time.perf_counter() - read_started

    tasks: list[dict[str, Any]] = []
    for source_position, row in enumerate(tqdm(selected, desc="instruction assets")):
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
            overlay = overlay_masks(source, [mask])
            overlay_name = f"{row['source_subset']}_r{row_index}_m{mask_index}_{task_type}.jpg"
            overlay.save(overlay_dir / overlay_name, quality=92)
            tasks.append(
                {
                    "row": row,
                    "mask_index": mask_index,
                    "task_type": task_type,
                    "source": source.copy(),
                    "overlay": overlay,
                    "overlay_name": overlay_name,
                }
            )

    type_counts = Counter(task["task_type"] for task in tasks)
    expected_per_type = args.num_cases // len(TASK_TYPES)
    if any(type_counts[name] != expected_per_type for name in TASK_TYPES):
        raise AssertionError(f"Task type assignment is not balanced: {type_counts}")

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
                    audit_messages(
                        task["source"],
                        task["overlay"],
                        task["row"],
                        task["task_type"],
                        task.get("previous_response"),
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
                    "overlay": task["overlay_name"],
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
                    "overlay": task["overlay_name"],
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
        raise RuntimeError(f"{len(failures)} instruction responses failed JSON validation")
    # A successful rerun must not leave an obsolete failure manifest from an
    # earlier attempt in the same output directory.
    (args.output_dir / "instruction_failures.jsonl").unlink(missing_ok=True)

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
        "source_subset_counts": dict(Counter(row["source_subset"] for row in selected)),
        "task_type_counts": dict(sorted(type_counts.items())),
        "area_stratum_counts": dict(sorted(Counter(row["area_stratum"] for row in selected).items())),
        "image_column_read_seconds": round(image_read_seconds, 3),
        "backend_load_seconds": round(backend_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "inference_cases_per_minute": round(len(tasks) / inference_seconds * 60.0, 3),
        "wall_seconds": round(elapsed, 3),
        "plan_jsonl": str(plan_path),
    }
    (args.output_dir / "instruction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
