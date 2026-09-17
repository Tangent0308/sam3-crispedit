"""Build a positive SAMTok index and materialize regional edit cases.

The source parquet stores image bytes inside a nested ``images`` column.  This
script first scans only lightweight annotation columns, removes ``No target``
rows, and writes a reusable positive index.  Image bytes are read only when
materializing a plan, and the exact source RLE masks are resized alongside the
Qwen editing canvas; masks are never regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


QWEN_TARGET_AREA = 1024 * 1024
CANVAS_MULTIPLE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-jsonl", type=Path, default=None)
    parser.add_argument(
        "--force-index", action="store_true", help="Rebuild positive_rows.jsonl."
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_problem(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def is_no_target(answer: Any, masks: Any) -> bool:
    normalized = str(answer or "").strip().lower().rstrip(".")
    return normalized == "no target" or not isinstance(masks, list) or not masks


def normalize_rle(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(value) for value in rle["size"]],
        "counts": str(counts),
    }


def build_positive_index(
    parquet_path: Path, index_path: Path, force: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    if index_path.exists() and not force:
        rows = load_jsonl(index_path)
        return rows, {
            "status": "reused",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "positive_rows": len(rows),
        }

    table = pq.read_table(
        parquet_path, columns=["source", "problem", "answer", "masks"]
    )
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row_index in tqdm(range(table.num_rows), desc="positive index"):
        source = str(table["source"][row_index].as_py() or "").lower()
        problem = table["problem"][row_index].as_py()
        answer = table["answer"][row_index].as_py()
        masks = table["masks"][row_index].as_py()
        if is_no_target(answer, masks):
            counts[f"{source}_no_target"] += 1
            continue
        normalized_masks = [normalize_rle(mask) for mask in masks]
        rows.append(
            {
                "parquet_row_index": row_index,
                "source_subset": source,
                "problem": first_problem(problem),
                "answer": str(answer),
                "num_masks": len(normalized_masks),
                "masks": normalized_masks,
            }
        )
        counts[f"{source}_positive"] += 1

    write_jsonl(index_path, rows)
    return rows, {
        "status": "built",
        "wall_seconds": round(time.perf_counter() - started, 3),
        "input_rows": table.num_rows,
        "positive_rows": len(rows),
        "excluded_rows": table.num_rows - len(rows),
        "counts": dict(sorted(counts.items())),
    }


def qwen_canvas_size(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    target_width = math.sqrt(QWEN_TARGET_AREA * ratio)
    target_height = target_width / ratio
    target_width = round(target_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    target_height = round(target_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    return int(target_width), int(target_height)


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    payload = normalize_rle(rle)
    payload["counts"] = payload["counts"].encode("ascii")
    decoded = mask_utils.decode(payload)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def tight_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Decoded mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def image_bytes_from_cell(value: Any) -> bytes:
    images = value.as_py() if hasattr(value, "as_py") else value
    if not isinstance(images, list) or not images:
        raise ValueError("The parquet image cell is empty")
    payload = images[0]
    if not isinstance(payload, dict) or not payload.get("bytes"):
        raise ValueError("The parquet image cell has no embedded bytes")
    return bytes(payload["bytes"])


def overlay_masks(image: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    colors = np.asarray(
        [[255, 32, 32], [0, 210, 255], [255, 210, 0], [200, 64, 255]],
        dtype=np.float32,
    )
    for index, mask in enumerate(masks):
        color = colors[index % len(colors)]
        base[mask] = np.round(base[mask] * 0.42 + color * 0.58).astype(np.uint8)
    return Image.fromarray(base)


def materialize_plan(
    parquet_path: Path,
    output_dir: Path,
    index_by_row: dict[int, dict[str, Any]],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    requested = [int(row["parquet_row_index"]) for row in plan]
    if len(requested) != len(set(requested)):
        raise ValueError("Plan contains duplicate parquet_row_index values")
    missing = [row_index for row_index in requested if row_index not in index_by_row]
    if missing:
        raise ValueError(
            f"Plan selects rows that are No target or missing masks: {missing}"
        )

    read_started = time.perf_counter()
    image_column = pq.read_table(parquet_path, columns=["images"])["images"]
    image_read_seconds = time.perf_counter() - read_started

    source_dir = output_dir / "sources"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    crop_dir = output_dir / "crops"
    for directory in (source_dir, mask_dir, overlay_dir, crop_dir):
        directory.mkdir(parents=True, exist_ok=True)

    annotations: list[dict[str, Any]] = []
    crop_records: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    source_use_counts: Counter[str] = Counter()
    case_index = 0

    for planned in tqdm(plan, desc="materialize source rows"):
        row_index = int(planned["parquet_row_index"])
        indexed = index_by_row[row_index]
        edits = planned.get("edits", [])
        edit_mask_indexes = [int(edit["mask_index"]) for edit in edits]
        expected_mask_indexes = list(range(int(indexed["num_masks"])))
        if sorted(edit_mask_indexes) != expected_mask_indexes:
            raise ValueError(
                f"Row {row_index}: plan mask indexes {sorted(edit_mask_indexes)} "
                f"must cover every source mask exactly once: {expected_mask_indexes}"
            )

        embedded = image_bytes_from_cell(image_column[row_index])
        with Image.open(io.BytesIO(embedded)) as handle:
            original = handle.convert("RGB")
        source_size = original.size
        canvas_size = qwen_canvas_size(*source_size)
        canvas = original.resize(canvas_size, Image.Resampling.LANCZOS)

        source_name = f"source_{indexed['source_subset']}_r{row_index}.png"
        source_path = source_dir / source_name
        canvas.save(source_path, format="PNG", optimize=True)
        source_sha256 = sha256_bytes(source_path.read_bytes())

        for edit in edits:
            mask_index = int(edit["mask_index"])
            raw_rle = indexed["masks"][mask_index]
            decoded = decode_rle(raw_rle)
            expected_shape = (source_size[1], source_size[0])
            if decoded.shape != expected_shape:
                raise ValueError(
                    f"Row {row_index} mask {mask_index}: RLE shape {decoded.shape} "
                    f"does not match image shape {expected_shape}"
                )
            resized = resize_mask(decoded, canvas_size)
            rle = encode_rle(resized)
            bbox = tight_bbox(resized)
            case_name = (
                f"{case_index:03d}_{indexed['source_subset']}_r{row_index}_"
                f"m{mask_index}_{planned['name']}_{edit['name']}"
            )
            image_name = case_name + ".png"
            mask_path = mask_dir / f"{case_name}.mask.png"
            Image.fromarray(resized.astype(np.uint8) * 255).save(mask_path)
            overlay_masks(canvas, [resized]).save(overlay_dir / image_name)
            crop_records.append(
                {
                    "image": f"{case_name}.mask.png",
                    "original_image": image_name,
                    "bbox": bbox,
                    "refer_object": str(edit["refer_object"]),
                    "new_instruction": str(edit["new_instruction"]),
                    "mask_index": mask_index,
                }
            )
            annotation = {
                "image": image_name,
                "source_image": source_name,
                "editing_instruction": str(edit["editing_instruction"]),
                "refer_object": [str(edit["refer_object"])],
                "mask": [rle],
                "task_type": str(edit["task_type"]),
                "source_subset": indexed["source_subset"],
                "parquet_row_index": row_index,
                "mask_index": mask_index,
            }
            annotations.append(annotation)
            provenance_rows.append(
                {
                    **annotation,
                    "source_parquet": str(parquet_path),
                    "problem": indexed["problem"],
                    "answer": indexed["answer"],
                    "source_width": source_size[0],
                    "source_height": source_size[1],
                    "canvas_width": canvas_size[0],
                    "canvas_height": canvas_size[1],
                    "embedded_image_sha256": sha256_bytes(embedded),
                    "canvas_image_sha256": source_sha256,
                    "bbox": bbox,
                    "mask_area_fraction": round(float(resized.mean()), 8),
                    "raw_rle_sha256": sha256_json(raw_rle),
                    "resized_rle": rle,
                }
            )
            task_counts[str(edit["task_type"])] += 1
            source_use_counts[source_name] += 1
            case_index += 1

    write_jsonl(output_dir / "annotations.jsonl", annotations)
    write_jsonl(crop_dir / "crop_instruction.jsonl", crop_records)
    write_jsonl(output_dir / "provenance.jsonl", provenance_rows)
    elapsed = time.perf_counter() - started
    summary = {
        "source_rows": len(plan),
        "unique_source_images": len(source_use_counts),
        "cases": len(annotations),
        "regions": len(crop_records),
        "task_type_counts": dict(sorted(task_counts.items())),
        "regions_per_case_counts": {"1": len(annotations)},
        "source_reuse_counts": dict(sorted(source_use_counts.items())),
        "all_source_masks_covered": True,
        "image_column_read_seconds": round(image_read_seconds, 3),
        "materialize_wall_seconds": round(elapsed, 3),
        "materialize_cases_per_second": round(len(annotations) / elapsed, 3),
    }
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "positive_rows.jsonl"
    positive_rows, index_summary = build_positive_index(
        args.parquet, index_path, args.force_index
    )
    result: dict[str, Any] = {"positive_index": index_summary}
    if args.plan_jsonl is not None:
        plan = load_jsonl(args.plan_jsonl)
        index_by_row = {
            int(row["parquet_row_index"]): row for row in positive_rows
        }
        result["materialization"] = materialize_plan(
            args.parquet, args.output_dir, index_by_row, plan
        )
    (args.output_dir / "prepare_run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
