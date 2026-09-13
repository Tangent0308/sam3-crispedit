#!/usr/bin/env python3
"""Validate ScaleEdit shard alignment, PNG masks, and instance RLE payloads."""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as mask_utils

from scaleedit.io import decode_image, discover_shards


def validate_mask_placeholder(row: dict) -> tuple[bool, list[str]]:
    """Validate a deliberate no-mask row without asking PIL to decode bytes."""

    if row.get("mask_png"):
        return False, []
    errors = []
    if str(row.get("qc_flag", "")) != "GROUND_FAIL":
        errors.append("missing mask_png outside GROUND_FAIL")
    if int(row.get("mask_sum") or 0) != 0:
        errors.append("missing mask_png with nonzero mask_sum")
    if int(row.get("mask_height") or 0) != 0 or int(row.get("mask_width") or 0) != 0:
        errors.append("missing mask_png with nonzero dimensions")
    if row.get("instance_masks"):
        errors.append("missing mask_png with instance masks")
    return True, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    for field in ("input_dir", "grounding_dir", "mask_dir", "report_json"):
        setattr(args, field, getattr(args, field).resolve())
    try:
        args.report_json.relative_to(args.input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("validation report must not be written under the source dataset")

    errors = []
    sample_ids = []
    areas = []
    instances = 0
    flags, sources, modes, tasks = Counter(), Counter(), Counter(), Counter()
    raw_by_name = {path.name: path for path in discover_shards(args.input_dir)}
    for ground_path in sorted(args.grounding_dir.glob("part-*.parquet")):
        raw_path = raw_by_name.get(ground_path.name)
        if raw_path is None:
            errors.append(f"missing source shard for {ground_path.name}")
            continue
        mask_path = args.mask_dir / ground_path.name
        if not mask_path.is_file():
            errors.append(f"missing mask output for {ground_path.name}")
            continue
        all_raw_rows = pq.read_table(raw_path).to_pylist()
        ground_rows = pq.read_table(ground_path).to_pylist()
        mask_rows = pq.read_table(mask_path).to_pylist()
        if len(ground_rows) != len(mask_rows):
            errors.append(
                f"row count mismatch {raw_path.name}: "
                f"{len(ground_rows)}/{len(mask_rows)}"
            )
            continue
        for position, (ground, row) in enumerate(zip(ground_rows, mask_rows)):
            row_idx = int(ground["row_idx"])
            if row_idx < 0 or row_idx >= len(all_raw_rows):
                errors.append(f"grounding row_idx out of range {raw_path.name}:{row_idx}")
                continue
            raw = all_raw_rows[row_idx]
            identity = str(raw.get("sample_id", ""))
            if {identity, str(ground.get("sample_id", "")), str(row.get("sample_id", ""))} != {
                identity
            }:
                errors.append(f"sample_id mismatch {raw_path.name}:{position}")
            if int(row["row_idx"]) != row_idx:
                errors.append(f"row_idx mismatch {raw_path.name}:{row_idx}")
            sample_ids.append(identity)
            source = decode_image(raw["source_image"])
            mode = str(json.loads(row["ground_json"]).get("mask_mode", "unresolved"))
            is_placeholder, placeholder_errors = validate_mask_placeholder(row)
            if is_placeholder:
                errors.extend(
                    f"{message} {raw_path.name}:{position}"
                    for message in placeholder_errors
                )
                areas.append(0.0)
                flags[str(row["qc_flag"])] += 1
                sources[str(row["mask_source"])] += 1
                modes[mode] += 1
                tasks[str(row["final_task"])] += 1
                continue
            mask = Image.open(io.BytesIO(row["mask_png"])).convert("L")
            if mask.size != source.size or mask.size != (row["mask_width"], row["mask_height"]):
                errors.append(f"mask size mismatch {raw_path.name}:{position}")
                continue
            mask_sum = int(np.count_nonzero(np.asarray(mask)))
            pixel_count = mask.width * mask.height
            area = mask_sum / max(pixel_count, 1)
            if mask_sum != int(row["mask_sum"]):
                errors.append(f"mask_sum mismatch {raw_path.name}:{position}")
            if not math.isclose(area, float(row["area_frac"]), abs_tol=1e-12):
                errors.append(f"area_frac mismatch {raw_path.name}:{position}")
            if mode == "full_image" and mask_sum != pixel_count:
                errors.append(f"incomplete full_image mask {raw_path.name}:{position}")
            if mode != "full_image" and mask_sum == 0:
                errors.append(f"empty non-global mask {raw_path.name}:{position}")
            for instance in row["instance_masks"]:
                instances += 1
                rle = {
                    "size": [int(value) for value in instance["rle_size"]],
                    "counts": instance["rle_counts"].encode("ascii"),
                }
                decoded = mask_utils.decode(rle)
                if tuple(decoded.shape) != (source.height, source.width):
                    errors.append(f"RLE size mismatch {raw_path.name}:{position}")
                if int(decoded.sum()) != int(instance["area"]):
                    errors.append(f"RLE area mismatch {raw_path.name}:{position}")
            areas.append(area)
            flags[str(row["qc_flag"])] += 1
            sources[str(row["mask_source"])] += 1
            modes[mode] += 1
            tasks[str(row["final_task"])] += 1

    duplicates = len(sample_ids) - len(set(sample_ids))
    if duplicates:
        errors.append(f"duplicate sample_id rows: {duplicates}")
    report = {
        "rows": len(sample_ids),
        "unique_sample_ids": len(set(sample_ids)),
        "task_count": len(tasks),
        "instances": instances,
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "qc_flags": dict(sorted(flags.items())),
        "mask_sources": dict(sorted(sources.items())),
        "mask_modes": dict(sorted(modes.items())),
        "area_frac": {
            "min": min(areas) if areas else None,
            "median": statistics.median(areas) if areas else None,
            "mean": statistics.fmean(areas) if areas else None,
            "max": max(areas) if areas else None,
            "empty": sum(value == 0 for value in areas),
            "full": sum(value == 1 for value in areas),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
