#!/usr/bin/env python3
"""Validate native RefEdit shard alignment, PNG masks, and instance RLEs."""

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
from tqdm import tqdm

from refedit import GROUND_PROMPT_VERSION, MASK_POLICY_VERSION
from refedit.io import discover_shards, sample_id
from refedit.selection import (
    load_prefilter_manifest_dir,
    validate_prefilter_rows,
)
from scaleedit.io import decode_image


def _placeholder_errors(row: dict) -> list[str]:
    errors = []
    if str(row.get("qc_flag", "")) != "GROUND_FAIL":
        errors.append("missing mask_png outside GROUND_FAIL")
    if int(row.get("mask_sum") or 0) != 0:
        errors.append("missing mask_png with nonzero mask_sum")
    if int(row.get("mask_height") or 0) or int(row.get("mask_width") or 0):
        errors.append("missing mask_png with nonzero dimensions")
    if row.get("instance_masks"):
        errors.append("missing mask_png with instance masks")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest-dir",
        type=Path,
        help="optional PASS-only prefilter manifest directory for a selected run",
    )
    args = parser.parse_args()
    for name in ("input_dir", "grounding_dir", "mask_dir", "report_json"):
        setattr(args, name, getattr(args, name).resolve())
    if args.selection_manifest_dir:
        args.selection_manifest_dir = args.selection_manifest_dir.resolve()
    try:
        args.report_json.relative_to(args.input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("validation report must not be written under RefEdit source data")

    errors = []
    sample_ids = []
    areas = []
    instance_count = 0
    qc_flags = Counter()
    mask_sources = Counter()
    mask_modes = Counter()
    tasks = Counter()
    grounding_statuses = Counter()
    problem_samples = []
    source_paths = discover_shards(args.input_dir)
    source_by_name = {path.name: path for path in source_paths}
    selected_by_shard = (
        load_prefilter_manifest_dir(args.selection_manifest_dir)
        if args.selection_manifest_dir
        else None
    )
    if selected_by_shard is not None:
        if set(selected_by_shard) != set(source_by_name):
            errors.append("selection manifest shard names do not match source shards")
        for name, rows in selected_by_shard.items():
            if name in source_by_name:
                validate_prefilter_rows(source_by_name[name], rows)
    ground_paths = sorted(args.grounding_dir.glob("train-*.parquet"))
    mask_paths = sorted(args.mask_dir.glob("train-*.parquet"))
    if {path.name for path in ground_paths} != set(source_by_name):
        errors.append("grounding shard names do not exactly match source shard names")
    if {path.name for path in mask_paths} != set(source_by_name):
        errors.append("mask shard names do not exactly match source shard names")

    for ground_path in tqdm(
        ground_paths, desc="Validate RefEdit shards", dynamic_ncols=True
    ):
        source_path = source_by_name.get(ground_path.name)
        mask_path = args.mask_dir / ground_path.name
        if source_path is None or not mask_path.is_file():
            continue
        raw_rows = pq.read_table(source_path).to_pylist()
        ground_rows = pq.read_table(ground_path).to_pylist()
        mask_rows = pq.read_table(mask_path).to_pylist()
        expected_rows = (
            selected_by_shard.get(ground_path.name, [])
            if selected_by_shard is not None
            else raw_rows
        )
        if not (len(expected_rows) == len(ground_rows) == len(mask_rows)):
            errors.append(
                f"row count mismatch {ground_path.name}: "
                f"expected={len(expected_rows)} source={len(raw_rows)} "
                f"ground={len(ground_rows)} mask={len(mask_rows)}"
            )
            continue

        for position, (ground, row) in enumerate(zip(ground_rows, mask_rows)):
            row_idx = int(ground["row_idx"])
            if row_idx < 0 or row_idx >= len(raw_rows):
                errors.append(f"row_idx out of range {ground_path.name}:{position}")
                continue
            raw = raw_rows[row_idx]
            identity = sample_id(raw["img_id"])
            if {
                identity,
                str(ground.get("sample_id", "")),
                str(row.get("sample_id", "")),
            } != {identity}:
                errors.append(f"sample_id mismatch {ground_path.name}:{position}")
            expected_row_idx = (
                int(expected_rows[position]["row_idx"])
                if selected_by_shard is not None
                else position
            )
            expected_identity = (
                str(expected_rows[position]["sample_id"])
                if selected_by_shard is not None
                else identity
            )
            if (
                int(row["row_idx"]) != row_idx
                or row_idx != expected_row_idx
                or identity != expected_identity
            ):
                errors.append(f"row_idx mismatch {ground_path.name}:{position}")
            if str(ground.get("prompt_version", "")) != GROUND_PROMPT_VERSION:
                errors.append(f"ground prompt version mismatch {ground_path.name}:{position}")
            if str(row.get("mask_policy_version", "")) != MASK_POLICY_VERSION:
                errors.append(f"mask policy version mismatch {ground_path.name}:{position}")

            try:
                payload = json.loads(row["ground_json"])
            except Exception as exc:
                errors.append(
                    f"invalid ground_json {ground_path.name}:{position}: {exc!r}"
                )
                payload = {}
            mode = str(payload.get("mask_mode", "unresolved"))
            qc_flag = str(row.get("qc_flag", ""))
            ground_status = str(ground.get("grounding_status", ""))
            sample_ids.append(identity)
            qc_flags[qc_flag] += 1
            mask_sources[str(row.get("mask_source", ""))] += 1
            mask_modes[mode] += 1
            tasks[str(row.get("final_task", ""))] += 1
            grounding_statuses[ground_status] += 1
            if qc_flag != "OK" or ground_status != "OK":
                problem_samples.append(
                    {
                        "sample_id": identity,
                        "shard": ground_path.name,
                        "row_idx": row_idx,
                        "grounding_status": ground_status,
                        "qc_flag": qc_flag,
                        "qc_flags": json.loads(row.get("qc_flags_json") or "[]"),
                    }
                )

            source = decode_image(raw["source_img"])
            target = decode_image(raw["target_img"])
            if source.size != target.size:
                errors.append(f"source/target size mismatch {ground_path.name}:{position}")
            mask_payload = row.get("mask_png") or b""
            if not mask_payload:
                for message in _placeholder_errors(row):
                    errors.append(f"{message} {ground_path.name}:{position}")
                areas.append(0.0)
                continue

            mask = Image.open(io.BytesIO(mask_payload)).convert("L")
            if mask.size != source.size or mask.size != (
                int(row["mask_width"]),
                int(row["mask_height"]),
            ):
                errors.append(f"mask size mismatch {ground_path.name}:{position}")
                continue
            mask_array = np.asarray(mask)
            mask_sum = int(np.count_nonzero(mask_array))
            pixel_count = mask.width * mask.height
            area = mask_sum / max(pixel_count, 1)
            if not set(np.unique(mask_array)).issubset({0, 255}):
                errors.append(f"non-binary mask PNG {ground_path.name}:{position}")
            if mask_sum != int(row["mask_sum"]):
                errors.append(f"mask_sum mismatch {ground_path.name}:{position}")
            if not math.isclose(area, float(row["area_frac"]), abs_tol=1e-12):
                errors.append(f"area_frac mismatch {ground_path.name}:{position}")
            if mode != "full_image" and mask_sum == 0:
                errors.append(f"empty regional mask {ground_path.name}:{position}")
            for instance in row.get("instance_masks") or []:
                instance_count += 1
                rle = {
                    "size": [int(value) for value in instance["rle_size"]],
                    "counts": instance["rle_counts"].encode("ascii"),
                }
                decoded = mask_utils.decode(rle)
                if tuple(decoded.shape) != (source.height, source.width):
                    errors.append(f"RLE size mismatch {ground_path.name}:{position}")
                if int(decoded.sum()) != int(instance["area"]):
                    errors.append(f"RLE area mismatch {ground_path.name}:{position}")
            areas.append(area)

    duplicates = len(sample_ids) - len(set(sample_ids))
    if duplicates:
        errors.append(f"duplicate sample_id rows: {duplicates}")
    expected_ids = (
        {
            str(row["sample_id"])
            for rows in selected_by_shard.values()
            for row in rows
        }
        if selected_by_shard is not None
        else {sample_id(value) for value in range(18_249)}
    )
    missing_ids = sorted(expected_ids - set(sample_ids))
    unexpected_ids = sorted(set(sample_ids) - expected_ids)
    if missing_ids:
        errors.append(f"missing sample_id rows: {len(missing_ids)}")
    if unexpected_ids:
        errors.append(f"unexpected sample_id rows: {len(unexpected_ids)}")

    report = {
        "source_shards": len(source_paths),
        "grounding_shards": len(ground_paths),
        "mask_shards": len(mask_paths),
        "rows": len(sample_ids),
        "unique_sample_ids": len(set(sample_ids)),
        "instances": instance_count,
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "missing_sample_ids": missing_ids,
        "unexpected_sample_ids": unexpected_ids,
        "grounding_statuses": dict(sorted(grounding_statuses.items())),
        "qc_flags": dict(sorted(qc_flags.items())),
        "mask_sources": dict(sorted(mask_sources.items())),
        "mask_modes": dict(sorted(mask_modes.items())),
        "tasks": dict(sorted(tasks.items())),
        "area_frac": {
            "min": min(areas) if areas else None,
            "median": statistics.median(areas) if areas else None,
            "mean": statistics.fmean(areas) if areas else None,
            "max": max(areas) if areas else None,
            "empty": sum(value == 0 for value in areas),
            "full": sum(value == 1 for value in areas),
        },
        "problem_sample_count": len(problem_samples),
        "problem_samples": problem_samples,
        "selection_manifest_dir": (
            str(args.selection_manifest_dir) if args.selection_manifest_dir else None
        ),
        "expected_selected_rows": len(expected_ids),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
