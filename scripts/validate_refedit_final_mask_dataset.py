#!/usr/bin/env python3
"""Validate the strict final RefEdit mask dataset and its manifest."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.final_dir = args.final_dir.resolve()
    args.report_json = args.report_json.resolve()
    if args.final_dir == args.input_dir or args.input_dir in args.final_dir.parents:
        raise ValueError("final mask dataset must not be inside RefEdit source data")

    source_by_name = {path.name: path for path in discover_shards(args.input_dir)}
    mask_paths = sorted((args.final_dir / "data").glob("train-*.parquet"))
    manifest_path = args.final_dir / "audit" / "final_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_rows = pq.read_table(manifest_path).to_pylist()
    manifest_index = {
        (
            Path(row["mask_relative_path"]).name,
            int(row["row_idx"]),
            str(row["sample_id"]),
        ): row
        for row in manifest_rows
    }

    errors = []
    identities = []
    areas = []
    instances = 0
    tasks = Counter()
    sources = Counter()
    if {path.name for path in mask_paths} != set(source_by_name):
        errors.append("final mask shard names do not exactly match source shard names")
    if len(manifest_index) != len(manifest_rows):
        errors.append("duplicate rows in final manifest")

    matched_manifest = set()
    for mask_path in tqdm(
        mask_paths, desc="Validate final RefEdit masks", dynamic_ncols=True
    ):
        source_path = source_by_name.get(mask_path.name)
        if source_path is None:
            continue
        source_rows = pq.read_table(
            source_path, columns=["img_id", "instruction"]
        ).to_pylist()
        previous_row_idx = -1
        for position, row in enumerate(pq.read_table(mask_path).to_pylist()):
            row_idx = int(row["row_idx"])
            identity = str(row["sample_id"])
            key = (mask_path.name, row_idx, identity)
            manifest = manifest_index.get(key)
            if manifest is None:
                errors.append(f"missing final manifest row {key}")
                continue
            matched_manifest.add(key)
            if row_idx < 0 or row_idx >= len(source_rows):
                errors.append(f"row_idx out of range {mask_path.name}:{position}")
                continue
            if row_idx <= previous_row_idx:
                errors.append(f"rows are not source-ordered {mask_path.name}:{position}")
            previous_row_idx = row_idx
            raw = source_rows[row_idx]
            if identity != sample_id(raw["img_id"]):
                errors.append(f"source identity mismatch {mask_path.name}:{position}")
            if str(row["final_instruction"]) != str(raw["instruction"]):
                errors.append(f"instruction mismatch {mask_path.name}:{position}")
            if str(row["qc_flag"]) != "OK" or str(row["grounding_status"]) != "OK":
                errors.append(f"non-OK row in final data {mask_path.name}:{position}")
            if str(row["prompt_version"]) != GROUND_PROMPT_VERSION:
                errors.append(f"ground policy mismatch {mask_path.name}:{position}")
            if str(row["mask_policy_version"]) != MASK_POLICY_VERSION:
                errors.append(f"mask policy mismatch {mask_path.name}:{position}")
            if str(manifest["prefilter_verdict"]) != "PASS":
                errors.append(f"non-PASS prefilter row {mask_path.name}:{position}")
            if (
                str(manifest["grounding_status"]) != str(row["grounding_status"])
                or str(manifest["mask_qc_flag"]) != str(row["qc_flag"])
                or str(manifest["mask_source"]) != str(row["mask_source"])
                or str(manifest["instruction"]) != str(row["final_instruction"])
            ):
                errors.append(f"manifest/mask mismatch {mask_path.name}:{position}")

            payload = row.get("mask_png") or b""
            if not payload:
                errors.append(f"missing mask PNG {mask_path.name}:{position}")
                continue
            mask = np.asarray(Image.open(io.BytesIO(payload)).convert("L"))
            if not set(np.unique(mask)).issubset({0, 255}):
                errors.append(f"non-binary mask {mask_path.name}:{position}")
            expected_size = (int(row["mask_height"]), int(row["mask_width"]))
            if mask.shape != expected_size:
                errors.append(f"mask size mismatch {mask_path.name}:{position}")
                continue
            mask_sum = int(np.count_nonzero(mask))
            area = mask_sum / max(mask.size, 1)
            if mask_sum != int(row["mask_sum"]):
                errors.append(f"mask_sum mismatch {mask_path.name}:{position}")
            if not math.isclose(area, float(row["area_frac"]), abs_tol=1e-12):
                errors.append(f"area mismatch {mask_path.name}:{position}")
            for instance in row.get("instance_masks") or []:
                instances += 1
                decoded = mask_utils.decode(
                    {
                        "size": [int(value) for value in instance["rle_size"]],
                        "counts": instance["rle_counts"].encode("ascii"),
                    }
                )
                if tuple(decoded.shape) != expected_size:
                    errors.append(f"RLE shape mismatch {mask_path.name}:{position}")
                if int(decoded.sum()) != int(instance["area"]):
                    errors.append(f"RLE area mismatch {mask_path.name}:{position}")
            identities.append(identity)
            areas.append(area)
            tasks[str(row["final_task"])] += 1
            sources[str(row["mask_source"])] += 1

    missing_manifest = sorted(set(manifest_index) - matched_manifest)
    if missing_manifest:
        errors.append(f"manifest rows missing from final data: {len(missing_manifest)}")
    duplicates = len(identities) - len(set(identities))
    if duplicates:
        errors.append(f"duplicate sample IDs: {duplicates}")
    report = {
        "source_shards": len(source_by_name),
        "mask_shards": len(mask_paths),
        "manifest_rows": len(manifest_rows),
        "rows": len(identities),
        "unique_sample_ids": len(set(identities)),
        "instances": instances,
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "tasks": dict(sorted(tasks.items())),
        "mask_sources": dict(sorted(sources.items())),
        "area_frac": {
            "min": min(areas) if areas else None,
            "median": statistics.median(areas) if areas else None,
            "mean": statistics.fmean(areas) if areas else None,
            "max": max(areas) if areas else None,
        },
        "ground_prompt_version": GROUND_PROMPT_VERSION,
        "mask_policy_version": MASK_POLICY_VERSION,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
