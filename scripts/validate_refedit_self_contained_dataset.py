#!/usr/bin/env python3
"""Validate the self-contained RefEdit final mask dataset with embedded images."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from pycocotools import mask as mask_utils
from tqdm import tqdm

from refedit import GROUND_PROMPT_VERSION, MASK_POLICY_VERSION
from refedit.io import sample_id
from scripts.build_refedit_self_contained_dataset import (
    MASK_COORDINATE_SPACE,
    SELF_CONTAINED_SCHEMA,
)
from scripts.build_unified_mask_dataset import (
    _decode_image,
    _decode_mask,
    _extract_image_bytes,
    _parse_ground_json,
)


def validate_self_contained_dataset(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir).resolve()
    data_dir = dataset_dir / "data"
    manifest_path = dataset_dir / "audit" / "final_manifest.parquet"
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    shard_paths = sorted(data_dir.glob("train-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"no train-*.parquet shards under {data_dir}")
    manifest_rows = pq.read_table(manifest_path).to_pylist()
    manifest_index = {
        (Path(str(row["mask_relative_path"])).name, int(row["row_idx"]), str(row["sample_id"])): row
        for row in manifest_rows
    }

    errors = []
    identities = []
    areas = []
    instances = 0
    tasks = Counter()
    mask_sources = Counter()
    mask_modes = Counter()
    format_pairs = Counter()
    matched_manifest = set()

    manifest_shards = {Path(str(row["mask_relative_path"])).name for row in manifest_rows}
    if {path.name for path in shard_paths} != manifest_shards:
        errors.append("self-contained shard names do not exactly match final manifest shard names")

    for shard_path in tqdm(shard_paths, desc="Validate self-contained RefEdit", dynamic_ncols=True):
        parquet = pq.ParquetFile(shard_path)
        if not parquet.schema_arrow.equals(SELF_CONTAINED_SCHEMA, check_metadata=True):
            errors.append(f"schema mismatch {shard_path.name}")
            continue
        rows = pq.read_table(shard_path).to_pylist()
        previous_row_idx = -1
        for position, row in enumerate(rows):
            row_idx = int(row["row_idx"])
            identity = str(row["sample_id"])
            img_id = int(row["img_id"])
            key = (shard_path.name, row_idx, identity)
            manifest_row = manifest_index.get(key)
            if manifest_row is None:
                errors.append(f"missing final manifest row {key}")
                continue
            matched_manifest.add(key)
            if row_idx <= previous_row_idx:
                errors.append(f"rows are not source-ordered {shard_path.name}:{position}")
            previous_row_idx = row_idx
            if identity != sample_id(img_id):
                errors.append(f"sample_id/img_id mismatch {shard_path.name}:{position}")
            if str(row["source_relative_path"]) != str(manifest_row["source_relative_path"]):
                errors.append(f"source_relative_path mismatch {shard_path.name}:{position}")
            if str(row["instruction"]) != str(manifest_row["instruction"]):
                errors.append(f"instruction mismatch {shard_path.name}:{position}")
            if str(row["final_instruction"]) != str(manifest_row["instruction"]):
                errors.append(f"final_instruction mismatch {shard_path.name}:{position}")
            if str(row["prefilter_verdict"]) != "PASS":
                errors.append(f"non-PASS row {shard_path.name}:{position}")
            if str(manifest_row["prefilter_verdict"]) != str(row["prefilter_verdict"]):
                errors.append(f"manifest prefilter mismatch {shard_path.name}:{position}")
            if str(row["grounding_status"]) != "OK":
                errors.append(f"non-OK grounding row {shard_path.name}:{position}")
            if str(row["qc_flag"]) != "OK":
                errors.append(f"non-OK qc row {shard_path.name}:{position}")
            if str(row["quality_status"]) != "strict_pass":
                errors.append(f"unexpected quality_status {shard_path.name}:{position}")
            if str(row["prompt_version"]) != GROUND_PROMPT_VERSION:
                errors.append(f"ground prompt version mismatch {shard_path.name}:{position}")
            if str(row["mask_policy_version"]) != MASK_POLICY_VERSION:
                errors.append(f"mask policy version mismatch {shard_path.name}:{position}")
            if str(row["mask_coordinate_space"]) != MASK_COORDINATE_SPACE:
                errors.append(f"mask coordinate space mismatch {shard_path.name}:{position}")
            if str(row["mask_source"]) != str(manifest_row["mask_source"]):
                errors.append(f"manifest mask_source mismatch {shard_path.name}:{position}")
            if not math.isclose(float(row["area_frac"]), float(manifest_row["area_frac"]), abs_tol=1e-12):
                errors.append(f"manifest area_frac mismatch {shard_path.name}:{position}")

            try:
                ground_payload = _parse_ground_json(row["ground_json"])
            except Exception as exc:
                errors.append(f"invalid ground_json {shard_path.name}:{position}: {exc!r}")
                continue
            mask_mode = str(ground_payload.get("mask_mode") or "")
            if not mask_mode:
                errors.append(f"missing mask_mode {shard_path.name}:{position}")
            elif mask_mode != str(row["mask_mode"]):
                errors.append(f"mask_mode mismatch {shard_path.name}:{position}")

            source_payload = _extract_image_bytes(row["source_img"])
            target_payload = _extract_image_bytes(row["target_img"])
            mask_payload = bytes(row.get("mask_png") or b"")
            if not source_payload or not target_payload or not mask_payload:
                errors.append(f"missing embedded payload {shard_path.name}:{position}")
                continue
            try:
                source_width, source_height, source_format = _decode_image(source_payload)
                target_width, target_height, target_format = _decode_image(target_payload)
            except Exception as exc:
                errors.append(f"invalid embedded image {shard_path.name}:{position}: {exc!r}")
                continue
            if (source_width, source_height) != (target_width, target_height):
                errors.append(f"source/target size mismatch {shard_path.name}:{position}")
            if source_width != int(row["source_width"]) or source_height != int(row["source_height"]):
                errors.append(f"source size metadata mismatch {shard_path.name}:{position}")
            if target_width != int(row["target_width"]) or target_height != int(row["target_height"]):
                errors.append(f"target size metadata mismatch {shard_path.name}:{position}")
            if source_format != str(row["source_format"]):
                errors.append(f"source format mismatch {shard_path.name}:{position}")
            if target_format != str(row["target_format"]):
                errors.append(f"target format mismatch {shard_path.name}:{position}")
            source_ar = source_width / max(source_height, 1)
            target_ar = target_width / max(target_height, 1)
            ar_delta = abs(target_ar / max(source_ar, 1e-8) - 1.0)
            if not math.isclose(ar_delta, float(row["ar_delta"]), abs_tol=1e-12):
                errors.append(f"ar_delta mismatch {shard_path.name}:{position}")

            try:
                mask_width, mask_height, mask_sum, mask_values = _decode_mask(mask_payload)
            except Exception as exc:
                errors.append(f"invalid mask_png {shard_path.name}:{position}: {exc!r}")
                continue
            if mask_values - {0, 255}:
                errors.append(f"non-binary mask {shard_path.name}:{position}")
            if (mask_width, mask_height) != (source_width, source_height):
                errors.append(f"mask/source size mismatch {shard_path.name}:{position}")
            if mask_width != int(row["mask_width"]) or mask_height != int(row["mask_height"]):
                errors.append(f"mask size metadata mismatch {shard_path.name}:{position}")
            if mask_sum != int(row["mask_sum"]):
                errors.append(f"mask_sum mismatch {shard_path.name}:{position}")
            area = mask_sum / max(mask_width * mask_height, 1)
            if not math.isclose(area, float(row["area_frac"]), abs_tol=1e-12):
                errors.append(f"area_frac mismatch {shard_path.name}:{position}")
            for instance in row.get("instance_masks") or []:
                instances += 1
                decoded = mask_utils.decode(
                    {
                        "size": [int(value) for value in instance["rle_size"]],
                        "counts": instance["rle_counts"].encode("ascii"),
                    }
                )
                if tuple(decoded.shape) != (source_height, source_width):
                    errors.append(f"RLE size mismatch {shard_path.name}:{position}")
                if int(decoded.sum()) != int(instance["area"]):
                    errors.append(f"RLE area mismatch {shard_path.name}:{position}")
            identities.append(identity)
            areas.append(area)
            tasks[str(row["final_task"])] += 1
            mask_sources[str(row["mask_source"])] += 1
            mask_modes[str(row["mask_mode"])] += 1
            format_pairs[f"{source_format}->{target_format}"] += 1

    missing_manifest = sorted(set(manifest_index) - matched_manifest)
    if missing_manifest:
        errors.append(f"manifest rows missing from self-contained data: {len(missing_manifest)}")
    duplicates = len(identities) - len(set(identities))
    if duplicates:
        errors.append(f"duplicate sample IDs: {duplicates}")

    return {
        "source_shards": len(shard_paths),
        "manifest_rows": len(manifest_rows),
        "rows": len(identities),
        "unique_sample_ids": len(set(identities)),
        "instances": instances,
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "tasks": dict(sorted(tasks.items())),
        "mask_sources": dict(sorted(mask_sources.items())),
        "mask_modes": dict(sorted(mask_modes.items())),
        "image_format_pairs": dict(sorted(format_pairs.items())),
        "area_frac": {
            "min": min(areas) if areas else None,
            "median": statistics.median(areas) if areas else None,
            "mean": statistics.fmean(areas) if areas else None,
            "max": max(areas) if areas else None,
        },
        "ground_prompt_version": GROUND_PROMPT_VERSION,
        "mask_policy_version": MASK_POLICY_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.report_json = args.report_json.resolve()
    report = validate_self_contained_dataset(args.dataset_dir)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["validation_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
