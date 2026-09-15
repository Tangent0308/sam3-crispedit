#!/usr/bin/env python3
"""Validate RefEdit quality-prefilter audit and PASS manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from refedit.io import discover_shards, sample_id
from refedit.quality_prefilter import QUALITY_PROMPT_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.quality_dir = args.quality_dir.resolve()
    args.report_json = args.report_json.resolve()

    source_by_name = {path.name: path for path in discover_shards(args.input_dir)}
    audit_by_name = {
        path.name: path for path in (args.quality_dir / "audit").glob("train-*.parquet")
    }
    manifest_by_name = {
        path.name: path
        for path in (args.quality_dir / "manifest").glob("train-*.parquet")
    }
    errors = []
    if set(audit_by_name) != set(source_by_name):
        errors.append("quality audit shard names do not match source shards")
    if set(manifest_by_name) != set(source_by_name):
        errors.append("quality manifest shard names do not match source shards")

    verdicts = Counter()
    dimensions = Counter()
    reasons = Counter()
    identities = []
    manifest_rows_total = 0
    parse_errors = 0
    for shard_name, source_path in tqdm(
        sorted(source_by_name.items()),
        desc="Validate RefEdit prefilter",
        dynamic_ncols=True,
    ):
        audit_path = audit_by_name.get(shard_name)
        manifest_path = manifest_by_name.get(shard_name)
        if audit_path is None or manifest_path is None:
            continue
        source_rows = pq.read_table(
            source_path, columns=["img_id", "instruction"]
        ).to_pylist()
        audit_rows = pq.read_table(audit_path).to_pylist()
        manifest_rows = pq.read_table(manifest_path).to_pylist()
        if len(audit_rows) != len(source_rows):
            errors.append(
                f"source/audit row count mismatch {shard_name}: "
                f"{len(source_rows)}!={len(audit_rows)}"
            )
            continue

        pass_rows = []
        for position, (source, audit) in enumerate(zip(source_rows, audit_rows)):
            identity = sample_id(source["img_id"])
            if (
                int(audit["row_idx"]) != position
                or str(audit["sample_id"]) != identity
                or int(audit["img_id"]) != int(source["img_id"])
                or str(audit["instruction"]) != str(source["instruction"])
                or Path(str(audit["source_relative_path"])).name != shard_name
            ):
                errors.append(f"source/audit mismatch {shard_name}:{position}")
            verdict = str(audit["verdict"])
            if bool(audit["keep"]) != (verdict == "PASS"):
                errors.append(f"keep/verdict mismatch {shard_name}:{position}")
            if str(audit["prompt_version"]) != QUALITY_PROMPT_VERSION:
                errors.append(f"prompt version mismatch {shard_name}:{position}")
            if not bool(audit["parse_ok"]):
                parse_errors += 1
            verdicts[verdict] += 1
            identities.append(identity)
            for value in json.loads(audit.get("failed_dimensions_json") or "[]"):
                dimensions[str(value)] += 1
            for value in json.loads(audit.get("reason_codes_json") or "[]"):
                reasons[str(value)] += 1
            if verdict == "PASS":
                pass_rows.append(audit)

        if len(manifest_rows) != len(pass_rows):
            errors.append(
                f"PASS/manifest row count mismatch {shard_name}: "
                f"{len(pass_rows)}!={len(manifest_rows)}"
            )
            continue
        for position, (manifest, audit) in enumerate(zip(manifest_rows, pass_rows)):
            if (
                str(manifest["sample_id"]) != str(audit["sample_id"])
                or int(manifest["row_idx"]) != int(audit["row_idx"])
                or int(manifest["img_id"]) != int(audit["img_id"])
                or str(manifest["instruction"]) != str(audit["instruction"])
                or str(manifest["prefilter_verdict"]) != "PASS"
                or str(manifest["prefilter_prompt_version"])
                != QUALITY_PROMPT_VERSION
            ):
                errors.append(f"audit/manifest mismatch {shard_name}:{position}")
        manifest_rows_total += len(manifest_rows)

    duplicates = len(identities) - len(set(identities))
    if duplicates:
        errors.append(f"duplicate sample IDs: {duplicates}")
    report = {
        "source_shards": len(source_by_name),
        "audit_shards": len(audit_by_name),
        "manifest_shards": len(manifest_by_name),
        "rows": len(identities),
        "unique_sample_ids": len(set(identities)),
        "manifest_rows": manifest_rows_total,
        "parse_errors": parse_errors,
        "verdicts": dict(sorted(verdicts.items())),
        "failed_dimensions": dict(dimensions.most_common()),
        "reason_codes": dict(reasons.most_common()),
        "prompt_version": QUALITY_PROMPT_VERSION,
        "validation_error_count": len(errors),
        "validation_errors": errors,
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
