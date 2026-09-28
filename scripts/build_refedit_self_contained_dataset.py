#!/usr/bin/env python3
"""Build a self-contained RefEdit final mask dataset with embedded source/target images."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from refedit import GROUND_PROMPT_VERSION, MASK_POLICY_VERSION
from refedit.io import discover_shards, sample_id, validate_schema
from scripts.build_unified_mask_dataset import (
    _atomic_write_json,
    _decode_image,
    _decode_mask,
    _extract_image_bytes,
    _parse_ground_json,
    _sha256_file,
    _write_parquet_atomic,
)
from scaleedit.mask_runner import INSTANCE_TYPE


SCHEMA_VERSION = "refedit_self_contained_mask_v1"
QUALITY_POLICY_VERSION = "prefilter_pass_grounding_ok_mask_qc_ok_v1"
MASK_COORDINATE_SPACE = "source_image"
IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _normalize_parquet_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_list(data_type):
        value_field = data_type.value_field
        return pa.list_(
            pa.field(
                "element",
                _normalize_parquet_type(value_field.type),
                nullable=value_field.nullable,
                metadata=value_field.metadata,
            )
        )
    if pa.types.is_large_list(data_type):
        value_field = data_type.value_field
        return pa.large_list(
            pa.field(
                "element",
                _normalize_parquet_type(value_field.type),
                nullable=value_field.nullable,
                metadata=value_field.metadata,
            )
        )
    if pa.types.is_struct(data_type):
        return pa.struct(
            [
                pa.field(
                    field.name,
                    _normalize_parquet_type(field.type),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
                for field in data_type
            ]
        )
    return data_type


PARQUET_INSTANCE_TYPE = _normalize_parquet_type(INSTANCE_TYPE)

SELF_CONTAINED_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("img_id", pa.int64(), nullable=False),
        pa.field("source_relative_path", pa.string(), nullable=False),
        pa.field("row_idx", pa.int64(), nullable=False),
        pa.field("instruction", pa.string(), nullable=False),
        pa.field("source_img", IMAGE_TYPE, nullable=False),
        pa.field("target_img", IMAGE_TYPE, nullable=False),
        pa.field("edit_task", pa.string(), nullable=False),
        pa.field("final_task", pa.string(), nullable=False),
        pa.field("original_instruction", pa.string(), nullable=False),
        pa.field("final_instruction", pa.string(), nullable=False),
        pa.field("ground_json", pa.string(), nullable=False),
        pa.field("mask_png", pa.binary(), nullable=False),
        pa.field(
            "instance_masks",
            pa.list_(pa.field("element", PARQUET_INSTANCE_TYPE)),
            nullable=False,
        ),
        pa.field("mask_mode", pa.string(), nullable=False),
        pa.field("mask_coordinate_space", pa.string(), nullable=False),
        pa.field("mask_source", pa.string(), nullable=False),
        pa.field("area_frac", pa.float64(), nullable=False),
        pa.field("qc_flag", pa.string(), nullable=False),
        pa.field("qc_flags_json", pa.string(), nullable=False),
        pa.field("mask_height", pa.int32(), nullable=False),
        pa.field("mask_width", pa.int32(), nullable=False),
        pa.field("mask_sum", pa.int64(), nullable=False),
        pa.field("source_width", pa.int32(), nullable=False),
        pa.field("source_height", pa.int32(), nullable=False),
        pa.field("target_width", pa.int32(), nullable=False),
        pa.field("target_height", pa.int32(), nullable=False),
        pa.field("source_format", pa.string(), nullable=False),
        pa.field("target_format", pa.string(), nullable=False),
        pa.field("ar_delta", pa.float64(), nullable=False),
        pa.field("grounding_status", pa.string(), nullable=False),
        pa.field("mllm_model", pa.string()),
        pa.field("prompt_version", pa.string()),
        pa.field("sam_version", pa.string()),
        pa.field("mask_policy_version", pa.string()),
        pa.field("mask_seconds", pa.float64()),
        pa.field("prefilter_verdict", pa.string(), nullable=False),
        pa.field("prefilter_confidence", pa.float32()),
        pa.field("prefilter_model_name", pa.string()),
        pa.field("prefilter_prompt_version", pa.string()),
        pa.field("quality_status", pa.string(), nullable=False),
    ],
    metadata={
        b"schema_version": SCHEMA_VERSION.encode(),
        b"quality_policy_version": QUALITY_POLICY_VERSION.encode(),
        b"mask_semantics": b"255=editable,0=preserve; source-image coordinates",
    },
)


def _image_cell(payload: bytes) -> dict[str, bytes | None]:
    return {"bytes": payload, "path": None}


def _manifest_key(shard_name: str, row_idx: int, identity: str) -> tuple[str, int, str]:
    return shard_name, int(row_idx), str(identity)


def _load_manifest_index(final_dir: Path) -> tuple[Path, dict[tuple[str, int, str], dict]]:
    manifest_path = final_dir / "audit" / "final_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = pq.read_table(manifest_path).to_pylist()
    index: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        key = _manifest_key(
            Path(str(row["mask_relative_path"])).name,
            int(row["row_idx"]),
            str(row["sample_id"]),
        )
        if key in index:
            raise ValueError(f"duplicate final manifest row: {key}")
        index[key] = row
    return manifest_path, index


def _copy_audit_files(final_dir: Path, output_root: Path) -> None:
    audit_root = output_root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    for name in ("final_manifest.parquet", "rejected_mask_qc.parquet"):
        src = final_dir / "audit" / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, audit_root / name)


def _schema_json() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "columns": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in SELF_CONTAINED_SCHEMA
        ],
        "mask_coordinate_space": MASK_COORDINATE_SPACE,
    }


def _write_readme(output_root: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# RefEdit self-contained final mask dataset

This directory is generated by `scripts/build_refedit_self_contained_dataset.py`.
Each accepted row embeds `source_img`, `target_img`, and `mask_png` directly in the
same parquet row; no external RefEdit source shards are required at read time.

- Schema: `{SCHEMA_VERSION}`
- Quality policy: `{QUALITY_POLICY_VERSION}`
- Rows: {summary['rows']:,}
- Shards: {summary['output_shards']}
- Total embedded image bytes: {summary['image_bytes']:,}
- Total embedded mask bytes: {summary['mask_bytes']:,}

Core columns:
- `sample_id`
- `instruction`
- `source_img`
- `target_img`
- `mask_png`
- `final_task`

All rows satisfy:
- `prefilter_verdict == PASS`
- `grounding_status == OK`
- `qc_flag == OK`
- binary `mask_png` in source-image coordinates
"""
    path = output_root / "README.md"
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _validate_output(output_root: Path, reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities: set[str] = set()
    checked_rows = 0
    for report in reports:
        data_path = output_root / str(report["data_path"])
        parquet = pq.ParquetFile(data_path)
        if not parquet.schema_arrow.equals(SELF_CONTAINED_SCHEMA, check_metadata=True):
            raise ValueError(f"schema mismatch in {data_path}")
        if parquet.metadata.num_rows != int(report["rows"]):
            raise ValueError(f"row count mismatch in {data_path}")
        columns = pq.read_table(
            data_path,
            columns=[
                "sample_id",
                "img_id",
                "prefilter_verdict",
                "grounding_status",
                "qc_flag",
                "quality_status",
            ],
        ).to_pydict()
        for identity, img_id, verdict, ground_status, qc_flag, quality_status in zip(
            columns["sample_id"],
            columns["img_id"],
            columns["prefilter_verdict"],
            columns["grounding_status"],
            columns["qc_flag"],
            columns["quality_status"],
        ):
            expected = sample_id(img_id)
            if str(identity) != expected:
                raise ValueError(f"sample_id/img_id mismatch in {data_path}: {identity!r}")
            if identity in identities:
                raise ValueError(f"duplicate sample_id in output: {identity}")
            identities.add(identity)
            if verdict != "PASS":
                raise ValueError(f"non-PASS row in {data_path}: {identity}")
            if ground_status != "OK":
                raise ValueError(f"non-OK grounding row in {data_path}: {identity}")
            if qc_flag != "OK":
                raise ValueError(f"non-OK qc row in {data_path}: {identity}")
            if quality_status != "strict_pass":
                raise ValueError(f"unexpected quality status in {data_path}: {identity}")
        checked_rows += parquet.metadata.num_rows
    return {
        "checked_shards": len(reports),
        "checked_rows": checked_rows,
        "unique_sample_ids": len(identities),
        "schema_consistent": True,
        "all_rows_strict_pass": True,
    }


def _build_row(
    shard_name: str,
    raw_row: Mapping[str, Any],
    final_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int]:
    row_idx = int(final_row["row_idx"])
    img_id = int(raw_row["img_id"])
    identity = sample_id(img_id)
    if str(final_row["sample_id"]) != identity:
        raise ValueError(f"sample_id mismatch in {shard_name}:{row_idx}")
    if str(manifest_row["sample_id"]) != identity:
        raise ValueError(f"manifest sample_id mismatch in {shard_name}:{row_idx}")
    if Path(str(final_row["source_relative_path"])).name != shard_name:
        raise ValueError(f"row/source shard mismatch in {shard_name}:{row_idx}")
    if Path(str(manifest_row["source_relative_path"])).name != shard_name:
        raise ValueError(f"manifest/source shard mismatch in {shard_name}:{row_idx}")
    instruction = str(raw_row["instruction"])
    if str(final_row["final_instruction"]) != instruction:
        raise ValueError(f"final/source instruction mismatch in {shard_name}:{row_idx}")
    if str(manifest_row["instruction"]) != instruction:
        raise ValueError(f"manifest/source instruction mismatch in {shard_name}:{row_idx}")
    if str(manifest_row["prefilter_verdict"]) != "PASS":
        raise ValueError(f"non-PASS manifest row in {shard_name}:{row_idx}")
    if str(final_row["grounding_status"]) != "OK":
        raise ValueError(f"non-OK grounding row in {shard_name}:{row_idx}")
    if str(final_row["qc_flag"]) != "OK":
        raise ValueError(f"non-OK qc row in {shard_name}:{row_idx}")
    if str(final_row["prompt_version"]) != GROUND_PROMPT_VERSION:
        raise ValueError(f"ground prompt version mismatch in {shard_name}:{row_idx}")
    if str(final_row["mask_policy_version"]) != MASK_POLICY_VERSION:
        raise ValueError(f"mask policy version mismatch in {shard_name}:{row_idx}")
    if str(manifest_row["ground_prompt_version"]) != str(final_row["prompt_version"]):
        raise ValueError(f"manifest/row prompt version mismatch in {shard_name}:{row_idx}")
    if str(manifest_row["mask_policy_version"]) != str(final_row["mask_policy_version"]):
        raise ValueError(f"manifest/row mask policy mismatch in {shard_name}:{row_idx}")
    if str(manifest_row["mask_qc_flag"]) != str(final_row["qc_flag"]):
        raise ValueError(f"manifest/row qc mismatch in {shard_name}:{row_idx}")
    if str(manifest_row["grounding_status"]) != str(final_row["grounding_status"]):
        raise ValueError(f"manifest/row grounding mismatch in {shard_name}:{row_idx}")

    source_bytes = _extract_image_bytes(raw_row["source_img"])
    target_bytes = _extract_image_bytes(raw_row["target_img"])
    mask_bytes = bytes(final_row.get("mask_png") or b"")
    if not source_bytes:
        raise ValueError(f"missing source_img bytes in {shard_name}:{row_idx}")
    if not target_bytes:
        raise ValueError(f"missing target_img bytes in {shard_name}:{row_idx}")
    if not mask_bytes:
        raise ValueError(f"missing mask_png bytes in {shard_name}:{row_idx}")

    source_width, source_height, source_format = _decode_image(source_bytes)
    target_width, target_height, target_format = _decode_image(target_bytes)
    if (source_width, source_height) != (target_width, target_height):
        raise ValueError(f"source/target size mismatch in {shard_name}:{row_idx}")
    source_ar = source_width / max(source_height, 1)
    target_ar = target_width / max(target_height, 1)
    ar_delta = abs(target_ar / max(source_ar, 1e-8) - 1.0)
    recorded_ar_delta = float(final_row.get("ar_delta") or 0.0)
    if not math.isclose(ar_delta, recorded_ar_delta, abs_tol=1e-12):
        raise ValueError(f"ar_delta mismatch in {shard_name}:{row_idx}")

    ground_payload = _parse_ground_json(final_row["ground_json"])
    mask_mode = str(ground_payload.get("mask_mode") or "")
    if not mask_mode:
        raise ValueError(f"missing mask_mode in {shard_name}:{row_idx}")

    mask_width, mask_height, mask_sum, mask_values = _decode_mask(mask_bytes)
    if mask_values - {0, 255}:
        raise ValueError(f"non-binary mask in {shard_name}:{row_idx}")
    if (mask_width, mask_height) != (source_width, source_height):
        raise ValueError(f"mask/source size mismatch in {shard_name}:{row_idx}")
    if mask_width != int(final_row["mask_width"]) or mask_height != int(final_row["mask_height"]):
        raise ValueError(f"recorded mask size mismatch in {shard_name}:{row_idx}")
    if mask_sum != int(final_row["mask_sum"]):
        raise ValueError(f"recorded mask_sum mismatch in {shard_name}:{row_idx}")
    area_frac = mask_sum / max(mask_width * mask_height, 1)
    if not math.isclose(area_frac, float(final_row["area_frac"]), abs_tol=1e-12):
        raise ValueError(f"area_frac mismatch in {shard_name}:{row_idx}")

    row = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": identity,
        "img_id": img_id,
        "source_relative_path": str(final_row["source_relative_path"]),
        "row_idx": row_idx,
        "instruction": instruction,
        "source_img": _image_cell(source_bytes),
        "target_img": _image_cell(target_bytes),
        "edit_task": str(final_row.get("edit_task") or final_row.get("final_task") or ""),
        "final_task": str(final_row.get("final_task") or ""),
        "original_instruction": str(final_row.get("original_instruction") or instruction),
        "final_instruction": str(final_row["final_instruction"]),
        "ground_json": str(final_row["ground_json"]),
        "mask_png": mask_bytes,
        "instance_masks": list(final_row.get("instance_masks") or []),
        "mask_mode": mask_mode,
        "mask_coordinate_space": MASK_COORDINATE_SPACE,
        "mask_source": str(final_row.get("mask_source") or ""),
        "area_frac": area_frac,
        "qc_flag": str(final_row["qc_flag"]),
        "qc_flags_json": str(final_row.get("qc_flags_json") or "[]"),
        "mask_height": mask_height,
        "mask_width": mask_width,
        "mask_sum": mask_sum,
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "source_format": source_format,
        "target_format": target_format,
        "ar_delta": ar_delta,
        "grounding_status": str(final_row["grounding_status"]),
        "mllm_model": str(final_row.get("mllm_model") or "") or None,
        "prompt_version": str(final_row.get("prompt_version") or "") or None,
        "sam_version": str(final_row.get("sam_version") or "") or None,
        "mask_policy_version": str(final_row.get("mask_policy_version") or "") or None,
        "mask_seconds": float(final_row["mask_seconds"]) if final_row.get("mask_seconds") is not None else None,
        "prefilter_verdict": str(manifest_row["prefilter_verdict"]),
        "prefilter_confidence": float(manifest_row["prefilter_confidence"]) if manifest_row.get("prefilter_confidence") is not None else None,
        "prefilter_model_name": str(manifest_row.get("prefilter_model_name") or "") or None,
        "prefilter_prompt_version": str(manifest_row.get("prefilter_prompt_version") or "") or None,
        "quality_status": "strict_pass",
    }
    return row, len(source_bytes) + len(target_bytes), len(mask_bytes)


def _process_shard(
    source_path: Path,
    final_dir: Path,
    manifest_index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    shard_name = source_path.name
    final_path = final_dir / "data" / shard_name
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    data_path = output_root / "data" / shard_name
    report_path = output_root / "audit" / "shard_reports" / f"{shard_name}.json"
    if resume and report_path.is_file() and data_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("complete") is True:
            return report
    for path in (data_path, report_path):
        if path.exists() and not resume:
            raise FileExistsError(f"existing output without --resume: {path}")

    validate_schema(source_path)
    raw_rows = pq.read_table(
        source_path, columns=["img_id", "source_img", "instruction", "target_img"]
    ).to_pylist()
    final_rows = pq.read_table(final_path).to_pylist()

    rows: list[dict[str, Any]] = []
    image_bytes = 0
    mask_bytes = 0
    tasks = Counter()
    mask_sources = Counter()
    previous_row_idx = -1
    for position, final_row in enumerate(final_rows):
        row_idx = int(final_row["row_idx"])
        if row_idx <= previous_row_idx:
            raise ValueError(f"rows are not source-ordered in {shard_name}:{position}")
        previous_row_idx = row_idx
        if row_idx < 0 or row_idx >= len(raw_rows):
            raise IndexError(f"row_idx out of range in {shard_name}:{row_idx}")
        raw_row = raw_rows[row_idx]
        identity = sample_id(raw_row["img_id"])
        manifest_row = manifest_index.get(_manifest_key(shard_name, row_idx, identity))
        if manifest_row is None:
            raise KeyError(f"missing manifest row for {shard_name}:{row_idx}:{identity}")
        row, image_payload_bytes, mask_payload_bytes = _build_row(
            shard_name, raw_row, final_row, manifest_row
        )
        rows.append(row)
        image_bytes += image_payload_bytes
        mask_bytes += mask_payload_bytes
        tasks[row["final_task"]] += 1
        mask_sources[row["mask_source"]] += 1

    table = pa.Table.from_pylist(rows, schema=SELF_CONTAINED_SCHEMA)
    data_bytes = _write_parquet_atomic(table, data_path)
    report = {
        "complete": True,
        "source_shard": shard_name,
        "rows": len(rows),
        "data_path": str(data_path.relative_to(output_root)),
        "data_bytes": data_bytes,
        "data_sha256": _sha256_file(data_path),
        "image_bytes": image_bytes,
        "mask_bytes": mask_bytes,
        "tasks": dict(sorted(tasks.items())),
        "mask_sources": dict(sorted(mask_sources.items())),
    }
    _atomic_write_json(report_path, report)
    return report


def build_self_contained_dataset(
    input_dir: Path,
    final_dir: Path,
    output_root: Path,
    *,
    workers: int = 8,
    resume: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_dir).resolve()
    final_dir = Path(final_dir).resolve()
    output_root = Path(output_root).resolve()
    if output_root == input_dir or input_dir in output_root.parents:
        raise ValueError("output root must not be inside RefEdit source data")
    if output_root == final_dir or final_dir in output_root.parents:
        raise ValueError("output root must not be inside RefEdit final sidecar data")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "audit" / "shard_reports").mkdir(parents=True, exist_ok=True)

    source_paths = discover_shards(input_dir)
    final_paths = sorted((final_dir / "data").glob("train-*.parquet"))
    final_names = {path.name for path in final_paths}
    source_names = {path.name for path in source_paths}
    if final_names != source_names:
        missing = sorted(source_names - final_names)
        extra = sorted(final_names - source_names)
        raise ValueError(f"final/source shard mismatch: missing={missing} extra={extra}")

    manifest_path, manifest_index = _load_manifest_index(final_dir)
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_shard = {
            executor.submit(
                _process_shard,
                source_path,
                final_dir,
                manifest_index,
                output_root,
                resume,
            ): source_path.name
            for source_path in source_paths
        }
        for future in as_completed(future_to_shard):
            shard_name = future_to_shard[future]
            report = future.result()
            reports.append(report)
            print(
                f"[{len(reports)}/{len(source_paths)}] {shard_name}: rows={report['rows']}",
                flush=True,
            )

    reports.sort(key=lambda item: item["source_shard"])
    _copy_audit_files(final_dir, output_root)
    _atomic_write_json(output_root / "schema.json", _schema_json())

    source_summary = json.loads((final_dir / "run_summary.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "rows": sum(int(report["rows"]) for report in reports),
        "output_shards": len(reports),
        "image_bytes": sum(int(report["image_bytes"]) for report in reports),
        "mask_bytes": sum(int(report["mask_bytes"]) for report in reports),
        "data_bytes": sum(int(report["data_bytes"]) for report in reports),
        "tasks": dict(
            sorted(
                Counter(
                    key
                    for report in reports
                    for key, value in report["tasks"].items()
                    for _ in range(int(value))
                ).items()
            )
        ),
        "mask_sources": dict(
            sorted(
                Counter(
                    key
                    for report in reports
                    for key, value in report["mask_sources"].items()
                    for _ in range(int(value))
                ).items()
            )
        ),
        "input_dir": str(input_dir),
        "final_dir": str(final_dir),
        "output_dir": str(output_root),
        "source_manifest_path": str(manifest_path),
        "source_final_rows": int(source_summary["final_rows"]),
        "ground_prompt_version": source_summary["ground_prompt_version"],
        "mask_policy_version": source_summary["mask_policy_version"],
    }
    summary["validation"] = _validate_output(output_root, reports)
    _atomic_write_json(output_root / "run_summary.json", summary)
    _write_readme(output_root, summary)
    success_path = output_root / "_SUCCESS"
    success_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    summary = build_self_contained_dataset(
        args.input_dir,
        args.final_dir,
        args.output_dir,
        workers=args.workers,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
