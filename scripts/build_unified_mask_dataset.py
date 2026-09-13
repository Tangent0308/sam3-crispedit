#!/usr/bin/env python3
"""Build a strict, self-contained ScaleEdit + CrispEdit mask training dataset.

The source datasets and mask sidecars are read-only.  Accepted rows are joined by
``source shard + row_idx`` and written with one common Arrow schema.  Rejected
rows contain metadata and reason codes only, never image payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


SCHEMA_VERSION = "edit_mask_train_v1"
QUALITY_POLICY_VERSION = "strict_qc_and_integrity_v1"

DEFAULT_CRISPEDIT_RAW_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M"
)
DEFAULT_CRISPEDIT_MASK_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask"
)
DEFAULT_SCALEEDIT_RAW_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "ScaleEdit-filtered-balanced-final-task-100k"
)
DEFAULT_SCALEEDIT_MASK_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "ScaleEdit-filtered-balanced-final-task-100k-vllm-labeled/masks"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "ScaleEdit-CrispEdit-mask-train"
)
DEFAULT_DENYLIST = Path(__file__).with_name("unified_mask_dataset_denylist.json")

CRISPEDIT_TYPE_MAP = {
    "add": "object_addition",
    "background": "background_replacement",
    "background change": "background_replacement",
    "color": "color_change",
    "motion": "action_editing",
    "motion change": "action_editing",
    "remove": "object_removal",
    "replace": "object_replacement",
    "style": "style_transfer",
}

TRAIN_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_row_idx", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("edit_type", pa.string(), nullable=False),
        pa.field("raw_edit_type", pa.string(), nullable=False),
        pa.field("instruction", pa.string(), nullable=False),
        pa.field("original_instruction", pa.string(), nullable=False),
        pa.field("source_image", pa.binary(), nullable=False),
        pa.field("edited_image", pa.binary(), nullable=False),
        pa.field("mask_png", pa.binary(), nullable=False),
        pa.field("source_image_format", pa.string(), nullable=False),
        pa.field("edited_image_format", pa.string(), nullable=False),
        pa.field("source_width", pa.int32(), nullable=False),
        pa.field("source_height", pa.int32(), nullable=False),
        pa.field("edited_width", pa.int32(), nullable=False),
        pa.field("edited_height", pa.int32(), nullable=False),
        pa.field("source_edited_aspect_ratio_delta", pa.float64(), nullable=False),
        pa.field("mask_width", pa.int32(), nullable=False),
        pa.field("mask_height", pa.int32(), nullable=False),
        pa.field("mask_area", pa.int64(), nullable=False),
        pa.field("mask_area_fraction", pa.float64(), nullable=False),
        pa.field("mask_coordinate_space", pa.string(), nullable=False),
        pa.field("mask_mode", pa.string(), nullable=False),
        pa.field("mask_source", pa.string(), nullable=False),
        pa.field("quality_status", pa.string(), nullable=False),
        pa.field("qc_flag", pa.string(), nullable=False),
        pa.field("qc_flags_json", pa.string(), nullable=False),
        pa.field("grounding_status", pa.string(), nullable=False),
        pa.field("ground_json", pa.string(), nullable=False),
        pa.field("instance_masks_json", pa.string(), nullable=False),
        pa.field("mllm_model", pa.string()),
        pa.field("prompt_version", pa.string()),
        pa.field("sam_version", pa.string()),
        pa.field("prefilter_verdict", pa.string()),
        pa.field("prefilter_confidence", pa.float64()),
        pa.field("prefilter_method", pa.string()),
        pa.field("prefilter_model_name", pa.string()),
        pa.field("prefilter_run_id", pa.string()),
        pa.field("metadata_json", pa.string(), nullable=False),
    ],
    metadata={
        b"schema_version": SCHEMA_VERSION.encode(),
        b"quality_policy_version": QUALITY_POLICY_VERSION.encode(),
        b"mask_semantics": b"255=editable,0=preserve; source-image coordinates",
    },
)

REJECT_SCHEMA = pa.schema(
    [
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_row_idx", pa.int64(), nullable=False),
        pa.field("edit_type", pa.string(), nullable=False),
        pa.field("raw_edit_type", pa.string(), nullable=False),
        pa.field("instruction", pa.string(), nullable=False),
        pa.field("qc_flag", pa.string()),
        pa.field("primary_reason", pa.string(), nullable=False),
        pa.field("all_reasons_json", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_root: Path
    mask_root: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=pretty,
        separators=None if pretty else (",", ":"),
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(_json_dumps(value, pretty=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _extract_image_bytes(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
    return None


def _decode_image(payload: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        image_format = _safe_string(image.format).upper()
    if width <= 0 or height <= 0 or not image_format:
        raise ValueError("image has invalid dimensions or format")
    return int(width), int(height), image_format


def _decode_mask(payload: bytes) -> tuple[int, int, int, set[int]]:
    with Image.open(io.BytesIO(payload)) as image:
        image_format = _safe_string(image.format).upper()
        mode = image.mode
        array = np.asarray(image)
    if image_format != "PNG":
        raise ValueError(f"mask format is {image_format or 'unknown'}, expected PNG")
    if mode != "L" or array.ndim != 2:
        raise ValueError(f"mask mode is {mode}, expected single-channel L")
    values = {int(value) for value in np.unique(array)}
    return int(array.shape[1]), int(array.shape[0]), int(np.count_nonzero(array)), values


def _parse_ground_json(value: Any) -> dict[str, Any]:
    text = _safe_string(value)
    if not text:
        raise ValueError("ground_json is empty")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("ground_json is not a JSON object")
    return parsed


def _crispedit_mask_mode(mask_row: Mapping[str, Any], ground: Mapping[str, Any]) -> str:
    status = _safe_string(mask_row.get("grounding_status")).upper()
    mask_source = _safe_string(mask_row.get("mask_source")).lower()
    canonical = _safe_string(mask_row.get("canonical_type")).lower()
    if "FULL_IMAGE" in status or mask_source == "full_image" or canonical == "style":
        return "full_image"
    if canonical == "background":
        observation = ground.get("observation")
        if isinstance(observation, Mapping):
            parsed = observation.get("parsed")
            if isinstance(parsed, Mapping):
                if parsed.get("background_mask_mode") == "full_image":
                    return "full_image"
        if ground.get("background_mask_mode") == "full_image":
            return "full_image"
        return "protect_foreground"
    return "regions"


def _edit_type(dataset: str, raw: Mapping[str, Any], mask: Mapping[str, Any]) -> str:
    if dataset == "scaleedit":
        return _safe_string(mask.get("final_task") or raw.get("final_task"))
    canonical = _safe_string(mask.get("canonical_type")).lower()
    raw_type = _safe_string(mask.get("raw_type") or raw.get("type")).lower()
    return CRISPEDIT_TYPE_MAP.get(canonical) or CRISPEDIT_TYPE_MAP.get(raw_type, canonical)


def _raw_edit_type(dataset: str, raw: Mapping[str, Any], mask: Mapping[str, Any]) -> str:
    if dataset == "scaleedit":
        return _safe_string(raw.get("edit_task") or mask.get("edit_task"))
    return _safe_string(mask.get("raw_type") or raw.get("type"))


def _instruction(dataset: str, raw: Mapping[str, Any], mask: Mapping[str, Any]) -> str:
    if dataset == "scaleedit":
        return _safe_string(mask.get("final_instruction") or raw.get("final_instruction"))
    return _safe_string(mask.get("instruction") or raw.get("instruction"))


def _source_sample_id(dataset: str, shard: str, row_idx: int, raw: Mapping[str, Any]) -> str:
    if dataset == "scaleedit":
        value = _safe_string(raw.get("sample_id"))
        if value:
            return value
    return f"{shard}#{row_idx}"


def _load_denylist(path: Path | None) -> dict[tuple[str, str, int], str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    denylist: dict[tuple[str, str, int], str] = {}
    for entry in entries:
        key = (
            _safe_string(entry["dataset"]).lower(),
            _safe_string(entry["source_shard"]),
            int(entry["source_row_idx"]),
        )
        if key in denylist:
            raise ValueError(f"duplicate denylist entry: {key}")
        denylist[key] = _safe_string(entry.get("reason")) or "manual quality rejection"
    return denylist


def _preliminary_reasons(
    dataset: str,
    shard: str,
    row_idx: int,
    mask: Mapping[str, Any],
    denylist: Mapping[tuple[str, str, int], str],
) -> list[str]:
    manual_reason = denylist.get((dataset, shard, row_idx))
    if manual_reason:
        return [f"manual_exclusion:{manual_reason}"]
    if dataset == "crispedit":
        if _safe_string(mask.get("prefilter_verdict")).upper() != "PASS":
            return ["prefilter_not_pass"]
        if _safe_string(mask.get("filter_decision")).lower() != "keep":
            return ["prefilter_not_keep"]
    qc_flag = _safe_string(mask.get("qc_flag"))
    if qc_flag != "OK":
        return [f"qc_not_ok:{qc_flag or 'missing'}"]
    try:
        mask_sum = int(mask.get("mask_sum") or 0)
    except (TypeError, ValueError):
        mask_sum = 0
    if mask_sum <= 0:
        return ["empty_mask"]
    area = _safe_float(mask.get("area_frac"))
    if area is None or not 0.0 < area <= 1.0:
        return ["invalid_area_fraction"]
    return []


def _source_fields(dataset: str) -> list[str]:
    if dataset == "crispedit":
        return ["input_img", "instruction", "output_img", "type"]
    return [
        "sample_id",
        "split",
        "source_relative_path",
        "manifest_row_index",
        "public_source_row_index",
        "edit_task",
        "final_task",
        "original_instruction",
        "final_instruction",
        "instruction_action",
        "category_action",
        "confidence",
        "source_image",
        "edited_image",
        "source_image_url",
        "source_image_origin",
        "source_image_width",
        "source_image_height",
        "edited_image_width",
        "edited_image_height",
    ]


def _standard_row(
    dataset: str,
    shard: str,
    row_idx: int,
    raw: Mapping[str, Any],
    mask: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    instruction = _instruction(dataset, raw, mask)
    original_instruction = (
        _safe_string(raw.get("original_instruction"))
        if dataset == "scaleedit"
        else instruction
    )
    edit_type = _edit_type(dataset, raw, mask)
    raw_edit_type = _raw_edit_type(dataset, raw, mask)
    if not instruction:
        reasons.append("missing_instruction")
    if not edit_type:
        reasons.append("missing_edit_type")
    if not raw_edit_type:
        reasons.append("missing_raw_edit_type")

    source_value = raw.get("source_image") if dataset == "scaleedit" else raw.get("input_img")
    edited_value = raw.get("edited_image") if dataset == "scaleedit" else raw.get("output_img")
    source_bytes = _extract_image_bytes(source_value)
    edited_bytes = _extract_image_bytes(edited_value)
    mask_bytes = _extract_image_bytes(mask.get("mask_png"))
    if not source_bytes:
        reasons.append("missing_source_image")
    if not edited_bytes:
        reasons.append("missing_edited_image")
    if not mask_bytes:
        reasons.append("missing_mask_png")
    if reasons:
        return None, reasons

    try:
        source_width, source_height, source_format = _decode_image(source_bytes)
    except Exception as error:
        reasons.append(f"invalid_source_image:{type(error).__name__}")
        source_width = source_height = 0
        source_format = ""
    try:
        edited_width, edited_height, edited_format = _decode_image(edited_bytes)
    except Exception as error:
        reasons.append(f"invalid_edited_image:{type(error).__name__}")
        edited_width = edited_height = 0
        edited_format = ""
    try:
        mask_width, mask_height, mask_area, mask_values = _decode_mask(mask_bytes)
    except Exception as error:
        reasons.append(f"invalid_mask_png:{type(error).__name__}")
        mask_width = mask_height = mask_area = 0
        mask_values = set()
    if reasons:
        return None, reasons

    source_aspect_ratio = source_width / max(source_height, 1)
    edited_aspect_ratio = edited_width / max(edited_height, 1)
    aspect_ratio_delta = abs(edited_aspect_ratio / max(source_aspect_ratio, 1e-8) - 1.0)
    if aspect_ratio_delta > 0.02:
        reasons.append("source_edited_aspect_ratio_mismatch")
    if (source_width, source_height) != (mask_width, mask_height):
        reasons.append("source_mask_size_mismatch")
    if mask_values - {0, 255}:
        reasons.append("mask_not_binary_0_255")
    if mask_area <= 0:
        reasons.append("empty_mask")
    recorded_width = int(mask.get("mask_width") or 0)
    recorded_height = int(mask.get("mask_height") or 0)
    recorded_area = int(mask.get("mask_sum") or 0)
    if (recorded_width, recorded_height) != (mask_width, mask_height):
        reasons.append("recorded_mask_size_mismatch")
    if recorded_area != mask_area:
        reasons.append("recorded_mask_area_mismatch")
    recorded_aspect_ratio_delta = _safe_float(mask.get("ar_delta"))
    if recorded_aspect_ratio_delta is None or not math.isclose(
        aspect_ratio_delta, recorded_aspect_ratio_delta, abs_tol=1e-12
    ):
        reasons.append("recorded_aspect_ratio_delta_mismatch")
    actual_fraction = mask_area / max(mask_width * mask_height, 1)
    recorded_fraction = _safe_float(mask.get("area_frac"))
    tolerance = max(1e-12, 1.5 / max(mask_width * mask_height, 1))
    if recorded_fraction is None or not math.isclose(
        actual_fraction, recorded_fraction, abs_tol=tolerance
    ):
        reasons.append("recorded_area_fraction_mismatch")

    source_sample_id = _source_sample_id(dataset, shard, row_idx, raw)
    if dataset == "scaleedit":
        sidecar_id = _safe_string(mask.get("sample_id"))
        if not sidecar_id or sidecar_id != source_sample_id:
            reasons.append("source_sidecar_sample_id_mismatch")
    else:
        source_instruction = _safe_string(raw.get("instruction"))
        sidecar_instruction = _safe_string(mask.get("instruction"))
        if source_instruction != sidecar_instruction:
            reasons.append("source_sidecar_instruction_mismatch")
    try:
        ground = _parse_ground_json(mask.get("ground_json"))
    except Exception as error:
        reasons.append(f"invalid_ground_json:{type(error).__name__}")
        ground = {}
    if reasons:
        return None, reasons

    mask_mode = (
        _safe_string(ground.get("mask_mode"))
        if dataset == "scaleedit"
        else _crispedit_mask_mode(mask, ground)
    )
    if mask_mode not in {"regions", "protect_foreground", "full_image"}:
        return None, ["invalid_mask_mode"]
    if mask_mode == "full_image" and mask_area != mask_width * mask_height:
        return None, ["incomplete_full_image_mask"]

    if dataset == "scaleedit":
        metadata = {
            "source_relative_path": raw.get("source_relative_path"),
            "manifest_row_index": raw.get("manifest_row_index"),
            "public_source_row_index": raw.get("public_source_row_index"),
            "instruction_action": raw.get("instruction_action"),
            "category_action": raw.get("category_action"),
            "category_confidence": raw.get("confidence"),
            "source_image_url": raw.get("source_image_url"),
            "source_image_origin": raw.get("source_image_origin"),
            "mask_policy_version": mask.get("mask_policy_version"),
        }
        split = _safe_string(raw.get("split")) or "train"
        prefilter = {
            "verdict": None,
            "confidence": None,
            "method": None,
            "model_name": None,
            "run_id": None,
        }
    else:
        metadata = {
            "prefilter_evidence_schema": mask.get("prefilter_evidence_schema"),
            "prefilter_reason": mask.get("prefilter_reason"),
            "filter_reason_codes": mask.get("filter_reason_codes"),
            "filter_mismatch_score": mask.get("filter_mismatch_score"),
        }
        split = "train"
        prefilter = {
            "verdict": _safe_string(mask.get("prefilter_verdict")) or None,
            "confidence": _safe_float(mask.get("prefilter_confidence")),
            "method": _safe_string(mask.get("prefilter_method")) or None,
            "model_name": _safe_string(mask.get("prefilter_model_name")) or None,
            "run_id": _safe_string(mask.get("prefilter_run_id")) or None,
        }

    row = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{dataset}:{source_sample_id}",
        "source_dataset": dataset,
        "source_sample_id": source_sample_id,
        "source_shard": shard,
        "source_row_idx": row_idx,
        "split": split,
        "edit_type": edit_type,
        "raw_edit_type": raw_edit_type,
        "instruction": instruction,
        "original_instruction": original_instruction or instruction,
        "source_image": source_bytes,
        "edited_image": edited_bytes,
        "mask_png": mask_bytes,
        "source_image_format": source_format,
        "edited_image_format": edited_format,
        "source_width": source_width,
        "source_height": source_height,
        "edited_width": edited_width,
        "edited_height": edited_height,
        "source_edited_aspect_ratio_delta": aspect_ratio_delta,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "mask_area": mask_area,
        "mask_area_fraction": actual_fraction,
        "mask_coordinate_space": "source_image",
        "mask_mode": mask_mode,
        "mask_source": _safe_string(mask.get("mask_source")),
        "quality_status": "strict_pass",
        "qc_flag": _safe_string(mask.get("qc_flag")),
        "qc_flags_json": _safe_string(mask.get("qc_flags_json")) or "[]",
        "grounding_status": _safe_string(mask.get("grounding_status")),
        "ground_json": _safe_string(mask.get("ground_json")),
        "instance_masks_json": _json_dumps(mask.get("instance_masks") or []),
        "mllm_model": _safe_string(mask.get("mllm_model")) or None,
        "prompt_version": _safe_string(mask.get("prompt_version")) or None,
        "sam_version": _safe_string(mask.get("sam_version")) or None,
        "prefilter_verdict": prefilter["verdict"],
        "prefilter_confidence": prefilter["confidence"],
        "prefilter_method": prefilter["method"],
        "prefilter_model_name": prefilter["model_name"],
        "prefilter_run_id": prefilter["run_id"],
        "metadata_json": _json_dumps(metadata),
    }
    return row, []


def _reject_row(
    dataset: str,
    shard: str,
    row_idx: int,
    raw: Mapping[str, Any],
    mask: Mapping[str, Any] | None,
    reasons: Sequence[str],
) -> dict[str, Any]:
    mask = mask or {}
    source_sample_id = _source_sample_id(dataset, shard, row_idx, raw)
    return {
        "source_dataset": dataset,
        "source_sample_id": source_sample_id,
        "source_shard": shard,
        "source_row_idx": row_idx,
        "edit_type": _edit_type(dataset, raw, mask),
        "raw_edit_type": _raw_edit_type(dataset, raw, mask),
        "instruction": _instruction(dataset, raw, mask),
        "qc_flag": _safe_string(mask.get("qc_flag")) or None,
        "primary_reason": reasons[0],
        "all_reasons_json": _json_dumps(list(reasons)),
    }


def _write_parquet_atomic(table: pa.Table, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    dictionary_candidates = {
        "schema_version",
        "source_dataset",
        "split",
        "edit_type",
        "raw_edit_type",
        "source_image_format",
        "edited_image_format",
        "mask_coordinate_space",
        "mask_mode",
        "mask_source",
        "quality_status",
        "qc_flag",
        "grounding_status",
        "mllm_model",
        "prompt_version",
        "sam_version",
        "prefilter_verdict",
        "prefilter_method",
        "prefilter_model_name",
        "prefilter_run_id",
        "primary_reason",
    }
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=3,
        use_dictionary=[
            name for name in table.schema.names if name in dictionary_candidates
        ],
        write_statistics=True,
    )
    os.replace(temporary, path)
    return path.stat().st_size


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _process_shard(
    spec: DatasetSpec,
    raw_path: Path,
    output_root: Path,
    denylist: Mapping[tuple[str, str, int], str],
    resume: bool,
) -> dict[str, Any]:
    shard = raw_path.name
    mask_path = spec.mask_root / shard
    data_path = output_root / spec.name / "data" / shard
    reject_path = output_root / "audit" / spec.name / shard
    report_path = output_root / "audit" / "shard_reports" / spec.name / (
        shard + ".json"
    )
    if resume and report_path.is_file() and data_path.is_file() and reject_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("complete") is True:
            return report
    for path in (data_path, reject_path, report_path):
        if path.exists():
            raise FileExistsError(
                f"partial output exists without a reusable report: {path}; "
                "choose a new output root or repair the partial run"
            )
    if not mask_path.is_file():
        raise FileNotFoundError(f"missing mask shard for {raw_path}: {mask_path}")

    mask_rows = pq.read_table(mask_path).to_pylist()
    mask_by_idx: dict[int, Mapping[str, Any]] = {}
    for mask in mask_rows:
        row_idx = int(mask["row_idx"])
        if row_idx in mask_by_idx:
            raise ValueError(f"duplicate mask row_idx {shard}:{row_idx}")
        mask_by_idx[row_idx] = mask

    source_file = pq.ParquetFile(raw_path)
    source_rows = source_file.metadata.num_rows
    out_of_range = sorted(index for index in mask_by_idx if not 0 <= index < source_rows)
    if out_of_range:
        raise ValueError(f"out-of-range mask row_idx in {shard}: {out_of_range[:8]}")
    available_fields = set(source_file.schema_arrow.names)
    required_fields = set(_source_fields(spec.name))
    missing_fields = sorted(required_fields - available_fields)
    if missing_fields:
        raise ValueError(f"source shard {shard} is missing columns: {missing_fields}")
    raw_rows = pq.read_table(raw_path, columns=_source_fields(spec.name)).to_pylist()

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    primary_reasons: Counter[str] = Counter()
    all_reasons: Counter[str] = Counter()
    edit_types: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    mask_sources: Counter[str] = Counter()
    areas: list[float] = []
    for row_idx, raw in enumerate(raw_rows):
        mask = mask_by_idx.get(row_idx)
        if mask is None:
            reasons = ["missing_mask_record"]
            rejected.append(_reject_row(spec.name, shard, row_idx, raw, None, reasons))
        else:
            reasons = _preliminary_reasons(
                spec.name, shard, row_idx, mask, denylist
            )
            if not reasons:
                standard, reasons = _standard_row(
                    spec.name, shard, row_idx, raw, mask
                )
                if standard is not None:
                    accepted.append(standard)
                    edit_types[standard["edit_type"]] += 1
                    mask_modes[standard["mask_mode"]] += 1
                    mask_sources[standard["mask_source"]] += 1
                    areas.append(float(standard["mask_area_fraction"]))
                    continue
            rejected.append(_reject_row(spec.name, shard, row_idx, raw, mask, reasons))
        primary_reasons[reasons[0]] += 1
        all_reasons.update(reasons)

    data_table = (
        pa.Table.from_pylist(accepted, schema=TRAIN_SCHEMA)
        if accepted
        else _empty_table(TRAIN_SCHEMA)
    )
    reject_table = (
        pa.Table.from_pylist(rejected, schema=REJECT_SCHEMA)
        if rejected
        else _empty_table(REJECT_SCHEMA)
    )
    data_bytes = _write_parquet_atomic(data_table, data_path)
    reject_bytes = _write_parquet_atomic(reject_table, reject_path)
    report = {
        "complete": True,
        "dataset": spec.name,
        "source_shard": shard,
        "mask_shard": mask_path.name,
        "source_rows": source_rows,
        "mask_rows": len(mask_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "primary_reasons": dict(sorted(primary_reasons.items())),
        "all_reasons": dict(sorted(all_reasons.items())),
        "edit_types": dict(sorted(edit_types.items())),
        "mask_modes": dict(sorted(mask_modes.items())),
        "mask_sources": dict(sorted(mask_sources.items())),
        "area_fraction_sum": math.fsum(areas),
        "area_fraction_min": min(areas) if areas else None,
        "area_fraction_max": max(areas) if areas else None,
        "data_path": str(data_path.relative_to(output_root)),
        "reject_path": str(reject_path.relative_to(output_root)),
        "data_bytes": data_bytes,
        "reject_bytes": reject_bytes,
        "data_sha256": _sha256_file(data_path),
        "completed_at": _utc_now(),
    }
    _atomic_write_json(report_path, report)
    return report


def _add_counter(target: Counter[str], values: Mapping[str, int]) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def _validate_complete_output(
    output_root: Path, reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sample_ids: set[str] = set()
    duplicate_ids: list[str] = []
    checked_rows = 0
    checked_rejections = 0
    for report in reports:
        data_path = output_root / str(report["data_path"])
        reject_path = output_root / str(report["reject_path"])
        data_file = pq.ParquetFile(data_path)
        reject_file = pq.ParquetFile(reject_path)
        if not data_file.schema_arrow.equals(TRAIN_SCHEMA, check_metadata=True):
            raise ValueError(f"unified schema mismatch: {data_path}")
        if not reject_file.schema_arrow.equals(REJECT_SCHEMA, check_metadata=True):
            raise ValueError(f"rejection schema mismatch: {reject_path}")
        expected_rows = int(report["accepted_rows"])
        expected_rejections = int(report["rejected_rows"])
        if data_file.metadata.num_rows != expected_rows:
            raise ValueError(f"accepted row-count mismatch: {data_path}")
        if reject_file.metadata.num_rows != expected_rejections:
            raise ValueError(f"rejection row-count mismatch: {reject_path}")
        columns = pq.read_table(
            data_path,
            columns=["sample_id", "source_dataset", "quality_status", "qc_flag"],
        ).to_pydict()
        for sample_id, dataset, quality_status, qc_flag in zip(
            columns["sample_id"],
            columns["source_dataset"],
            columns["quality_status"],
            columns["qc_flag"],
        ):
            if sample_id in sample_ids and len(duplicate_ids) < 20:
                duplicate_ids.append(sample_id)
            sample_ids.add(sample_id)
            if dataset != report["dataset"]:
                raise ValueError(f"source_dataset mismatch in {data_path}: {dataset}")
            if quality_status != "strict_pass" or qc_flag != "OK":
                raise ValueError(f"non-passing row in {data_path}: {sample_id}")
        checked_rows += expected_rows
        checked_rejections += expected_rejections
    if duplicate_ids:
        raise ValueError(f"duplicate global sample IDs: {duplicate_ids}")
    return {
        "checked_shards": len(reports),
        "checked_rows": checked_rows,
        "checked_rejections": checked_rejections,
        "unique_sample_ids": len(sample_ids),
        "schema_consistent": True,
        "all_rows_strict_pass": True,
    }


def _aggregate_reports(
    output_root: Path,
    specs: Sequence[DatasetSpec],
    reports: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for spec in specs:
        selected = [report for report in reports if report["dataset"] == spec.name]
        primary: Counter[str] = Counter()
        all_reasons: Counter[str] = Counter()
        edit_types: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        for report in selected:
            _add_counter(primary, report["primary_reasons"])
            _add_counter(all_reasons, report["all_reasons"])
            _add_counter(edit_types, report["edit_types"])
            _add_counter(modes, report["mask_modes"])
            _add_counter(sources, report["mask_sources"])
        accepted = sum(int(report["accepted_rows"]) for report in selected)
        rejected = sum(int(report["rejected_rows"]) for report in selected)
        area_sum = math.fsum(float(report["area_fraction_sum"]) for report in selected)
        area_min_values = [
            float(report["area_fraction_min"])
            for report in selected
            if report["area_fraction_min"] is not None
        ]
        area_max_values = [
            float(report["area_fraction_max"])
            for report in selected
            if report["area_fraction_max"] is not None
        ]
        by_dataset[spec.name] = {
            "source_root": str(spec.raw_root),
            "mask_root": str(spec.mask_root),
            "shards": len(selected),
            "source_rows": sum(int(report["source_rows"]) for report in selected),
            "mask_rows": sum(int(report["mask_rows"]) for report in selected),
            "accepted_rows": accepted,
            "rejected_rows": rejected,
            "acceptance_rate": accepted / max(accepted + rejected, 1),
            "primary_reasons": dict(sorted(primary.items())),
            "all_reasons": dict(sorted(all_reasons.items())),
            "accepted_edit_types": dict(sorted(edit_types.items())),
            "accepted_mask_modes": dict(sorted(modes.items())),
            "accepted_mask_sources": dict(sorted(sources.items())),
            "accepted_area_fraction": {
                "min": min(area_min_values) if area_min_values else None,
                "mean": area_sum / accepted if accepted else None,
                "max": max(area_max_values) if area_max_values else None,
            },
            "data_bytes": sum(int(report["data_bytes"]) for report in selected),
            "reject_bytes": sum(int(report["reject_bytes"]) for report in selected),
        }
    accepted_total = sum(value["accepted_rows"] for value in by_dataset.values())
    rejected_total = sum(value["rejected_rows"] for value in by_dataset.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "complete": True,
        "created_at": _utc_now(),
        "accepted_rows": accepted_total,
        "rejected_rows": rejected_total,
        "source_rows": accepted_total + rejected_total,
        "datasets": by_dataset,
        "paths": {
            "crispedit_data": "crispedit/data/*.parquet",
            "scaleedit_data": "scaleedit/data/*.parquet",
            "rejection_audit": "audit/{crispedit,scaleedit}/*.parquet",
            "shard_index": "shards.parquet",
        },
        "run_config": dict(run_config),
    }
    index_schema = pa.schema(
        [
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("source_shard", pa.string(), nullable=False),
            pa.field("source_rows", pa.int64(), nullable=False),
            pa.field("mask_rows", pa.int64(), nullable=False),
            pa.field("accepted_rows", pa.int64(), nullable=False),
            pa.field("rejected_rows", pa.int64(), nullable=False),
            pa.field("data_path", pa.string(), nullable=False),
            pa.field("reject_path", pa.string(), nullable=False),
            pa.field("data_bytes", pa.int64(), nullable=False),
            pa.field("data_sha256", pa.string(), nullable=False),
        ]
    )
    index_rows = [
        {key: report[key] for key in index_schema.names}
        for report in sorted(reports, key=lambda item: (item["dataset"], item["source_shard"]))
    ]
    index_table = pa.Table.from_pylist(index_rows, schema=index_schema)
    index_path = output_root / "shards.parquet"
    if index_path.exists():
        index_path.unlink()
    _write_parquet_atomic(index_table, index_path)
    manifest["validation"] = _validate_complete_output(output_root, reports)
    _atomic_write_json(output_root / "manifest.json", manifest)
    _atomic_write_json(
        output_root / "schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "columns": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in TRAIN_SCHEMA
            ],
            "mask_semantics": "255=editable, 0=preserve, source-image coordinates",
        },
    )
    return manifest


def _write_dataset_readme(output_root: Path, manifest: Mapping[str, Any]) -> None:
    text = f"""# Unified ScaleEdit + CrispEdit mask training data

This directory is generated by `scripts/build_unified_mask_dataset.py` and is
self-contained.  The original datasets were read only.

- Schema: `{SCHEMA_VERSION}`
- Quality policy: `{QUALITY_POLICY_VERSION}`
- Accepted rows: {manifest['accepted_rows']:,}
- Rejected rows: {manifest['rejected_rows']:,}
- CrispEdit data: `crispedit/data/*.parquet`
- ScaleEdit data: `scaleedit/data/*.parquet`
- Rejection audit: `audit/{{crispedit,scaleedit}}/*.parquet`
- Summary: `manifest.json`
- Exact schema: `schema.json`
- Per-shard checksums and row counts: `shards.parquet`

Core training columns are `instruction`, `source_image`, `edited_image`, and
`mask_png`.  Every accepted mask is an 8-bit PNG with 255 for editable pixels
and 0 for preserved pixels, in `source_image` coordinates.  All accepted rows
have `quality_status=strict_pass` and `qc_flag=OK`.

Example:

```python
import pyarrow.dataset as ds

root = {str(output_root)!r}
data = ds.dataset(
    [
        ds.dataset(f"{{root}}/crispedit/data", format="parquet"),
        ds.dataset(f"{{root}}/scaleedit/data", format="parquet"),
    ],
)
batch = next(data.to_batches(columns=[
    "sample_id", "edit_type", "instruction",
    "source_image", "edited_image", "mask_png",
]))
```
"""
    path = output_root / "README.md"
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_dataset(
    specs: Sequence[DatasetSpec],
    output_root: Path,
    denylist_path: Path | None,
    *,
    workers: int,
    resume: bool,
    limit_shards: int | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    source_roots = {
        root
        for spec in specs
        for root in (spec.raw_root.resolve(), spec.mask_root.resolve())
    }
    if output_root in source_roots:
        raise ValueError("output root must differ from every source and mask root")
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"output root is not empty: {output_root}; use --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    denylist = _load_denylist(denylist_path)
    jobs: list[tuple[DatasetSpec, Path]] = []
    for spec in specs:
        if not spec.raw_root.is_dir():
            raise FileNotFoundError(f"missing raw root: {spec.raw_root}")
        if not spec.mask_root.is_dir():
            raise FileNotFoundError(f"missing mask root: {spec.mask_root}")
        raw_paths = sorted(spec.raw_root.glob("*.parquet"))
        if limit_shards is not None:
            raw_paths = raw_paths[:limit_shards]
        if not raw_paths:
            raise FileNotFoundError(f"no parquet shards under {spec.raw_root}")
        jobs.extend((spec, path) for path in raw_paths)

    run_config = {
        "schema_version": SCHEMA_VERSION,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "started_at": _utc_now(),
        "output_root": str(output_root),
        "workers": workers,
        "resume": resume,
        "limit_shards": limit_shards,
        "denylist_path": str(denylist_path.resolve()) if denylist_path else None,
        "denylist_entries": len(denylist),
        "datasets": [
            {
                "name": spec.name,
                "raw_root": str(spec.raw_root.resolve()),
                "mask_root": str(spec.mask_root.resolve()),
            }
            for spec in specs
        ],
    }
    _atomic_write_json(output_root / "run_config.json", run_config)
    reports: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_job = {
            executor.submit(
                _process_shard, spec, raw_path, output_root, denylist, resume
            ): (spec.name, raw_path.name)
            for spec, raw_path in jobs
        }
        for future in as_completed(future_to_job):
            dataset, shard = future_to_job[future]
            try:
                report = future.result()
            except Exception as error:
                print(
                    f"ERROR dataset={dataset} shard={shard}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            reports.append(report)
            completed += 1
            print(
                f"[{completed}/{len(jobs)}] {dataset}/{shard}: "
                f"accepted={report['accepted_rows']} rejected={report['rejected_rows']}",
                flush=True,
            )

    manifest = _aggregate_reports(output_root, specs, reports, run_config)
    _write_dataset_readme(output_root, manifest)
    success_path = output_root / "_SUCCESS"
    success_path.write_text(
        _json_dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "accepted_rows": manifest["accepted_rows"],
                "completed_at": manifest["created_at"],
            },
            pretty=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("crispedit", "scaleedit"),
        default=("crispedit", "scaleedit"),
    )
    parser.add_argument("--crispedit-raw-root", type=Path, default=DEFAULT_CRISPEDIT_RAW_ROOT)
    parser.add_argument("--crispedit-mask-root", type=Path, default=DEFAULT_CRISPEDIT_MASK_ROOT)
    parser.add_argument("--scaleedit-raw-root", type=Path, default=DEFAULT_SCALEEDIT_RAW_ROOT)
    parser.add_argument("--scaleedit-mask-root", type=Path, default=DEFAULT_SCALEEDIT_MASK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-shards", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    specs_by_name = {
        "crispedit": DatasetSpec(
            "crispedit", args.crispedit_raw_root.resolve(), args.crispedit_mask_root.resolve()
        ),
        "scaleedit": DatasetSpec(
            "scaleedit", args.scaleedit_raw_root.resolve(), args.scaleedit_mask_root.resolve()
        ),
    }
    specs = [specs_by_name[name] for name in dict.fromkeys(args.datasets)]
    manifest = build_dataset(
        specs,
        args.output_root,
        args.denylist,
        workers=args.workers,
        resume=args.resume,
        limit_shards=args.limit_shards,
    )
    print(_json_dumps(manifest, pretty=True), flush=True)


if __name__ == "__main__":
    main()
