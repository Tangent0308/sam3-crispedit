"""Strict join of dense quality and sparse scene manifests, by original row_idx."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCENE_FIELDS = [
    ("scene_decision", pa.string()), ("scene_pass", pa.bool_()),
    ("scene_reason", pa.string()), ("scene_run_id", pa.string()),
    ("scene_prompt_version", pa.string()), ("scene_filter_method", pa.string()),
    ("scene_model_name", pa.string()), ("scene_parse_ok", pa.bool_()),
    ("mask_selection_reason", pa.string()),
]


def scene_fields(row=None):
    row = row or {}
    return {name: row.get(name, False if pa.types.is_boolean(dtype) else "")
            for name, dtype in SCENE_FIELDS}


def load_indexed(path):
    if path is None:
        return {}
    rows = pq.read_table(path).to_pylist()
    result = {}
    for row in rows:
        index = row.get("row_idx")
        if not isinstance(index, int) or index < 0 or index in result:
            raise ValueError(f"invalid/duplicate row_idx in {path}: {index}")
        result[index] = row
    return result


def load_filters(quality_path, difficulty_path, raw_count):
    quality = load_indexed(quality_path)
    if quality_path and set(quality) != set(range(raw_count)):
        raise ValueError(f"quality manifest must cover all {raw_count} raw rows: {quality_path}")
    if not difficulty_path:
        return quality, {}
    if not quality_path:
        raise ValueError("difficulty manifest requires a quality manifest")
    scene = load_indexed(difficulty_path)
    expected = set()
    for index, row in quality.items():
        verdict = row.get("prefilter_verdict")
        if verdict not in {"PASS", "FAIL", "UNSURE", "ERROR"}:
            raise ValueError(f"invalid quality verdict at {quality_path}:{index}")
        decision = row.get("filter_decision", row.get("prefilter_decision"))
        if decision != ("keep" if verdict == "PASS" else "drop"):
            raise ValueError(f"inconsistent quality decision at {quality_path}:{index}")
        if verdict == "PASS":
            expected.add(index)
    if set(scene) != expected:
        raise ValueError(f"scene rows must equal quality PASS rows: {difficulty_path}; "
                         f"missing={sorted(expected-set(scene))[:10]}, extra={sorted(set(scene)-expected)[:10]}")
    for index, row in scene.items():
        if row.get("source_prefilter_verdict") != "PASS" or row.get("source_prefilter_run_id") != quality[index].get("prefilter_run_id"):
            raise ValueError(f"scene/quality provenance mismatch: {difficulty_path}:{index}")
        if row.get("scene_decision") not in {"PASS", "DROP"} or row.get("scene_pass") is not (row["scene_decision"] == "PASS"):
            raise ValueError(f"inconsistent scene decision: {difficulty_path}:{index}")
        if row["scene_decision"] == "PASS" and not row.get("scene_parse_ok"):
            raise ValueError(f"unparsed scene PASS: {difficulty_path}:{index}")
    return quality, scene


def apply_scene(pre, row, enabled):
    """Preserve quality verdict; filter_decision becomes the final two-stage gate."""
    result = {**pre, **scene_fields(row)}
    if pre["filter_decision"] != "keep":
        result["mask_selection_reason"] = "QUALITY_DROP"
    elif enabled and (row is None or row["scene_decision"] != "PASS"):
        result["filter_decision"] = "drop"
        result["mask_selection_reason"] = "SCENE_DROP"
    else:
        result["mask_selection_reason"] = "SELECTED"
    return result
