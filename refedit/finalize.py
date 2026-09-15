"""Build a strict RefEdit mask dataset from prefilter and mask outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from refedit import GROUND_PROMPT_VERSION, MASK_POLICY_VERSION
from refedit.io import discover_shards, sample_id
from refedit.selection import (
    load_prefilter_manifest_dir,
    validate_prefilter_rows,
)
from scaleedit.mask_runner import MASK_SCHEMA


FINAL_MANIFEST_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("sample_id", pa.string()),
        ("img_id", pa.int64()),
        ("source_relative_path", pa.string()),
        ("mask_relative_path", pa.string()),
        ("task", pa.string()),
        ("instruction", pa.string()),
        ("prefilter_verdict", pa.string()),
        ("prefilter_confidence", pa.float32()),
        ("prefilter_model_name", pa.string()),
        ("prefilter_prompt_version", pa.string()),
        ("grounding_status", pa.string()),
        ("mask_qc_flag", pa.string()),
        ("mask_source", pa.string()),
        ("area_frac", pa.float64()),
        ("ground_prompt_version", pa.string()),
        ("mask_policy_version", pa.string()),
    ]
)

REJECTED_MANIFEST_SCHEMA = pa.schema(
    list(FINAL_MANIFEST_SCHEMA) + [("rejection_reason", pa.string())]
)


def _index_rows(rows: Iterable[dict], label: str) -> Dict[Tuple[int, str], dict]:
    result = {}
    for row in rows:
        key = (int(row["row_idx"]), str(row["sample_id"]))
        if key in result:
            raise ValueError(f"duplicate {label} identity: {key}")
        result[key] = row
    return result


def _write_table(path: Path, rows: List[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        temporary,
        compression="zstd",
    )
    temporary.replace(path)


def _manifest_row(selected: dict, mask: dict, shard_name: str) -> dict:
    return {
        "row_idx": int(selected["row_idx"]),
        "sample_id": str(selected["sample_id"]),
        "img_id": int(selected["img_id"]),
        "source_relative_path": str(selected["source_relative_path"]),
        "mask_relative_path": f"data/{shard_name}",
        "task": str(selected.get("task", mask.get("final_task", ""))),
        "instruction": str(selected["instruction"]),
        "prefilter_verdict": str(selected["prefilter_verdict"]),
        "prefilter_confidence": float(selected.get("prefilter_confidence", 0.0)),
        "prefilter_model_name": str(selected.get("prefilter_model_name", "")),
        "prefilter_prompt_version": str(
            selected.get("prefilter_prompt_version", "")
        ),
        "grounding_status": str(mask.get("grounding_status", "")),
        "mask_qc_flag": str(mask.get("qc_flag", "")),
        "mask_source": str(mask.get("mask_source", "")),
        "area_frac": float(mask.get("area_frac", float("nan"))),
        "ground_prompt_version": str(mask.get("prompt_version", "")),
        "mask_policy_version": str(mask.get("mask_policy_version", "")),
    }


def build_final_dataset(
    input_dir: Path,
    prefilter_manifest_dir: Path,
    grounding_dir: Path,
    mask_dir: Path,
    output_dir: Path,
) -> dict:
    """Materialize prefilter-PASS and mask-QC-OK rows with strict joins."""

    input_dir = Path(input_dir).resolve()
    prefilter_manifest_dir = Path(prefilter_manifest_dir).resolve()
    grounding_dir = Path(grounding_dir).resolve()
    mask_dir = Path(mask_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("final mask output must not be inside RefEdit source data")

    source_by_name = {path.name: path for path in discover_shards(input_dir)}
    selected_by_shard = load_prefilter_manifest_dir(prefilter_manifest_dir)
    if set(selected_by_shard) != set(source_by_name):
        missing = sorted(set(source_by_name) - set(selected_by_shard))
        extra = sorted(set(selected_by_shard) - set(source_by_name))
        raise ValueError(
            f"prefilter/source shard mismatch: missing={missing} extra={extra}"
        )

    final_manifest: List[dict] = []
    rejected_manifest: List[dict] = []
    mask_qc = Counter()
    tasks = Counter()
    mask_sources = Counter()
    selected_total = 0
    for shard_name, source_path in sorted(source_by_name.items()):
        selected_rows = selected_by_shard[shard_name]
        validate_prefilter_rows(source_path, selected_rows)
        selected_total += len(selected_rows)

        ground_path = grounding_dir / shard_name
        mask_path = mask_dir / shard_name
        if not ground_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"missing grounding/mask input for selected shard {shard_name}"
            )
        ground_rows = pq.read_table(ground_path).to_pylist()
        mask_rows = pq.read_table(mask_path).to_pylist()
        ground_index = _index_rows(ground_rows, f"grounding {shard_name}")
        mask_index = _index_rows(mask_rows, f"mask {shard_name}")
        source_rows = pq.read_table(
            source_path, columns=["img_id", "instruction"]
        ).to_pylist()

        final_rows = []
        for selected in selected_rows:
            key = (int(selected["row_idx"]), str(selected["sample_id"]))
            ground = ground_index.get(key)
            mask = mask_index.get(key)
            if ground is None or mask is None:
                raise KeyError(f"selected sample missing labels in {shard_name}: {key}")
            raw = source_rows[key[0]]
            expected = sample_id(raw["img_id"])
            if expected != key[1]:
                raise ValueError(f"source identity mismatch in {shard_name}: {key}")
            if str(ground["sample_id"]) != str(mask["sample_id"]):
                raise ValueError(f"ground/mask identity mismatch in {shard_name}: {key}")
            if Path(str(ground["source_relative_path"])).name != shard_name or Path(
                str(mask["source_relative_path"])
            ).name != shard_name:
                raise ValueError(f"label/source shard mismatch in {shard_name}: {key}")
            if str(ground["final_instruction"]) != str(raw["instruction"]):
                raise ValueError(
                    f"ground/source instruction mismatch in {shard_name}: {key}"
                )
            if str(mask["final_instruction"]) != str(raw["instruction"]):
                raise ValueError(
                    f"mask/source instruction mismatch in {shard_name}: {key}"
                )
            if str(ground["prompt_version"]) != GROUND_PROMPT_VERSION:
                raise ValueError(f"stale grounding policy in {shard_name}: {key}")
            if str(mask["prompt_version"]) != GROUND_PROMPT_VERSION:
                raise ValueError(f"stale mask grounding policy in {shard_name}: {key}")
            if str(mask["mask_policy_version"]) != MASK_POLICY_VERSION:
                raise ValueError(f"stale mask policy in {shard_name}: {key}")
            if str(ground["grounding_status"]) != str(mask["grounding_status"]):
                raise ValueError(f"ground/mask status mismatch in {shard_name}: {key}")

            manifest = _manifest_row(selected, mask, shard_name)
            qc = str(mask["qc_flag"])
            grounding_status = str(mask["grounding_status"])
            mask_qc[qc] += 1
            tasks[str(mask["final_task"])] += 1
            mask_sources[str(mask["mask_source"])] += 1
            rejection_reasons = []
            if grounding_status != "OK":
                rejection_reasons.append(f"grounding_status={grounding_status}")
            if qc != "OK":
                rejection_reasons.append(f"mask_qc_flag={qc}")
            if not mask.get("mask_png"):
                rejection_reasons.append("missing_mask_png")
            if rejection_reasons:
                rejected_manifest.append(
                    {**manifest, "rejection_reason": ";".join(rejection_reasons)}
                )
            else:
                final_rows.append(mask)
                final_manifest.append(manifest)

        _write_table(output_dir / "data" / shard_name, final_rows, MASK_SCHEMA)

    _write_table(
        output_dir / "audit" / "final_manifest.parquet",
        final_manifest,
        FINAL_MANIFEST_SCHEMA,
    )
    _write_table(
        output_dir / "audit" / "rejected_mask_qc.parquet",
        rejected_manifest,
        REJECTED_MANIFEST_SCHEMA,
    )
    summary = {
        "source_rows": sum(
            pq.ParquetFile(path).metadata.num_rows
            for path in source_by_name.values()
        ),
        "source_shards": len(source_by_name),
        "prefilter_pass_rows": selected_total,
        "final_rows": len(final_manifest),
        "rejected_after_mask": len(rejected_manifest),
        "output_shards": len(list((output_dir / "data").glob("train-*.parquet"))),
        "mask_qc_flags_on_prefilter_pass": dict(sorted(mask_qc.items())),
        "tasks_on_prefilter_pass": dict(sorted(tasks.items())),
        "mask_sources_on_prefilter_pass": dict(sorted(mask_sources.items())),
        "ground_prompt_version": GROUND_PROMPT_VERSION,
        "mask_policy_version": MASK_POLICY_VERSION,
        "input_dir": str(input_dir),
        "prefilter_manifest_dir": str(prefilter_manifest_dir),
        "grounding_dir": str(grounding_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
