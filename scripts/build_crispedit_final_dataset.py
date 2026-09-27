#!/usr/bin/env python3
"""Join final CrispEdit mask rows to source images and all stage audit fields."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DATA_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling")
RUN_NAME = "labeling_4node_crispedit_full_localenv_20260925"
STAGES = (
    ("quality", "prefilter/quality/manifest"),
    ("quality_audit", "prefilter/quality/audit"),
    ("scene", "prefilter/scene/manifest"),
    ("scene_audit", "prefilter/scene/audit"),
    ("grounding", f"runs/{RUN_NAME}/labels/grounding"),
    ("mask", f"runs/{RUN_NAME}/labels/mask"),
)


def schema_signature(path: Path) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, str(field.type)) for field in pq.ParquetFile(path).schema_arrow)


def union_stage_schemas(stage_roots: dict[str, Path], shard_names: list[str], workers: int):
    jobs = [(stage, root / name) for stage, root in stage_roots.items() for name in shard_names]
    field_types: dict[str, dict[str, pa.DataType]] = {stage: {} for stage in stage_roots}
    field_orders: dict[str, list[str]] = {}
    for stage, root in stage_roots.items():
        sample_fields = pq.ParquetFile(root / shard_names[0]).schema_arrow
        field_orders[stage] = [field.name for field in sample_fields]
        field_types[stage].update({field.name: field.type for field in sample_fields})

    def inspect(job):
        stage, path = job
        parquet = pq.ParquetFile(path)
        return stage, path, [(field.name, field.type) for field in parquet.schema_arrow]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(inspect, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inspecting stage schemas", unit="shard"):
            stage, path, fields = future.result()
            if ("row_idx", pa.int64()) not in fields:
                raise ValueError(f"{stage} lacks int64 row_idx: {path}")
            for name, data_type in fields:
                previous = field_types[stage].get(name)
                if previous is not None and previous != data_type:
                    raise TypeError(f"{stage}.{name} has incompatible types {previous} and {data_type} in {path}")
                field_types[stage][name] = data_type
    return {
        stage: [pa.field(name, fields[name]) for name in field_orders[stage] + sorted(set(fields) - set(field_orders[stage]))]
        for stage, fields in field_types.items()
    }


def take_stage_rows(path: Path, stage: str, union_fields: list[pa.Field], row_indices: list[int]):
    parquet = pq.ParquetFile(path)
    actual = {field.name: field.type for field in parquet.schema_arrow}
    expected = {field.name: field.type for field in union_fields}
    if any(name not in expected or expected[name] != data_type for name, data_type in actual.items()):
        raise ValueError(f"{stage} schema has unknown/incompatible fields in {path}")
    table = parquet.read()
    source_indices = table.column("row_idx").to_pylist()
    positions = {}
    for position, row_idx in enumerate(source_indices):
        row_idx = int(row_idx)
        if row_idx in positions:
            raise ValueError(f"duplicate row_idx={row_idx} in {path}")
        positions[row_idx] = position
    missing = [row_idx for row_idx in row_indices if row_idx not in positions]
    if missing:
        raise ValueError(f"{path} missing selected row_idx values: {missing[:10]}")
    selected = table.take(pa.array([positions[row_idx] for row_idx in row_indices], type=pa.int64()))
    arrays = [
        selected.column(field.name) if field.name in actual else pa.nulls(len(row_indices), type=field.type)
        for field in union_fields
    ]
    return pa.Table.from_arrays(arrays, schema=pa.schema(union_fields))


def slug_for(shard_name: str, source_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", source_type.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive category for {shard_name}: {source_type!r}")
    return slug


def process_shard(
    mask_path: Path,
    source_root: Path,
    stage_roots: dict[str, Path],
    source_signature: tuple[tuple[str, str], ...],
    stage_fields: dict[str, list[pa.Field]],
    output_schema: pa.Schema,
    build_dir: Path,
) -> dict:
    shard_name = mask_path.name
    source_path = source_root / shard_name
    if not source_path.is_file():
        raise FileNotFoundError(f"missing source shard: {source_path}")
    source_parquet = pq.ParquetFile(source_path)
    if tuple((field.name, str(field.type)) for field in source_parquet.schema_arrow) != source_signature:
        raise ValueError(f"source schema differs in {source_path}")
    mask_table = pq.ParquetFile(mask_path).read()
    row_indices = [int(value) for value in mask_table.column("row_idx").to_pylist()]
    if len(set(row_indices)) != len(row_indices):
        raise ValueError(f"duplicate mask row_idx values in {mask_path}")
    row_indices.sort()
    if not row_indices:
        raise ValueError(f"final mask shard is empty: {mask_path}")
    if min(row_indices) < 0 or max(row_indices) >= source_parquet.metadata.num_rows:
        raise IndexError(f"{shard_name}: row_idx outside source rows {source_parquet.metadata.num_rows}")
    source_table = source_parquet.read().take(pa.array(row_indices, type=pa.int64()))

    stage_tables = {}
    for stage, _ in STAGES:
        stage_path = stage_roots[stage] / shard_name
        if not stage_path.is_file():
            raise FileNotFoundError(f"missing {stage} shard: {stage_path}")
        stage_tables[stage] = take_stage_rows(stage_path, stage, stage_fields[stage], row_indices)

    if stage_tables["quality"].column("prefilter_verdict").to_pylist() != ["PASS"] * len(row_indices):
        raise ValueError(f"{shard_name} contains rows without quality PASS")
    if stage_tables["scene"].column("scene_decision").to_pylist() != ["PASS"] * len(row_indices):
        raise ValueError(f"{shard_name} contains rows without scene PASS")
    if stage_tables["scene"].column("scene_pass").to_pylist() != [True] * len(row_indices):
        raise ValueError(f"{shard_name} contains rows without scene_pass")

    arrays = [pa.array([shard_name] * len(row_indices), type=pa.string()), pa.array(row_indices, type=pa.int64())]
    arrays.extend(source_table.columns)
    for stage, _ in STAGES:
        arrays.extend(stage_tables[stage].columns)
    table = pa.Table.from_arrays(arrays, schema=output_schema)
    source_types = source_table.column("type").to_pylist()
    qc_values = stage_tables["mask"].column("qc_flag").to_pylist()
    types, qc_flags = Counter(source_types), Counter(qc_values)
    category = slug_for(shard_name, source_types[0])
    destination = build_dir / "shards" / category / shard_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd", compression_level=3, row_group_size=256)
    return {"shard": shard_name, "category": category, "rows": len(row_indices), "types": dict(types), "qc_flags": dict(qc_flags)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--schema-workers", type=int, default=48)
    parser.add_argument("--resume-build-dir", type=Path,
                        help="Resume an incomplete output directory created by this script")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    run_root = data_root / "runs" / args.run_name
    source_root = data_root / "source" / "CrispEdit-2M"
    mask_root = run_root / "labels" / "mask"
    output_dir = args.output_dir or data_root / "final_dataset_39k"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    build_dir = args.resume_build_dir.resolve() if args.resume_build_dir else output_dir.with_name(f".{output_dir.name}.building.{os.getpid()}")
    if args.resume_build_dir:
        if not build_dir.is_dir() or build_dir.parent != output_dir.parent:
            raise ValueError(f"resume build dir must be an existing sibling of output: {build_dir}")
    elif build_dir.exists():
        raise FileExistsError(f"build directory already exists: {build_dir}")

    mask_files = sorted(mask_root.glob("*.parquet"))
    if not mask_files:
        raise FileNotFoundError(f"no final mask shards found in {mask_root}")

    stage_roots = {name: data_root / relative for name, relative in STAGES}
    stage_fields = union_stage_schemas(stage_roots, [path.name for path in mask_files], args.schema_workers)
    for name, fields in stage_fields.items():
        if not any(field.name == "row_idx" and field.type == pa.int64() for field in fields):
            raise ValueError(f"{name} parquet lacks int64 row_idx")

    source_sample = source_root / mask_files[0].name
    if not source_sample.is_file():
        raise FileNotFoundError(f"missing source sample shard: {source_sample}")
    source_signature = schema_signature(source_sample)

    output_fields = [
        pa.field("source_shard", pa.string()),
        pa.field("row_idx", pa.int64()),
    ]
    output_fields.extend(
        pa.field(field.name, field.type)
        for field in pq.ParquetFile(source_sample).schema_arrow
    )
    for stage, _ in STAGES:
        output_fields.extend(pa.field(f"{stage}__{field.name}", field.type) for field in stage_fields[stage])
    output_schema = pa.schema(output_fields)

    counts_by_type: Counter[str] = Counter()
    counts_by_qc: Counter[str] = Counter()
    rows_written = 0
    shard_manifest = []

    if args.resume_build_dir:
        existing_files = list((build_dir / "shards").rglob("*.parquet")) if (build_dir / "shards").exists() else []
        valid_names = {path.name for path in mask_files}
        existing_by_name = {}
        expected_fields = {field.name: field.type for field in output_schema}
        for path in existing_files:
            if path.name not in valid_names or path.name in existing_by_name:
                raise ValueError(f"unexpected or duplicate partial shard: {path}")
            parquet = pq.ParquetFile(path)
            actual_fields = {field.name: field.type for field in parquet.schema_arrow}
            if any(name not in expected_fields or expected_fields[name] != data_type for name, data_type in actual_fields.items()):
                raise ValueError(f"partial shard has incompatible schema: {path}")
            required = {"source_shard", "row_idx", "type", "mask__qc_flag"}
            if not required.issubset(actual_fields):
                raise ValueError(f"partial shard lacks required columns: {path}")
            summary_table = parquet.read(columns=["source_shard", "row_idx", "type", "mask__qc_flag"])
            source_shards = summary_table.column("source_shard").to_pylist()
            if any(value != path.name for value in source_shards):
                raise ValueError(f"source_shard mismatch in partial file: {path}")
            types = Counter(summary_table.column("type").to_pylist())
            qc_flags = Counter(summary_table.column("mask__qc_flag").to_pylist())
            result = {
                "shard": path.name,
                "category": path.parent.name,
                "rows": parquet.metadata.num_rows,
                "types": dict(types),
                "qc_flags": dict(qc_flags),
            }
            existing_by_name[path.name] = result
            rows_written += result["rows"]
            counts_by_type.update(types)
            counts_by_qc.update(qc_flags)
            shard_manifest.append({key: result[key] for key in ("shard", "category", "rows")})
        print(f"Validated {len(existing_by_name)} existing shard files for resume", flush=True)
    else:
        build_dir.mkdir(parents=True)
        existing_by_name = {}

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending_mask_files = [path for path in mask_files if path.name not in existing_by_name]
        futures = {
            pool.submit(
                process_shard,
                mask_path,
                source_root,
                stage_roots,
                source_signature,
                stage_fields,
                output_schema,
                build_dir,
            ): mask_path.name
            for mask_path in pending_mask_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging CrispEdit final dataset", unit="shard"):
            result = future.result()
            rows_written += result["rows"]
            counts_by_type.update(result["types"])
            counts_by_qc.update(result["qc_flags"])
            shard_manifest.append({key: result[key] for key in ("shard", "category", "rows")})
    shard_manifest.sort(key=lambda item: item["shard"])

    expected_summary = json.loads((run_root / "labels/validation_summary.json").read_text())
    expected_rows = int(expected_summary["counts"]["rows"])
    if rows_written != expected_rows:
        raise ValueError(f"row count mismatch: merged={rows_written}, validated mask rows={expected_rows}")
    if len(shard_manifest) != int(expected_summary["shards"]):
        raise ValueError(f"shard count mismatch: merged={len(shard_manifest)}, validated mask shards={expected_summary['shards']}")
    if counts_by_qc != Counter(expected_summary["flags"]):
        raise ValueError(f"QC count mismatch: merged={dict(counts_by_qc)}, validator={expected_summary['flags']}")
    if counts_by_type != Counter(expected_summary["selected_types"]):
        raise ValueError(f"type count mismatch: merged={dict(counts_by_type)}, validator={expected_summary['selected_types']}")

    manifest = {
        "dataset": "CrispEdit final merged source and labeling dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "quality_manifest_root": str(stage_roots["quality"]),
        "quality_audit_root": str(stage_roots["quality_audit"]),
        "scene_manifest_root": str(stage_roots["scene"]),
        "scene_audit_root": str(stage_roots["scene_audit"]),
        "grounding_root": str(stage_roots["grounding"]),
        "mask_root": str(mask_root),
        "run_name": args.run_name,
        "rows": rows_written,
        "shards": len(shard_manifest),
        "rows_by_source_type": dict(sorted(counts_by_type.items())),
        "rows_by_mask_qc_flag": dict(sorted(counts_by_qc.items())),
        "join_key": ["source_shard", "row_idx"],
        "source_fields": list(pq.ParquetFile(source_sample).schema_arrow.names),
        "stage_fields": {stage: [f"{stage}__{field.name}" for field in fields] for stage, fields in stage_fields.items()},
        "schema": [{"name": field.name, "type": str(field.type)} for field in output_schema],
        "shard_files": shard_manifest,
    }
    (build_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    pq.write_metadata(output_schema, build_dir / "_common_metadata")
    readme = f"""# CrispEdit final merged dataset (~39k)

This dataset contains {rows_written:,} rows that reached final mask labeling. It is stored at `{output_dir}`. Each row joins the original source pair and instruction with all persisted fields from the quality filter, scene filter, grounding, and mask stages.

## Layout

- `shards/<edit_type>/<source-shard>.parquet`: merged, row-aligned parquet shards.
- `dataset_manifest.json`: source paths, row and shard counts, per-type/QC counts, complete output schema, and shard index.
- `COMPLETE`: written only after all shards and consistency checks finish.

The stable join key is `source_shard` plus `row_idx`. Original source fields retain their names (`input_img`, `output_img`, `instruction`, `type`). Stage fields are prefixed with `quality__`, `quality_audit__`, `scene__`, `scene_audit__`, `grounding__`, or `mask__`; the audit prefixes include raw model responses, attempts, and timing fields where those were persisted. Image columns retain the original `{bytes, path}` structs. Mask data retains the original PNG and per-instance RLE/audit fields.

Each final row was checked to have a quality `PASS`, scene `PASS`, grounding row, and mask row. Output counts are listed in `dataset_manifest.json`.
"""
    (build_dir / "README.md").write_text(readme)
    (build_dir / "COMPLETE").write_text(f"rows={rows_written}\nshards={len(shard_manifest)}\n")
    os.rename(build_dir, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rows": rows_written, "shards": len(shard_manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
