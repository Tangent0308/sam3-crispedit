#!/usr/bin/env python3
"""Merge final ScaleEdit labels with source pairs and every persisted stage field."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


BASE = Path("/mnt/bn/strategy-mllm-train/user/tanyue")
RUN_ID = "labeling_4node_scaleedit_300k_20260926"
STAGES = (
    ("quality", "quality/manifest"),
    ("quality_audit", "quality/audit"),
    ("scene", "scene/manifest"),
    ("scene_audit", "scene/audit"),
    ("grounding", "grounding"),
    ("mask", "mask"),
)


def inspect_schemas(roots: dict[str, Path], names: list[str], workers: int):
    """Read all footers so rare fields are not lost from the output schema."""
    field_types: dict[str, dict[str, pa.DataType]] = {name: {} for name in roots}
    row_counts: dict[str, dict[str, int]] = {name: {} for name in roots}

    def inspect(stage: str, name: str):
        path = roots[stage] / name
        pf = pq.ParquetFile(path)
        return stage, name, [(f.name, f.type) for f in pf.schema_arrow], pf.metadata.num_rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(inspect, stage, name) for stage in roots for name in names]
        for future in tqdm(as_completed(jobs), total=len(jobs), desc="Inspecting schemas", unit="file"):
            stage, name, fields, rows = future.result()
            row_counts[stage][name] = rows
            for field_name, data_type in fields:
                previous = field_types[stage].get(field_name)
                if previous is not None and previous != data_type:
                    raise TypeError(f"{stage}.{field_name}: {previous} vs {data_type} in {name}")
                if previous is None:
                    field_types[stage][field_name] = data_type
    ordered = {}
    for stage, types in field_types.items():
        # Footer completion order is nondeterministic; use stable field order from
        # one shard, then append any rare extra fields alphabetically.
        first = pq.ParquetFile(roots[stage] / names[0]).schema_arrow.names
        ordered[stage] = [pa.field(name, types[name]) for name in first]
        ordered[stage] += [pa.field(name, types[name]) for name in sorted(set(types) - set(first))]
    return ordered, row_counts


def selected_table(path: Path, fields: list[pa.Field], row_indices: list[int], stage: str):
    table = pq.read_table(path)
    if stage == "source":
        if row_indices and (row_indices[0] < 0 or row_indices[-1] >= len(table)):
            raise IndexError(f"source row_idx out of bounds: {path}")
        positions = row_indices
    else:
        indices = table.column("row_idx").to_pylist()
        index_to_position = {}
        for position, index in enumerate(indices):
            if index in index_to_position:
                raise ValueError(f"duplicate row_idx={index}: {path}")
            index_to_position[index] = position
        missing = [index for index in row_indices if index not in index_to_position]
        if missing:
            raise ValueError(f"missing {stage} row_idx {missing[:5]} in {path}")
        positions = [index_to_position[index] for index in row_indices]
    selected = table.take(pa.array(positions, type=pa.int64()))
    arrays = [
        selected.column(field.name) if field.name in selected.schema.names else pa.nulls(len(selected), type=field.type)
        for field in fields
    ]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def check_identity(shard: str, source: pa.Table, stages: dict[str, pa.Table]):
    for stage, table in stages.items():
        for field in ("sample_id", "final_task", "final_instruction", "source_relative_path"):
            if field in table.schema.names and table.column(field).to_pylist() != source.column(field).to_pylist():
                raise ValueError(f"{shard}: {stage}.{field} differs from source")
    for stage in ("quality", "quality_audit", "scene", "scene_audit"):
        table = stages[stage]
        if table.column("verdict").to_pylist() != ["PASS"] * len(source):
            raise ValueError(f"{shard}: non-PASS {stage} verdict")
        if table.column("keep").to_pylist() != [True] * len(source):
            raise ValueError(f"{shard}: non-kept {stage} row")


def build_shard(name: str, roots: dict[str, Path], schemas: dict[str, list[pa.Field]],
                output_schema: pa.Schema, build_dir: Path):
    mask_table = pq.read_table(roots["mask"] / name, columns=["row_idx"])
    row_indices = sorted(int(value) for value in mask_table.column("row_idx").to_pylist())
    if not row_indices:
        return {"shard": name, "rows": 0, "qc_flags": {}, "tasks": {}}
    if len(set(row_indices)) != len(row_indices):
        raise ValueError(f"duplicate mask row_idx in {name}")
    source = selected_table(roots["source"] / name, schemas["source"], row_indices, "source")
    stages = {
        stage: selected_table(roots[stage] / name, schemas[stage], row_indices, stage)
        for stage, _ in STAGES
    }
    check_identity(name, source, stages)
    arrays = [pa.array([name] * len(row_indices), type=pa.string()), pa.array(row_indices, type=pa.int64())]
    arrays.extend(source.columns)
    for stage, _ in STAGES:
        arrays.extend(stages[stage].columns)
    merged = pa.Table.from_arrays(arrays, schema=output_schema)
    destination = build_dir / "shards" / name
    temporary = destination.with_suffix(".parquet.tmp")
    pq.write_table(merged, temporary, compression="zstd", compression_level=3, row_group_size=256)
    os.replace(temporary, destination)
    return {
        "shard": name,
        "rows": len(row_indices),
        "qc_flags": dict(Counter(stages["mask"].column("qc_flag").to_pylist())),
        "tasks": dict(Counter(source.column("final_task").to_pylist())),
    }


def inspect_existing(path: Path, name: str, schema: pa.Schema, expected_rows: int):
    pf = pq.ParquetFile(path)
    if pf.schema_arrow != schema or pf.metadata.num_rows != expected_rows:
        raise ValueError(f"incomplete or incompatible existing merged shard: {path}")
    table = pf.read(columns=["source_shard", "row_idx", "final_task", "mask__qc_flag"])
    if any(shard != name for shard in table.column("source_shard").to_pylist()):
        raise ValueError(f"source_shard mismatch: {path}")
    indices = table.column("row_idx").to_pylist()
    if indices != sorted(set(indices)):
        raise ValueError(f"row_idx missing order or duplicate: {path}")
    return {
        "shard": name,
        "rows": len(table),
        "qc_flags": dict(Counter(table.column("mask__qc_flag").to_pylist())),
        "tasks": dict(Counter(table.column("final_task").to_pylist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=BASE / "datasets/ScaleEdit-filtered-source")
    parser.add_argument("--run-dir", type=Path, default=BASE / "experiments/ScaleEdit" / RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=BASE / "scaleedit")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--schema-workers", type=int, default=48)
    parser.add_argument("--resume-build-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.schema_workers < 1:
        parser.error("worker counts must be positive")
    source_dir, run_dir, output_dir = args.source_dir.resolve(), args.run_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    build_dir = (args.resume_build_dir.resolve() if args.resume_build_dir else
                 output_dir.with_name(f".{output_dir.name}.building.{os.getpid()}"))
    if build_dir.parent != output_dir.parent:
        raise ValueError("build directory must be a sibling of final output")
    if build_dir.exists() and not args.resume_build_dir:
        raise FileExistsError(f"build directory already exists: {build_dir}")
    if args.resume_build_dir and not build_dir.is_dir():
        raise FileNotFoundError(build_dir)
    roots = {"source": source_dir, **{stage: run_dir / relative for stage, relative in STAGES}}
    names = sorted(path.name for path in roots["mask"].glob("*.parquet"))
    if not names:
        raise FileNotFoundError(f"no mask shards in {roots['mask']}")
    schemas, rows = inspect_schemas(roots, names, args.schema_workers)
    if any(rows[stage][name] != rows["mask"][name] for stage in ("grounding",) for name in names):
        raise ValueError("grounding/mask row counts differ")
    for stage in roots:
        if stage != "source" and not any(f.name == "row_idx" and f.type == pa.int64() for f in schemas[stage]):
            raise ValueError(f"{stage} lacks int64 row_idx")
    fields = [pa.field("source_shard", pa.string()), pa.field("row_idx", pa.int64())]
    fields.extend(schemas["source"])
    for stage, _ in STAGES:
        fields.extend(pa.field(f"{stage}__{f.name}", f.type) for f in schemas[stage])
    output_schema = pa.schema(fields)
    expected = json.loads((run_dir / "reports/run_manifest.json").read_text())["stages"]
    if sum(rows["mask"].values()) != expected["mask"]["counts"]["rows"]:
        raise ValueError("mask parquet footers differ from formal run report")
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "shards").mkdir(exist_ok=True)
    nonempty = [name for name in names if rows["mask"][name] > 0]
    existing = {}
    for path in (build_dir / "shards").glob("*.parquet"):
        if path.name not in nonempty:
            raise ValueError(f"unexpected output shard: {path}")
        existing[path.name] = inspect_existing(path, path.name, output_schema, rows["mask"][path.name])
    print(f"Build directory: {build_dir}; resuming {len(existing)} verified shards", flush=True)
    results = list(existing.values())
    pending = [name for name in names if name not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_shard, name, roots, schemas, output_schema, build_dir) for name in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging ScaleEdit", unit="shard"):
            results.append(future.result())
    results.sort(key=lambda item: item["shard"])
    totals = Counter()
    flags = Counter()
    tasks = Counter()
    for result in results:
        totals["rows"] += result["rows"]
        flags.update(result["qc_flags"])
        tasks.update(result["tasks"])
    if totals["rows"] != expected["mask"]["counts"]["rows"]:
        raise ValueError(f"merged row count {totals['rows']} differs from run report")
    if dict(flags) != {key: expected["mask"]["counts"][key] for key in ("OK", "MASK_REVIEW", "GROUND_FAIL")}:
        raise ValueError(f"merged QC counts differ from run report: {flags}")
    if sum(tasks.values()) != totals["rows"]:
        raise ValueError("task counts do not sum to merged rows")
    for result in results:
        if result["rows"] and not (build_dir / "shards" / result["shard"]).is_file():
            raise FileNotFoundError(result["shard"])
    manifest = {
        "dataset": "ScaleEdit source and final labeling, row-aligned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir), "run_dir": str(run_dir),
        "join_key": ["source_shard", "row_idx"],
        "rows": totals["rows"], "shards": len(nonempty),
        "empty_mask_shards": len(names) - len(nonempty),
        "rows_by_qc_flag": dict(sorted(flags.items())),
        "rows_by_final_task": dict(sorted(tasks.items())),
        "source_fields": [f.name for f in schemas["source"]],
        "stage_fields": {stage: [f"{stage}__{f.name}" for f in schemas[stage]] for stage, _ in STAGES},
        "schema": [{"name": f.name, "type": str(f.type)} for f in output_schema],
        "shard_files": results,
    }
    (build_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    pq.write_metadata(output_schema, build_dir / "_common_metadata")
    (build_dir / "README.md").write_text(
        f"# ScaleEdit merged labeling dataset\n\n"
        f"{totals['rows']:,} rows that passed both filters and reached final mask labeling. "
        f"The {len(nonempty):,} nonempty Parquet files are in `shards/`. "
        "Each row includes the original source/edited image bytes, instructions, task and provenance fields, "
        "both prefilter results and audits, grounding results, final PNG mask, per-instance RLE and QC fields.\n\n"
        "Join identity is `(source_shard, row_idx)`. Source columns retain their names; stage columns "
        "use `quality__`, `quality_audit__`, `scene__`, `scene_audit__`, `grounding__`, and `mask__` prefixes. "
        "`mask__qc_flag` distinguishes `OK`, `MASK_REVIEW`, and `GROUND_FAIL`; all are retained, "
        "and `OK` is automatic QC rather than human validation.\n\n"
        "See `dataset_manifest.json` for exact provenance, counts, shard index and full schema. "
        "`COMPLETE` is written only after all consistency checks pass.\n"
    )
    (build_dir / "COMPLETE").write_text(f"rows={totals['rows']}\nshards={len(nonempty)}\n")
    os.rename(build_dir, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rows": totals["rows"],
                      "shards": len(nonempty), "qc_flags": dict(flags)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
