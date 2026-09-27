#!/usr/bin/env python3
"""Normalize optional schema-evolution columns across final CrispEdit shards."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def normalize_one(path: Path, expected: pa.Schema) -> str:
    parquet = pq.ParquetFile(path)
    actual_schema = parquet.schema_arrow
    if actual_schema.equals(expected, check_metadata=False):
        return "already_normalized"

    expected_types = {field.name: field.type for field in expected}
    actual_types = {field.name: field.type for field in actual_schema}
    if any(name not in expected_types or expected_types[name] != data_type for name, data_type in actual_types.items()):
        raise ValueError(f"Unexpected/incompatible schema in {path}")
    table = parquet.read()
    columns = [
        table.column(field.name) if field.name in actual_types else pa.nulls(table.num_rows, type=field.type)
        for field in expected
    ]
    normalized = pa.Table.from_arrays(columns, schema=expected)
    temp_path = path.with_name(path.name + ".schema.tmp")
    if temp_path.exists():
        raise FileExistsError(f"refusing to overwrite stale temp file: {temp_path}")
    try:
        pq.write_table(normalized, temp_path, compression="zstd", compression_level=3, row_group_size=256)
        # Verify the replacement before atomically swapping it in.
        check = pq.ParquetFile(temp_path)
        if check.metadata.num_rows != parquet.metadata.num_rows:
            raise ValueError(f"row count changed while normalizing {path}")
        if not check.schema_arrow.equals(expected, check_metadata=False):
            raise ValueError(f"schema did not normalize correctly: {path}")
        os.replace(temp_path, path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return "normalized"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    root = args.dataset_dir.resolve()
    schema_path = root / "_common_metadata"
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    expected = pq.read_schema(schema_path)
    files = sorted((root / "shards").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {root / 'shards'}")

    counts = {"normalized": 0, "already_normalized": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(normalize_one, path, expected): path for path in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing final shard schemas", unit="shard"):
            counts[future.result()] += 1
    print({"files": len(files), **counts, "columns": len(expected)})


if __name__ == "__main__":
    main()
