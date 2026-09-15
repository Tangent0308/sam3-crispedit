"""Validated RefEdit quality-prefilter selections."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq

from refedit.io import sample_id


PREFILTER_REQUIRED_COLUMNS = {
    "row_idx",
    "sample_id",
    "img_id",
    "source_relative_path",
    "instruction",
    "prefilter_verdict",
}


def load_prefilter_manifest_dir(manifest_dir: Path) -> Dict[str, List[dict]]:
    """Load PASS rows grouped by native source-shard basename.

    The quality runner writes one manifest shard per source shard and includes
    only PASS rows.  We still validate the verdict and provenance explicitly so
    a wrong directory cannot silently select unrelated examples.
    """

    manifest_dir = Path(manifest_dir).resolve()
    paths = sorted(manifest_dir.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no train-*.parquet prefilter manifests under {manifest_dir}"
        )
    result: Dict[str, List[dict]] = {}
    seen = set()
    for path in paths:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(PREFILTER_REQUIRED_COLUMNS - names)
        if missing:
            raise ValueError(f"{path.name} misses prefilter columns: {missing}")
        rows = pq.read_table(path).to_pylist()
        previous_row_idx = -1
        for row in rows:
            if str(row["prefilter_verdict"]) != "PASS":
                raise ValueError(
                    f"non-PASS row in prefilter manifest {path.name}: "
                    f"{row['sample_id']}"
                )
            if Path(str(row["source_relative_path"])).name != path.name:
                raise ValueError(
                    f"source shard mismatch in prefilter manifest {path.name}: "
                    f"{row['source_relative_path']}"
                )
            row_idx = int(row["row_idx"])
            if row_idx <= previous_row_idx:
                raise ValueError(
                    f"prefilter rows are not strictly source-ordered in {path.name}"
                )
            previous_row_idx = row_idx
            identity = str(row["sample_id"])
            if identity != sample_id(row["img_id"]):
                raise ValueError(
                    f"img_id/sample_id mismatch in prefilter manifest: {identity}"
                )
            if identity in seen:
                raise ValueError(f"duplicate prefilter sample_id: {identity}")
            seen.add(identity)
        result[path.name] = rows
    return result


def validate_prefilter_rows(source_path: Path, rows: List[dict]) -> None:
    """Validate manifest identities and row indices against a source shard."""

    source_rows = pq.read_table(
        source_path, columns=["img_id", "instruction"]
    ).to_pylist()
    for selected in rows:
        row_idx = int(selected["row_idx"])
        if row_idx < 0 or row_idx >= len(source_rows):
            raise IndexError(
                f"prefilter row_idx out of range in {source_path.name}: {row_idx}"
            )
        source = source_rows[row_idx]
        expected = sample_id(source["img_id"])
        if expected != str(selected["sample_id"]):
            raise ValueError(
                f"prefilter/source identity mismatch in {source_path.name}:{row_idx}: "
                f"{selected['sample_id']!r}!={expected!r}"
            )
        if int(selected["img_id"]) != int(source["img_id"]):
            raise ValueError(
                f"prefilter/source img_id mismatch in {source_path.name}:{row_idx}"
            )
        if str(selected["instruction"]) != str(source["instruction"]):
            raise ValueError(
                f"prefilter/source instruction mismatch in "
                f"{source_path.name}:{row_idx}"
            )
