"""Read-only helpers for native Hugging Face RefEdit parquet shards."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pyarrow.parquet as pq


REQUIRED_COLUMNS = {"img_id", "source_img", "instruction", "target_img"}


def discover_shards(root: Path) -> List[Path]:
    shards = sorted((root / "data").glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no RefEdit train shards under {root / 'data'}")
    return shards


def sample_id(img_id: object) -> str:
    return f"refedit:{int(img_id)}"


def validate_schema(path: Path) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(REQUIRED_COLUMNS - names)
    if missing:
        raise ValueError(f"{path.name} misses required columns: {missing}")


def iter_row_batches(
    path: Path,
    batch_size: int,
    columns: Sequence[str] = ("img_id", "source_img", "instruction", "target_img"),
) -> Iterable[List[Tuple[int, Dict]]]:
    row_idx = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=max(1, batch_size), columns=list(columns)
    ):
        rows: List[Tuple[int, Dict]] = []
        for record in batch.to_pylist():
            rows.append((row_idx, record))
            row_idx += 1
        yield rows
