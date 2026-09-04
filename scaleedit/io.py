"""Read-only helpers for ScaleEdit parquet shards."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pyarrow.parquet as pq
from PIL import Image


def discover_shards(root: Path) -> List[Path]:
    """Return only materialized ScaleEdit data shards, never metadata caches."""

    shards = sorted(root.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no part-*.parquet shards under {root}")
    return shards


def image_bytes(value: Any) -> bytes:
    """Accept ScaleEdit binary cells and Arrow/HF image structs."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    raise TypeError(f"unsupported image cell: {type(value)!r}")


def decode_image(value: Any) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes(value))).convert("RGB")


def image_size(value: Any) -> Tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes(value))) as image:
        return image.size


def read_rows(path: Path, columns: Sequence[str] | None = None) -> List[Dict]:
    return pq.read_table(path, columns=columns).to_pylist()


def iter_row_batches(path: Path, batch_size: int) -> Iterable[List[Tuple[int, Dict]]]:
    """Stream a shard while preserving the shard-local row index."""

    row_idx = 0
    pending: List[Tuple[int, Dict]] = []
    parquet = pq.ParquetFile(path)
    for record_batch in parquet.iter_batches(batch_size=max(1, batch_size)):
        for record in record_batch.to_pylist():
            pending.append((row_idx, record))
            row_idx += 1
            if len(pending) >= batch_size:
                yield pending
                pending = []
    if pending:
        yield pending
