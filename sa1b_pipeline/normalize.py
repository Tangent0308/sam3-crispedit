"""Normalize SA-1B companion records before visual selection.

The handoff's dense TFRecords have varied feature names across shards.  This
adapter accepts those records when TensorFlow is installed and also supports
JSON/JSONL/Parquet exports, preserving the original shard/index provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .io import write_jsonl


def _value(feature: Any, key: str = "") -> Any:
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        values = list(feature.bytes_list.value)
        if key.lower() in {"image", "image_bytes", "encoded_image", "jpg", "jpeg", "png"}:
            return values[0] if len(values) == 1 else values
        return values[0].decode("utf-8", "replace") if len(values) == 1 else [x.decode("utf-8", "replace") for x in values]
    if kind == "int64_list":
        values = list(feature.int64_list.value)
        return values[0] if len(values) == 1 else values
    if kind == "float_list":
        values = list(feature.float_list.value)
        return values[0] if len(values) == 1 else values
    return None


def iter_tfrecord(path: str | Path) -> Iterable[dict[str, Any]]:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to read SA-1B TFRecords; export JSONL first otherwise") from exc
    for offset, raw in enumerate(tf.data.TFRecordDataset([str(path)])):
        example = tf.train.Example.FromString(bytes(raw))
        features = {key: _value(value, key) for key, value in example.features.feature.items()}
        objects = features.get("objects_json", features.get("objects", []))
        if isinstance(objects, str):
            try:
                objects = json.loads(objects)
            except json.JSONDecodeError:
                objects = []
        image = features.get("image") or features.get("image_bytes")
        # Dense shards often keep image bytes in a separate TFRecord.  Such
        # records remain traceable and can be joined by key in a later stage.
        yield {"source_id": str(features.get("key", offset)), "image_bytes": image,
               "caption": features.get("caption", ""), "tags": features.get("tags", []),
               "objects": objects, "provenance": {"record_file": str(path), "record_offset": offset}}


def normalize_records(records: Iterable[dict[str, Any]], output_dir: Path, base_dir: Path | None = None) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for index, record in enumerate(records):
        record = dict(record)
        source_id = str(record.get("source_id", record.get("image_id", index)))
        image_path = record.get("image_path") or record.get("image") or record.get("file_name")
        if not image_path and record.get("image_bytes"):
            suffix = ".jpg"
            raw = record["image_bytes"]
            if isinstance(raw, str):
                raw = raw.encode("latin1")
            image_path = str(output_dir / "images" / f"{source_id}{suffix}")
            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            Path(image_path).write_bytes(raw)
        if not image_path:
            continue
        if base_dir and not Path(str(image_path)).is_absolute():
            image_path = str((base_dir / str(image_path)).resolve())
        objects = record.get("objects", record.get("annotations", []))
        for obj in objects if isinstance(objects, list) else []:
            if isinstance(obj, dict):
                mask = obj.get("mask_path", obj.get("mask"))
                if isinstance(mask, str) and base_dir and not Path(mask).is_absolute():
                    obj["mask_path"] = str((base_dir / mask).resolve())
        row = {"source_id": source_id, "image_path": str(image_path),
               "caption": record.get("caption", ""), "tags": record.get("tags", []),
               "objects": objects,
               "provenance": record.get("provenance", {})}
        result.append(row)
    write_jsonl(output_dir / "candidates.jsonl", result)
    return result


def join_sa1b_records(image_dir: Path, dense_dir: Path, output_dir: Path,
                      ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Join the handoff's separate image and dense-label TFRecord trees.

    `ids` is strongly recommended for the 11M-record corpus; without it this
    function is correct but necessarily scans both trees.
    """
    image_rows: dict[str, dict[str, Any]] = {}
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() not in {".index", ".json", ".txt"}:
            for row in iter_tfrecord(path):
                key = str(row["source_id"])
                if ids is None or key in ids:
                    image_rows[key] = row
    merged = []
    for path in sorted(dense_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".index", ".json", ".txt"}:
            continue
        for row in iter_tfrecord(path):
            key = str(row["source_id"])
            if ids is not None and key not in ids:
                continue
            image = image_rows.get(key)
            if image is None:
                continue
            merged.append({"source_id": key, "image_bytes": image.get("image_bytes"),
                           "objects": row.get("objects", []), "caption": row.get("caption", image.get("caption", "")),
                           "tags": row.get("tags", image.get("tags", [])),
                           "provenance": {"image": image.get("provenance", {}), "dense": row.get("provenance", {})}})
    return normalize_records(merged, output_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON/JSONL or TFRecord file")
    source.add_argument("--image-records-dir", type=Path, help="SA-1B image TFRecord tree")
    p.add_argument("--dense-records-dir", type=Path, help="SA-1B dense annotation TFRecord tree")
    p.add_argument("--id-file", type=Path, help="newline-delimited keys to join/sample")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if a.image_records_dir:
        if not a.dense_records_dir:
            p.error("--dense-records-dir is required with --image-records-dir")
        ids = set(a.id_file.read_text().split()) if a.id_file else None
        rows = join_sa1b_records(a.image_records_dir, a.dense_records_dir, a.output_dir, ids)
        print(json.dumps({"records": len(rows), "manifest": str(a.output_dir / 'candidates.jsonl')}))
        return
    if a.input.suffix.lower() in {".json", ".jsonl"}:
        from .io import read_jsonl
        if a.input.suffix.lower() == ".jsonl":
            records = read_jsonl(a.input)
        else:
            value = json.loads(a.input.read_text(encoding="utf-8")); records = value if isinstance(value, list) else value.get("records", [])
    else:
        records = iter_tfrecord(a.input)
    rows = normalize_records(records, a.output_dir, a.input.parent if a.input else None)
    print(json.dumps({"records": len(rows), "manifest": str(a.output_dir / 'candidates.jsonl')}))


if __name__ == "__main__":
    main()
