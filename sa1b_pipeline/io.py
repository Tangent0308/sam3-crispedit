"""Input/output adapters for normalized SA-1B records and checkpoint JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import Candidate, CandidateObject


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _objects(raw: dict[str, Any]) -> list[CandidateObject]:
    values = raw.get("objects") or raw.get("objects_json") or raw.get("annotations") or []
    if isinstance(values, str):
        values = json.loads(values)
    result = []
    for value in values:
        if isinstance(value, str):
            value = json.loads(value)
        result.append(CandidateObject(
            object_id=str(value.get("object_id", value.get("id", value.get("annotation_id", "")))),
            category=str(value.get("category", value.get("label", value.get("name", "object")))),
            description=str(value.get("description", value.get("caption", ""))),
            mask_path=value.get("mask_path", value.get("mask")) if isinstance(value.get("mask_path", value.get("mask")), str) else None,
            rle=value.get("rle", value.get("segmentation")) if isinstance(value.get("rle", value.get("segmentation")), dict) else None,
            bbox=value.get("bbox", value.get("box")),
            area_fraction=value.get("area_fraction"),
            parent_id=value.get("parent_id"),
            metadata={**(value.get("metadata") or {}), **{k: v for k, v in value.items() if k not in {"metadata","object_id", "id", "annotation_id", "category", "label", "name", "description", "caption", "mask_path", "mask", "rle", "segmentation", "bbox", "box", "area_fraction", "parent_id"}}},
        ))
    return result


def candidate_from_dict(raw: dict[str, Any], base_dir: str | Path | None = None) -> Candidate:
    base = Path(base_dir) if base_dir else None
    image = raw.get("image_path") or raw.get("image") or raw.get("file_name")
    if not image:
        raise ValueError("SA-1B record has no image path")
    image = str((base / image).resolve()) if base and not Path(image).is_absolute() else str(image)
    objects = _objects(raw)
    if base:
        for obj in objects:
            if obj.mask_path and not Path(obj.mask_path).is_absolute():
                obj.mask_path = str((base / obj.mask_path).resolve())
            for key in ('execution_mask_path', 'protected_mask_path'):
                if obj.metadata.get(key) and not Path(obj.metadata[key]).is_absolute():
                    obj.metadata[key] = str((base / obj.metadata[key]).resolve())
    return Candidate(
        source_id=str(raw.get("source_id", raw.get("image_id", raw.get("id", "")))),
        image_path=image,
        objects=objects,
        caption=str(raw.get("caption", raw.get("image_caption", ""))),
        tags=list(raw.get("tags", []) or []),
        provenance=dict(raw.get("provenance", {})),
        image_sha256=raw.get("image_sha256", raw.get("sha256")),
    )


def load_candidates(path: str | Path) -> list[Candidate]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("records", value.get("candidates", []))
    else:
        raise ValueError("use a normalized .json or .jsonl SA-1B manifest")
    return [candidate_from_dict(row, path.parent) for row in rows]
