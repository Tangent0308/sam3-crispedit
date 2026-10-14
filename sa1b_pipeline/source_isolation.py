"""Fail-closed source isolation for grouped train/eval data."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Iterable

from .geometry import sha256
from .io import read_jsonl, write_jsonl
from .production_sources import DENSE_INDEX, IMAGE_INDEX, read_record


def normalize_source_id(value: Any) -> str:
    return str(value or "").removeprefix("v2-").removeprefix("sa1b-")


def source_keys_from_row(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids, hashes = set(), set()
    source = row.get("source", row)
    if isinstance(source, dict):
        for key in ("source_id", "image_id", "family_id", "id"):
            if source.get(key):
                ids.add(normalize_source_id(source[key]))
        for key in ("image_sha256", "sha256", "image_hash"):
            if source.get(key):
                hashes.add(str(source[key]))
        path = source.get("image_path") or source.get("image")
        if isinstance(path, str) and Path(path).is_file():
            try:
                hashes.add(sha256(path))
            except OSError:
                pass
    return ids, hashes


def collect_manifest_source_keys(paths: Iterable[str | Path], *, fail_on_parse: bool = True) -> dict[str, Any]:
    """Collect source IDs and image byte hashes from all supplied manifests."""
    ids, hashes, files = set(), set(), []
    for value in paths:
        path = Path(value)
        if not path.exists():
            continue
        files.append(str(path.resolve()))
        if path.suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if fail_on_parse:
                        raise ValueError(f"invalid JSONL at {path}:{line_no}")
                    continue
                row_ids, row_hashes = source_keys_from_row(row)
                ids.update(row_ids); hashes.update(row_hashes)
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else value.get("records", value.get("candidates", []))
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows or []:
                if isinstance(row, dict):
                    row_ids, row_hashes = source_keys_from_row(row)
                    ids.update(row_ids); hashes.update(row_hashes)
    return {"source_ids": sorted(ids), "image_sha256": sorted(hashes), "manifests": files}


def discover_training_manifests(dataset_root: str | Path, *, exclude_roots: Iterable[str | Path] = (),
                               max_depth: int = 4) -> list[Path]:
    """Discover only manifest-like files, never image/cache payloads."""
    root = Path(dataset_root)
    excluded = [Path(p).resolve() for p in exclude_roots]
    result = []
    skip_dirs = {"source_cache", "image_hashes", "images", "media", "units", "model_cache",
                 ".runtime", ".git", "__pycache__", "panels", "audit_overlays", "verification"}
    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current).resolve()
        depth = len(current_path.relative_to(root.resolve()).parts)
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if any(current_path == ex or ex in current_path.parents for ex in excluded):
            dirs[:] = []
            continue
        if depth > max_depth:
            dirs[:] = []
            continue
        for name in files:
            path = current_path / name
            if path.suffix == ".jsonl" and path.stat().st_size <= 2 * 1024**3:
                result.append(path)
            elif path.suffix == ".json" and path.stat().st_size <= 256 * 1024**2 and name not in {"config.json", "complete.json", "state.json", "owner.json"}:
                result.append(path)
    return sorted(set(result))


def _raw_source_keys(source_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = Path(source_root)
    with (source_root / IMAGE_INDEX).open("rb") as stream:
        images = pickle.load(stream)
    with (source_root / DENSE_INDEX).open("rb") as stream:
        dense = pickle.load(stream)
    return images, dense


def reserve_eval_sources(*, source_root: str | Path, output: str | Path, count: int,
                         used_source_ids: set[str], used_hashes: set[str], seed: int = 0) -> dict[str, Any]:
    """Reserve eval IDs before any new training source is sampled.

    The reservation stores both random-access index records and raw image byte
    hashes. A later train preparation must consume this reservation and reject
    either identity, which prevents accidental split leakage even if an ID is
    renamed in a downstream manifest.
    """
    if count < 1:
        raise ValueError("eval count must be positive")
    images, dense = _raw_source_keys(source_root)
    eligible = [sid for sid in sorted(images) if normalize_source_id(sid) not in used_source_ids and sid in dense]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = []
    hashes = set(used_hashes)
    for sid in eligible:
        if len(chosen) >= count:
            break
        entry = {"source_id": sid, "image": images[sid], "annotation": dense[sid]}
        try:
            raw = read_record(source_root, images[sid], "sa1b_tfd_1k")["jpg"]
        except (OSError, KeyError, ValueError, EOFError):
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        chosen.append({**entry, "image_sha256": digest, "split": "eval"})
    if len(chosen) != count:
        raise RuntimeError(f"could reserve only {len(chosen)}/{count} eval sources")
    result = {"schema_version": "sa1b.samtok.source_isolation.v1", "seed": seed,
              "count": len(chosen), "reserved_ids": sorted(x["source_id"] for x in chosen),
              "reserved_image_sha256": sorted(x["image_sha256"] for x in chosen),
              "excluded_prior_source_ids": sorted(used_source_ids),
              "excluded_prior_image_sha256": sorted(used_hashes), "entries": chosen}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def load_reservation(path: str | Path) -> tuple[set[str], set[str]]:
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != "sa1b.samtok.source_isolation.v1":
        raise ValueError("unsupported source reservation")
    return {normalize_source_id(x) for x in value.get("reserved_ids", [])}, set(value.get("reserved_image_sha256", []))


def assert_no_eval_leakage(rows: Iterable[dict[str, Any]], reservation: str | Path) -> None:
    eval_ids, eval_hashes = load_reservation(reservation)
    for row in rows:
        ids, hashes = source_keys_from_row(row)
        overlap = ids & eval_ids
        hash_overlap = hashes & eval_hashes
        if overlap or hash_overlap:
            raise ValueError(f"eval source leaked into training: ids={sorted(overlap)} hashes={sorted(hash_overlap)}")
