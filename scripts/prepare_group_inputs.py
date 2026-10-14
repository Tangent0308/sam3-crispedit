#!/usr/bin/env python3
"""Reserve leakage-free eval sources and materialize grouped-data candidates."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sa1b_pipeline.production_sources import materialize
from sa1b_pipeline.production_state import atomic_json
from sa1b_pipeline.source_isolation import (collect_manifest_source_keys, discover_training_manifests,
                                             normalize_source_id, reserve_eval_sources)
from sa1b_pipeline.io import write_jsonl


def _rows(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets"))
    p.add_argument("--frozen-root", type=Path, required=True)
    p.add_argument("--benchmark-exclusion", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--eval-count", type=int, default=200)
    p.add_argument("--train-limit", type=int, default=8000,
                   help="number of source candidates to materialize; use 0 to skip")
    p.add_argument("--materialize-missing-train", action="store_true",
                   help="read non-materialized 50k reservoir records from SG2 HDFS")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--skip-discovery", action="store_true",
                   help="use only benchmark and frozen reservoir for the reservation registry")
    p.add_argument("--existing-manifest", action="append", type=Path, default=[],
                   help="explicit prior training manifest to include in eval exclusion; repeatable")
    p.add_argument("--materialize-workers", type=int, default=8,
                   help="bounded concurrent HDFS reads for missing train sources")
    args = p.parse_args(argv)
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    protected = args.frozen_root.resolve()
    if out == protected or protected in out.parents or out in protected.parents:
        raise RuntimeError("group input root must be separate from the frozen production root")
    # ``collect_manifest_source_keys`` returns a dict; use explicit names to
    # make the exclusion provenance clear in the persisted record.
    benchmark_keys = collect_manifest_source_keys([args.benchmark_exclusion])
    benchmark_ids = set(benchmark_keys["source_ids"])
    benchmark_hashes = set(benchmark_keys["image_sha256"])
    used_ids = set(benchmark_ids)
    used_hashes = set(benchmark_hashes)
    reservoir = args.frozen_root / "inputs/reservoir.jsonl"
    if reservoir.exists():
        for row in _rows(reservoir):
            used_ids.add(normalize_source_id(row.get("source_id")))
    manifests = list(args.existing_manifest)
    if not args.skip_discovery:
        manifests.extend(discover_training_manifests(args.dataset_root,
                                                     exclude_roots=[out, protected / "source_cache"]))
    # Preserve provenance while avoiding duplicate parsing when a path is both
    # explicitly supplied and discovered by the bounded scan.
    manifests = list(dict.fromkeys(str(Path(x).resolve()) for x in manifests))
    discovered = collect_manifest_source_keys(manifests, fail_on_parse=False)
    # Existing training rows are included in the registry for eval exclusion;
    # this does not prevent grouped training from reusing a valid 50k source.
    used_ids.update(discovered["source_ids"])
    used_hashes.update(discovered["image_sha256"])
    reservation_path = out / "inputs/eval_reservation.json"
    if reservation_path.exists():
        reservation = json.loads(reservation_path.read_text())
    else:
        reservation = reserve_eval_sources(source_root=args.source_root, output=reservation_path,
                                           count=args.eval_count, used_source_ids=used_ids,
                                           used_hashes=used_hashes, seed=args.seed)
    eval_ids = set(reservation["reserved_ids"])
    # Training may reuse 50k source IDs, but never the reserved eval IDs or
    # benchmark sources. Existing unrelated manifests only constrain eval
    # reservation and are not an arbitrary training exclusion list.
    exclusion = {"source_ids": benchmark_ids | eval_ids, "image_hashes": benchmark_hashes | set(reservation["reserved_image_sha256"]),
                 "benchmark_sha256": "group-source-isolation"}
    if args.train_limit == 0:
        train_entries = []
    else:
        train_entries = _rows(reservoir) if reservoir.exists() else []
        train_entries = [r for r in train_entries if normalize_source_id(r.get("source_id")) not in eval_ids]
        train_entries = train_entries[:args.train_limit]
    train_candidates, train_rejections = [], []
    missing_train = []
    for entry in train_entries:
        # The frozen 50k run already materialized many source images. Reusing
        # those read-only files is allowed for grouped training and avoids an
        # unnecessary second HDFS round trip; final export copies media into
        # the grouped root with ``portable``.
        sid = str(entry.get("source_id"))
        frozen_marker = args.frozen_root / "sources" / sid / "complete.json"
        if frozen_marker.is_file():
            frozen = json.loads(frozen_marker.read_text())
            if frozen.get("status") == "ready":
                candidate = dict(frozen["candidate"])
                candidate.setdefault("provenance", {})["group_training_source"] = "frozen_50k_read_only"
                train_candidates.append(candidate)
                continue
        if not args.materialize_missing_train:
            train_rejections.append({"source_id": sid, "status": "not_materialized_in_frozen_root"})
            continue
        missing_train.append(entry)
    if missing_train:
        workers = max(1, min(int(args.materialize_workers), len(missing_train)))
        # Random-access reads are independent. Results are restored to source
        # order after the bounded pool completes so manifests remain stable
        # across resume and worker counts.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(materialize, entry, out, args.source_root, exclusion): i
                       for i, entry in enumerate(missing_train)}
            ordered = [None] * len(missing_train)
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        for result in ordered:
            if result.get("status") == "ready":
                train_candidates.append(result["candidate"])
            else:
                train_rejections.append(result)
    eval_candidates, eval_rejections = [], []
    eval_exclusion = {"source_ids": benchmark_ids, "image_hashes": benchmark_hashes,
                      "benchmark_sha256": "group-source-isolation"}
    # HDFS random reads are independent; bounded concurrency reduces wall time
    # without changing the deterministic reservation or atomic source claims.
    def materialize_eval(entry):
        return materialize(entry, out, args.source_root, eval_exclusion)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(materialize_eval, entry) for entry in reservation["entries"]]
        for future in as_completed(futures):
            result = future.result()
            if result.get("status") == "ready":
                eval_candidates.append(result["candidate"])
            else:
                eval_rejections.append(result)
    write_jsonl(out / "inputs/train_candidates.jsonl", train_candidates)
    write_jsonl(out / "inputs/eval_candidates.jsonl", eval_candidates)
    atomic_json(out / "inputs/manifest.json", {
        "schema_version": "sa1b.samtok.group.inputs.v1", "train_candidates": len(train_candidates),
        "eval_candidates": len(eval_candidates), "train_rejections": len(train_rejections),
        "eval_rejections": len(eval_rejections), "eval_reservation": str(reservation_path),
        "benchmark_exclusion": str(args.benchmark_exclusion), "frozen_root": str(protected),
        "source_root": str(args.source_root), "discovered_manifests": discovered["manifests"],
        "eval_ids": sorted(eval_ids), "eval_hashes": sorted(reservation["reserved_image_sha256"]),
    })
    print(json.dumps({"train_candidates": len(train_candidates), "eval_candidates": len(eval_candidates),
                      "eval_reservation": str(reservation_path), "root": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
