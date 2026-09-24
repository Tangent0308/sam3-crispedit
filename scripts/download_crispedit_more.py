#!/usr/bin/env python3
"""Download a balanced, resumable CrispEdit shard expansion from Hugging Face."""

import argparse
import json
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


REPO = "WeiChow/CrispEdit-2M"
TYPES = ("add", "color", "remove", "replace")
BASE = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def spread(names: list[str], count: int) -> list[str]:
    if len(names) < count:
        raise ValueError(f"Only {len(names)} eligible files, need {count}")
    if count == 1:
        return [names[len(names) // 2]]
    return [names[(i * (len(names) - 1)) // (count - 1)] for i in range(count)]


def make_plan(source: Path, per_type: int) -> dict:
    info = HfApi().repo_info(REPO, repo_type="dataset")
    remote = [entry.rfilename for entry in info.siblings]
    existing = {path.name for path in source.glob("*.parquet")}
    chosen = []
    for kind in TYPES:
        candidates = sorted(
            (name for name in remote
             if re.fullmatch(r"data/" + re.escape(kind) + r"_\d+\.parquet", name)
             and Path(name).name not in existing),
            key=lambda name: int(Path(name).stem.rsplit("_", 1)[1]),
        )
        chosen.extend(spread(candidates, per_type))
    assert len(chosen) == len(set(chosen)) == len(TYPES) * per_type
    return {
        "repo": REPO, "revision": info.sha, "files": chosen,
        "initial_source_shards": len(existing),
        "planned_utc": datetime.now(timezone.utc).isoformat(),
        "expected_rows_approx": len(chosen) * 256,
    }


def validate(path: Path) -> int:
    parquet = pq.ParquetFile(path)
    missing = {"input_img", "output_img", "instruction", "type"} - set(parquet.schema_arrow.names)
    if missing or parquet.metadata.num_rows <= 0:
        raise ValueError(f"Invalid CrispEdit shard {path}: missing={missing}, rows={parquet.metadata.num_rows}")
    return parquet.metadata.num_rows


def download_one(name: str, plan: dict, source: Path, staging: Path) -> tuple[str, int, bool]:
    destination = source / Path(name).name
    if destination.exists():
        return name, validate(destination), False
    error = None
    for attempt in range(4):
        try:
            path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset", filename=name,
                revision=plan["revision"], local_dir=staging))
            rows = validate(path)
            # Same filesystem; exclusive link prevents a concurrent run replacing user data.
            try:
                os.link(path, destination)
            except FileExistsError:
                rows = validate(destination)
                return name, rows, False
            path.unlink()
            return name, rows, True
        except Exception as exc:
            error = exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Failed after four attempts: {name}: {error!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=BASE / "CrispEdit-2M")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=195)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.per_type <= 0 or args.workers <= 0:
        parser.error("--per-type and --workers must be positive")
    source = args.source_dir.resolve()
    if not source.is_dir():
        parser.error(f"source directory missing: {source}")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["repo"] != REPO or len(plan["files"]) != args.per_type * len(TYPES):
            parser.error("Existing plan conflicts with requested repo/per-type count")
    else:
        plan = make_plan(source, args.per_type)
        atomic_json(plan_path, plan)
    print(f"Pinned {plan['repo']} @ {plan['revision']}", flush=True)
    print(f"Selected {len(plan['files'])} shards; per type {args.per_type}; approx {plan['expected_rows_approx']} rows", flush=True)
    if args.plan_only:
        return
    staging = run_dir / "staging"
    marker = staging / ".crispedit_download_owned"
    staging.mkdir(exist_ok=True)
    if not marker.exists():
        if any(staging.iterdir()):
            raise RuntimeError(f"Refusing to use nonempty unowned staging directory: {staging}")
        marker.write_text(REPO + "\n" + plan["revision"] + "\n")
    elif marker.read_text() != REPO + "\n" + plan["revision"] + "\n":
        raise RuntimeError(f"Staging ownership/revision mismatch: {staging}")
    results = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, name, plan, source, staging): name for name in plan["files"]}
        with tqdm(total=len(futures), unit="shard", dynamic_ncols=True, mininterval=5) as progress:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    _, rows, added = future.result()
                    results[name] = {"rows": rows, "added": added}
                except Exception as exc:
                    errors[name] = repr(exc)
                    print(f"ERROR {name}: {exc!r}", flush=True)
                progress.update(1)
                progress.set_postfix(ok=len(results), errors=len(errors))
    summary = {
        "repo": REPO, "revision": plan["revision"], "selected_shards": len(plan["files"]),
        "verified_shards": len(results), "verified_rows": sum(item["rows"] for item in results.values()),
        "added_shards_this_attempt": sum(item["added"] for item in results.values()),
        "by_type": dict(Counter(Path(name).name.rsplit("_", 1)[0] for name in results)),
        "errors": errors, "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "summary.json", summary)
    if errors or len(results) != len(plan["files"]):
        raise RuntimeError(f"Download incomplete: {len(errors)} errors; resume same --run-dir")
    if marker.read_text() != REPO + "\n" + plan["revision"] + "\n":
        raise RuntimeError("Staging marker changed during download")
    shutil.rmtree(staging)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
