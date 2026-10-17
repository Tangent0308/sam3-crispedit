#!/usr/bin/env python3
"""Prepare, resume, monitor and export current grouped-data production on 4x8 GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.group_production import (
    accept,
    choose,
    describe,
    propose,
    split_cells,
    target_cells,
)
from sa1b_pipeline.production_state import (
    atomic_json,
    file_hash,
    immutable_json,
    require_json,
    portable,
    read_rows,
    write_rows,
)

REPO = Path(__file__).resolve().parents[1]
PROTECTED = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014"
)
CHILDREN = set()
CHILDREN_LOCK = threading.Lock()

MODEL_BASE = "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/"


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def config(root):
    return json.loads((root / "config.json").read_text())


class LockBusyError(RuntimeError):
    pass


@contextlib.contextmanager
def owner_lock(path):
    """Atomic mkdir works on the shared mount even when flock is unsupported."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        owner = (
            json.loads((path / "owner.json").read_text())
            if (path / "owner.json").exists()
            else {}
        )
        if owner.get("host") == socket.gethostname() and owner.get("pid"):
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                shutil.rmtree(path)
                path.mkdir()
            else:
                raise LockBusyError(f"active owner: {owner}")
        else:
            raise LockBusyError(
                f"owner lock remains at {path}; verify previous worker has stopped before recover-lock"
            )
    atomic_json(
        path / "owner.json",
        {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": time.time(),
            "trial": os.environ.get("ARNOLD_TRIAL_ID"),
        },
    )
    try:
        yield
    finally:
        shutil.rmtree(path)


def prepare(args):
    from sa1b_pipeline.source_isolation import load_reservation, normalize_source_id

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    targets = {
        "twins": target_cells("twins", args.twins_target),
        "multi": target_cells("multi", args.multi_target),
    }
    c = {
        "schema": "group-production-1",
        "world_size": 4,
        "gpus_per_node": 8,
        "protocol": "2.0",
        "mask_semantics": "2.0",
        "instruction_style": "benchmark-concise-3-blind-1",
        "steps": 40,
        "multi_commit_policy": "parallel-components-v1",
        "multi_audit_devices": "6,7",
        "multi_groups_in_flight": 16,
        "targets": targets,
        "rank_targets": {k: split_cells(v) for k, v in targets.items()},
        "planner_model": MODEL_BASE + "Qwen3.8-Flash-Next-FP8",
        "planner_tp": 4,
        "blind_model": MODEL_BASE + "Qwen3.8-27B",
        "audit_model": MODEL_BASE + "Qwen3.8-27B",
        "editor_model": MODEL_BASE + "Qwen-Image-2.1",
        "site_model": "/mnt/bn/strategy-mllm-train/common/models/sam3",
        "wave_size": 128,
        "source_chunk_size": 64,
        "max_attempts": {"twins": 150000, "multi": 100000},
        "source_manifest_sha256": file_hash(args.manifest),
        "reservation_sha256": file_hash(args.reservation),
        "max_accepted_per_source_per_dataset": 2,
        "published": False,
    }
    immutable_json(root / "config.json", c)
    if (root / "inputs/ready.json").exists():
        print("Inputs already prepared; no rewrite.")
        return
    (root / "inputs").mkdir(exist_ok=True)
    reserved, hashes = load_reservation(args.reservation)
    shutil.copyfile(args.reservation, root / "inputs/eval_reservation.json")
    seen_ids = set()
    seen_hashes = set()
    counts = Counter()
    copied = 0

    def materialize(row):
        src = Path(row["image_path"])
        h = file_hash(src)
        if row.get("image_sha256") and h != row["image_sha256"]:
            raise ValueError("source image changed: " + str(src))
        if normalize_source_id(row["source_id"]) in reserved or h in hashes:
            raise ValueError("eval source leak")
        dst = root / "sources" / h[:2] / (h + src.suffix.lower())
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            tmp = dst.with_name(dst.name + "." + uuid.uuid4().hex + ".tmp")
            shutil.copyfile(src, tmp)
            if file_hash(tmp) != h:
                raise ValueError("source copy checksum mismatch")
            tmp.replace(dst)
        elif file_hash(dst) != h:
            raise ValueError("existing source copy corrupted")
        row["image_path"] = str(dst)
        row["image_sha256"] = h
        for obj in row.get("objects", []):
            if obj.get("mask_path"):
                obj["mask_path"] = str(
                    root
                    / portable(
                        obj["mask_path"],
                        root,
                        key="mask_path",
                        base=args.manifest.parent,
                    )
                )
        return row

    streams = [
        (root / "inputs" / f"train_rank{i}.jsonl.tmp").open("w") for i in range(4)
    ]
    try:
        with cf.ThreadPoolExecutor(max_workers=16) as pool:
            iterator = rows(args.manifest)
            while True:
                import itertools

                batch = list(itertools.islice(iterator, 128))
                if not batch:
                    break
                for row in pool.map(materialize, batch):
                    sid = normalize_source_id(row["source_id"])
                    h = row["image_sha256"]
                    if sid in seen_ids or h in seen_hashes:
                        continue
                    seen_ids.add(sid)
                    seen_hashes.add(h)
                    rank = int(h[:16], 16) % 4
                    streams[rank].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[rank] += 1
                    copied += 1
                atomic_json(
                    root / "inputs/preparation_progress.json",
                    {"sources": copied, "by_rank": dict(counts), "time": time.time()},
                )
                print(
                    json.dumps({"prepared_sources": copied, "by_rank": dict(counts)}),
                    flush=True,
                )
        for f in streams:
            f.close()
        for rank in range(4):
            (root / "inputs" / f"train_rank{rank}.jsonl.tmp").replace(
                root / "inputs" / f"train_rank{rank}.jsonl"
            )
        ready = {
            "sources": copied,
            "by_rank": dict(counts),
            "eval_isolated": True,
            "files": {
                f"train_rank{i}.jsonl": file_hash(
                    root / "inputs" / f"train_rank{i}.jsonl"
                )
                for i in range(4)
            },
        }
        atomic_json(root / "inputs/ready.json", ready)
    finally:
        for f in streams:
            f.close()


def stage_env(devices):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=devices,
        SA1B_FORCE_DIRECT_VLLM="1",
        SA1B_MM_ENCODER_ATTN_BACKEND="TORCH_SDPA",
        DIFFUSION_ATTENTION_BACKEND="TORCH_SDPA",
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="4",
    )
    return env


def command(wave, name, cmd, devices):
    immutable_json(wave / (name + "_command.json"), {"argv": cmd, "devices": devices})
    done = wave / (name + "_exit.json")
    if done.exists() and json.loads(done.read_text())["exit_code"] == 0:
        return
    start = time.time()
    atomic_json(wave / "active_stage.json", {"stage": name, "started": start})
    with (wave / (name + ".log")).open("a") as log:
        child = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=stage_env(devices),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with CHILDREN_LOCK:
            CHILDREN.add(child)
        try:
            returncode = child.wait()
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            with CHILDREN_LOCK:
                CHILDREN.discard(child)
    result = {"exit_code": returncode, "seconds": time.time() - start, "started": start}
    with (wave / "stage_attempts.jsonl").open("a") as f:
        f.write(json.dumps({"stage": name, **result}) + "\n")
    atomic_json(done, result)
    if result["exit_code"]:
        raise RuntimeError(f"{name} failed; see {wave}/{name}.log; resume same root")


def run_wave(args, c, kind, wave, chosen):
    manifest = wave / "inputs/manifest.jsonl"
    if not manifest.exists():
        write_rows(
            manifest,
            list(
                {r["candidate"]["source_id"]: r["candidate"] for r in chosen}.values()
            ),
        )
    common = [
        str(REPO / "scripts/run_group_pipeline.py"),
        "--kind",
        kind,
        "--manifest",
        str(manifest),
        "--output-root",
        str(wave),
        "--target",
        str(len(chosen)),
        "--split",
        "train",
        "--reservation",
        str(args.root / "inputs/eval_reservation.json"),
        "--outline-planning-crops",
        "--locate-planning-scene",
        "--structured-planning-json",
        "--site-model",
        c["site_model"],
    ]
    if kind == "twins":
        common += ["--quota-from-selection"]
    # Multi operation combinations are selected against final cells. A refill
    # wave may intentionally be skewed; full final quotas are enforced by this
    # coordinator, rather than a per-wave equal-count heuristic.
    if kind == "multi":
        common += ["--dynamic-operation-quotas"]
    cmd = [
        args.mllm_python,
        *common,
        "--plan-only",
        "--planner-model",
        c["planner_model"],
        "--planner-tp",
        str(c["planner_tp"]),
        "--planner-batch-size",
        "8",
        "--max-new-tokens",
        "2400" if kind == "multi" else "1600",
    ]
    if kind == "twins":
        cmd += ["--defer-blind"]
    command(wave, "planning", cmd, "0,1,2,3")
    stage = str(REPO / "scripts/group_production/stage.py")
    if kind == "twins":
        command(
            wave,
            "blind",
            [
                args.mllm_python,
                stage,
                "--stage",
                "blind",
                "--wave",
                str(wave),
                "--kind",
                kind,
                "--model",
                c["blind_model"],
            ],
            "0",
        )
    if not read_rows(wave / "planned.jsonl"):
        write_rows(wave / "generated.jsonl", [])
        write_rows(wave / "audited.jsonl", [])
        return
    if kind == "multi" and c.get("multi_commit_policy", "all-or-nothing") in {"keep-passed-v1", "parallel-components-v1"}:
        audit_devices = c.get("multi_audit_devices", "6,7")
        audit_ids = set(audit_devices.split(","))
        editor_devices = ",".join(x for x in args.editor_devices.split(",") if x not in audit_ids)
        if not editor_devices:
            raise ValueError("transactional multi needs separate editor and audit GPUs")
        command(wave, "transactional_generation_audit", [
            args.mllm_python, str(REPO / "scripts/run_multi_transactional.py"),
            "--planned-manifest", str(wave / "planned.jsonl"), "--output-root", str(wave),
            "--editor-python", args.editor_python, "--audit-python", args.mllm_python,
            "--editor-model", c["editor_model"], "--audit-model", c["audit_model"],
            "--editor-devices", editor_devices, "--audit-devices", audit_devices,
            "--groups-in-flight", str(c.get("multi_groups_in_flight", 16)),
            "--generation-mode", "parallel-components" if c["multi_commit_policy"] == "parallel-components-v1" else "serial",
        ], ",".join(dict.fromkeys(editor_devices.split(",") + audit_devices.split(","))))
        return
    command(
        wave,
        "generation",
        [
            args.editor_python,
            *common,
            "--generate-only",
            "--editor-model",
            c["editor_model"],
            "--editor-devices",
            args.editor_devices,
        ],
        args.editor_devices,
    )
    generated = read_rows(wave / "generated.jsonl")
    if not generated:
        write_rows(wave / "audited.jsonl", [])
        return
    # Publish the shared audit contract once before spawning readers. Avoid
    # concurrent rename/visibility races on the mounted shared filesystem.
    from sa1b_pipeline.group_audit import group_audit_contract
    contract = group_audit_contract(kind, c["audit_model"])
    immutable_json(wave / "audit_checkpoints/contract.json", contract)
    require_json(wave / "audit_checkpoints/contract.json", contract)
    workers = min(4, len(generated))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                command,
                wave,
                f"audit{i}",
                [
                    args.mllm_python,
                    stage,
                    "--stage",
                    "audit",
                    "--wave",
                    str(wave),
                    "--kind",
                    kind,
                    "--model",
                    c["audit_model"],
                    "--worker",
                    str(i),
                    "--workers",
                    str(workers),
                ],
                str(i),
            )
            for i in range(workers)
        ]
        for future in futures:
            future.result()
    write_rows(
        wave / "audited.jsonl",
        [
            g
            for i in range(workers)
            for g in read_rows(wave / f"audited_worker{i}.jsonl")
        ],
    )


def run_kind(args, c, kind):
    from sa1b_pipeline.group_schema import validate_multi_record, validate_twin_record

    root = args.root / "nodes" / f"node{args.rank}" / kind
    root.mkdir(parents=True, exist_ok=True)
    targets = c["rank_targets"][kind][args.rank]
    candidates = list(rows(args.root / "inputs" / f"train_rank{args.rank}.jsonl"))
    # Stable per-kind shuffle avoids repeatedly preferring the first images.
    import random

    random.Random(20261016 + args.rank).shuffle(candidates)
    started_path = root / "started.json"
    if not started_path.exists():
        atomic_json(started_path, {"time": time.time()})
    started = json.loads(started_path.read_text())["time"]
    groups = []
    attempted = []
    pool = []
    cursor = 0
    validator = validate_twin_record if kind == "twins" else validate_multi_record
    for wave_no in range(1, c["max_attempts"][kind] + 1):
        accepted = accept(groups, targets, kind)
        state = describe(accepted, targets, kind)
        state.update(
            kind=kind,
            rank=args.rank,
            attempted=len(attempted),
            time=time.time(),
            started=started,
            wave=wave_no,
            status="running",
        )
        atomic_json(root / "progress.json", state)
        write_rows(root / "accepted.jsonl", accepted)
        if state["complete"]:
            break
        wave = root / "waves" / f"{wave_no:05d}"
        selection = wave / "inputs/selected.jsonl"
        if selection.exists():
            chosen = read_rows(selection)
        else:
            if len(attempted) >= c["max_attempts"][kind] // 4:
                state["status"] = "attempt_budget_exhausted"
                break
            while True:
                chosen = choose(
                    pool,
                    attempted,
                    accepted,
                    targets,
                    kind,
                    min(c["wave_size"], c["max_attempts"][kind] // 4 - len(attempted)),
                )
                if len(chosen) >= c["wave_size"] or cursor >= len(candidates):
                    break
                stop = min(len(candidates), cursor + c["source_chunk_size"])
                chunk = root / "pool" / f"{cursor:06d}.jsonl"
                if chunk.exists():
                    additions = read_rows(chunk)
                else:
                    wanted = [k for k, v in state["remaining"].items() if v]
                    jobs = [
                        (cand, kind, wanted, 20261016 + cursor + i)
                        for i, cand in enumerate(candidates[cursor:stop])
                    ]
                    import multiprocessing as mp

                    with cf.ProcessPoolExecutor(
                        max_workers=args.selection_workers,
                        mp_context=mp.get_context("spawn"),
                    ) as workers:
                        additions = [
                            row
                            for result in workers.map(propose, jobs, chunksize=1)
                            for row in result
                        ]
                    write_rows(chunk, additions)
                pool.extend(additions)
                cursor = stop
                atomic_json(
                    root / "pool/progress.json",
                    {
                        "sources_scanned": cursor,
                        "total_sources": len(candidates),
                        "pool_rows": len(pool),
                    },
                )
            if not chosen:
                state["status"] = "candidate_pool_exhausted"
                break
            write_rows(selection, chosen)
        if not (wave / "audited.jsonl").exists():
            run_wave(args, c, kind, wave, chosen)
        audited = read_rows(wave / "audited.jsonl")
        for g in audited:
            from sa1b_pipeline.group_production import passed

            if passed(g, kind):
                try:
                    validator(g, expected_split="train")
                except (ValueError, KeyError, TypeError) as exc:
                    g.setdefault("export_validation", {})["error"] = str(exc)
                    if kind == "twins":
                        g["group_checks"]["group_pass"] = False
                    else:
                        g["joint_audit"]["pass"] = False
        groups.extend(audited)
        attempted.extend(chosen)
        # No reuse of the exact source/objects proposal, even after a rejection.
        print(
            json.dumps(
                {
                    "kind": kind,
                    "rank": args.rank,
                    "wave": wave_no,
                    "attempted": len(attempted),
                    **describe(groups, targets, kind),
                }
            ),
            flush=True,
        )
    accepted = accept(groups, targets, kind)
    state = {
        **state,
        **describe(accepted, targets, kind),
        "attempted": len(attempted),
        "time": time.time(),
    }
    if state["complete"]:
        state["status"] = "automatic_quota_complete"
    atomic_json(root / "progress.json", state)
    write_rows(root / "accepted.jsonl", accepted)
    return state


def summary(root):
    c = config(root)
    result = {"time": time.time(), "published": False, "datasets": {}}
    for kind in ("twins", "multi"):
        states = []
        for rank in range(4):
            p = root / "nodes" / f"node{rank}" / kind / "progress.json"
            state = (
                json.loads(p.read_text())
                if p.exists()
                else {
                    "rank": rank,
                    "accepted": 0,
                    "attempted": 0,
                    "status": "not_started",
                }
            )
            if state.get("status") == "running":
                wave = p.parent / "waves" / f"{state.get('wave', 1):05d}"
                stage = wave / "active_stage.json"
                if stage.exists():
                    state["active_stage"] = json.loads(stage.read_text())
                state["current_wave_completed"] = {
                    "planned": len(list((wave / "planned").glob("*.json"))),
                    "generated": len(list((wave / "generated").glob("*/group.json"))),
                    "audited": max(
                        0, len(list((wave / "audit_checkpoints").glob("*.json"))) - 1
                    ),
                }
                fail = p.parents[1] / "failure.json"
                if fail.exists() and json.loads(fail.read_text()).get(
                    "time", 0
                ) > state.get("time", 0):
                    state["status"] = "failed"
            states.append(state)
        accepted = sum(x["accepted"] for x in states)
        attempted = sum(x.get("attempted", 0) for x in states)
        target = sum(c["targets"][kind].values())
        starts = [x["started"] for x in states if x.get("started")]
        elapsed = time.time() - min(starts) if starts else 0
        rate = accepted / elapsed if accepted >= 32 and elapsed > 0 else None
        result["datasets"][kind] = {
            "accepted": accepted,
            "target": target,
            "attempted": attempted,
            "ranks": states,
            "elapsed_seconds": elapsed,
            "auto_groups_per_hour": rate * 3600 if rate else None,
            "estimated_remaining_seconds": (target - accepted) / rate if rate else None,
        }
    return result


def monitor(root, stop=None):
    stop = stop or threading.Event()
    (root / "logs").mkdir(exist_ok=True)
    while not stop.is_set():
        s = summary(root)
        atomic_json(root / "progress.json", s)
        lines = []
        for kind, v in s["datasets"].items():
            n, t = v["accepted"], v["target"]
            filled = int(30 * n / max(1, t))
            eta = (
                f"{v['estimated_remaining_seconds'] / 3600:.1f}h"
                if v["estimated_remaining_seconds"] is not None
                else "warming-up"
            )
            phases = ",".join(
                f"r{x['rank']}:{x.get('active_stage', {}).get('stage', x['status'])}"
                for x in v["ranks"]
            )
            lines.append(
                f"{kind:5} [{'=' * filled}{'.' * (30 - filled)}] {n:,}/{t:,} {n / max(1, t):.1%} attempted={v['attempted']:,} ETA={eta} {phases}"
            )
        line = (
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            + " | "
            + " | ".join(lines)
        )
        with (root / "logs/progress_tqdm.log").open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)
        if stop.wait(300):
            break


def node(args):
    def interrupted(signum, frame):
        with CHILDREN_LOCK:
            for child in list(CHILDREN):
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise RuntimeError(
            f"worker interrupted by signal {signum}; persisted checkpoints remain resumable"
        )

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    c = config(args.root)
    if (
        int(os.environ.get("ARNOLD_WORKER_NUM", "0")) != 4
        or int(os.environ.get("ARNOLD_WORKER_GPU", "0")) != 8
    ):
        raise ValueError(
            "node command requires actual Arnold 4x8 topology; do not fake environment"
        )
    if str(args.rank) != os.environ.get("ARNOLD_ID") or args.rank not in range(4):
        raise ValueError("rank mismatch")
    ready = json.loads((args.root / "inputs/ready.json").read_text())
    manifest = args.root / "inputs" / f"train_rank{args.rank}.jsonl"
    if file_hash(manifest) != ready["files"][manifest.name]:
        raise ValueError("input shard changed")
    if file_hash(args.root / "inputs/eval_reservation.json") != c["reservation_sha256"]:
        raise ValueError("reservation changed")
    with owner_lock(args.root / "control" / f"node{args.rank}.lock.d"):
        stop = threading.Event()
        thread = None
        if args.rank == 0:
            thread = threading.Thread(
                target=monitor, args=(args.root, stop), daemon=True
            )
            thread.start()
        try:
            if not args.skip_model_cache:
                from sa1b_pipeline.model_cache import stage_model

                mapping = {}
                for key in (
                    "planner_model",
                    "blind_model",
                    "editor_model",
                    "site_model",
                ):
                    model = c[key]
                    mapping[model] = stage_model(
                        model,
                        args.cache_root,
                        progress=lambda x: print(json.dumps(x), flush=True),
                    )
                atomic_json(
                    args.root / "nodes" / f"node{args.rank}" / "model_map.json", mapping
                )
                os.environ["SA1B_LOCAL_MODEL_MAP"] = json.dumps(mapping)
            for kind in ("twins", "multi"):
                run_kind(args, c, kind)
            atomic_json(
                args.root / "nodes" / f"node{args.rank}" / "done.json",
                {"time": time.time(), "status": "run_finished_check_deficits"},
            )
            # Keep rank 0's monitor alive until every worker has a terminal
            # outcome, so early completion does not stop the 5-minute log.
            if args.rank == 0:
                while True:
                    terminal = []
                    for i in range(4):
                        node_root = args.root / "nodes" / f"node{i}"
                        terminal.append(
                            (node_root / "done.json").exists()
                            or (
                                (node_root / "failure.json").exists()
                                and not (
                                    args.root / "control" / f"node{i}.lock.d"
                                ).exists()
                            )
                        )
                    if all(terminal):
                        break
                    time.sleep(30)
        except BaseException as exc:
            atomic_json(
                args.root / "nodes" / f"node{args.rank}" / "failure.json",
                {"time": time.time(), "error": str(exc)},
            )
            raise
        finally:
            stop.set()
            if thread:
                thread.join(2)
    try:
        export(args.root, require_all=True)
    except LockBusyError:
        pass
    except FileNotFoundError:
        pass  # Other ranks are still running.


def export(root, require_all=False):
    from sa1b_pipeline.group_production import source
    from sa1b_pipeline.group_schema import validate_multi_record, validate_twin_record

    c = config(root)
    if require_all and not all(
        (root / "nodes" / f"node{i}" / "done.json").exists() for i in range(4)
    ):
        raise FileNotFoundError("ranks incomplete")
    with owner_lock(root / "control/export.lock.d"):
        reports = {}
        cache = {}
        for kind in ("twins", "multi"):
            values = []
            for i in range(4):
                p = root / "nodes" / f"node{i}" / kind / "accepted.jsonl"
                if p.exists():
                    values.extend(read_rows(p))
            validator = (
                validate_twin_record if kind == "twins" else validate_multi_record
            )
            for g in values:
                validator(g, expected_split="train")
            ids = [g.get("twin_id") or g.get("multi_id") for g in values]
            if (
                len(ids) != len(set(ids))
                or max(Counter(source(g) for g in values).values(), default=0) > 2
            ):
                raise ValueError("duplicate groups or source cap exceeded")
            report = describe(values, c["targets"][kind], kind)
            if len(accept(values, c["targets"][kind], kind)) != len(values):
                raise ValueError("invalid final acceptance/quota")
            filename = (
                f"{kind}_train.jsonl"
                if report["complete"]
                else f"{kind}_train.partial.jsonl"
            )
            write_rows(
                root / "final" / filename,
                [portable(g, root, base=root, cache=cache) for g in values],
            )
            reports[kind] = {
                **report,
                "manifest": str(root / "final" / filename),
                "automatic_only": True,
            }
        atomic_json(
            root / "final/summary.json",
            {"datasets": reports, "published": False, "media_base": str(root)},
        )
        print(json.dumps(reports), flush=True)


def bundle(root):
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip():
        raise ValueError("commit the tested code before freezing deployment")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    dest = root / "deployment"
    dest.mkdir(exist_ok=True)
    if (dest / "code.tar.gz").exists():
        raise ValueError("deployment already frozen; use a new root for code changes")
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(dest / "code.tar.gz"), commit],
        cwd=REPO,
        check=True,
    )
    (dest / "code.sha256").write_text(file_hash(dest / "code.tar.gz") + "\n")
    (dest / "config.sha256").write_text(file_hash(root / "config.json") + "\n")
    shutil.copyfile(
        REPO / "scripts/group_production/launch_4node.sh", dest / "launch_4node.sh"
    )
    (dest / "arnold_entry_full.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport GROUP_RUN_ROOT="
        + __import__("shlex").quote(str(root))
        + '\nexport GROUP_RUNTIME_ROOT=/opt/tiger/sa1b-group-production-runtime\nexport GROUP_SKIP_SETUP=0\nexec bash "$GROUP_RUN_ROOT/deployment/launch_4node.sh"\n'
    )
    c = config(root)
    models = {}
    for key in (
        "planner_model",
        "blind_model",
        "audit_model",
        "editor_model",
        "site_model",
    ):
        path = Path(c[key])
        models[c[key]] = {
            str(p.relative_to(path)): {
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in sorted(path.rglob("*"))
            if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(path).parts)
        }
    atomic_json(dest / "models.json", models)
    atomic_json(
        dest / "manifest.json",
        {
            "commit": commit,
            "code_sha256": file_hash(dest / "code.tar.gz"),
            "config_sha256": file_hash(root / "config.json"),
            "input_ready_sha256": file_hash(root / "inputs/ready.json"),
            "model_inventory_sha256": file_hash(dest / "models.json"),
        },
    )
    print(json.dumps({"root": str(root), "commit": commit, "ready": True}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--reservation", type=Path, required=True)
    prep.add_argument("--twins-target", type=int, default=15000)
    prep.add_argument("--multi-target", type=int, default=5000)
    n = sub.add_parser("node")
    n.add_argument("--rank", type=int, required=True)
    n.add_argument("--mllm-python", required=True)
    n.add_argument("--editor-python", required=True)
    n.add_argument("--editor-devices", default="0,1,2,3,4,5,6,7")
    n.add_argument("--selection-workers", type=int, default=8)
    n.add_argument(
        "--cache-root", type=Path, default=Path("/opt/tiger/sa1b-group-model-cache")
    )
    n.add_argument("--skip-model-cache", action="store_true")
    for name in ("status", "monitor", "export", "bundle"):
        sub.add_parser(name)
    rec = sub.add_parser("recover-lock")
    rec.add_argument("--rank", type=int, required=True)
    rec.add_argument("--previous-workers-stopped", action="store_true", required=True)
    args = p.parse_args()
    args.root = args.root.resolve()
    if (
        args.root == PROTECTED
        or PROTECTED in args.root.parents
        or args.root in PROTECTED.parents
    ):
        raise ValueError("protected production root")
    if args.command == "prepare":
        with owner_lock(args.root / "control/prepare.lock.d"):
            prepare(args)
    elif args.command == "node":
        node(args)
    elif args.command == "status":
        print(json.dumps(summary(args.root), ensure_ascii=False, indent=2))
    elif args.command == "monitor":
        monitor(args.root)
    elif args.command == "export":
        export(args.root)
    elif args.command == "bundle":
        bundle(args.root)
    else:
        if args.rank not in range(4):
            raise ValueError("invalid rank")
        lock = args.root / "control" / f"node{args.rank}.lock.d"
        if lock.exists():
            dest = lock.with_name(lock.name + ".recovered." + str(time.time_ns()))
            lock.rename(dest)
            print("Retained previous lock at " + str(dest))


if __name__ == "__main__":
    main()
