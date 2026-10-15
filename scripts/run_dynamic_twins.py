#!/usr/bin/env python3
"""Resume bounded twin generation waves, replenishing final accepted quota cells.

A single coordinator owns one isolated root (flock). Existing stage checkpoints
are reused on restart; per-stage runtimes are explicit because editor and vLLM
have separate environments. This driver never publishes or writes the 50k root.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.dynamic_twin_balance import (
    accepted_groups,
    choose_wave,
    progress,
    quotas,
)
from sa1b_pipeline.group_audit import refresh_twin_add_checks
from sa1b_pipeline.group_selection import (
    select_twin_candidates,
    twin_operation_priority,
)
from sa1b_pipeline.io import load_candidates
from sa1b_pipeline.production_state import (
    atomic_json,
    immutable_json,
    read_rows,
    write_rows,
)
from sa1b_pipeline.source_isolation import assert_no_eval_leakage, load_reservation


def run(args):
    root = args.output_root.resolve()
    frozen = Path(
        "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014"
    )
    if root == frozen or frozen in root.parents or root in frozen.parents:
        raise ValueError("protected production root")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    immutable_json(
        root / "dynamic_config.json",
        {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k
            not in {
                "selection_only",
                "stop_after",
                "audit_model",
                "audit_tp",
                "audit_devices",
                "candidate_extension",
            }
        },
    )
    audit_config = {
        "model": args.audit_model or args.mllm_model,
        "tp": args.audit_tp,
        "devices": args.audit_devices or args.mllm_devices,
    }
    if (root / "audit_config.json").exists():
        immutable_json(root / "audit_config.json", audit_config)
    targets = quotas(args.target, args.same_fraction)
    poolpath = root / "inputs/candidate_pool.jsonl"
    if not poolpath.exists():
        reserved, _ = load_reservation(args.reservation)
        candidates = [
            {**c.to_dict(), "objects": [o.__dict__ for o in c.objects]}
            for c in load_candidates(args.manifest)
        ]
        candidates = [
            c
            for c in candidates
            if str(c["source_id"]).removeprefix("sa1b-") not in reserved
        ]
        assert_no_eval_leakage(candidates, args.reservation)
        random.Random(args.seed).shuffle(candidates)
        pool = []
        for i, key in enumerate(targets):
            if not targets[key]:
                continue
            op, rel = key.split(":")
            rows = select_twin_candidates(
                candidates,
                target=args.pool_per_cell,
                operation=op,
                category_relation=rel,
                seed=args.seed + i,
                member_count=2,
                max_groups_per_source=2,
                reserved_source_ids=reserved,
            )
            rows.sort(
                key=lambda r: twin_operation_priority(r["objects"], op), reverse=True
            )
            pool.extend(rows)
            print(json.dumps({"pool_cell": key, "candidates": len(rows)}), flush=True)
        write_rows(poolpath, pool)
    pool = read_rows(poolpath)
    if args.candidate_extension:
        extension = read_rows(args.candidate_extension)
        assert_no_eval_leakage((r["candidate"] for r in extension), args.reservation)
        digest = hashlib.sha256(args.candidate_extension.read_bytes()).hexdigest()
        stored = root / "inputs/pool_extensions" / (digest + ".jsonl")
        if not stored.exists():
            write_rows(stored, extension)
    for extension in sorted((root / "inputs/pool_extensions").glob("*.jsonl")):
        pool.extend(read_rows(extension))
    pool.sort(
        key=lambda r: twin_operation_priority(r["objects"], r["operation"]),
        reverse=True,
    )
    if args.selection_only:
        return
    attempted = []
    groups = []
    stage_script = Path(__file__).with_name("run_group_pipeline.py")
    for wave_no in range(1, args.max_attempts + 1):
        status = progress(groups, attempted, targets)
        atomic_json(root / "progress.json", status)
        if status["complete"]:
            break
        wave = root / "waves" / f"{wave_no:03d}"
        selection = wave / "inputs/selected.jsonl"
        if selection.exists():
            chosen = read_rows(selection)
        else:
            chosen = choose_wave(
                pool,
                attempted,
                accepted_groups(groups, targets),
                targets,
                wave_size=args.wave_size,
                max_attempts=args.max_attempts,
            )
            if not chosen:
                break
            write_rows(selection, chosen)
        common = [
            str(stage_script),
            "--kind",
            "twins",
            "--manifest",
            str(args.manifest.resolve()),
            "--output-root",
            str(wave),
            "--target",
            str(len(chosen)),
            "--split",
            "train",
            "--reservation",
            str(args.reservation.resolve()),
            "--outline-planning-crops",
            "--locate-planning-scene",
            "--structured-planning-json",
            "--quota-from-selection",
        ]
        if args.site_model:
            common += ["--site-model", args.site_model]
        commands = {
            "planning": [
                args.planner_python,
                *common,
                "--plan-only",
                "--planner-model",
                args.mllm_model,
                "--planner-batch-size",
                "4",
                "--max-new-tokens",
                "1600",
            ],
            "generation": [
                args.editor_python,
                *common,
                "--generate-only",
                "--editor-model",
                args.editor_model,
                "--editor-devices",
                args.editor_devices,
            ],
            "audit": [
                args.planner_python,
                *common,
                "--audit-only",
                "--audit-model",
                args.audit_model or args.mllm_model,
                "--audit-tp",
                str(args.audit_tp),
                "--twin-audit-mode",
                "joint",
            ],
        }
        atomic_json(wave / "commands.json", commands)
        for stage, cmd in commands.items():
            if stage == "audit":
                immutable_json(root / "audit_config.json", audit_config)
            exitpath = wave / (stage + "_exit.json")
            if (
                exitpath.exists()
                and json.loads(exitpath.read_text()).get("exit_code") == 0
            ):
                if args.stop_after == stage:
                    return
                continue
            if stage != "planning" and not read_rows(wave / "planned.jsonl"):
                write_rows(wave / "audited.jsonl", [])
                break
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=(args.audit_devices or args.mllm_devices)
                if stage == "audit"
                else args.mllm_devices,
                SA1B_FORCE_DIRECT_VLLM="1",
                SA1B_MM_ENCODER_ATTN_BACKEND="TORCH_SDPA",
                DIFFUSION_ATTENTION_BACKEND="TORCH_SDPA",
                PYTHONUNBUFFERED="1",
                WANDB_MODE="disabled",
                PYTHONPATH=str(stage_script.parents[1]),
            )
            print(
                json.dumps(
                    {
                        "wave": wave_no,
                        "stage": stage,
                        "attempted_before": len(attempted),
                        "accepted": status["accepted"],
                        "target": args.target,
                    }
                ),
                flush=True,
            )
            start = time.monotonic()
            with (wave / (stage + ".log")).open("a") as f:
                result = subprocess.run(
                    cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False
                )
            outcome = {
                "exit_code": result.returncode,
                "seconds": time.monotonic() - start,
            }
            # Keep failed startup/recovery costs when a later retry succeeds.
            with (wave / "stage_attempts.jsonl").open("a") as history:
                history.write(json.dumps({"stage": stage, **outcome}) + "\n")
            atomic_json(exitpath, outcome)
            if result.returncode:
                raise RuntimeError(
                    f"{stage} failed in {wave}; restart same command after fixing the cause"
                )
            if args.stop_after == stage:
                atomic_json(
                    root / "progress.json",
                    {
                        **status,
                        "stage_complete": stage,
                        "wave": wave_no,
                        "stop_reason": "requested_stage_boundary",
                    },
                )
                return
        attempted.extend(chosen)
        if (wave / "audited.jsonl").exists():
            cached_groups = read_rows(wave / "audited.jsonl")
            for group in cached_groups:
                refresh_twin_add_checks(group)
            groups.extend(cached_groups)
        status = progress(groups, attempted, targets)
        atomic_json(root / "progress.json", status)
        write_rows(root / "accepted.jsonl", accepted_groups(groups, targets))
        print(json.dumps(status), flush=True)
    status = progress(groups, attempted, targets)
    status["stop_reason"] = (
        "target_reached"
        if status["complete"]
        else (
            "attempt_budget_exhausted"
            if len(attempted) >= args.max_attempts
            else "candidate_pool_exhausted"
        )
    )
    atomic_json(root / "progress.json", status)
    write_rows(root / "accepted.jsonl", accepted_groups(groups, targets))
    print(json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reservation", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--target", type=int, default=12)
    p.add_argument("--wave-size", type=int, default=12)
    p.add_argument("--max-attempts", type=int, default=24)
    p.add_argument("--same-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=202610152)
    p.add_argument("--pool-per-cell", type=int, default=40)
    p.add_argument("--planner-python", required=True)
    p.add_argument("--editor-python", required=True)
    p.add_argument("--mllm-model", required=True)
    p.add_argument("--editor-model", required=True)
    p.add_argument("--site-model")
    p.add_argument("--audit-model")
    p.add_argument("--audit-tp", type=int, default=1)
    p.add_argument("--audit-devices")
    p.add_argument("--mllm-devices", default="2")
    p.add_argument("--editor-devices", default="2,3")
    p.add_argument(
        "--candidate-extension",
        type=Path,
        help="register an immutable additional candidate pool; completed waves remain unchanged",
    )
    p.add_argument("--selection-only", action="store_true")
    p.add_argument("--stop-after", choices=["planning", "generation"])
    args = p.parse_args()
    if min(args.target, args.wave_size, args.max_attempts, args.pool_per_cell) < 1:
        p.error("counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
