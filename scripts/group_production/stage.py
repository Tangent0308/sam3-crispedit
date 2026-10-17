#!/usr/bin/env python3
"""Separate blind/audit runtime; atomic per-group audit checkpoints on resume."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sa1b_pipeline.production_state import (
    atomic_json,
    digest,
    require_json,
    read_rows,
    write_rows,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["blind", "audit"], required=True)
    p.add_argument("--wave", type=Path, required=True)
    p.add_argument("--kind", choices=["twins", "multi"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    root = args.wave
    if args.stage == "blind":
        from sa1b_pipeline.blind_identifiability import run_blind_probes
        from sa1b_pipeline.mllm import VLLMSelector

        groups = [
            json.loads(p.read_text()) for p in sorted((root / "planned").glob("*.json"))
        ]
        if groups:
            model = VLLMSelector(
                args.model,
                tensor_parallel_size=1,
                batch_size=args.batch_size,
                max_new_tokens=1000,
            )
            try:
                run_blind_probes(groups, model, root / "blind_probes")
            finally:
                model.close()
            for g in groups:
                atomic_json(root / "planned" / f"{g['twin_id']}.json", g)
        write_rows(
            root / "planned.jsonl",
            [
                g
                for g in groups
                if g.get("group_checks", {}).get("blind_disambiguation", {}).get("pass")
                is True
            ],
        )
        return
    from sa1b_pipeline.audit_backend import VLLMAuditor
    from sa1b_pipeline.group_audit import audit_multi_groups, audit_twin_groups_joint

    groups = read_rows(root / "generated.jsonl")[args.worker :: args.workers]
    cache = root / "audit_checkpoints"
    cache.mkdir(exist_ok=True)
    from sa1b_pipeline.group_audit import group_audit_contract
    # Only the coordinator publishes this shared contract; workers never write it.
    require_json(cache / "contract.json", group_audit_contract(args.kind, args.model))
    pending = []
    for g in groups:
        gid = g.get("twin_id") or g["multi_id"]
        path = cache / (gid + ".json")
        fingerprint = digest(g)
        if path.exists():
            if json.loads(path.read_text())["input_sha256"] != fingerprint:
                raise ValueError("audit input changed: " + gid)
        else:
            pending.append((g, fingerprint, path))
    if pending:
        model = VLLMAuditor(
            args.model,
            batch_size=args.batch_size,
            max_tokens=1000,
            check_operation=False,
            tensor_parallel_size=1,
        )
        try:
            fn = audit_twin_groups_joint if args.kind == "twins" else audit_multi_groups
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                output = fn([x[0] for x in batch], model, root / "audit_panels")
                if len(output) != len(batch):
                    raise RuntimeError("incomplete audit batch")
                for g, (_, fingerprint, path) in zip(output, batch):
                    atomic_json(path, {"input_sha256": fingerprint, "group": g})
                print(
                    json.dumps(
                        {
                            "worker": args.worker,
                            "audited": min(start + len(batch), len(pending)),
                            "total": len(pending),
                        }
                    ),
                    flush=True,
                )
        finally:
            model.close()
    write_rows(
        root / f"audited_worker{args.worker}.jsonl",
        [
            json.loads(
                (cache / ((g.get("twin_id") or g["multi_id"]) + ".json")).read_text()
            )["group"]
            for g in groups
        ],
    )


if __name__ == "__main__":
    main()
