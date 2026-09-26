"""Four-node shared-filesystem runner for add/replace/attribute derivations.

This is intentionally separate from ``run_multinode_labeling``: the existing
remove run is frozen and its manifest/output are never reused or overwritten.
Each source is assigned to one node, while all three non-remove cases for every
dataset mask stay on that node.  Planning uses the generic dataset-mask MLLM
planner; editing uses the existing Qwen-Image-2.1 regional editor on eight local
GPUs.  Audit is optional so a smoke run can validate generation first.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

from synthesis_pipeline.labeling_checkpoint import ensure_sources

REPO = Path(__file__).resolve().parents[1]
TASK_TYPES = ("add", "replace", "attribute")
PROFILE = {
    "planner": "dataset_mask_v6_28_semantic_surface_and_assembly_guard",
    "editor": "context_grounded_v4_qwen21",
    "editor_backend": "vllm-omni",
    "editor_prompt_policy": "typed-v1",
    "steps": 40,
    "seed": 0,
    "audit": "optional_audit_edit_pairs_v15",
}


def read_rows(path: Path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def failed(root: Path):
    files = sorted(Path(root).glob("control/*.failed.json"))
    return files[0] if files else None


def wait_for(paths, root: Path, timeout: int):
    start = time.monotonic()
    while True:
        problem = failed(root)
        if problem:
            raise RuntimeError(f"peer failed: {problem}: {problem.read_text()[:2000]}")
        if all(Path(path).is_file() for path in paths):
            return
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"timeout waiting for {[str(p) for p in paths if not Path(p).exists()]}")
        time.sleep(2)


def split_sources(rows, nodes):
    groups = {}
    names = set()
    for row in rows:
        image = row["image"]
        if image in names or not re.fullmatch(r"\d+_[\w.-]+\.png", image):
            raise ValueError(f"invalid or duplicate image id: {image}")
        if row.get("task_type") not in TASK_TYPES:
            raise ValueError(f"unexpected task type: {row.get('task_type')}")
        if not row.get("mask") or str(row.get("answer", "")).strip().lower().rstrip(".") == "no target":
            raise ValueError(f"invalid/No target input: {image}")
        names.add(image)
        groups.setdefault(row["source_image"], []).append(row)
    shards = [[] for _ in range(nodes)]
    for group in groups.values():
        owner = min(range(nodes), key=lambda index: (len(shards[index]), index))
        shards[owner].extend(group)
    order = {row["image"]: index for index, row in enumerate(rows)}
    for shard in shards:
        shard.sort(key=lambda row: order[row["image"]])
    return shards


def run_stage(command, log_path: Path, root: Path, timeout: int, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", json.dumps(command), "LOG", log_path, flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                problem = failed(root)
                if problem:
                    raise RuntimeError(f"peer failed: {problem}")
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"stage timeout: {log_path}")
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"stage exited {process.returncode}: {log_path}")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    return time.monotonic() - started


def run_planning_pool(data_root: Path, planning_root: Path, rows, gpu_ids, coordination: Path, timeout: int):
    """Run the generic two-pass planner on all local GPUs, then merge shards."""
    started = time.monotonic()
    shard_root = planning_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    groups = {}
    for row in rows:
        groups.setdefault(row["source_image"], []).append(row)
    shards = [[] for _ in gpu_ids]
    for group in groups.values():
        owner = min(range(len(shards)), key=lambda index: (len(shards[index]), index))
        shards[owner].extend(group)
    mllm_python = os.environ.get("SAMTOK_MLLM_PYTHON", sys.executable)
    model_id = os.environ.get("SAMTOK_QWEN38_MODEL", "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B")
    jobs = []
    for index, (gpu, subset) in enumerate(zip(gpu_ids, shards)):
        if not subset:
            continue
        shard_data = shard_root / f"data{index}"
        shard_out = shard_root / f"out{index}"
        shard_data.mkdir(parents=True, exist_ok=True)
        (shard_data / "sources").symlink_to((data_root / "sources").resolve())
        write_rows(shard_data / "annotations.jsonl", subset)
        write_rows(shard_data / "input_annotations.jsonl", rows)
        log = (coordination / "logs" / f"planning.gpu{gpu}.log").open("w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8"}
        command = [mllm_python, "-m", "synthesis_pipeline.plan_dataset_regions",
                   "--data-root", str(shard_data), "--out-root", str(shard_out),
                   "--model-id", model_id]
        jobs.append((index, subset, subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    errors = []
    pending = list(jobs)
    while pending:
        if failed(coordination):
            raise RuntimeError(f"peer failed while planning: {failed(coordination)}")
        if time.monotonic() - started > timeout:
            for _, _, process, _ in pending:
                process.terminate()
            raise TimeoutError("multitype planner pool timed out")
        still_running = []
        for item in pending:
            index, subset, process, log = item
            code = process.poll()
            if code is None:
                still_running.append(item)
            else:
                log.close()
                if code:
                    errors.append({"shard": index, "exit_code": code})
        pending = still_running
        if pending:
            time.sleep(2)
    if errors:
        raise RuntimeError(f"planner failures: {errors}")

    order = {row["image"]: i for i, row in enumerate(rows)}
    merged = {}
    for index, subset, _, _ in jobs:
        path = shard_root / f"out{index}/regions/annotations.jsonl"
        if path.exists():
            for row in read_rows(path):
                merged[row["image"]] = row
    ordered = [merged[name] for name in sorted(merged, key=order.get)]
    for stage in ("plan", "scope", "regions"):
        (planning_root / stage).mkdir(parents=True, exist_ok=True)
    write_rows(planning_root / "regions/annotations.jsonl", ordered)
    write_rows(planning_root / "regions/input_annotations.jsonl", rows)
    (planning_root / "regions/sources").symlink_to((data_root / "sources").resolve())
    for stage in ("plan", "scope"):
        stage_rows = {}
        for index, subset, _, _ in jobs:
            path = shard_root / f"out{index}/{stage}/annotations.jsonl"
            if path.exists():
                for row in read_rows(path):
                    stage_rows[row["image"]] = row
        write_rows(planning_root / stage / "annotations.jsonl", [stage_rows[name] for name in sorted(stage_rows, key=order.get)])
    atomic(planning_root / "summary.json", {
        "input_cases": len(rows), "accepted": len(ordered), "workers": len(jobs),
        "wall_seconds": round(time.monotonic() - started, 3), "planner": PROFILE["planner"]
    })
    return time.monotonic() - started


def prepare_shards(data_root: Path, run_root: Path, nodes: int, resume: bool):
    rows = read_rows(data_root / "annotations.jsonl")
    if not rows:
        raise ValueError("empty multitype input manifest")
    shards = split_sources(rows, nodes)
    partition = run_root / "reports/partition.json"
    if resume and partition.exists():
        old = json.loads(partition.read_text())
        if old["input_sha256"] != digest(data_root / "annotations.jsonl"):
            raise ValueError("resume input manifest differs from original run")
    for rank, shard in enumerate(shards):
        node_data = run_root / "inputs" / f"node{rank}"
        node_data.mkdir(parents=True, exist_ok=resume)
        ensure_sources(node_data, data_root / "sources")
        write_rows(node_data / "annotations.jsonl", shard)
        write_rows(node_data / "input_annotations.jsonl", shard)
    atomic(
        partition,
        {
            "cases": len(rows),
            "source_images": len({row["source_image"] for row in rows}),
            "counts": [len(shard) for shard in shards],
            "task_type_counts": dict(Counter(row["task_type"] for row in rows)),
            "input_sha256": digest(data_root / "annotations.jsonl"),
        },
    )


def merge_results(run_root: Path, nodes: int):
    inventory, planned, generated, audited, passed = [], [], [], [], []
    for rank in range(nodes):
        node_root = run_root / "nodes" / f"node{rank}"
        input_rows = read_rows(run_root / "inputs" / f"node{rank}" / "annotations.jsonl")
        plan_path = node_root / "planning/regions/annotations.jsonl"
        plan_rows = read_rows(plan_path) if plan_path.exists() else []
        generated_path = node_root / "generation/context_grounded_v4_qwen21/annotations.jsonl"
        generated_rows = read_rows(generated_path) if generated_path.exists() else []
        audit_path = node_root / "audit/edit_audit.jsonl"
        audit_rows = read_rows(audit_path) if audit_path.exists() else []
        plan_by = {row["image"]: row for row in plan_rows}
        gen_by = {row["image"]: row for row in generated_rows}
        audit_by = {row["image"]: row for row in audit_rows}
        for row in input_rows:
            name = row["image"]
            plan = plan_by.get(name)
            audit = audit_by.get(name)
            status = str(audit.get("quality", audit.get("decision", "fail"))) if audit else (
                "generated" if name in gen_by else "no_plan"
            )
            item = {
                "image": name,
                "task_type": row["task_type"],
                "source_image": row["source_image"],
                "node": rank,
                "plan_status": "accepted" if plan else "rejected_or_missing",
                "generation_status": "generated" if name in gen_by else "not_generated",
                "audit_status": status,
                "edited_path": str(node_root / "generation/context_grounded_v4_qwen21/edited" / name)
                if name in gen_by
                else None,
                "original_mask": row["mask"],
            }
            inventory.append(item)
            if plan:
                planned.append(plan)
            if name in gen_by:
                generated.append({**gen_by[name], **item})
            if audit:
                audited.append({**audit, "task_type": row["task_type"], "node": rank})
                if str(audit.get("quality", audit.get("decision", ""))).lower() == "pass":
                    passed.append({**gen_by.get(name, {}), **item, "quality_label": "model_pass_not_human_verified"})
    order = {row["image"]: index for index, row in enumerate(inventory)}
    for values in (inventory, planned, generated, audited, passed):
        values.sort(key=lambda row: order.get(row["image"], 10**12))
    write_rows(run_root / "results/all_cases.jsonl", inventory)
    write_rows(run_root / "results/planned.jsonl", planned)
    write_rows(run_root / "results/generated.jsonl", generated)
    write_rows(run_root / "results/audit.jsonl", audited)
    write_rows(run_root / "results/model_pass.jsonl", passed)
    # Keep type-specific manifests ready for the later four-way merge without
    # duplicating large PNGs.  The image paths remain node-local and immutable.
    for task_type in TASK_TYPES:
        type_root = run_root / "results" / "by_type" / task_type
        type_root.mkdir(parents=True, exist_ok=True)
        write_rows(type_root / "all_cases.jsonl", [r for r in inventory if r["task_type"] == task_type])
        write_rows(type_root / "generated.jsonl", [r for r in generated if r["task_type"] == task_type])
        write_rows(type_root / "audit.jsonl", [r for r in audited if r.get("task_type") == task_type])
        write_rows(type_root / "model_pass.jsonl", [r for r in passed if r["task_type"] == task_type])
    result = {
        "input_cases": len(inventory),
        "input_task_type_counts": dict(Counter(row["task_type"] for row in inventory)),
        "planned_cases": len(planned),
        "generated_cases": len(generated),
        "audited_cases": len(audited),
        "decisions": dict(Counter(row["audit_status"] for row in inventory)),
        "profile": PROFILE,
        "quality_note": "Model pass is not independently reviewed ground truth",
    }
    atomic(run_root / "reports/final.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("ARNOLD_ID", "0")))
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--timeout", type=int, default=604800)
    parser.add_argument("--join-timeout", type=int, default=10800)
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--skip-audit", action="store_true", help="Stop after generation; useful for smoke tests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt-id", default="")
    args = parser.parse_args()
    if args.nodes != 4 or not 0 <= args.rank < args.nodes:
        parser.error("this entry requires four ranks (0..3)")
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not args.local_test and len(gpu_ids) != 8:
        parser.error("production requires eight GPUs per node")
    if args.resume != bool(args.attempt_id):
        parser.error("--resume requires a new --attempt-id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("invalid run id")
    args.run_root = args.run_root.resolve()
    args.data_root = args.data_root.resolve()
    coordination = args.run_root / "attempts" / args.attempt_id if args.resume else args.run_root
    control = coordination / "control"
    control.mkdir(parents=True, exist_ok=True)
    (args.run_root / "control").mkdir(parents=True, exist_ok=True)
    lock = (args.run_root / "control" / f"execution.multitype.node{args.rank}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    claim = control / f"node{args.rank}.claim"
    with claim.open("x") as handle:
        handle.write(f"{socket.gethostname()} {os.getpid()}\n")
    try:
        if args.rank == 0:
            prepare_shards(args.data_root, args.run_root, args.nodes, args.resume)
            atomic(control / "partition.ok.json", {"ready": True})
        wait_for([control / "partition.ok.json"], coordination, args.join_timeout)
        node_data = args.run_root / "inputs" / f"node{args.rank}"
        node_root = args.run_root / "nodes" / f"node{args.rank}"
        node_root.mkdir(parents=True, exist_ok=args.resume)
        rows = read_rows(node_data / "annotations.jsonl")
        stage_times = {}
        progress = lambda stage: print(f"node{args.rank}: {stage}, input={len(rows)}", flush=True)
        planning = node_root / "planning"
        regions = planning / "regions/annotations.jsonl"
        if not (args.resume and regions.exists()):
            progress("planning_grounding_scope")
            stage_times["planning"] = run_planning_pool(
                node_data, planning, rows, gpu_ids,
                coordination, args.timeout,
            )
        else:
            progress("planning_reused")
        planned_rows = read_rows(regions) if regions.exists() else []
        generation = node_root / "generation"
        generated_manifest = generation / "context_grounded_v4_qwen21/annotations.jsonl"
        if planned_rows and not (args.resume and generated_manifest.exists()):
            progress("editing_qwen21_8gpu")
            editor_python = os.environ.get("SAMTOK_EDITOR_PYTHON", sys.executable)
            manifest_cmd = [editor_python, "-m", "synthesis_pipeline.experiment_edit_quality",
                            "--data-root", str(planning / "regions"), "--out-root", str(generation),
                            "--variant", PROFILE["editor"], "--all-cases", "--manifest-only"]
            stage_times["manifest"] = run_stage(manifest_cmd, coordination / "logs" / f"manifest.node{args.rank}.log", coordination, args.timeout)
            jobs = []
            env = {"DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA"}
            site = Path(editor_python).parent.parent / "lib/python3.12/site-packages/nvidia"
            env["LD_LIBRARY_PATH"] = ":".join([str(site / "cu13/lib"), str(site / "cuda_runtime/lib"), os.environ.get("LD_LIBRARY_PATH", "")])
            for shard, gpu in enumerate(gpu_ids):
                log = (coordination / "logs" / f"editing.node{args.rank}.gpu{gpu}.log").open("w")
                command = [editor_python, "-m", "synthesis_pipeline.experiment_edit_quality",
                           "--data-root", str(planning / "regions"), "--out-root", str(generation),
                           "--variant", PROFILE["editor"], "--all-cases", "--model-id",
                           os.environ.get("SAMTOK_QWEN21_MODEL", "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1"),
                           "--steps", str(PROFILE["steps"]), "--qwen21-prompt-policy", PROFILE["editor_prompt_policy"],
                           "--qwen21-backend", PROFILE["editor_backend"], "--shard", str(shard), "--shards", str(len(gpu_ids))]
                process_env = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
                jobs.append((subprocess.Popen(command, cwd=REPO, env=process_env, stdout=log, stderr=subprocess.STDOUT), log))
            errors = []
            for process, log in jobs:
                code = process.wait(); log.close()
                if code: errors.append(code)
            if errors: raise RuntimeError(f"editor failures: {errors}")
        else:
            progress("generation_reused_or_no_plan")
        if not args.skip_audit:
            progress("audit")
            # Audit is deliberately a separate generic binary stage.  It can be
            # disabled for a generation smoke run without changing the editor.
            if generated_manifest.exists():
                audit_root = node_root / "audit"
                if not (args.resume and (audit_root / "edit_audit.jsonl").exists()):
                    audit_python = os.environ.get("SAMTOK_MLLM_PYTHON", sys.executable)
                    stage_times["audit"] = run_stage(
                        [audit_python, "-m", "synthesis_pipeline.audit_edit_pairs",
                         "--annotations-jsonl", str(generated_manifest),
                         "--source-dir", str(planning / "regions/sources"),
                         "--edited-dir", str(generation / "context_grounded_v4_qwen21/edited"),
                         "--out-dir", str(audit_root), "--vlm", "qwen8b-vllm",
                         "--vlm-model-id", os.environ.get("SAMTOK_QWEN38_MODEL", "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"),
                         "--batch-size", "8"],
                        coordination / "logs" / f"audit.node{args.rank}.log", coordination, args.timeout,
                    )
        progress("node_done")
        atomic(control / f"node{args.rank}.done.json", {"timings": stage_times})
        wait_for([control / f"node{i}.done.json" for i in range(args.nodes)], coordination, args.timeout)
        if args.rank == 0:
            result = merge_results(args.run_root, args.nodes)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            atomic(control / "finalize.ok.json", result)
        wait_for([control / "finalize.ok.json"], coordination, args.timeout)
    except BaseException:
        atomic(control / f"node{args.rank}.failed.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
