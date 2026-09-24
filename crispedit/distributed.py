#!/usr/bin/env python3
"""Coordinate the complete CrispEdit quality, scene, grounding and SAM3 pipeline."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


BASE = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets")
MODEL = Path("/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B")
REPO = Path(__file__).resolve().parent.parent


def code_digest() -> str:
    digest = hashlib.sha256()
    paths = sorted((REPO / 'crispedit').rglob('*.py')) + [
        REPO / 'scripts/validate_crispedit_mask_pipeline.py', REPO / 'pyproject.toml']
    for path in paths:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def marker(path: Path, value: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value + "\n")
    os.replace(temporary, path)


def wait_for(path: Path, control: Path, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while not path.is_file():
        failed = list(control.glob("*.failed"))
        if failed:
            raise RuntimeError(f"Peer failed: {failed[0]}: {failed[0].read_text()[:500]}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(2)


def await_nodes(stage: str, control: Path, count: int, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while True:
        failed = list(control.glob("*.failed"))
        if failed:
            raise RuntimeError(f"{stage} peer failed: {failed[0]}: {failed[0].read_text()[:500]}")
        if all((control / f"{stage}.node{rank}.done").exists() for rank in range(count)):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {stage} on {count} nodes")
        time.sleep(2)


def stage_paths(root: Path, name: str) -> tuple[Path, Path]:
    return root / "audit" / name, root / "manifest" / name


def output_complete(root: Path, name: str, expected_rows: int | None = None) -> bool:
    audit, manifest = stage_paths(root, name)
    if not audit.exists() and not manifest.exists():
        return False
    if not audit.is_file() or not manifest.is_file():
        raise ValueError(f"Partial audit/manifest output for {name} at {root}")
    first = pq.read_table(audit, columns=["row_idx"]).column("row_idx").to_pylist()
    second = pq.read_table(manifest, columns=["row_idx"]).column("row_idx").to_pylist()
    if first != second or len(first) != len(set(first)):
        raise ValueError(f"Unaligned audit/manifest for {name} at {root}")
    if expected_rows is not None and first != list(range(expected_rows)):
        raise ValueError(f"Unexpected dense quality rows for {name} at {root}")
    return True


def assignments(names: list[str], source: Path, nodes: int) -> list[list[str]]:
    weights = {name: pq.ParquetFile(source / name).metadata.num_rows for name in names}
    result = [[] for _ in range(nodes)]
    loads = [0] * nodes
    for name in sorted(names, key=lambda item: (-weights[item], item)):
        rank = min(range(nodes), key=lambda item: (loads[item], len(result[item]), item))
        result[rank].append(name)
        loads[rank] += weights[name]
    return [sorted(group) for group in result]


def build_plan(args) -> dict:
    if args.download_run_dir:
        summary_path = args.download_run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Download has not completed: {summary_path}")
        download = json.loads(summary_path.read_text())
        if download["errors"] or download["verified_shards"] != download["selected_shards"]:
            raise ValueError("Download summary reports incomplete files")
        if (args.download_run_dir / "staging").exists():
            raise ValueError("Download staging still exists; wait for cleanup")
        selected = json.loads((args.download_run_dir / "plan.json").read_text())["files"]
        if not all((args.source_dir / Path(item).name).is_file() for item in selected):
            raise FileNotFoundError("A planned download shard is missing from source directory")
    from crispedit.common import supported_shard
    paths = [p for p in sorted(args.source_dir.glob("*.parquet"))
             if supported_shard(p)]
    if not paths:
        raise ValueError("No source parquet files found")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate source basenames")
    q_pending, s_pending = [], []
    for path in tqdm(paths, desc="CrispEdit four-node plan", unit="shard", mininterval=5):
        rows = pq.ParquetFile(path).metadata.num_rows
        quality_ready = output_complete(args.quality_dir, path.name, rows)
        scene_ready = output_complete(args.scene_dir, path.name)
        if not quality_ready:
            if scene_ready:
                raise ValueError(f"Scene exists without quality: {path.name}")
            q_pending.append(path.name)
        if not scene_ready:
            s_pending.append(path.name)
    return {
        "protocol": "crispedit_complete_pipeline",
        "nodes": args.nodes,
        "source_dir": str(args.source_dir),
        "quality_dir": str(args.quality_dir),
        "scene_dir": str(args.scene_dir),
        "model_path": str(args.model_path),
        "label_dir": str(args.label_dir),
        "checkpoint_path": str(args.checkpoint_path),
        "settings": run_settings(args),
        "source_snapshot": source_snapshot(args.source_dir, names),
        "code_sha256": code_digest(),
        "source_shards": names,
        "quality_assignments": assignments(q_pending, args.source_dir, args.nodes),
        "scene_assignments": assignments(s_pending, args.source_dir, args.nodes),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def execute(args, command: list[str], stage: str) -> None:
    """Stop this node's entire worker process group if any peer fails."""
    log = args.run_dir / 'logs' / f'{stage}.node{args.rank}.log'
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS=environment.get('OMP_NUM_THREADS', '4'))
    with log.open('a') as handle:
        handle.write('COMMAND ' + json.dumps(command) + '\n')
        handle.flush()
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                failed = list(args.control.glob('*.failed'))
                if failed:
                    raise RuntimeError(f'Peer failed: {failed[0]}')
                time.sleep(2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def stage_runner(args, plan: dict, stage: str, names: list[str]) -> None:
    local = args.run_dir / "work" / stage / f"node{args.rank}"
    local.mkdir(parents=True, exist_ok=True)
    shards = local / "shards.txt"
    shards.write_text("".join(name + "\n" for name in names))
    stage_log = args.run_dir / "logs" / f"{stage}.node{args.rank}.log"
    if not names:
        stage_log.write_text("No pending shards on this node.\n")
        return
    stage_script = "crispedit_pair_prefilter.py" if stage == "quality" else "crispedit_benchmark_scene_filter.py"
    command = [str(args.python), "-u", str(REPO / stage_script),
               "--input-dir", str(args.source_dir), "--output-dir", str(local),
               "--model-path", str(args.model_path), "--devices", args.devices,
               "--tensor-parallel-size", str(args.tensor_parallel_size),
               "--batch-size", str(args.batch_size), "--vllm-max-num-seqs", str(args.batch_size),
               "--vllm-max-model-len", "8192", "--vllm-gpu-memory-utilization", "0.85",
               "--progress-mininterval", "5", "--shard-list-file", str(shards)]
    if stage == "quality":
        command += ["--max-new-tokens", "1024"]
    else:
        command += ["--prefilter-manifest-dir", str(args.quality_dir / "manifest"),
                    "--max-new-tokens", "256"]
    print(f"node{args.rank} {stage}: {len(names)} shards; log={stage_log}", flush=True)
    execute(args, command, stage)
    summary = json.loads((local / "run_summary.json").read_text())
    if (summary["shards"] != len(names) or summary["worker_errors"] or
            any(code != 0 for code in summary["worker_exit_codes"])):
        raise RuntimeError(f"Invalid {stage} node{args.rank} summary: {summary}")
    expected = [name for name in names if stage == "quality" or quality_pass_rows(args.quality_dir, name)]
    if stage == "scene":
        # The scene runner intentionally omits shards whose quality PASS count is zero.
        if summary["shards"] != len(expected):
            raise RuntimeError(f"Missing scene outputs on node{args.rank}")
    for name in expected:
        if not output_complete(local, name):
            raise RuntimeError(f"Missing completed {stage} shard {name} on node{args.rank}")


def quality_pass_rows(root: Path, name: str) -> list[int]:
    table = pq.read_table(root / "manifest" / name, columns=["row_idx", "prefilter_verdict"])
    return [int(i) for i, verdict in zip(table["row_idx"].to_pylist(),
                                        table["prefilter_verdict"].to_pylist()) if verdict == "PASS"]


def merge_stage(args, plan: dict, stage: str) -> dict:
    destination = args.quality_dir if stage == "quality" else args.scene_dir
    assigned = plan[f"{stage}_assignments"]
    for rank, names in enumerate(assigned):
        local = args.run_dir / "work" / stage / f"node{rank}"
        for name in names:
            if stage == "scene" and not quality_pass_rows(args.quality_dir, name):
                # A shard with zero quality PASS still needs an empty scene manifest.
                if not output_complete(local, name):
                    from crispedit.difficulty.scene_runner import AUDIT_SCHEMA, MANIFEST_SCHEMA
                    (local / "audit").mkdir(parents=True, exist_ok=True)
                    (local / "manifest").mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.Table.from_batches([], schema=AUDIT_SCHEMA), local / "audit" / name)
                    pq.write_table(pa.Table.from_batches([], schema=MANIFEST_SCHEMA), local / "manifest" / name)
            if not output_complete(local, name):
                raise RuntimeError(f"Cannot merge missing {stage} output: node{rank}/{name}")
            for kind in ("audit", "manifest"):
                source = local / kind / name
                target = destination / kind / name
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except FileExistsError:
                    if not os.path.samefile(source, target):
                        raise FileExistsError(f"Refusing to overwrite existing {stage} result: {target}")
    return summarize(args, plan, stage)


def summarize(args, plan: dict, stage: str) -> dict:
    root = args.quality_dir if stage == "quality" else args.scene_dir
    # Historical background/style results may share this output directory.
    # Preserve their statistics without scheduling inference for those types.
    summary_names = sorted(set(plan['source_shards']) | {
        p.name for p in (root / 'manifest').glob('*.parquet')
        if (args.source_dir / p.name).is_file()})
    count = Counter()
    rows = 0
    calls = 0
    parse_errors = 0
    for name in summary_names:
        source_rows = pq.ParquetFile(args.source_dir / name).metadata.num_rows
        if stage == "quality":
            output_complete(root, name, source_rows)
            table = pq.read_table(root / "manifest" / name,
                                  columns=["row_idx", "prefilter_verdict", "prefilter_parse_ok"])
            values = table["prefilter_verdict"].to_pylist()
            count.update(values)
            parse_errors += sum(not value for value in table["prefilter_parse_ok"].to_pylist())
        else:
            output_complete(root, name)
            table = pq.read_table(root / "manifest" / name,
                                  columns=["row_idx", "scene_decision", "scene_model_called", "scene_parse_ok"])
            indices = table["row_idx"].to_pylist()
            if indices != quality_pass_rows(args.quality_dir, name):
                raise ValueError(f"Scene rows differ from quality PASS: {name}")
            values = table["scene_decision"].to_pylist()
            count.update(values)
            calls += sum(table["scene_model_called"].to_pylist())
            parse_errors += sum(not value for value in table["scene_parse_ok"].to_pylist())
        rows += len(values)
    summary_path = root / "run_summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    output = dict(previous)
    output.update({"shards": len(summary_names), "rows": rows,
                   "expected_shards": len(summary_names), "expected_rows": rows,
                   "active_source_shards": len(plan['source_shards']),
                   "preserved_out_of_scope_shards": len(set(summary_names) - set(plan['source_shards'])),
                   "parse_errors": parse_errors, "worker_errors": [],
                   "consolidated_utc": datetime.now(timezone.utc).isoformat(),
                   "four_node_run_dir": str(args.run_dir),
                   "incremental_shards": sum(map(len, plan[f"{stage}_assignments"]))})
    if stage == "quality":
        output["verdicts"] = dict(count)
    else:
        output["decisions"] = dict(count)
        output["active_model_called_rows"] = calls
        old_calls = int(previous.get("model_calls", 0))
        new_calls = sum(json.loads(path.read_text()).get("model_calls", 0)
                        for rank in range(args.nodes)
                        if (path := args.run_dir / "work" / stage / f"node{rank}" / "run_summary.json").is_file())
        output["model_calls"] = old_calls if previous.get("four_node_run_dir") == str(args.run_dir) else old_calls + new_calls
        output["model_calls_scope"] = "Historical executed requests; active_model_called_rows excludes manual removals."
    if previous.get("four_node_run_dir") != str(args.run_dir):
        output["four_node_incremental_runs"] = [
            {"node": rank, **json.loads(path.read_text())}
            for rank in range(args.nodes)
            if (path := args.run_dir / "work" / stage / f"node{rank}" / "run_summary.json").is_file()
        ]
    write_json(summary_path, output)
    write_json(args.run_dir / "reports" / f"{stage}_summary.json", output)
    return output


def source_snapshot(root, names):
    return {name: {'bytes': (root / name).stat().st_size,
                   'mtime_ns': (root / name).stat().st_mtime_ns,
                   'rows': pq.ParquetFile(root / name).metadata.num_rows} for name in names}


def run_settings(args):
    return {key: getattr(args, key) for key in (
        'tensor_parallel_size', 'batch_size', 'grounding_tp', 'grounding_batch_size', 'local_test')}


def build_label_plan(args, plan):
    from crispedit.mask.selection import load_filters, apply_scene
    from crispedit.mask.grounding_runner import prefilter_fields
    rows = {}
    for name in tqdm(plan['source_shards'], desc='select double PASS', mininterval=5):
        quality, scene = load_filters(args.quality_dir / 'manifest' / name,
                                      args.scene_dir / 'manifest' / name,
                                      plan['source_snapshot'][name]['rows'])
        indices = [i for i in sorted(quality)
                   if apply_scene(prefilter_fields(quality[i]), scene.get(i), True)['filter_decision'] == 'keep']
        if indices:
            rows[name] = indices
    groups, loads = [[] for _ in range(args.nodes)], [0] * args.nodes
    for name in sorted(rows, key=lambda n: (-len(rows[n]), n)):
        rank = min(range(args.nodes), key=lambda r: (loads[r], r))
        groups[rank].append(name)
        loads[rank] += len(rows[name])
    manifest_digest = hashlib.sha256()
    for name in plan['source_shards']:
        for root in (args.quality_dir, args.scene_dir):
            manifest_digest.update((root / 'manifest' / name).read_bytes())
    return {'rows': rows, 'assignments': groups, 'node_rows': loads,
            'manifest_sha256': manifest_digest.hexdigest()}


def selection_file(args, labels, rank=None):
    names = sorted(labels['rows']) if rank is None else sorted(labels['assignments'][rank])
    path = args.run_dir / ('selection.json' if rank is None else f'selection.node{rank}.json')
    write_json(path, {'cases': [{'shard': name, 'row_idx': i} for name in names for i in labels['rows'][name]]})
    return path


def label_runner(args, labels, stage):
    names = labels['assignments'][args.rank]
    if not names:
        (args.run_dir / 'logs' / f'{stage}.node{args.rank}.log').write_text('No double-PASS rows on this node.\n')
        return
    selection = selection_file(args, labels, args.rank)
    output = args.run_dir / 'work' / stage / f'node{args.rank}'
    if stage == 'grounding':
        command = [str(args.python), '-u', str(REPO / 'crispedit_mllm_grounding.py'),
                   '--input-dir', str(args.source_dir), '--output-dir', str(output),
                   '--keep-manifest-dir', str(args.quality_dir / 'manifest'),
                   '--difficulty-manifest-dir', str(args.scene_dir / 'manifest'),
                   '--model-path', str(args.model_path), '--devices', args.devices,
                   '--tensor-parallel-size', str(args.grounding_tp),
                   '--inference-backend', 'vllm', '--grounding-mode', 'two-pass',
                   '--batch-size', str(args.grounding_batch_size),
                   '--request-batch-size', str(args.grounding_batch_size),
                   '--max-images-per-generate', '0', '--max-new-tokens', '1536',
                   '--observation-max-new-tokens', '3072',
                   '--vllm-gpu-memory-utilization', '0.85', '--vllm-max-model-len', '32768',
                   '--vllm-max-images-per-prompt', '16', '--vllm-mm-encoder-tp-mode', 'data']
    else:
        command = [str(args.sam_python), '-u', str(REPO / 'crispedit_grounded_mask_runner.py'),
                   '--input-dir', str(args.source_dir), '--grounding-dir', str(args.label_dir / 'grounding'),
                   '--output-dir', str(output), '--checkpoint-path', str(args.checkpoint_path),
                   '--devices', args.devices]
    command += ['--selection-file', str(selection), '--progress-mininterval', '5', '--fail-fast']
    execute(args, command, stage)
    verify_label_shards(output, names, labels, stage)


def verify_label_shards(root, names, labels, stage):
    for name in names:
        table = pq.read_table(root / name)
        if table['row_idx'].to_pylist() != labels['rows'][name]:
            raise ValueError(f'{stage} row alignment failed: {name}')
        if any(r['mask_selection_reason'] != 'SELECTED' for r in table.select(['mask_selection_reason']).to_pylist()):
            raise ValueError(f'{stage} contains a non-double-PASS row: {name}')
        if stage == 'grounding':
            for r in table.select(['ground_parse_ok', 'ground_json']).to_pylist():
                payload = json.loads(r['ground_json'])
                if not r['ground_parse_ok'] or payload.get('runtime_error'):
                    raise ValueError(f'Grounding parse/runtime error: {name}')
        elif 'ERROR' in table['qc_flag'].to_pylist():
            raise ValueError(f'SAM runtime error: {name}')


def merge_labels(args, labels, stage):
    destination = args.label_dir / stage
    destination.mkdir(parents=True, exist_ok=True)
    for rank, names in enumerate(labels['assignments']):
        local = args.run_dir / 'work' / stage / f'node{rank}'
        verify_label_shards(local, names, labels, stage)
        for name in names:
            source, target = local / name, destination / name
            try:
                os.link(source, target)
            except FileExistsError:
                if not os.path.samefile(source, target):
                    raise FileExistsError(f'Refusing to replace another mask run: {target}')


def run(args) -> None:
    args.source_dir = args.source_dir.resolve()
    args.quality_dir = args.quality_dir.resolve()
    args.scene_dir = args.scene_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.run_dir = args.run_dir.resolve()
    args.label_dir = (args.label_dir or args.run_dir / 'labels').resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()
    if args.download_run_dir:
        args.download_run_dir = args.download_run_dir.resolve()
    if args.nodes != 4 or args.rank not in range(args.nodes):
        raise ValueError("This launcher requires four ranks 0..3")
    if not args.python.is_file() or not args.model_path.is_dir():
        raise FileNotFoundError("Python environment or Qwen3.8 model directory missing")
    if not args.sam_python.is_file() or not args.checkpoint_path.is_file():
        raise FileNotFoundError('SAM Python or checkpoint missing')
    if not args.local_test and len(args.devices.split(',')) != 8:
        raise ValueError('Production requires eight GPU devices per node')
    import re
    token = args.resume_token if args.resume else 'initial'
    if not token or not re.fullmatch(r'[A-Za-z0-9._-]+', token) or (args.resume and token == 'initial'):
        raise ValueError('Resume requires a new shared attempt token')
    control = args.run_dir / 'control' / token
    args.control = control
    control.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    # One process per rank, including across resume attempts.
    import fcntl
    lock = (args.run_dir / f'node{args.rank}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control / f'joined.node{args.rank}.json').exists():
        raise ValueError('Attempt already used; resume with a new token')
    plan_path = args.run_dir / "plan.json"
    if args.resume and not plan_path.exists():
        raise ValueError('Cannot resume without a saved plan')
    if args.rank == 0 and not plan_path.exists():
        try:
            write_json(plan_path, build_plan(args))
            marker(control / "plan.ready")
        except Exception as exc:
            marker(control / "plan.failed", repr(exc))
            raise
    elif args.rank == 0:
        marker(control / "plan.ready")
    wait_for(control / "plan.ready", control, args.wait_seconds)
    plan = json.loads(plan_path.read_text())
    if (plan["nodes"] != args.nodes or plan["source_dir"] != str(args.source_dir) or
            plan["quality_dir"] != str(args.quality_dir) or plan["scene_dir"] != str(args.scene_dir) or
            plan["model_path"] != str(args.model_path) or
            plan['label_dir'] != str(args.label_dir) or plan['checkpoint_path'] != str(args.checkpoint_path) or
            plan['settings'] != run_settings(args)):
        raise ValueError("Saved plan does not match this node's configuration")
    if plan["code_sha256"] != code_digest():
        raise ValueError("This node's pipeline code differs from the saved run plan")
    if plan['source_snapshot'] != source_snapshot(args.source_dir, plan['source_shards']):
        raise ValueError('Source snapshot changed since planning')
    print(f"node{args.rank}: quality={len(plan['quality_assignments'][args.rank])} "
          f"scene={len(plan['scene_assignments'][args.rank])}; run={args.run_dir}", flush=True)
    if args.plan_only:
        return
    write_json(control / f'joined.node{args.rank}.json', {'hostname': socket.gethostname(), 'rank': args.rank})
    for rank in range(args.nodes):
        wait_for(control / f'joined.node{rank}.json', control, args.wait_seconds)
    hosts = [json.loads((control / f'joined.node{r}.json').read_text())['hostname'] for r in range(args.nodes)]
    if not args.local_test and len(set(hosts)) != args.nodes:
        raise ValueError('Four distinct physical hosts required; --local-test is for validation only')
    for stage in ("quality", "scene"):
        if stage == "scene":
            wait_for(control / "quality.merged", control, args.wait_seconds)
        done = control / f"{stage}.node{args.rank}.done"
        if not done.exists():
            try:
                names = plan[f"{stage}_assignments"][args.rank]
                if stage == "scene":
                    names = [name for name in names if quality_pass_rows(args.quality_dir, name)]
                stage_runner(args, plan, stage, names)
                marker(done)
            except Exception as exc:
                marker(control / f"{stage}.node{args.rank}.failed", repr(exc))
                raise
        if args.rank == 0:
            await_nodes(stage, control, args.nodes, args.wait_seconds)
            merged = control / f"{stage}.merged"
            if not merged.exists():
                try:
                    summary = merge_stage(args, plan, stage)
                    marker(merged)
                    print(f"{stage} merged: {summary.get('verdicts', summary.get('decisions'))}", flush=True)
                except Exception as exc:
                    marker(control / f"{stage}.merge.failed", repr(exc))
                    raise
        else:
            wait_for(control / f"{stage}.merged", control, args.wait_seconds)
    label_plan_path = args.run_dir / 'label_plan.json'
    if args.rank == 0:
        owner = args.label_dir / 'run_owner.json'
        if args.label_dir.exists() and any(args.label_dir.iterdir()) and not owner.exists():
            raise ValueError(f'Label directory already contains another run: {args.label_dir}')
        if owner.exists() and json.loads(owner.read_text())['run_dir'] != str(args.run_dir):
            raise ValueError('Label directory belongs to another run')
        write_json(owner, {'run_dir': str(args.run_dir), 'code_sha256': plan['code_sha256']})
        labels = build_label_plan(args, plan)
        if label_plan_path.exists() and json.loads(label_plan_path.read_text()) != labels:
            raise ValueError('Double-PASS selection changed; use a new run')
        write_json(label_plan_path, labels)
        selection_file(args, labels)
        marker(control / 'labels.ready')
    wait_for(control / 'labels.ready', control, args.wait_seconds)
    labels = json.loads(label_plan_path.read_text())
    for stage in ('grounding', 'mask'):
        try:
            label_runner(args, labels, stage)
            marker(control / f'{stage}.node{args.rank}.done')
            if args.rank == 0:
                await_nodes(stage, control, args.nodes, args.wait_seconds)
                merge_labels(args, labels, stage)
                marker(control / f'{stage}.merged')
            wait_for(control / f'{stage}.merged', control, args.wait_seconds)
        except Exception as exc:
            marker(control / f'{stage}.node{args.rank}.failed', repr(exc))
            raise
    if args.rank == 0:
        execute(args, [str(args.sam_python), '-u', str(REPO / 'scripts/validate_crispedit_mask_pipeline.py'),
                       '--input-dir', str(args.source_dir), '--quality-dir', str(args.quality_dir / 'manifest'),
                       '--difficulty-dir', str(args.scene_dir / 'manifest'), '--run-dir', str(args.label_dir),
                       '--selection-file', str(args.run_dir / 'selection.json')], 'validate')
        summary = json.loads((args.label_dir / 'validation_summary.json').read_text())
        write_json(args.run_dir / 'reports' / 'mask_summary.json', summary)
        write_json(args.run_dir / 'reports' / 'run_manifest.json', {
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': plan['code_sha256'],
            'hosts': hosts, 'local_test': args.local_test, 'label_dir': str(args.label_dir),
            'selected_rows': sum(labels['node_rows']), 'mask_summary': summary})
        marker(control / 'complete.ok')
    wait_for(control / 'complete.ok', control, args.wait_seconds)
    print(f'node{args.rank}: complete pipeline validated; masks={args.label_dir / "mask"}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=BASE / "CrispEdit-2M")
    parser.add_argument("--quality-dir", type=Path, default=BASE / "CrispEdit-2M-qwen38-pair-prefilter")
    parser.add_argument("--scene-dir", type=Path, default=BASE / "CrispEdit-2M-difficult-local-edit")
    parser.add_argument("--download-run-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument('--label-dir', type=Path, help='New mask output directory; default RUN_DIR/labels')
    parser.add_argument('--checkpoint-path', type=Path, default=Path('/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt'))
    parser.add_argument('--sam-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--grounding-tp', type=int, default=2)
    parser.add_argument('--grounding-batch-size', type=int, default=16)
    parser.add_argument('--local-test', action='store_true', help='Explicit single-host four-rank GPU smoke test')
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--nodes", type=int, default=int(os.environ.get("ARNOLD_WORKER_NUM", "4")))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("ARNOLD_ID", "0")))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--wait-seconds", type=int, default=86400)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse this run after all prior node processes have stopped")
    parser.add_argument("--resume-token", default="",
                        help="New attempt ID shared by all four nodes; required with --resume")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if hasattr(args, 'control'):
            marker(args.control / f'pipeline.node{args.rank}.failed', repr(exc))
        raise


if __name__ == "__main__":
    main()
