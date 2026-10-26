#!/usr/bin/env python3
"""Single-target v2.1 wave: evidence-gated multi planning, then independent single-target generation.

Models, planning flags, editor finishing and audit settings are those of the production multi path
(run_dual_validation.py with the production flags of run_evidence_production_4node.py). Three single-target
changes, all in new code (the twins/multi scripts are unchanged):
* generation edits every target from the original photograph and exports every passing target
  (run_single_transactional.py, sa1b_pipeline/single_parallel.py);
* service request/response queues live on local disk (--service-root). Next to the run on the network
  filesystem a stat takes ~7 ms and with hundreds of polling clients ~1.6 s (median), so requests reached the
  planner workers in small batches; worker logs are copied into OUTPUT_ROOT/services afterwards;
* planning submits all requests of a wave at once (--planning-batch, default all groups), so the planner
  workers' queues stay full; bounded rounds of 128 groups left 41% of the planner time idle in wave01;
* optional (--planning-chunks K): the wave's groups are planned as K parts at the same time on the shared
  planner pool, as --concurrent-kinds plans twins and multi together. Each part's single-threaded steps
  (building gate requests, building and checking every planned group: 334 s of 1,247 s planning in a
  300-group wave, with the planner GPUs idle) then run side by side.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.inference_services import ServicePool, RemoteSelector
from sa1b_pipeline.production_state import atomic_json, read_rows, write_rows, immutable_json, file_hash

FINISH = {'removal_effects': None, 'regrain': True, 'tone_match': False, 'add_cleanup': True,
          'prompt_rules': True, 'focus_match': True}   # run_dual_validation --edit-finish


def keep_logs(pool, role, root):
    """Copy the workers' logs and ready markers next to the run, then drop the local queue directory."""
    if not pool.directory.exists():
        return
    for worker in sorted(pool.directory.glob('[0-9]*')):
        target = root/'services'/role/pool.directory.name/worker.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ('worker.log', 'ready.json'):
            if (worker/name).exists():
                shutil.copy2(worker/name, target/name)
    shutil.rmtree(pool.directory, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-root', type=Path, required=True, help='Contains multi/inputs selected.jsonl + manifest.jsonl')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--planner-model', required=True)
    p.add_argument('--editor-model', required=True)
    p.add_argument('--site-model', required=True)
    p.add_argument('--mllm-python', required=True)
    p.add_argument('--editor-python', required=True)
    p.add_argument('--reservation', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--planner-devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--audit-devices', default='5,6,7')
    p.add_argument('--service-root', type=Path, default=Path('/tmp/sa1b-single-services'),
                   help='Local directory for the request/response queues (each pool adds its own subdirectory)')
    p.add_argument('--planning-batch', type=int, default=0,
                   help='Groups (and planner requests) submitted per planning round; 0: all groups at once')
    p.add_argument('--planning-chunks', type=int, default=1,
                   help='Plan the groups as this many parts at the same time (multi/chunk_K); 1: one part')
    p.add_argument('--groups-in-flight', type=int, default=16)
    p.add_argument('--single-target-groups', action='store_true',
                   help='Opt-in (single-target production): planning keeps groups with one target, so targets of '
                        'operations without need can be left out and a group with one feasible target is kept '
                        '(sa1b_pipeline/single_planning.py; this process only)')
    args = p.parse_args()
    planner_devices = args.planner_devices.split(',')
    audit_devices = [d for d in args.audit_devices.split(',') if d]
    for devices in (planner_devices, audit_devices):
        if not devices or len(set(devices)) != len(devices) or any(d not in list('01234567') for d in devices):
            p.error('devices must be distinct physical GPU IDs 0..7')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('unset CUDA_VISIBLE_DEVICES; this entry uses physical GPUs 0..7')
    root = args.output_root; root.mkdir(parents=True, exist_ok=True)
    os.environ.update(WANDB_MODE='disabled', WANDB_DISABLED='true', OMP_NUM_THREADS='4')
    from scripts import run_group_pipeline, run_single_transactional
    from scripts.run_group_production_4node import owner_lock
    if args.single_target_groups:
        from sa1b_pipeline import single_planning
        single_planning.enable(run_group_pipeline)
    with owner_lock(root/'single.lock'):
        immutable_json(root/'single_contract.json', {
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'code': {str(f.relative_to(Path(__file__).resolve().parents[1])): file_hash(f)
                     for folder in ('sa1b_pipeline', 'scripts', 'utils')
                     for f in (Path(__file__).resolve().parents[1]/folder).rglob('*.py')}})
        timings = json.loads((root/'timings.json').read_text()) if (root/'timings.json').exists() else {}

        def stage(name, fn):
            if timings.get(name, {}).get('complete'):
                return
            t = time.monotonic(); fn()
            timings[name] = {'seconds': time.monotonic()-t, 'complete': True}
            atomic_json(root/'timings.json', timings)
            print(json.dumps({'stage': name, **timings[name]}), flush=True)
        start = time.monotonic()
        out = root/'multi'; inputs = out/'inputs'; inputs.mkdir(parents=True, exist_ok=True)
        for name in ('selected.jsonl', 'manifest.jsonl'):
            src = args.inputs_root/'multi'/'inputs'/name; dst = inputs/name
            if dst.exists() and dst.read_bytes() != src.read_bytes():
                raise ValueError('changed selected inputs')
            if not dst.exists():
                shutil.copy2(src, dst)
        selected = read_rows(inputs/'selected.jsonl')
        n_groups = len(selected)
        chunks = max(1, min(args.planning_chunks, n_groups))
        if chunks == 1:
            parts = [(out, n_groups)]
        else:
            parts = []
            for k in range(chunks):
                part = out/f'chunk_{k}'; (part/'inputs').mkdir(parents=True, exist_ok=True)
                rows = selected[k::chunks]
                text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
                if (part/'inputs/selected.jsonl').exists() and (part/'inputs/selected.jsonl').read_text() != text:
                    raise ValueError('changed planning chunk inputs')
                (part/'inputs/selected.jsonl').write_text(text)
                parts.append((part, len(rows)))

        # The production multi planning command (run_dual_validation, kind multi, production flags).
        def planning_cmd(part, n):
            return ['--kind', 'multi', '--manifest', str(inputs/'manifest.jsonl'), '--output-root', str(part),
               '--target', str(n), '--reservation', str(args.reservation),
               '--outline-planning-crops', '--locate-planning-scene', '--structured-planning-json',
               '--skip-unused-planning-views', '--plan-only', '--planner-model', args.planner_model,
               '--planner-tp', '1', '--planner-batch-size', '16', '--max-new-tokens', '1024',
               '--planner-thinking-budget', '512', '--per-mask-planning', '--planning-operation-retry', '0',
               '--planning-source-hints', 'none', '--seed', str(args.seed), '--site-model', args.site_model,
               '--planning-view-workers', '16', '--multi-plan-style', 'content-v1', '--dynamic-operation-quotas',
               '--evidence-gates', '--evidence-gate-lighting', '--evidence-gate-medium', '--evidence-gate-viewpoint',
               '--replace-retry', '--explicit-text', '--twin-single-add', '--garment-extent-check']
        if not timings.get('multi_planning', {}).get('complete'):
            planners = ServicePool(python=args.mllm_python, role='planner', devices=planner_devices,
                                   model=args.planner_model, directory=args.service_root/'planner', batch_size=16,
                                   planner_options={'max_new_tokens': 1024, 'thinking_token_budget': 512})
            planners.start()

            def plan(part, n):
                # Planning stores per-run state on its planner object: one light selector per part.
                selector = RemoteSelector(planners)
                # One planning round per part: plan_batch sends `batch_size` groups and requests at a time.
                selector.batch_size = args.planning_batch or max(n, 1) * 4
                run_group_pipeline.main(planning_cmd(part, n), planner_override=selector)

            def plan_all():
                with ThreadPoolExecutor(max_workers=len(parts)) as pool:
                    for future in [pool.submit(plan, part, n) for part, n in parts]:
                        future.result()
                if len(parts) > 1:
                    write_rows(out/'planned.jsonl', [g for part, _ in parts for g in read_rows(part/'planned.jsonl')])
            try:
                stage('multi_planning', plan_all)
            finally:
                planners.close(); keep_logs(planners, 'planner', root)
        editor_devices = [d for d in '01234567' if d not in audit_devices]
        if not editor_devices:
            p.error('no GPU left for editors')
        editor_tools = {'text_verify': True, 'finish': {**FINISH, 'removal_effects': args.site_model}}
        editors = ServicePool(python=args.editor_python, role='editor', devices=editor_devices, model=args.editor_model,
                              directory=args.service_root/'editor', editor_tools=editor_tools, dispatch='least_loaded')
        audits = ServicePool(python=args.mllm_python, role='audit', devices=audit_devices, model=args.planner_model,
                             directory=args.service_root/'audit', batch_size=8, dispatch='least_loaded')
        editors.priority_scheduling = audits.priority_scheduling = True
        audits.keep_singles = True
        audits.ref_reconcile = True
        from sa1b_pipeline.edit_verifier import SPLIT
        audits.verifier = {'describe': False, 'thinking': 512, 'by_operation': dict(SPLIT)}
        gen = ['--planned-manifest', str(out/'planned.jsonl'), '--output-root', str(root/'single'),
               '--editor-model', args.editor_model, '--audit-model', args.planner_model,
               '--editor-python', args.editor_python, '--audit-python', args.mllm_python,
               '--editor-devices', ','.join(editor_devices), '--audit-devices', ','.join(audit_devices),
               '--audit-batch-size', '8', '--groups-in-flight', str(args.groups_in_flight),
               '--local-generation-reference', '--save-crop-evidence']
        try:
            stage('single_generation', lambda: run_single_transactional.main(gen, services=(editors, audits)))
        finally:
            editors.close(); audits.close()
            keep_logs(editors, 'editor', root); keep_logs(audits, 'audit', root)
        atomic_json(root/'session.json', {'seconds': time.monotonic()-start, 'timings': timings, 'steps': 40,
                                         'published': False})


if __name__ == '__main__':
    main()
