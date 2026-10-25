#!/usr/bin/env python3
"""Generate multi-target groups; audit final components once before exact fusion."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse
import json
import math
import sys
import signal
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.inference_services import ServicePool, RemoteEditor, RemoteAuditor
from sa1b_pipeline.multi_transactional import generate_committed_multi_group, POLICY
from sa1b_pipeline.multi_parallel import POLICY as PARALLEL_POLICY
from sa1b_pipeline.production_state import read_rows, write_rows, atomic_json, immutable_json, file_hash


def main(argv=None, *, services=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planned-manifest', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--editor-model', required=True); p.add_argument('--audit-model', required=True)
    p.add_argument('--editor-python', required=True); p.add_argument('--audit-python', required=True)
    p.add_argument('--editor-devices', default='0,1,2,3,4,5'); p.add_argument('--audit-devices', default='6,7')
    p.add_argument('--groups-in-flight', type=int, default=16)
    p.add_argument('--audit-batch-size', type=int, default=4)
    p.add_argument('--generation-mode', choices=('serial', 'parallel-components'), default='parallel-components')
    p.add_argument('--audit-policy', choices=('step-and-joint', 'component-final-v2', 'group-final-v3', 'packed-components-v4'), default=None,
                   help='Audit each final independent/dependent region once; fusion needs no additional MLLM call.')
    p.add_argument('--save-crop-evidence', action='store_true')
    p.add_argument('--local-generation-reference', action='store_true',
                   help='Use grounded source nouns inside crops; preserve published scene references.')
    p.add_argument('--keep-singles', action='store_true',
                   help='Opt-in: a group left with exactly one passing target writes it to singles.jsonl')
    args = p.parse_args(argv)
    if args.audit_policy is None:
        args.audit_policy = 'component-final-v2' if args.generation_mode == 'parallel-components' else 'step-and-joint'
    if args.generation_mode == 'serial' and args.audit_policy != 'step-and-joint':
        p.error('serial mode requires --audit-policy step-and-joint')
    edit_gpus = [x.strip() for x in args.editor_devices.split(',') if x.strip()]
    audit_gpus = [x.strip() for x in args.audit_devices.split(',') if x.strip()]
    if set(edit_gpus) & set(audit_gpus) or args.groups_in_flight < 1 or args.audit_batch_size < 1:
        p.error('editor/audit GPUs must be disjoint; concurrency and batch size must be positive')
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    from scripts.run_group_production_4node import owner_lock
    with owner_lock(root/'transactional.lock'):
        plans = read_rows(args.planned_manifest)
        if args.local_generation_reference:
            from sa1b_pipeline.group_generation import _unit_dict
            for group in plans:
                for member in group['units']:
                    member['unit'] = {**_unit_dict(member), 'generation_reference_frame': 'crop-entity-v1'}
        if len({g['multi_id'] for g in plans}) != len(plans):
            raise ValueError('duplicate multi IDs')
        immutable_json(root/'transactional_run.json', {'policy': POLICY if args.generation_mode == 'serial' else PARALLEL_POLICY,
                       'save_crop_evidence': args.save_crop_evidence,
                       'local_generation_reference': args.local_generation_reference,
                       'audit_policy': args.audit_policy,
                       'planned_sha256': file_hash(args.planned_manifest),
                       'editor_model': args.editor_model, 'audit_model': args.audit_model, 'steps': 40,
                       **({'keep_singles': True} if args.keep_singles else {})})
        editors = ServicePool(python=args.editor_python, role='editor', devices=edit_gpus,
                              model=args.editor_model, directory=root/'services/editor')
        audits = ServicePool(python=args.audit_python, role='audit', devices=audit_gpus,
                             model=args.audit_model, directory=root/'services/audit', batch_size=args.audit_batch_size)
        if services is not None:
            editors, audits = services
            if editors.model != args.editor_model or audits.model != args.audit_model or audits.batch_size != args.audit_batch_size:
                raise ValueError('resident services do not match multi run contract')
        if args.keep_singles:
            audits.keep_singles = True
        auditor = RemoteAuditor(audits); started = time.monotonic(); results = []
        def run(g):
            if args.generation_mode == 'parallel-components':
                from sa1b_pipeline.multi_parallel import generate_parallel_multi_group
                return generate_parallel_multi_group(g, root/'generated'/g['multi_id'],
                    lambda: RemoteEditor(editors), auditor, save_crop_evidence=args.save_crop_evidence,
                    audit_policy=args.audit_policy)
            return generate_committed_multi_group(g, root/'generated'/g['multi_id'], RemoteEditor(editors), auditor,
                                                  save_crop_evidence=args.save_crop_evidence)
        failures = []
        # Data-dependent errors of single groups are recorded and the group is left out; an infrastructure
        # failure (a stopped service worker, a timeout) still stops the run. Too many failures stop it too.
        data_errors = (ValueError, KeyError, IndexError, AssertionError, ZeroDivisionError)
        def run_isolated(g):
            try:
                return run(g)
            except data_errors as exc:
                import traceback
                record = {'multi_id': g['multi_id'], 'error': repr(exc), 'traceback': traceback.format_exc(),
                          'time': time.time()}
                with (root/'failed_groups.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps({'group_failed': g['multi_id'], 'error': repr(exc)}), flush=True)
                return None
        try:
            # Load all independent services together; the first audit should
            # not pay a model cold start while editors wait for its decision.
            # Completed groups still verify their input/media hashes in run().
            # Replaying a completed wave must not reload eight GPU models.
            if any(not (root/'generated'/g['multi_id']/'committed.json').is_file() for g in plans):
                editors.start(); audits.start()
            with ThreadPoolExecutor(max_workers=args.groups_in_flight) as executor:
                futures = [executor.submit(run_isolated, g) for g in plans]
                try:
                    for future in as_completed(futures):
                        result = future.result()
                        if result is None:
                            failures.append(1)
                            if len(failures) > max(3, math.ceil(.05 * len(plans))):
                                raise RuntimeError(f'{len(failures)} multi groups failed with data errors; '
                                                   f'see {root}/failed_groups.jsonl')
                            continue
                        results.append(result)
                        atomic_json(root/'transactional_progress.json', {
                            'completed': len(results), 'planned': len(plans),
                            'automatic_pass': sum(g['joint_audit']['pass'] for g in results),
                            'retained_targets': sum(g['n_targets'] for g in results),
                            'seconds': time.monotonic()-started})
                except BaseException:
                    for future in futures: future.cancel()
                    import traceback
                    traceback.print_exc()  # visible even when a concurrent job reports first
                    if services is None:  # shared services belong to the caller (other jobs use them)
                        editors.close(); audits.close()
                    raise
        finally:
            if services is None:
                editors.close(); audits.close()
        results.sort(key=lambda g: g['multi_id'])
        write_rows(root/'audited.jsonl', results)
        write_rows(root/'generated.jsonl', [g for g in results if g['joint_audit']['status']=='complete'])
        accepted = [g for g in results if g['joint_audit']['pass']]
        write_rows(root/'accepted.jsonl', accepted)
        if args.keep_singles:
            from sa1b_pipeline.single_salvage import multi_single
            write_rows(root/'singles.jsonl', [s for g in results if (s := multi_single(g))])
        atomic_json(root/'transactional_done.json', {'seconds': time.monotonic()-started,
                    'groups': len(results), 'automatic_pass': len(accepted),
                    'retained_target_distribution': dict(Counter(g['n_targets'] for g in accepted)),
                    'accepted_operations': dict(Counter(op for g in accepted for op in g['operations'])),
                    'note': 'automatic acceptance only; production quota/export checks remain separate'})
        print(json.dumps({'completed': len(results), 'automatic_pass': len(accepted), 'root': str(root)}))


if __name__ == '__main__':
    main()
