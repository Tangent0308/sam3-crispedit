#!/usr/bin/env python3
"""Generate planned multi-target groups with per-step commit/rollback and one final audit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse
import json
import sys
import signal
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.inference_services import ServicePool, RemoteEditor, RemoteAuditor
from sa1b_pipeline.multi_transactional import generate_committed_multi_group, POLICY
from sa1b_pipeline.production_state import read_rows, write_rows, atomic_json, immutable_json, file_hash


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
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
    p.add_argument('--save-crop-evidence', action='store_true')
    args = p.parse_args(argv)
    edit_gpus = [x.strip() for x in args.editor_devices.split(',') if x.strip()]
    audit_gpus = [x.strip() for x in args.audit_devices.split(',') if x.strip()]
    if set(edit_gpus) & set(audit_gpus) or args.groups_in_flight < 1 or args.audit_batch_size < 1:
        p.error('editor/audit GPUs must be disjoint; concurrency and batch size must be positive')
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    from scripts.run_group_production_4node import owner_lock
    with owner_lock(root/'transactional.lock'):
        plans = read_rows(args.planned_manifest)
        if len({g['multi_id'] for g in plans}) != len(plans):
            raise ValueError('duplicate multi IDs')
        immutable_json(root/'transactional_run.json', {'policy': POLICY if args.generation_mode == 'serial' else 'parallel-components-v1',
                       'save_crop_evidence': args.save_crop_evidence,
                       'planned_sha256': file_hash(args.planned_manifest),
                       'editor_model': args.editor_model, 'audit_model': args.audit_model, 'steps': 40})
        editors = ServicePool(python=args.editor_python, role='editor', devices=edit_gpus,
                              model=args.editor_model, directory=root/'services/editor')
        audits = ServicePool(python=args.audit_python, role='audit', devices=audit_gpus,
                             model=args.audit_model, directory=root/'services/audit', batch_size=args.audit_batch_size)
        auditor = RemoteAuditor(audits); started = time.monotonic(); results = []
        def run(g):
            if args.generation_mode == 'parallel-components':
                from sa1b_pipeline.multi_parallel import generate_parallel_multi_group
                return generate_parallel_multi_group(g, root/'generated'/g['multi_id'],
                    lambda: RemoteEditor(editors), auditor, save_crop_evidence=args.save_crop_evidence)
            return generate_committed_multi_group(g, root/'generated'/g['multi_id'], RemoteEditor(editors), auditor,
                                                  save_crop_evidence=args.save_crop_evidence)
        try:
            with ThreadPoolExecutor(max_workers=args.groups_in_flight) as executor:
                futures = [executor.submit(run, g) for g in plans]
                try:
                    for future in as_completed(futures):
                        results.append(future.result())
                        atomic_json(root/'transactional_progress.json', {
                            'completed': len(results), 'planned': len(plans),
                            'automatic_pass': sum(g['joint_audit']['pass'] for g in results),
                            'retained_targets': sum(g['n_targets'] for g in results),
                            'seconds': time.monotonic()-started})
                except BaseException:
                    for future in futures: future.cancel()
                    editors.close(); audits.close()
                    raise
        finally:
            editors.close(); audits.close()
        results.sort(key=lambda g: g['multi_id'])
        write_rows(root/'audited.jsonl', results)
        write_rows(root/'generated.jsonl', [g for g in results if g['joint_audit']['status']=='complete'])
        accepted = [g for g in results if g['joint_audit']['pass']]
        write_rows(root/'accepted.jsonl', accepted)
        atomic_json(root/'transactional_done.json', {'seconds': time.monotonic()-started,
                    'groups': len(results), 'automatic_pass': len(accepted),
                    'retained_target_distribution': dict(Counter(g['n_targets'] for g in accepted)),
                    'accepted_operations': dict(Counter(op for g in accepted for op in g['operations'])),
                    'note': 'automatic acceptance only; production quota/export checks remain separate'})
        print(json.dumps({'completed': len(results), 'automatic_pass': len(accepted), 'root': str(root)}))


if __name__ == '__main__':
    main()
