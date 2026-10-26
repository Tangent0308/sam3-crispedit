#!/usr/bin/env python3
"""Single-target samples from evidence-gated multi plans: every target edited from the original photograph,
checked and audited on its own (sa1b_pipeline/single_parallel.py). Single-target v2.1; the multi runner
(run_multi_transactional.py) is unchanged."""
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
from sa1b_pipeline.single_parallel import POLICY, generate_independent_singles, singles_of
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
    p.add_argument('--editor-devices', default='0,1,2,3,4'); p.add_argument('--audit-devices', default='5,6,7')
    p.add_argument('--service-root', type=Path, help='Request/response queues (default: OUTPUT_ROOT/services)')
    p.add_argument('--groups-in-flight', type=int, default=16)
    p.add_argument('--audit-batch-size', type=int, default=8)
    p.add_argument('--save-crop-evidence', action='store_true')
    p.add_argument('--local-generation-reference', action='store_true',
                   help='Use grounded source nouns inside crops; preserve published scene references.')
    args = p.parse_args(argv)
    edit_gpus = [x.strip() for x in args.editor_devices.split(',') if x.strip()]
    audit_gpus = [x.strip() for x in args.audit_devices.split(',') if x.strip()]
    if set(edit_gpus) & set(audit_gpus) or args.groups_in_flight < 1 or args.audit_batch_size < 1:
        p.error('editor/audit GPUs must be disjoint; concurrency and batch size must be positive')
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    service_root = args.service_root or root / 'services'
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
        immutable_json(root/'transactional_run.json', {'policy': POLICY,
                       'save_crop_evidence': args.save_crop_evidence,
                       'local_generation_reference': args.local_generation_reference,
                       'audit_policy': 'packed-components-v4',
                       'planned_sha256': file_hash(args.planned_manifest),
                       'editor_model': args.editor_model, 'audit_model': args.audit_model, 'steps': 40})
        editors = ServicePool(python=args.editor_python, role='editor', devices=edit_gpus,
                              model=args.editor_model, directory=service_root/'editor')
        audits = ServicePool(python=args.audit_python, role='audit', devices=audit_gpus,
                             model=args.audit_model, directory=service_root/'audit', batch_size=args.audit_batch_size)
        if services is not None:
            editors, audits = services
            if editors.model != args.editor_model or audits.model != args.audit_model or audits.batch_size != args.audit_batch_size:
                raise ValueError('resident services do not match single run contract')
        audits.keep_singles = True   # a group with one valid target still audits it
        auditor = RemoteAuditor(audits); started = time.monotonic(); results = []

        def run(g):
            return generate_independent_singles(g, root/'generated'/g['multi_id'], lambda: RemoteEditor(editors),
                                                auditor, save_crop_evidence=args.save_crop_evidence)
        failures = []
        # As in the multi runner: data errors of one group are recorded and the group is left out; an
        # infrastructure failure (a stopped worker, a timeout) still stops the run, and so do too many failures.
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
                                raise RuntimeError(f'{len(failures)} groups failed with data errors; '
                                                   f'see {root}/failed_groups.jsonl')
                            continue
                        results.append(result)
                        atomic_json(root/'transactional_progress.json', {
                            'completed': len(results), 'planned': len(plans),
                            'singles': sum(g['n_targets'] for g in results),
                            'seconds': time.monotonic()-started})
                except BaseException:
                    for future in futures: future.cancel()
                    import traceback
                    traceback.print_exc()
                    if services is None:
                        editors.close(); audits.close()
                    raise
        finally:
            if services is None:
                editors.close(); audits.close()
        results.sort(key=lambda g: g['multi_id'])
        write_rows(root/'audited.jsonl', results)
        singles = [s for g in results for s in singles_of(g)]
        write_rows(root/'singles.jsonl', singles)
        atomic_json(root/'transactional_done.json', {'seconds': time.monotonic()-started,
                    'groups': len(results), 'failed_groups': len(failures), 'singles': len(singles),
                    'singles_by_operation': dict(Counter(s['operation'] for s in singles)),
                    'planned_targets': sum(len(g['retention']['decisions']) for g in results),
                    'note': 'automatic checks only; no manual review'})
        print(json.dumps({'completed': len(results), 'singles': len(singles), 'root': str(root)}))


if __name__ == '__main__':
    main()
