"""Apply current conservative label-admission rules to saved model replies.

No model calls and no alteration of the original experiment. These are still
model candidates, not manually verified training labels.
"""
import argparse
import json
from pathlib import Path
from synthesis_pipeline.audit_quality_v3 import (
    read, parse_rewrite, rewrite_policy_error, corrected_annotation,
)
from synthesis_pipeline.audit_edit_pairs import write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations-jsonl', type=Path, required=True)
    p.add_argument('--audit-jsonl', type=Path, required=True)
    p.add_argument('--rewrites-jsonl', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=False)
    rows = {r['image']: r for r in read(args.annotations_jsonl)}
    audits = {r['image']: r for r in read(args.audit_jsonl)}
    results, candidates = [], []
    for record in read(args.rewrites_jsonl):
        row = rows[record['image']]
        val = parse_rewrite(record['raw_response'])
        error = rewrite_policy_error(row, val)
        result = {**record, 'prior_status': record['status'], 'policy_error': error,
                  'status': 'candidate' if error is None else 'rejected_policy_or_parse',
                  'policy_version': 'single_target_same_operation_v1'}
        results.append(result)
        corrected = corrected_annotation(row, result, audits[row['image']])
        if corrected is not None:
            candidates.append(corrected)
    write_jsonl(args.out_root / 'rewrites.jsonl', results)
    write_jsonl(args.out_root / 'model_accepted_annotations.jsonl', candidates)
    summary = dict(replies=len(results), model_candidates=len(candidates), vlm_calls=0,
                   verification='not_manually_verified')
    (args.out_root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
