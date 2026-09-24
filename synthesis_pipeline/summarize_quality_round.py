"""Summarize all denominators and measured stage costs of a frozen new cohort."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit-root',type=Path)
    p.add_argument('--evaluation-root',type=Path)
    a=p.parse_args();root=a.root
    inputs=read(root/'annotations.jsonl');plans=read(root/'plan/annotations.jsonl')
    regions=read(root/'regions/annotations.jsonl');final=read(root/'final/annotations.jsonl')
    raw=root/'generation/context_grounded_v4'
    summaries=[json.loads(x.read_text()) for x in sorted(raw.glob('summary_shard*.json'))]
    records=[r for s in summaries for r in s['records']]
    stats=dict(input_regions=len(inputs),fresh_sources=len({r['source_image'] for r in inputs}),
        input_types=dict(Counter(r['task_type'] for r in inputs)),planning_accepted=len(plans),
        source_mask_resolved=sum(r['region_contract']['status']!='unresolved' for r in regions),
        generated=len(list((raw/'edited').glob('*.png'))),composed=len(final),
        composed_types=dict(Counter(r['task_type'] for r in final)),
        composition_rejected=len(read(root/'final/composition_rejected/annotations.jsonl')),
        measured_stages={})
    for stage,folder in [('planning',root/'plan'),('source_masks',root/'regions'),('composition',root/'composed'),('audit',a.audit_root)]:
        if folder and (folder/'summary.json').exists():
            value=json.loads((folder/'summary.json').read_text())
            stats['measured_stages'][stage]={k:v for k,v in value.items() if k!='shards'}
    if records:
        times=[r['seconds'] for r in records]
        critical=max(s['model_load_seconds']+sum(r['seconds'] for r in s['records']) for s in summaries)
        stats['measured_stages']['diffusion']=dict(workers_completed=len(summaries),cases=len(records),
            single_case_mean_seconds=sum(times)/len(times),single_case_min_seconds=min(times),
            single_case_max_seconds=max(times),slowest_worker_load_plus_cases_seconds=critical,
            approximate_cases_per_minute=len(records)*60/critical,
            timing_note='Worker load + per-case generation; excludes orchestration startup, SAM, audit and manual review.')
    if a.evaluation_root and (a.evaluation_root/'evaluation.json').exists():
        stats['evaluation']=json.loads((a.evaluation_root/'evaluation.json').read_text())
    stats['note']='Counts retain all frozen inputs. Model acceptance is not assistant verification. Stage timings are not an uninterrupted end-to-end benchmark.'
    (root/'cohort_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
