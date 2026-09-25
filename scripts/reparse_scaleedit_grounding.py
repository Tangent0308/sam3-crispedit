#!/usr/bin/env python3
"""Revalidate persisted raw model JSON after parser-only fixes; no coordinates are invented."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from tqdm import tqdm
from scaleedit import policy
from scaleedit.runner import GROUND_SCHEMA, code_digest, write_table


def reparse(row):
    row=dict(row)
    payload=json.loads(row['ground_json'])
    payload['parser_replay']={'previous_qc':row['qc_flag'],'previous_error':row['error'],
                              'model_called':False,'coordinates_modified':False}
    try:
        observation=policy.parse_edit_units(payload['observation']['attempts'][-1]['raw_text'])
        payload['observation'].update(parsed=observation,parse_ok=True,error='')
        if not observation['changes']:
            raise ValueError('no_realized_edits')
        payload['boxes']={'source':[],'target':[]}
        seen=set()
        for request in payload['requests']:
            side=request['grounding_image']
            if side in seen:
                raise ValueError('Duplicate grounding side')
            seen.add(side)
            context=policy.side_context(observation,side)
            boxes,unresolved=policy.parse_located_units(request['attempts'][-1]['raw_text'],context,side)
            geometry={u['change_id']:u['geometry'] for u in context['changes']}
            for box in boxes: box['geometry']=geometry[box['change_id']]
            payload['boxes'][side]=boxes
            request.update(parse_ok=True,error='',unresolved=unresolved)
        expected={u['image_side'] for u in observation['changes']}
        if seen!=expected or not any(payload['boxes'].values()):
            raise ValueError('Incomplete grounding coverage')
        row.update(qc_flag='MASK_REVIEW' if any(q['unresolved'] for q in payload['requests']) else 'OK',error='')
    except Exception as exc:
        row.update(qc_flag='GROUND_FAIL',error=repr(exc))
    row['ground_json']=json.dumps(payload,ensure_ascii=False)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('Use a new output directory; original audits are never overwritten')
    files=sorted(args.input_dir.glob('*.parquet'))
    if not files or not (args.input_dir/'run_summary.json').exists():
        raise SystemExit('Require a completed grounding run, including its run_summary.json')
    digest=code_digest()
    counts={}
    for path in tqdm(files,desc='Reparse saved grounding',unit='shard'):
        rows=[reparse(row) for row in pq.read_table(path).to_pylist()]
        sig=hashlib.sha256(path.read_bytes()+digest.encode()+b'parser-only-replay').hexdigest()
        write_table(args.output_dir/path.name,rows,GROUND_SCHEMA,sig)
        for row in rows: counts[row['qc_flag']]=counts.get(row['qc_flag'],0)+1
    result=dict(rows=sum(counts.values()),counts=counts,source=str(args.input_dir.resolve()),
                method='parser-only-replay',mllm_calls=0,code_digest=digest)
    (args.output_dir/'run_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__': main()
