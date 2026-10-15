"""Organize immutable automatic-pass exports into a portable training release."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time
import numpy as np
from sa1b_pipeline.production_state import atomic_json, write_rows, file_hash
from sa1b_pipeline.operation_balance import automatic_pass


def emit(**kw):
    print(json.dumps({'time': time.time(), **kw}), flush=True)


def organize(root, workers=24):
    root = root.resolve()
    release = root/'release'
    release.mkdir(exist_ok=True)
    lock=release/'.organizing'
    lock.mkdir()  # no concurrent publishers or automatic stale-lock stealing
    try:
        old_summary=json.loads((root/'final/summary.json').read_text())
        assert file_hash(root/'final/auto_pass.jsonl') == old_summary['manifest_sha256']
        rows=[json.loads(l) for l in (root/'final/auto_pass.jsonl').open() if l.strip()]
        assert len(rows)==50000 and all(automatic_pass(r,require_operation=True) for r in rows)
        mapping={}; train=[]; audit=[]; seen=set()
        fields=('job_id','case_id','unit_id','source_id','object_id','operation','instruction',
                'instruction_noref','instruction_protocol_version','instruction_spec','ref_phrase',
                'scope','mask_role','bbox','point','support_policy','attribute_kind','payload')
        for row in rows:
            sid=row['source_id']
            assert re.fullmatch(r'[A-Za-z0-9_-]+',sid)
            sample_id=hashlib.sha256(row['job_id'].encode()).hexdigest()[:24]
            assert sample_id not in seen
            seen.add(sample_id)
            new={k:row[k] for k in fields if k in row}
            new['sample_id']=sample_id
            for key,category in [('source_image','images/source'),('edited_image','images/target'),
                                 ('mask','masks/region'),('support_mask','masks/support')]:
                original=row[key]
                suffix=Path(original).suffix.lower()
                name=sid if key=='source_image' else sample_id
                destination=f'{category}/{name[:2]}/{name}{suffix}'
                if original in mapping: assert mapping[original]==destination
                mapping[original]=destination
                new[key]=destination
            new['image_size']=row['source_provenance']['image_size']
            new['audit']={k:row['semantic_audit'][k] for k in ('decision','operation_correct','target_done','non_target_preserved','natural')}
            new['manual_review']='not_performed'
            train.append(new)
        def remap(value):
            if isinstance(value,dict): return {k:remap(v) for k,v in value.items()}
            if isinstance(value,list): return [remap(v) for v in value]
            if isinstance(value,str): return mapping.get(value,value)
            return value
        audit=[{'sample_id':t['sample_id'],**remap(r)} for t,r in zip(train,rows)]
        for directory in sorted({str(Path(v).parent) for v in mapping.values()}):
            (release/directory).mkdir(parents=True,exist_ok=True)
        def link(item):
            original,destination=item
            src=root/original; dst=release/destination
            if Path(original).is_absolute() or '..' in Path(original).parts: raise ValueError(original)
            try: os.link(src,dst)
            except FileExistsError:
                if not os.path.samefile(src,dst): raise ValueError(f'Conflicting release media: {dst}')
            a,b=src.stat(),dst.stat()
            assert a.st_size>0 and (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
            return {'path':destination,'original_path':original,'bytes':b.st_size,'storage':'hardlink'}
        media=[]; last=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(link,item) for item in mapping.items()]
            for f in as_completed(futures):
                media.append(f.result())
                if time.monotonic()-last>20 or len(media)==len(mapping):
                    event={'stage':'link_media','completed':len(media),'total':len(mapping)}
                    atomic_json(release/'build_progress.json',event);emit(**event);last=time.monotonic()
        media.sort(key=lambda x:x['path'])
        attempted=Counter(); passed=Counter(); failures=Counter(); pernode={}; timings=defaultdict(list)
        for rank in range(4):
            n=Counter(); p=Counter()
            for line in (root/f'nodes/node{rank}/validation.jsonl').open():
                r=json.loads(line); op=r['operation']; attempted[op]+=1;n[op]+=1
                if automatic_pass(r,require_operation=True): passed[op]+=1;p[op]+=1
                else:
                    for reason in r.get('failure_reasons',[]):failures[str(reason)]+=1
                    for reason in (r.get('semantic_audit') or {}).get('failure_reasons',[]):failures['semantic:'+str(reason)]+=1
                for key in ('generation_seconds','audit_seconds'):
                    if isinstance(r.get(key),(float,int)):timings[key].append(r[key])
            pernode[str(rank)]={'generated':dict(n),'passed':dict(p)}
        sources=Counter(r['source_id'] for r in train)
        timing_stats={k:{'count':len(v),'mean':float(np.mean(v)),'median':float(np.median(v)),
                         'p90':float(np.percentile(v,90))} for k,v in timings.items()}
        stats={'samples':len(train),'operations':dict(Counter(r['operation'] for r in train)),
               'unique_source_images':len(sources),'unique_source_masks':len(seen),
               'generated_total':sum(attempted.values()),'generated_by_type':dict(attempted),
               'pipeline_passes_by_type':dict(passed),
               'pipeline_pass_rate':sum(passed.values())/sum(attempted.values()),
               'pipeline_pass_rate_by_type':{op:passed[op]/n for op,n in attempted.items()},
               'nodes':pernode,'timings_seconds':timing_stats,
               'samples_per_source':{'min':min(sources.values()),'max':max(sources.values()),
                   'mean':len(train)/len(sources),'median':float(np.median(list(sources.values())))},
               'media_files':len(media),'media_logical_bytes':sum(m['bytes'] for m in media),
               'media_by_directory':{c:{'files':sum(m['path'].startswith(c+'/') for m in media),
                    'bytes':sum(m['bytes'] for m in media if m['path'].startswith(c+'/'))}
                    for c in ['images/source','images/target','masks/region','masks/support']},
               'failure_reason_counts':dict(failures.most_common()),
               'counting_unit':'one independently edited source mask',
               'manual_review':'not_performed', 'source_export_sha256':old_summary['manifest_sha256']}
        assert dict(passed)==dict(Counter(r['operation'] for r in train))
        write_rows(release/'annotations/train.jsonl',train)
        write_rows(release/'annotations/audit.jsonl',audit)
        write_rows(release/'annotations/media_manifest.jsonl',media)
        atomic_json(release/'statistics.json',stats)
        report={'status':'complete','schema_version':'samtok-single-edit-1.0','samples':len(train),
                'media_files':len(media),'all_media_verified_same_inode_as_original':True,
                'source_export_sha256':old_summary['manifest_sha256'],
                'train_sha256':file_hash(release/'annotations/train.jsonl'),
                'audit_sha256':file_hash(release/'annotations/audit.jsonl'),
                'manual_review':'not_performed','completed_unix':time.time()}
        atomic_json(release/'release.json',report)
        atomic_json(release/'build_progress.json',report)
        emit(**report)
    finally: lock.rmdir()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=24);a=p.parse_args();organize(a.root,a.workers)
