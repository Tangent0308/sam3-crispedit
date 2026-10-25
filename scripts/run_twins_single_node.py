#!/usr/bin/env python3
"""Single 8-GPU group production with resident 27B services and final quotas.

The historical script name is retained for existing twin entrypoints; --kind
also supports multi groups, with quotas based on the retained final targets.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.group_production import target_cells, choose, propose, accept, describe, passed
from sa1b_pipeline.group_schema import validate_twin_record, validate_multi_record
from sa1b_pipeline.inference_services import ServicePool, RemoteSelector
from sa1b_pipeline.production_state import atomic_json, immutable_json, read_rows, write_rows, portable, file_hash, digest
from scripts.run_group_production_4node import owner_lock
from scripts.run_twins_streaming import main as run_stream


def load_source_candidates(manifests, path_map=None):
    """Relocate source inputs only; never reinterpret a run's checkpoints."""
    from sa1b_pipeline.archive_relocation import ArchivePaths
    paths = ArchivePaths.load(path_map) if path_map else None
    return [paths.record(row) if paths else row
            for manifest in manifests for row in read_rows(manifest)]


def run_multi_wave(wave, args, reservation, pools):
    from scripts.run_group_pipeline import main as plan
    from scripts.run_multi_transactional import main as generate
    selected=read_rows(wave/'inputs/selected.jsonl')
    write_rows(wave/'inputs/manifest.jsonl',[s['candidate'] for s in selected])
    plan(['--kind','multi','--manifest',str(wave/'inputs/manifest.jsonl'),'--output-root',str(wave),
          '--target',str(len(selected)),'--reservation',str(reservation),
          '--multi-plan-style','content-v1','--per-mask-planning','--planning-source-hints','none',
          '--planning-operation-retry','0','--outline-planning-crops','--locate-planning-scene',
          '--structured-planning-json','--skip-unused-planning-views','--plan-only',
          '--planner-model',args.model,'--planner-batch-size',str(args.planner_batch_size),
          '--planner-thinking-budget',str(args.planner_thinking_budget),'--max-new-tokens','2300',
          '--site-model',args.site_model],planner_override=RemoteSelector(pools[2]))
    generate(['--planned-manifest',str(wave/'planned.jsonl'),'--output-root',str(wave),
              '--editor-model',args.editor_model,'--audit-model',args.model,
              '--editor-python',args.editor_python,'--audit-python',args.mllm_python,
              '--editor-devices','1,2,3,4,5,6','--audit-devices','7',
              '--audit-batch-size',str(args.audit_batch_size),'--groups-in-flight','32',
              '--local-generation-reference','--generation-mode','parallel-components',
              '--audit-policy','component-final-v2'],services=pools[:2])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['twins','multi'],default='twins')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True,
                   help='existing prepared group inputs; read only, never resumes its workers')
    p.add_argument('--archive-path-map',type=Path,
                   help='explicit relocation map for source inputs only; hashed in the new run contract')
    p.add_argument('--target',type=int,default=15000)
    p.add_argument('--max-attempts',type=int,default=120000)
    p.add_argument('--wave-size',type=int,default=128)
    p.add_argument('--max-waves',type=int,default=10000)
    p.add_argument('--selection-workers',type=int,default=8)
    p.add_argument('--planner-batch-size',type=int,default=16)
    p.add_argument('--planner-thinking-budget',type=int,default=512)
    p.add_argument('--audit-batch-size',type=int,default=8)
    p.add_argument('--mllm-python',required=True)
    p.add_argument('--editor-python',required=True)
    p.add_argument('--cache-root',type=Path,default=Path('/opt/tiger/sa1b-group-model-cache'))
    p.add_argument('--skip-model-cache',action='store_true')
    base='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/'
    p.add_argument('--model',default=base+'Qwen3.8-27B')
    p.add_argument('--editor-model',default=base+'Qwen-Image-2.1')
    p.add_argument('--site-model',default=base+'sam3')
    args=p.parse_args(argv)
    if min(args.target,args.max_attempts,args.wave_size,args.max_waves,args.selection_workers,
           args.planner_batch_size,args.audit_batch_size,args.planner_thinking_budget)<1:
        p.error('targets, limits, workers and batch sizes must be positive')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('unset CUDA_VISIBLE_DEVICES; this entry owns GPU 0 through 7')
    root=args.root.resolve(); source=args.source_root.resolve()
    if root==source or root.is_relative_to(source) or source.is_relative_to(root):
        p.error('use an independent output root, outside the frozen source run')
    root.mkdir(parents=True,exist_ok=True)
    def stop(signum,frame):raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    with owner_lock(root/'production.lock'):
        manifests=sorted((source/'inputs').glob('train_rank*.jsonl'))
        if not manifests:raise FileNotFoundError('prepared train_rank*.jsonl inputs are missing')
        reservation=source/'inputs/eval_reservation.json'
        repo=Path(__file__).resolve().parents[1]
        code_hash=digest({str(f.relative_to(repo)):file_hash(f) for folder in ('sa1b_pipeline','scripts')
                          for f in sorted((repo/folder).rglob('*.py'))})
        planner_options={'max_new_tokens':2300,'thinking_token_budget':args.planner_thinking_budget}
        relocation_contract = ({'source_path_map': {
            'path': str(args.archive_path_map.resolve()),
            'sha256': file_hash(args.archive_path_map)}}
            if args.archive_path_map else {})
        immutable_json(root/'config.json',{'version':'groups-27b-single-v3','kind':args.kind,'code_sha256':code_hash,'target':args.target,
            **relocation_contract,
            'inputs':{str(f):file_hash(f) for f in manifests},'reservation':file_hash(reservation),
            'model':args.model,'editor_model':args.editor_model,'site_model':args.site_model,
            'planner_batch_size':args.planner_batch_size,'audit_batch_size':args.audit_batch_size,'wave_size':args.wave_size,
            'planning_style':'content-v1','per_mask_planning':True,'planner_options':planner_options,
            'planning_contract':'per-mask-grounded-content-3',
            'audit_mode':'joint-observed-operation-v5' if args.kind=='twins' else 'component-final-v2',
            'generation_reference_frame':'crop-entity-v1',
            'instruction_protocol':'2.0','mask_semantics':'2.0','steps':40,
            'topology':{'planner':[0],'editor':[1,2,3,4,5,6],'blind_and_audit':[7]}})
        targets=target_cells(args.kind,args.target)
        candidates=load_source_candidates(manifests,args.archive_path_map)
        if len({r['source_id'] for r in candidates})!=len(candidates):
            raise ValueError('source shards contain duplicate source IDs')
        from sa1b_pipeline.source_isolation import assert_no_eval_leakage
        assert_no_eval_leakage(candidates,reservation)
        random.Random(20261017).shuffle(candidates)
        if not args.skip_model_cache:
            from sa1b_pipeline.model_cache import stage_model
            mapping={model:stage_model(model,args.cache_root,progress=lambda x:print(json.dumps(x),flush=True))
                     for model in (args.model,args.editor_model,args.site_model)}
            os.environ['SA1B_LOCAL_MODEL_MAP']=json.dumps(mapping)
        editors=ServicePool(python=args.editor_python,role='twins' if args.kind=='twins' else 'editor',devices=[str(i) for i in range(1,7)],
                            model=args.editor_model,directory=root/'services/editor')
        audits=ServicePool(python=args.mllm_python,role='audit',devices=['7'],model=args.model,
                           directory=root/'services/audit',batch_size=args.audit_batch_size)
        planners=ServicePool(python=args.mllm_python,role='planner',devices=['0'],model=args.model,
                             directory=root/'services/planner',batch_size=args.planner_batch_size,planner_options=planner_options)
        pools=(editors,audits,planners)
        pool=[];attempted=[];groups=[];cursor=0;start=time.monotonic();status='wave_limit'
        try:
            for wave_no in range(1,args.max_waves+1):
                accepted=accept(groups,targets,args.kind);state=describe(accepted,targets,args.kind)
                state.update(attempted=len(attempted),wave=wave_no,status='running',session_seconds=time.monotonic()-start)
                atomic_json(root/'progress.json',state);write_rows(root/'accepted.jsonl',accepted)
                print(json.dumps(state),flush=True)
                if state['complete']:status='automatic_quota_complete';break
                wave=root/'waves'/f'{wave_no:05d}';selected=wave/'inputs/selected.jsonl'
                if selected.exists():chosen=read_rows(selected)
                else:
                    limit=min(args.wave_size,args.max_attempts-len(attempted))
                    if limit<=0:status='attempt_budget_exhausted';break
                    while True:
                        chosen=choose(pool,attempted,accepted,targets,args.kind,limit)
                        if len(chosen)>=limit or cursor>=len(candidates):break
                        stop_at=min(len(candidates),cursor+64);cache=root/'pool'/f'{cursor:06d}.jsonl'
                        if cache.exists():additions=read_rows(cache)
                        else:
                            jobs=[(c,args.kind,[k for k,n in targets.items() if n],20261017+cursor+i)
                                  for i,c in enumerate(candidates[cursor:stop_at])]
                            with ProcessPoolExecutor(max_workers=args.selection_workers,mp_context=mp.get_context('spawn')) as ex:
                                additions=[r for result in ex.map(propose,jobs) for r in result]
                            write_rows(cache,additions)
                        pool.extend(additions);cursor=stop_at
                        atomic_json(root/'pool/progress.json',{'sources_scanned':cursor,'total_sources':len(candidates),'pool_rows':len(pool)})
                    if not chosen:status='candidate_pool_exhausted';break
                    write_rows(selected,chosen)
                # Completed waves are immutable; resident services survive all
                # new waves, but are never loaded just to replay checkpoints.
                if args.kind=='multi':
                    run_multi_wave(wave,args,reservation,pools)
                else:
                    run_stream(['--selected-manifest',str(selected),'--output-root',str(wave),
                        '--planner-model',args.model,'--audit-model',args.model,'--editor-model',args.editor_model,
                        '--site-model',args.site_model,'--reservation',str(reservation),'--local-generation-reference',
                        '--mllm-python',args.mllm_python,'--editor-python',args.editor_python,
                        '--twin-planning-style','content-v1','--per-mask-planning',
                        '--planner-thinking-budget',str(args.planner_thinking_budget),
                        '--planner-batch-size',str(args.planner_batch_size),'--audit-batch-size',str(args.audit_batch_size)],services=pools)
                audited=read_rows(wave/'audited.jsonl')
                for g in audited:
                    if passed(g,args.kind):
                        try:(validate_twin_record if args.kind=='twins' else validate_multi_record)(g,expected_split='train')
                        except (ValueError,KeyError,TypeError) as exc:
                            if args.kind=='twins':g['group_checks']['group_pass']=False
                            else:g['joint_audit']['pass']=False
                            g['export_validation']={'error':str(exc)}
                groups.extend(audited);attempted.extend(chosen)
        finally:
            for service in pools:service.close()
        accepted=accept(groups,targets,args.kind);report=describe(accepted,targets,args.kind)
        if report['complete']:status='automatic_quota_complete'
        report.update(status=status,attempted=len(attempted),session_seconds=time.monotonic()-start,published=False)
        write_rows(root/'accepted.jsonl',accepted);atomic_json(root/'progress.json',report)
        cache={};filename=f'{args.kind}_train.jsonl' if report['complete'] else f'{args.kind}_train.partial.jsonl'
        for g in accepted:(validate_twin_record if args.kind=='twins' else validate_multi_record)(g,expected_split='train')
        write_rows(root/'final'/filename,[portable(g,root,base=root,cache=cache) for g in accepted])
        atomic_json(root/'final/summary.json',{**report,'manifest':str(root/'final'/filename),'media_base':str(root)})
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
