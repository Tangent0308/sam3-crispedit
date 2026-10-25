#!/usr/bin/env python3
"""Full twin/multi validation sharing resident models, with resumable stages."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.inference_services import ServicePool,RemoteSelector
from sa1b_pipeline.production_state import atomic_json,read_rows,immutable_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-root',type=Path,required=True,help='Contains twins/inputs and multi/inputs selected.jsonl + manifest.jsonl')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--planner-model',required=True)
    p.add_argument('--editor-model',required=True)
    p.add_argument('--site-model',required=True)
    p.add_argument('--mllm-python',required=True)
    p.add_argument('--editor-python',required=True)
    p.add_argument('--reservation',type=Path,required=True)
    p.add_argument('--audit-policy',choices=['component-final-v2','group-final-v3','packed-components-v4'],default='packed-components-v4')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--planning-source-hints',choices=['none','category','annotation'],default='none')
    p.add_argument('--planner-thinking-budget',type=int,default=512)
    p.add_argument('--planner-max-tokens',type=int,default=1024)
    p.add_argument('--planner-devices',default='0',help='Independent TP=1 planner replicas, e.g. 0,1; released before editing')
    p.add_argument('--grounding-model', help='Optional SAM3 evidence service on GPU5; editors then use GPUs0..4')
    p.add_argument('--reference-alignment-repair', action='store_true',
                   help='Experimental: global SAM evidence on GPU5; all candidates considered; '
                        'one retry per target, capped at floor(25%% of batch targets) minus truncation retries; '
                        'revisions checked again, unresolved evidence is never semantic approval')
    p.add_argument('--lama-repair-model', help='Experimental: one LaMa proposal for small near-noop removals')
    p.add_argument('--restore-color-texture', action='store_true')
    p.add_argument('--kinds', default='twins,multi', help='Comma list: twins and/or multi (skip an empty kind)')
    p.add_argument('--text-verify', action='store_true',
                   help='Opt-in: explicit wording in planning, generator text rules, OCR-verified editing with '
                        'bounded retries and glyph repair (editor env needs RapidOCR in SA1B_OCR_LIB)')
    p.add_argument('--evidence-gate-add-site', action='store_true',
                   help='Opt-in with --evidence-gates: add Refs may name the attachment site on the anchor')
    p.add_argument('--replace-retry', action='store_true',
                   help='Opt-in with --evidence-gates: one re-plan for replace targets dropped as appearance-only')
    p.add_argument('--evidence-gate-medium', action='store_true',
                   help='Opt-in with --evidence-gates: drop additions inside pictures or on sculptures')
    p.add_argument('--reader-retry', action='store_true',
                   help='Opt-in with --evidence-gates, not recommended: one re-plan for targets whose only gate failure was '
                        'the reader check (run-4 wave-1 inputs: 2 of 5 rescued, +22 s planning)')
    p.add_argument('--stop-after-planning', action='store_true',
                   help='Experiments: stop after the planning stages (no editor/audit services)')
    p.add_argument('--evidence-gate-viewpoint', action='store_true',
                   help='Opt-in with --evidence-gates: drop additions in photos taken from high above (aerial)')
    p.add_argument('--evidence-gate-lighting', action='store_true',
                   help='Opt-in with --evidence-gates: drop colour edits of parts in shade or backlit')
    p.add_argument('--edit-finish', action='store_true',
                   help='Opt-in editor finishing: SAM3 (--site-model) shadow/reflection removal with removal prompt rules, '
                        'grain matching, one-item add cleanup, focus match for new items (focus_match)')
    p.add_argument('--audit-verifier', choices=['off', 'nothink', 'describe', 'think512', 'split'], default='off',
                   help='Opt-in: after the main audit passes, one focused yes/no defect check per target '
                        '(edit_verifier) on the audit service; a hard defect rejects the target. split: reasoning '
                        '(512) for remove/replace, none for add, no check for attribute (edit_verifier.SPLIT)')
    p.add_argument('--audit-ref-reconcile', action='store_true',
                   help='Opt-in: when the audit doubts a Ref, the planning reader settles it (repairs are reader-checked '
                        'with one short call; a reader-verified draft is kept)')
    p.add_argument('--audit-devices', default='6,7',
                   help='Physical GPUs of the generation-time audit service; editors use the remaining GPUs '
                        '(minus GPU5 with --grounding-model). Audits outweigh edits per wave with --audit-verifier.')
    p.add_argument('--generation-retry', action='store_true',
                   help='Opt-in, not recommended: a twin group rejected for image reasons gets one new generation (seed 1) '
                        'of its failed members and one new audit (twin_streaming.retry_plan); 1 of 5 retried groups passed')
    p.add_argument('--priority-scheduling', action='store_true',
                   help='Opt-in: requests go to the least loaded worker, and multi-target edits and audits (chains of '
                        'dependent steps) are served before twin requests; changes order only, not results')
    p.add_argument('--coordinated-quotas', action='store_true',
                   help='Waves balanced by an external coordinator (run_evidence_waves.py): multi planning skips its '
                        'per-wave aggregate operation-balance check (--dynamic-operation-quotas); per-group rules stay. '
                        'A production-size wave selects by remaining need, e.g. 139 groups with remove 36 below the mean')
    p.add_argument('--planning-view-workers', type=int, default=1,
                   help='Processes preparing planning views per kind before planning (same results; production: 16). '
                        'Serial preparation left the loaded planners idle for 422 s in a 239-group twin wave')
    p.add_argument('--concurrent-kinds', action='store_true',
                   help='Plan twins and multi at the same time, then generate them at the same time, on the shared '
                        'services (stages otherwise run one after another)')
    p.add_argument('--keep-singles', action='store_true',
                   help='Opt-in: keep members of non-accepted twin groups and the last target of a multi group that '
                        'pass every check of their own as single-target samples (singles.jsonl)')
    p.add_argument('--twin-single-add', action='store_true',
                   help='Opt-in: twin add members add one compact item each')
    p.add_argument('--garment-extent-check', action='store_true',
                   help='Opt-in: drop replacements of shorts/short skirts by long trousers or long dresses at planning')
    p.add_argument('--evidence-gates', action='store_true',
                   help='Opt-in: independent reader/identity/payload checks on the resident planner without reasoning; '
                        'inputs may over-generate a third twin candidate (twin_keep=2)')
    args=p.parse_args()
    planner_devices=args.planner_devices.split(',')
    if not planner_devices or len(set(planner_devices))!=len(planner_devices) or any(d not in list('01234567') for d in planner_devices):
        p.error('planner devices must be distinct physical GPU IDs 0..7')
    if args.reference_alignment_repair and '5' in planner_devices:
        p.error('reference alignment reserves GPU5 during planning')
    kinds=[k for k in args.kinds.split(',') if k]
    if not kinds or set(kinds)-{'twins','multi'}:
        p.error('--kinds must list twins and/or multi')
    if args.planner_thinking_budget < 1 or args.planner_max_tokens < args.planner_thinking_budget+256:
        p.error('planner budget must be positive with at least 256 final-answer tokens')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('unset CUDA_VISIBLE_DEVICES; this entry uses physical GPUs 0..7')
    root=args.output_root;root.mkdir(parents=True,exist_ok=True)
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OMP_NUM_THREADS='4')
    from scripts import run_group_pipeline,run_twins_streaming,run_multi_transactional
    from scripts.run_group_production_4node import owner_lock
    with owner_lock(root/'dual.lock'):
        contract={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  'code':{str(f.relative_to(Path(__file__).resolve().parents[1])):file_hash(f)
                          for folder in ('sa1b_pipeline','scripts','utils')
                          for f in (Path(__file__).resolve().parents[1]/folder).rglob('*.py')}}
        immutable_json(root/'dual_contract.json',contract)
        timings=json.loads((root/'timings.json').read_text()) if (root/'timings.json').exists() else {}
        timing_lock=threading.Lock()
        def stage(name,fn):
            with timing_lock:
                if name in timings and timings[name].get('complete'):return
            t=time.monotonic();fn()
            with timing_lock:
                timings[name]={'seconds':time.monotonic()-t,'complete':True}
                atomic_json(root/'timings.json',timings)
            print(json.dumps({'stage':name,**timings[name]}),flush=True)
        def run_kinds(jobs):
            # jobs: [(stage name, fn)]; with --concurrent-kinds both kinds share the services at once.
            if not args.concurrent_kinds or len(jobs)<2:
                for name,fn in jobs: stage(name,fn)
                return
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                for future in [pool.submit(stage,name,fn) for name,fn in jobs]:
                    future.result()
        start=time.monotonic()
        planners=ServicePool(python=args.mllm_python,role='planner',devices=planner_devices,model=args.planner_model,
            directory=root/'services/planner',batch_size=16,planner_options={'max_new_tokens':args.planner_max_tokens,'thinking_token_budget':args.planner_thinking_budget})
        # Load the planner GPUs now, while planning prepares its views (workers otherwise start on the first
        # request); a resumed run whose planning is complete does not load them at all.
        if any(not timings.get(kind+'_planning',{}).get('complete') for kind in kinds):
            planners.start()
        selector = RemoteSelector(planners)
        reference_resolver = None
        try:
            if args.reference_alignment_repair:
                from sa1b_pipeline.site_segmentation import Sam3SiteResolver
                from sa1b_pipeline.reference_alignment import ReferenceAlignment
                reference_resolver = Sam3SiteResolver(args.grounding_model or args.site_model, device='cuda:5')
                selector.reference_inspector = ReferenceAlignment(
                    reference_resolver, root/'services/reference_alignment')
            planning_jobs=[]
            for kind in kinds:
                out=root/kind;inputs=out/'inputs';inputs.mkdir(parents=True,exist_ok=True)
                for name in ('selected.jsonl','manifest.jsonl'):
                    src=args.inputs_root/kind/'inputs'/name;dst=inputs/name
                    if dst.exists() and dst.read_bytes()!=src.read_bytes():raise ValueError('changed selected inputs')
                    if not dst.exists():shutil.copy2(src,dst)
                cmd=['--kind',kind,'--manifest',str(inputs/'manifest.jsonl'),'--output-root',str(out),
                    '--target',str(len(read_rows(inputs/'selected.jsonl'))),'--reservation',str(args.reservation),
                    '--outline-planning-crops','--locate-planning-scene','--structured-planning-json','--skip-unused-planning-views',
                    '--plan-only','--planner-model',args.planner_model,'--planner-tp','1','--planner-batch-size','16',
                    '--max-new-tokens',str(args.planner_max_tokens),'--planner-thinking-budget',str(args.planner_thinking_budget),'--per-mask-planning',
                    '--planning-operation-retry','0','--planning-source-hints',args.planning_source_hints,'--seed',str(args.seed),'--site-model',args.site_model,
                    '--planning-view-workers',str(args.planning_view_workers)]
                cmd+=['--twin-planning-style','content-v1','--quota-from-selection','--defer-blind'] if kind=='twins' else ['--multi-plan-style','content-v1']
                if kind=='multi' and args.coordinated_quotas:
                    cmd+=['--dynamic-operation-quotas']
                if args.evidence_gates:
                    cmd+=['--evidence-gates']
                if args.evidence_gate_add_site:
                    cmd+=['--evidence-gate-add-site']
                if args.evidence_gate_lighting:
                    cmd+=['--evidence-gate-lighting']
                if args.evidence_gate_medium:
                    cmd+=['--evidence-gate-medium']
                if args.evidence_gate_viewpoint:
                    cmd+=['--evidence-gate-viewpoint']
                if args.replace_retry:
                    cmd+=['--replace-retry']
                if args.reader_retry:
                    cmd+=['--reader-retry']
                if args.text_verify:
                    cmd+=['--explicit-text']
                if args.twin_single_add:
                    cmd+=['--twin-single-add']
                if args.garment_extent_check:
                    cmd+=['--garment-extent-check']
                # Planning stores per-run state (evidence gates, options) on its planner object:
                # one light selector per kind over the same service pool.
                kind_selector=RemoteSelector(planners)
                if getattr(selector,'reference_inspector',None) is not None:
                    kind_selector.reference_inspector=selector.reference_inspector
                planning_jobs.append((kind+'_planning',lambda cmd=cmd,sel=kind_selector:run_group_pipeline.main(cmd,planner_override=sel)))
            run_kinds(planning_jobs)
        finally:
            planners.close()
            if reference_resolver is not None:
                del selector.reference_inspector, reference_resolver
                import torch
                with torch.cuda.device(5):
                    torch.cuda.empty_cache()
        if args.stop_after_planning:
            atomic_json(root/'session.json',{'seconds':time.monotonic()-start,'timings':timings,'steps':40,
                                             'published':False,'stopped_after_planning':True})
            return
        audit_devices=[d for d in args.audit_devices.split(',') if d]
        if not audit_devices or len(set(audit_devices))!=len(audit_devices) or any(d not in list('01234567') for d in audit_devices):
            p.error('audit devices must be distinct physical GPU IDs 0..7')
        editor_devices=''.join(d for d in '01234567' if d not in audit_devices and not (args.grounding_model and d=='5'))
        if not editor_devices:
            p.error('no GPU left for editors')
        editor_tools = {}
        if args.lama_repair_model: editor_tools['lama_model'] = args.lama_repair_model
        if args.restore_color_texture: editor_tools['restore_texture'] = True
        if args.text_verify: editor_tools['text_verify'] = True
        if args.edit_finish:
            # Settings from the 2026-10-23 paired A/B (151 pairs): tone_match off (sky ghost),
            # no min_context crop growth (replace 5:2 against), no add/replace material sentence
            # (neutral/slightly negative). Removal shadow/reflection handling is the clear gain.
            # focus_match replaces defocus_match (2026-10-24 blind replay of 291 outputs).
            editor_tools['finish'] = {'removal_effects': args.site_model, 'regrain': True, 'tone_match': False,
                                      'add_cleanup': True, 'prompt_rules': True, 'focus_match': True}
        dispatch='least_loaded' if args.priority_scheduling else 'round_robin'
        editors=ServicePool(python=args.editor_python,role='editor',devices=list(editor_devices),model=args.editor_model,directory=root/'services/editor',editor_tools=editor_tools,dispatch=dispatch)
        audits=ServicePool(python=args.mllm_python,role='audit',devices=audit_devices,model=args.planner_model,directory=root/'services/audit',batch_size=8,dispatch=dispatch)
        editors.priority_scheduling=audits.priority_scheduling=args.priority_scheduling
        editors.generation_retry=args.generation_retry
        if args.keep_singles:
            audits.keep_singles=True
        if args.audit_ref_reconcile:
            audits.ref_reconcile=True
        if args.audit_verifier=='split':
            from sa1b_pipeline.edit_verifier import SPLIT
            audits.verifier={'describe':False,'thinking':512,'by_operation':dict(SPLIT)}
        elif args.audit_verifier!='off':
            audits.verifier={'describe':args.audit_verifier=='describe','thinking':512 if args.audit_verifier=='think512' else 0}
        grounder = None
        if args.grounding_model:
            from sa1b_pipeline.inference_services import RemoteGrounding
            grounder=ServicePool(python=args.mllm_python,role='grounding',devices=['5'],
                model=args.grounding_model,directory=root/'services/grounding',batch_size=1)
            audits.grounding=RemoteGrounding(grounder)
        try:
            if grounder:
                from sa1b_pipeline.grounding_evidence import preflight_groups
                from sa1b_pipeline.production_state import write_rows
                for kind in kinds:
                    def preflight(kind=kind):
                        out=root/kind
                        kept,rejected=preflight_groups(read_rows(out/'planned.jsonl'),kind,audits.grounding)
                        write_rows(out/'grounded_planned.jsonl',kept)
                        atomic_json(out/'source_grounding_rejected.json',rejected)
                    stage(kind+'_source_grounding',preflight)
            generation_jobs=[]
            for kind in kinds:
                manifest='grounded_planned.jsonl' if grounder else 'planned.jsonl'
                cmd=['--planned-manifest',str(root/kind/manifest),'--output-root',str(root/kind/'generation'),
                    '--editor-model',args.editor_model,'--audit-model',args.planner_model,'--editor-python',args.editor_python,
                    '--mllm-python' if kind=='twins' else '--audit-python',args.mllm_python,
                    '--editor-devices',','.join(editor_devices),'--audit-devices',','.join(audit_devices),'--audit-batch-size','8','--groups-in-flight','16','--local-generation-reference']
                if kind=='twins':
                    cmd+=['--site-model',args.site_model]
                    generation_jobs.append((kind+'_generation',lambda cmd=cmd:run_twins_streaming.main(cmd,services=(editors,audits,None))))
                else:
                    cmd+=['--generation-mode','parallel-components','--audit-policy',args.audit_policy,'--save-crop-evidence']
                    if args.keep_singles:
                        cmd+=['--keep-singles']
                    generation_jobs.append((kind+'_generation',lambda cmd=cmd:run_multi_transactional.main(cmd,services=(editors,audits))))
            run_kinds(generation_jobs)
        finally:
            editors.close();audits.close()
            if grounder:grounder.close()
        atomic_json(root/'session.json',{'seconds':time.monotonic()-start,'timings':timings,'steps':40,'published':False})


if __name__=='__main__':main()
