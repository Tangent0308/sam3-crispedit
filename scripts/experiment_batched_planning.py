"""Batched one-request-per-mask vLLM experiment with bounded reasoning comparison."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
from collections import Counter
from dataclasses import replace
from PIL import Image


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--budgets',default='0,256,512');p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--submission-window',type=int,default=0,
                help='Pending requests submitted together; GPU concurrency stays batch-size. '
                     'Larger windows require fixed-input/replay/add-first priorities.')
 p.add_argument('--model',default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')
 p.add_argument('--tensor-parallel-size',type=int,default=1)
 p.add_argument('--gpu-memory-utilization',type=float,default=.85)
 p.add_argument('--moe-backend',default=None,
                help='Optional runtime MoE kernel backend, recorded in run config')
 p.add_argument('--disable-flashinfer-autotune',action='store_true')
 p.add_argument('--engram-cpu-offload',action='store_true',
                help='Explicitly offload N-gram lookup table for supported models')
 p.add_argument('--reasoning-parser',choices=['qwen3','none'],default='qwen3',
                help='Use none for native non-thinking Instruct checkpoints')
 p.add_argument('--scene-max-edge',type=int,default=1280)
 p.add_argument('--crop-max-edge',type=int,default=896)
 p.add_argument('--fill-alpha',type=float,default=.50)
 p.add_argument('--priority-policy',choices=['replay','balanced','shared-balanced','fixed-input','add-first'],default='replay')
 p.add_argument('--operation-mode',choices=['fallback','assigned'],default='fallback',
                help='assigned permits the scheduled operation or no_edit only')
 p.add_argument('--shared-quota-db',type=Path,
                help='Local SQLite ledger shared by disjoint planner replicas; actual accepted plans plus in-flight preferences')
 p.add_argument('--skip-reason-max-chars',type=int,default=0,
                help='Optional decoding cap on auxiliary skipped-type reasons; instruction fields remain unrestricted')
 p.add_argument('--priority-seed',type=int,default=20261012)
 p.add_argument('--production-resume',action='store_true',help='Durable per-mask intent/results; uncertain interrupted requests are not resubmitted')
 p.add_argument('--operation-weights',help='JSON weights for assigned balanced scheduling, derived from final quota deficits')
 p.add_argument('--priority-replay-root',type=Path)
 p.add_argument('--view-hints',choices=['none','arrow','peer'],default='none')
 p.add_argument('--scene-mask-outline',action='store_true',
                help='Exact external mask contour in the full scene; target RGB stays unchanged')
 p.add_argument('--view-workers',type=int,default=1,
                help='CPU view preparation workers; model requests remain one per mask')
 p.add_argument('--source-cache-mb',type=int,default=512,
                help='Bounded decoded RGB cache; zero disables it, without changing model inputs')
 p.add_argument('--contrast-first',action='store_true',
                help='Decode target identity and nearest rival before the final reference')
 p.add_argument('--wide-context',action='store_true')
 p.add_argument('--crop-padding-factor',type=float,
                help='Override the crop context margin as a multiple of target extent')
 p.add_argument('--pixel-evidence',action='store_true')
 p.add_argument('--context-min-fraction',type=float,default=.18)
 p.add_argument('--context-min-pixels',type=int,default=0)
 p.add_argument('--prompt-style',choices=['compact','concise','binding','annotated','grounded','grounded-neutral','grounded-balanced','grounded-supported','options','natural-priority','retouch'],default='compact')
 p.add_argument('--structured-output',choices=['none','typed','typed-choice','typed-options'],default='none')
 p.add_argument('--view',choices=['dark','isolated','filled','dual-filled'],default='dark');a=p.parse_args()
 weights=json.loads(a.operation_weights) if a.operation_weights else None
 if weights is not None and (set(weights)!={'add','replace','remove','attribute'} or any(not isinstance(v,(int,float)) or not __import__('math').isfinite(v) or v<0 for v in weights.values()) or not any(weights.values()) or a.priority_policy!='balanced' or a.operation_mode!='assigned'):
  p.error('operation weights require four finite nonnegative values, positive sum, balanced assigned mode')
 if a.production_resume and (a.budgets!='256' or a.priority_policy!='balanced' or a.operation_mode!='assigned'):
  p.error('production journal supports the recommended single-budget assigned balanced recipe')
 if (a.prompt_style=='options') != (a.structured_output=='typed-options'):p.error('options prompt and typed-options decoding must be used together')
 if a.crop_max_edge!=896 and not a.wide_context:p.error('crop-max-edge currently requires wide-context')
 if a.view_hints!='none' and a.view!='dark':p.error('visual callouts currently require the dark context view')
 if a.scene_mask_outline and a.view!='dark':p.error('scene mask outline requires dark crop view')
 if a.scene_mask_outline and a.prompt_style not in ('grounded','grounded-balanced'):p.error('scene mask outline currently uses a grounded prompt')
 if a.reasoning_parser=='none' and any(int(b) for b in a.budgets.split(',')):p.error('no reasoning parser requires zero thinking budget')
 if not 0<a.gpu_memory_utilization<1:p.error('gpu memory utilization must be between zero and one')
 if a.view_workers<1:p.error('view-workers must be positive')
 if a.source_cache_mb<0:p.error('source-cache-mb must be nonnegative')
 if a.batch_size<1 or a.submission_window<0:p.error('invalid batch size or submission window')
 submission_window=a.submission_window or a.batch_size
 if submission_window<a.batch_size:p.error('submission window must be at least batch-size')
 if submission_window>a.batch_size and a.priority_policy in ('balanced','shared-balanced'):
  p.error('larger windows require fixed priorities; balanced mode needs completed-result feedback')
 if (a.priority_policy=='shared-balanced') != bool(a.shared_quota_db):p.error('shared-balanced requires --shared-quota-db, exclusively')
 if a.shared_quota_db and len(a.budgets.split(','))!=1:p.error('shared quota requires a single reasoning budget')
 if a.skip_reason_max_chars<0 or (a.skip_reason_max_chars and a.structured_output!='typed-choice'):
  p.error('skip reason cap must be nonnegative and requires typed-choice')
 if a.operation_mode=='assigned' and (a.structured_output!='typed-choice' or a.prompt_style!='grounded-balanced'):
  p.error('assigned mode requires grounded-balanced prompt and typed-choice decoding')
 if a.contrast_first and a.structured_output!='typed':p.error('contrast-first requires typed schema')
 if a.crop_padding_factor is not None and a.crop_padding_factor<0:p.error('crop-padding-factor must be nonnegative')
 crop_padding_factor=a.crop_padding_factor if a.crop_padding_factor is not None else (1.3 if a.wide_context else .45)
 if not 0<a.fill_alpha<1:p.error('fill-alpha must be between 0 and 1')
 if a.structured_output!='none' and a.prompt_style=='concise':p.error('typed schema requires the compact reference fields')
 single_phrase=a.prompt_style in {'grounded','grounded-neutral','grounded-balanced','grounded-supported','natural-priority','retouch'}
 from sa1b_pipeline import per_mask
 from sa1b_pipeline.planning_candidates import compact_prompt,concise_prompt,binding_prompt,annotated_prompt,grounded_prompt,grounded_neutral_prompt,grounded_balanced_prompt,grounded_supported_prompt,natural_priority_prompt,retouch_capability_prompt,expand_compact,CANDIDATE_VERSION
 from sa1b_pipeline.io import candidate_from_dict,write_jsonl
 from sa1b_pipeline.instructions import build_unit
 from sa1b_pipeline.schema import SelectionCase,validate_case
 from sa1b_pipeline.mllm import parse_json
 from transformers import AutoProcessor
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 from sa1b_pipeline.planning_schema import typed_compact_schema
 from sa1b_pipeline.planning_options import options_schema,options_prompt,select_option
 base=a.priority_replay_root or Path('/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/05_history/legacy/sa1b_edit_work/per_mask_paired_v2_4_20261012')
 replay={(r['source_id'],r['object_id']):r['operation_priority'] for r in map(json.loads,(base/'run/audit.jsonl').read_text().splitlines()) if r['stage']=='single_call_plan'} if a.priority_policy=='replay' else {}
 rows=[json.loads(l) for l in a.manifest.read_text().splitlines()];jobs=[]
 # Flat one-mask rows can repeat a source.  Preserve all supplied peers rather
 # than silently replacing the source's inventory with its final mask.
 source_objects={}
 for r in rows:
  source_objects.setdefault(r['source_id'],[]).extend(r['objects'])
 for i in range(max(len(r['objects']) for r in rows)):
  for r in rows:
   if i<len(r['objects']):jobs.append({**r,'objects':[r['objects'][i]]})
 job_keys=[(str(r['source_id']),str(r['objects'][0]['object_id'])) for r in jobs]
 if len(job_keys)!=len(set(job_keys)):raise ValueError('duplicate source/mask would cause repeated model calls')
 from sa1b_pipeline.operation_schedule import fixed_balanced_schedule
 fixed_priorities=fixed_balanced_schedule(
  [(r['source_id'],r['objects'][0]['object_id']) for r in jobs],a.priority_seed) if a.priority_policy=='fixed-input' else {}
 if a.root.exists() and not a.production_resume:raise ValueError('fresh root required')
 a.root.mkdir(parents=True,exist_ok=a.production_resume);snap=a.root/'code_snapshot';snap.mkdir(exist_ok=a.production_resume)
 for f in ['sa1b_pipeline/instruction_style.py','sa1b_pipeline/planning_candidates.py',__file__,'sa1b_pipeline/per_mask.py','sa1b_pipeline/image_cache.py','sa1b_pipeline/planning_schema.py','sa1b_pipeline/planning_visual_hints.py','sa1b_pipeline/operation_schedule.py','sa1b_pipeline/shared_operation_quota.py','sa1b_pipeline/planning_options.py']:
  dst=snap/Path(f).name
  if a.production_resume and dst.exists() and dst.read_bytes()!=Path(f).read_bytes():raise ValueError('planner code changed on resume')
  shutil.copy2(f,dst)
 journal=None
 if a.production_resume:
  from sa1b_pipeline.production_state import PlanningJournal
  contract={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
  contract['manifest_sha256']=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
  journal=PlanningJournal(a.root/'journal',contract,resume=True)
 model=a.model
 t=time.perf_counter();processor=AutoProcessor.from_pretrained(model)
 extra_engine={}
 if a.moe_backend is not None:extra_engine['moe_backend']=a.moe_backend
 if a.disable_flashinfer_autotune:extra_engine['enable_flashinfer_autotune']=False
 if a.engram_cpu_offload:extra_engine['engram_config']={'cpu_offload':True}
 llm=LLM(model=model,dtype='bfloat16',tensor_parallel_size=a.tensor_parallel_size,max_model_len=16384,max_num_seqs=a.batch_size,max_num_batched_tokens=8192,
  gpu_memory_utilization=a.gpu_memory_utilization,enforce_eager=True,trust_remote_code=True,limit_mm_per_prompt={'image':2,'video':0},
  mm_encoder_attn_backend='TORCH_SDPA',enable_prefix_caching=True,
  structured_outputs_config={'reasoning_parser':'qwen3','enable_in_reasoning':False} if a.structured_output!='none' and a.reasoning_parser=='qwen3' else None,
  reasoning_config={'reasoning_parser':'qwen3','reasoning_start_str':'<think>','reasoning_end_str':'</think>'} if a.reasoning_parser=='qwen3' else None,**extra_engine)
 load_seconds=time.perf_counter()-t
 for budget in map(int,a.budgets.split(',')):
  root=a.root/f'think_{budget}';root.mkdir(exist_ok=a.production_resume);(root/'run').mkdir(exist_ok=a.production_resume)
  version=f'{CANDIDATE_VERSION}-{a.prompt_style}-{a.view}-scene{a.scene_max_edge}-wide{int(a.wide_context)}-think-{budget}';per_mask.VERSION=version
  version+=f'-crop{a.crop_max_edge}-alpha{a.fill_alpha:g}-schema{a.structured_output}-pixels{int(a.pixel_evidence)}';per_mask.VERSION=version
  version+=f'-hints{a.view_hints}';per_mask.VERSION=version
  if a.contrast_first:version+='-contrastfirst';per_mask.VERSION=version
  if a.crop_padding_factor is not None:version+=f'-padding{crop_padding_factor:g}';per_mask.VERSION=version
  if a.scene_mask_outline:version+='-sceneoutline';per_mask.VERSION=version
  write_jsonl(root/'input.jsonl',rows)
  config={'version':version,'model':model,'tensor_parallel_size':a.tensor_parallel_size,'scene_max_edge':a.scene_max_edge,'backend':'vllm','view':a.view,'reasoning_budget':budget,'batch_size':a.batch_size,'one_request_per_mask':True,'structured_output':a.structured_output,'replayed_priorities':str(base),'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snap.iterdir()}}
  config.update(priority_policy=a.priority_policy,prompt_style=a.prompt_style,wide_context=a.wide_context,crop_max_edge=a.crop_max_edge,fill_alpha=a.fill_alpha,context_min_fraction=a.context_min_fraction,context_min_pixels=a.context_min_pixels,replayed_priorities=str(base) if a.priority_policy=='replay' else None)
  config['single_phrase_contract']=single_phrase
  config['answer_token_limit']=1000 if a.structured_output=='typed-options' else 650
  config['pixel_evidence']=a.pixel_evidence
  config['view_hints']=a.view_hints
  config['scene_mask_outline']=a.scene_mask_outline
  config['priority_seed']=a.priority_seed if a.priority_policy=='fixed-input' else None
  config['view_workers']=a.view_workers
  config['source_cache_mb']=a.source_cache_mb
  config['submission_window']=submission_window
  config['contrast_first']=a.contrast_first
  config['crop_padding_factor']=crop_padding_factor
  config['structured_reasoning_parser']='qwen3' if a.structured_output!='none' and a.reasoning_parser=='qwen3' else None
  config['reasoning_parser']=a.reasoning_parser
  config['gpu_memory_utilization']=a.gpu_memory_utilization
  config['extra_engine_options']=extra_engine
  config['limit_mm_per_prompt']={'image':2,'video':0}
  config['skip_reason_max_chars']=a.skip_reason_max_chars
  config['operation_mode']=a.operation_mode
  quota=None;quota_owner=str(root.resolve())
  if a.shared_quota_db:
   from sa1b_pipeline.shared_operation_quota import SharedOperationQuota
   quota=SharedOperationQuota(a.shared_quota_db,config)
   config['shared_quota_db']=str(a.shared_quota_db.resolve())
  (root/'run_config.json').write_text(json.dumps(config,indent=2))
  from sa1b_pipeline.image_cache import SourceImageCache
  source_cache=SourceImageCache(a.source_cache_mb*1024*1024)
  accepted=[];audits=[];batches=[];started=time.perf_counter();prepared=[];counts=Counter()
  previously_submitted=set()
  if journal is not None:
   for entry in journal.entries():
    previously_submitted.add(tuple(entry['key']))
    if entry['status']=='complete':
     audits.extend([entry['audit'],entry['terminal']])
     if entry['case'] is not None:
      accepted.append(entry['case']);counts[entry['case']['units'][0]['operation']]+=1
    else:
     audits.append({'source_id':entry['key'][0],'object_id':entry['key'][1],'stage':'terminal','status':'interrupted_unknown','reason':'request intent persisted; not retried'})
  remaining_jobs=[(i,j) for i,j in enumerate(jobs) if (str(j['source_id']),str(j['objects'][0]['object_id'])) not in previously_submitted]
  def prepare_job(index_job):
   index,job=index_job
   c=candidate_from_dict(job);o=c.objects[0]
   # Separate output directories prevent concurrent writes for duplicate masks.
   view_dir=root/'views'/f'job-{index:04d}' if a.view_workers>1 else root/'views'
   paths,geometry,mp,digest,box=per_mask.single_call_views(
    c,o,view_dir,crop_padding_factor=crop_padding_factor,
    crop_min_fraction=a.context_min_fraction if a.wide_context else 0.,
    crop_min_pixels=a.context_min_pixels if a.wide_context else 16,
    crop_max_edge=a.crop_max_edge,scene_max_edge=a.scene_max_edge,source_cache=source_cache)
   if a.view=='isolated':
    from PIL import ImageFilter
    import numpy as np
    panel=Image.open(paths[1]).convert('RGB');size=(panel.width,panel.height-38)
    original=panel.crop((0,38,panel.width,panel.height))
    bm=Image.open(mp).convert('L').crop(box).resize(size,Image.Resampling.NEAREST)
    crop=Image.new('RGB',size,(225,225,225));crop.paste(original,mask=bm)
    ring=(np.asarray(bm.filter(ImageFilter.MaxFilter(5)))>0)&(np.asarray(bm)==0)
    crop.paste((255,0,180),mask=Image.fromarray(ring.astype('uint8')*255));panel.paste(crop,(0,38));panel.save(paths[1],compress_level=1)
   if a.view in ['filled','dual-filled']:
    from PIL import ImageFilter,ImageDraw
    import numpy as np
    panel=Image.open(paths[1]).convert('RGB');size=(panel.width,panel.height-38)
    original=Image.open(c.image_path).convert('RGB').crop(box).resize(size,Image.Resampling.LANCZOS)
    bm=Image.open(mp).convert('L').crop(box).resize(size,Image.Resampling.NEAREST)
    crop=original.copy();tinted=Image.blend(original,Image.new('RGB',size,(255,0,160)),a.fill_alpha)
    crop.paste(tinted,mask=bm)
    ring=(np.asarray(bm.filter(ImageFilter.MaxFilter(5)))>0)&(np.asarray(bm)==0)
    crop.paste((255,0,160),mask=Image.fromarray(ring.astype('uint8')*255))
    panel.paste('white',(0,0,panel.width,38));ImageDraw.Draw(panel).text((8,8),'EDIT ONLY THE PINK-FILLED REGION',fill=(170,0,110))
    panel.paste(crop,(0,38));panel.save(paths[1],compress_level=1)
    if a.view=='dual-filled':
     scene=Image.open(c.image_path).convert('RGB');scene.thumbnail((a.scene_max_edge,a.scene_max_edge),Image.Resampling.LANCZOS)
     sm=Image.open(mp).convert('L').resize(scene.size,Image.Resampling.NEAREST)
     scene.paste(Image.blend(scene,Image.new('RGB',scene.size,(255,0,160)),.30),mask=sm)
     edge=(np.asarray(sm.filter(ImageFilter.MaxFilter(5)))>0)&(np.asarray(sm)==0)
     scene.paste((255,0,160),mask=Image.fromarray(edge.astype('uint8')*255));scene.save(paths[0],compress_level=1)
   w,h=Image.open(c.image_path).size
   if a.scene_mask_outline:
    from sa1b_pipeline.planning_visual_hints import annotate_full_scene
    annotate_full_scene(c.image_path,mp,paths[0],a.scene_max_edge)
   if a.view_hints!='none':
    from sa1b_pipeline.planning_visual_hints import annotate_crop
    original_obj=next(x for x in source_objects[c.source_id] if str(x['object_id'])==o.object_id)
    evidence=annotate_crop(paths[1],mp,(w,h),box,original_obj,source_objects[c.source_id],a.view_hints)
    o.metadata={**o.metadata,'visual_callout':evidence}
   o.metadata={**o.metadata,'full_scene_geometry':{'target_bbox_xyxy_fraction':[round(v/d,3) for v,d in zip(geometry['bbox'],[w,h,w,h])],'crop_bbox_xyxy_fraction':[round(v/d,3) for v,d in zip(box,[w,h,w,h])]}}
   if a.pixel_evidence:
    with Image.open(c.image_path) as raw_image, Image.open(mp) as raw_mask:
     o.metadata['selected_pixel_evidence']=per_mask.selected_pixel_evidence(raw_image,raw_mask)
   return (index,c,o,paths,geometry,mp,digest,box)
  if a.view_workers==1:
   prepared=list(map(prepare_job,remaining_jobs))
  else:
   from concurrent.futures import ThreadPoolExecutor
   with ThreadPoolExecutor(max_workers=a.view_workers) as pool:
    # map preserves the source/mask order and therefore the type schedule.
    prepared=list(pool.map(prepare_job,remaining_jobs))
  view_preparation_seconds=time.perf_counter()-started
  structured=StructuredOutputsParams(json=typed_compact_schema(contrast_first=a.contrast_first,single_phrase=single_phrase)) if a.structured_output=='typed' else None
  if a.structured_output=='typed-options':structured=StructuredOutputsParams(json=options_schema())
  params=SamplingParams(temperature=0,top_p=1,max_tokens=budget+config['answer_token_limit'],thinking_token_budget=budget if budget else None,structured_outputs=structured)
  for start in range(0,len(prepared),submission_window):
   batch=[];reserved=counts.copy();ops=['add','replace','remove','attribute']
   for index,c,o,paths,g,mp,digest,box in prepared[start:start+submission_window]:
    if a.priority_policy=='replay':priority=replay[c.source_id,o.object_id]
    elif a.priority_policy=='fixed-input':priority=fixed_priorities[c.source_id,o.object_id]
    elif a.priority_policy=='add-first':priority=['add','replace','remove','attribute']
    elif quota is not None:priority=quota.reserve(json.dumps([c.source_id,o.object_id]),digest,quota_owner)
    else:priority=sorted([op for op in ops if weights is None or weights[op]>0],key=lambda op:(reserved[op]/(weights[op] if weights else 1),ops.index(op)))
    reserved[priority[0]]+=1
    if a.operation_mode=='assigned':priority=priority[:1]
    prompt={'compact':compact_prompt,'concise':concise_prompt,'binding':binding_prompt,'annotated':annotated_prompt,'grounded':grounded_prompt,'grounded-neutral':grounded_neutral_prompt,'grounded-balanced':grounded_balanced_prompt,'grounded-supported':grounded_supported_prompt,'options':options_prompt,'natural-priority':natural_priority_prompt,'retouch':retouch_capability_prompt}[a.prompt_style](o,priority)
    if a.operation_mode=='assigned':
     from sa1b_pipeline.planning_candidates import assigned_operation_prompt
     prompt=assigned_operation_prompt(o,priority)
    if a.scene_mask_outline:
     prompt=prompt.replace('Image 1 is the full scene with a cyan locating rectangle.',
      'Image 1 is the full scene with the EXACT same target mask outlined in magenta/white. '
      'The full-scene contour follows the actual selected pixels, including holes, not a rectangular selection. '
      'Target pixels keep their original appearance in both images. Use the full-scene contour to locate '
      'this exact instance among its neighbours, then write a reference valid without any annotation.')
    if a.contrast_first:
     ref_order='full_ref' if single_phrase else 'ref_core, distinguish'
     prompt+=f'\nOUTPUT ORDER: kind, scope, source_color, rival, {ref_order}, op, change, optional keep. Identify the visible target and closest confusable rival before writing the minimal final reference. The final reference must exclude that rival using verified features; do not leave essential distinctions only in rival. Return one plan.'
    if a.view_hints!='none':
     from sa1b_pipeline.planning_visual_hints import visual_hint_prompt
     prompt=visual_hint_prompt(prompt,o.metadata['visual_callout'])
    if a.structured_output=='typed-choice':
     prompt+='\nAfter op and before change, include skipped: an object mapping EACH earlier type in the supplied priority list to one brief factual reason it cannot yield a natural edit here. Use {} when taking the first type. For no_edit, cover all listed types. Do not skip a type merely because recolouring is easier, the object is small, or a new silhouette differs from the mask. A detachable shirt button, logo, background tree or empty box can be removed naturally; the ban on removing anatomy does not extend to those objects. Never invent a reason to fit a preferred answer. These are concise decision summaries, not a reasoning transcript.'
     if a.skip_reason_max_chars:
      prompt+=f' Each skipped reason must be one short factual clause, at most {a.skip_reason_max_chars} characters. State it once without repeating or expanding it; then continue to change.'
    if a.view in ['filled','dual-filled']:
     first,rest=prompt.split('\n\n',1)
     prompt=('Write one realistic local photo-edit request for the supplied mask. Exactly TWO images: '
      'Image 1 is the full original scene with a cyan locating rectangle. Image 2 is an enlarged crop: '
      'the target pixels have a translucent PINK FILL plus a pink boundary. Only PINK-FILLED pixels are selected; '
      'holes and untinted neighboring objects are excluded. Pink is an artificial annotation, NOT the target color; '
      'read the actual color from Image 1. Identify the pink-filled target in Image 2 first, then locate that same '
      'instance in Image 1 and distinguish its closest similar neighbor.\n\n'+rest)
     if a.view=='dual-filled':
      prompt=prompt.replace('Image 1 is the full original scene with a cyan locating rectangle.',
       'Image 1 is the full scene with the SAME target mask lightly pink-tinted and outlined; all surrounding objects retain original colors.')
      prompt=prompt.replace('read the actual color from Image 1.',
       'infer the original color using the weak tint in Image 1 and the scene context; source_color may be null if uncertain.')
    images=[Image.open(path).convert('RGB') for path in paths]
    content=[{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt}]
    rendered=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=bool(budget))
    batch.append((index,c,o,priority,paths,g,mp,digest,box,prompt,{'prompt':rendered,'multi_modal_data':{'image':images}}))
   t=time.perf_counter()
   batch_params=[SamplingParams(temperature=0,top_p=1,max_tokens=budget+650,
    thinking_token_budget=budget if budget else None,
    structured_outputs=StructuredOutputsParams(json=typed_compact_schema(r[3],single_phrase=single_phrase,skip_reason_max_chars=a.skip_reason_max_chars))) for r in batch] if a.structured_output=='typed-choice' else params
   if journal is not None:journal.submit([[r[1].source_id,r[2].object_id] for r in batch])
   outputs=llm.generate([r[-1] for r in batch],batch_params,use_tqdm=False);dt=time.perf_counter()-t
   if len(outputs)!=len(batch):raise RuntimeError('planner returned an incomplete batch')
   batches.append({'start':start,'count':len(batch),'seconds':dt})
   for info,output in zip(batch,outputs):
    index,c,o,priority,paths,g,mp,digest,box,prompt,_=info
    generated=output.outputs[0];answer=generated.text.rsplit('</think>',1)[-1]
    try:
     parsed=parse_json(answer)
     raw=select_option(parsed,o.object_id,priority) if a.structured_output=='typed-options' else expand_compact(parsed,o.object_id)
    except ValueError as exc:raw={'_parse_error':str(exc)}
    metrics={'input_tokens':len(output.prompt_token_ids or []),'output_tokens':len(generated.token_ids),'answer_tokens':len(processor.tokenizer.encode(answer,add_special_tokens=False)),'reasoning_end_marker_present':'</think>' in generated.text,'finish_reason':generated.finish_reason,'batch_seconds':dt,'batch_size':len(batch),'thinking_budget':budget}
    raw['_generation_metrics']=metrics;raw['_raw_output']=answer
    audit={'source_id':c.source_id,'object_id':o.object_id,'planner_version':version,'stage':'single_call_plan','images':paths,'prompt':prompt,'result':raw,'seconds':dt/len(batch),'seconds_kind':'amortized_batch_wall','binding_hash':digest,'preferred_operation':priority[0],'operation_priority':priority,'evaluation_job_index':index}
    terminal={'source_id':c.source_id,'object_id':o.object_id,'planner_version':version,'stage':'terminal','evaluation_job_index':index}
    accepted_case=None
    try:
     per_mask.validate_single_plan(raw,o.object_id,priority)
     unit=build_unit(raw,{**o.__dict__,'category':raw['category']},mp,g,[])
     c.image_sha256=per_mask.sha256(c.image_path)
     source=replace(c,objects=[replace(o,mask_path=mp)],provenance={**c.provenance,'image_size':list(Image.open(c.image_path).size),'planning_mode':version})
     case=SelectionCase('sa1b-'+c.source_id+'-mask-'+hashlib.sha256(o.object_id.encode()).hexdigest()[:12],source,[unit],{'target_count':1,'target_features':[g]},{'planning_mode':version,'model_plan':raw,'model_id':model,'binding_hash':digest,'crop_box':box,'model_calls':1,'semantic_audit_performed':False,'joint_multi_target':False,'preferred_operation':priority[0],'operation_priority':priority,'operation_fallback':unit.operation!=priority[0]})
     validate_case(case,1,1);accepted.append(case.to_dict());counts[unit.operation]+=1;terminal.update(status='accepted',operation=unit.operation)
     accepted_case=case.to_dict()
    except (ValueError,KeyError,TypeError) as exc:terminal.update(status='rejected',reason=str(exc))
    if journal is not None:journal.finish([c.source_id,o.object_id],audit,terminal,accepted_case)
    audits.extend([audit,terminal]);jobroot=root/'jobs'/f'{index:04d}';jobroot.mkdir(parents=True,exist_ok=a.production_resume)
    write_jsonl(jobroot/'audit.jsonl',[audit,terminal])
    if quota is not None:quota.finish(json.dumps([c.source_id,o.object_id]),quota_owner,terminal.get('operation'))
   write_jsonl(root/'planned.jsonl',accepted);write_jsonl(root/'run/audit.jsonl',audits)
   print(json.dumps({'budget':budget,'completed':start+len(batch),'accepted':len(accepted),'batch_seconds':dt}),flush=True)
  write_jsonl(root/'planned.jsonl',accepted);write_jsonl(root/'run/audit.jsonl',audits)
  summary={'input_masks':len(jobs),'model_calls':sum(r['stage']=='single_call_plan' for r in audits),'interrupted_unknown':sum(r.get('status')=='interrupted_unknown' for r in audits),'accepted_masks':len(accepted),'accepted_operations':dict(Counter(c['units'][0]['operation'] for c in accepted)),'wall_seconds':time.perf_counter()-started,'model_load_seconds':load_seconds,'batches':batches,'target_images_generated':False,'assistant_visual_review_completed':False}
  summary.update(view_preparation_seconds=view_preparation_seconds,
                 inference_batch_seconds=sum(b['seconds'] for b in batches),
                 source_image_cache=source_cache.stats())
  summary['request_preparation_and_postprocessing_seconds']=summary['wall_seconds']-summary['view_preparation_seconds']-summary['inference_batch_seconds']
  if quota is not None:
   summary['shared_quota_snapshot']=quota.counts();quota.close()
  if journal is not None:
   from sa1b_pipeline.production_state import atomic_json
   atomic_json(root/'run/summary.json',summary)
  else:(root/'run/summary.json').write_text(json.dumps(summary,indent=2))
 print('completed all budgets',flush=True)
if __name__=='__main__':main()
