"""Freeze fresh source-disjoint dev/holdout regions BEFORE model planning."""
import argparse
import hashlib
import io
import json
import random
import os
from collections import Counter
from pathlib import Path

from PIL import Image
from synthesis_pipeline.prepare_samtok_data import load_jsonl,write_jsonl,decode_rle,resize_mask,encode_rle,qwen_canvas_size
from synthesis_pipeline.generate_samtok_plan import normalized_area
from synthesis_pipeline.reference_binding import bind_reference
from synthesis_pipeline.split_audit_eval import split_rows


def pixel_hash(data):
    with Image.open(io.BytesIO(data)) as im:
        im=im.convert('RGB')
        return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()


def history_files(root):
    """Scan manifests once; avoid repeated network walks through image assets."""
    names={'annotations.jsonl','input_annotations.jsonl','candidate_source_rows.jsonl',
           'sampled_source_rows.jsonl','source_hash_cache.jsonl','candidate_pixel_hash_cache.jsonl'}
    found={name:[] for name in names}
    # These directories store images or rendered presentation assets, never
    # sampling manifests. All cohort/planner/audit directories remain visible.
    assets={'sources','edited','inputs','diagnostics','support','crops',
            '__pycache__','.git'}
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in assets]
        for name in names.intersection(files):found[name].append(Path(directory)/name)
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet',type=Path,required=True)
    p.add_argument('--positive-index',type=Path,required=True)
    p.add_argument('--history-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--mask-counts',default='2',help='Allowed source region counts; every selected region is reused')
    p.add_argument('--task-types',default='add,remove,replace,attribute',
                   help='Frozen task cycle; use remove for a removal-only diagnostic cohort')
    p.add_argument('--sources-per-stratum',type=int,default=2,
                   help='Sources per dataset/area quintile; total regions = 20 * this value')
    args=p.parse_args()
    mask_counts={int(x) for x in args.mask_counts.split(',')}
    if not mask_counts or min(mask_counts)<1:p.error('Positive mask counts required')
    task_types=args.task_types.split(',')
    if not task_types or any(t not in {'add','remove','replace','attribute'} for t in task_types):
        p.error('Invalid task-types cycle')
    if args.sources_per_stratum<1:p.error('sources-per-stratum must be positive')
    if args.out_root.exists():raise FileExistsError(args.out_root)
    excluded=set();known_hashes={};manifests=[]
    history=history_files(args.history_root)
    for name in ['annotations.jsonl','input_annotations.jsonl','candidate_source_rows.jsonl',
                 'sampled_source_rows.jsonl','source_hash_cache.jsonl']:
        for file in history[name]:
            for row in load_jsonl(file):
                if 'parquet_row_index' in row:
                    index=int(row['parquet_row_index']);excluded.add(index)
                    sha=row.get('source_pixel_sha256')
                    if sha:
                        previous=known_hashes.setdefault(index,sha)
                        if previous!=sha:
                            raise ValueError(f'Conflicting source hash for parquet row {index}')
            manifests.append(str(file))
    # Hashes computed while probing earlier candidate pools are reusable but
    # do not imply that those rows were selected.  Keep this cache separate
    # from the legacy source_hash_cache, whose rows are historical exclusions.
    for file in history['candidate_pixel_hash_cache.jsonl']:
        for row in load_jsonl(file):
            if 'parquet_row_index' not in row or not row.get('source_pixel_sha256'):
                continue
            index=int(row['parquet_row_index']);sha=row['source_pixel_sha256']
            previous=known_hashes.setdefault(index,sha)
            if previous!=sha:
                raise ValueError(f'Conflicting candidate source hash for parquet row {index}')
        manifests.append(str(file))
    rows=load_jsonl(args.positive_index)
    import pyarrow.parquet as pq
    images=pq.read_table(args.parquet,columns=['images'])['images']
    missing_hashes=sorted(excluded-set(known_hashes))
    reused_hash_count=len(known_hashes)
    for index in missing_hashes:
        known_hashes[index]=pixel_hash(images[index].as_py()[0]['bytes'])
    used_hashes=set(known_hashes.values())
    prior_hashes=set(used_hashes)
    rng=random.Random(args.seed);selected=[];duplicates=0;selection_diagnostics={}
    pools={};all_selected_indices=set()

    def source_hash(row):
        index=int(row['parquet_row_index'])
        sha=known_hashes.get(index)
        if sha is None:
            data=images[index].as_py()[0]['bytes'];sha=pixel_hash(data)
            known_hashes[index]=sha
        return sha

    def persist_candidate_hashes(path):
        write_jsonl(path,[
            {'parquet_row_index':index,'source_pixel_sha256':sha}
            for index,sha in sorted(known_hashes.items())
        ])

    for subset in ['gres','ver']:
        pool=[r for r in rows if r['source_subset']==subset and r['num_masks'] in mask_counts and r['parquet_row_index'] not in excluded]
        pool.sort(key=lambda r:min(normalized_area(m) for m in r['masks']))
        pools[subset]=pool
        subset_selected=[];selected_indices=set();shortfalls={}
        for stratum in range(5):
            bucket=pool[round(len(pool)*stratum/5):round(len(pool)*(stratum+1)/5)]
            rng.shuffle(bucket);chosen=0
            for row in bucket:
                sha=source_hash(row)
                if sha in used_hashes:duplicates+=1;continue
                item={**row,'source_pixel_sha256':sha,'area_stratum':stratum,
                      'selection_mode':'stratum_quota'}
                selected.append(item);subset_selected.append(item)
                selected_indices.add(int(row['parquet_row_index']))
                all_selected_indices.add(int(row['parquet_row_index']))
                used_hashes.add(sha);chosen+=1
                if chosen==args.sources_per_stratum:break
            if chosen!=args.sources_per_stratum:
                shortfalls[str(stratum)]=args.sources_per_stratum-chosen

        # A heavily reused stratum can run out of source-disjoint images late
        # in iteration. Preserve the requested dataset quota by backfilling
        # from any remaining area stratum instead of failing or reusing data.
        target=5*args.sources_per_stratum
        remaining=[r for r in pool if int(r['parquet_row_index']) not in selected_indices]
        rng.shuffle(remaining)
        for row in remaining:
            if len(subset_selected)>=target:break
            sha=source_hash(row)
            if sha in used_hashes:duplicates+=1;continue
            area=min(normalized_area(mask) for mask in row['masks'])
            rank=sum(min(normalized_area(mask) for mask in candidate['masks']) <= area
                     for candidate in pool)
            stratum=min(4,max(0,int(5*rank/max(len(pool),1))))
            item={**row,'source_pixel_sha256':sha,'area_stratum':stratum,
                  'selection_mode':'subset_backfill'}
            selected.append(item);subset_selected.append(item)
            selected_indices.add(int(row['parquet_row_index']))
            all_selected_indices.add(int(row['parquet_row_index']));used_hashes.add(sha)
        selection_diagnostics[subset]={
            'target_sources':target,
            'stratum_shortfalls':shortfalls,
            'backfilled_sources':sum(r['selection_mode']=='subset_backfill'
                                     for r in subset_selected),
            'actual_area_strata':dict(Counter(r['area_stratum'] for r in subset_selected)),
            'selected_before_global_backfill':len(subset_selected),
        }

    # If one dataset is exhausted, preserve the total new-source budget from
    # the other dataset.  This is preferable to reusing an old image or
    # silently shrinking the evaluation batch.
    total_target=10*args.sources_per_stratum
    global_remaining=[]
    for subset,pool in pools.items():
        size=max(len(pool),1)
        for rank,row in enumerate(pool):
            index=int(row['parquet_row_index'])
            if index in all_selected_indices:continue
            global_remaining.append((row,min(4,int(5*rank/size))))
    rng.shuffle(global_remaining)
    for row,stratum in global_remaining:
        if len(selected)>=total_target:break
        sha=source_hash(row)
        if sha in used_hashes:duplicates+=1;continue
        item={**row,'source_pixel_sha256':sha,'area_stratum':stratum,
              'selection_mode':'global_backfill'}
        selected.append(item);all_selected_indices.add(int(row['parquet_row_index']))
        used_hashes.add(sha)
    shared_cache=args.history_root/'candidate_pixel_hash_cache.jsonl'
    persist_candidate_hashes(shared_cache)
    if len(selected)!=total_target:
        raise ValueError(
            f'Insufficient source-disjoint rows globally: needed {total_target}, '
            f'found {len(selected)}; diagnostics={selection_diagnostics}'
        )
    for subset in ['gres','ver']:
        chosen=[r for r in selected if r['source_subset']==subset]
        selection_diagnostics[subset]['selected_after_global_backfill']=len(chosen)
        selection_diagnostics[subset]['global_backfilled_sources']=sum(
            r['selection_mode']=='global_backfill' for r in chosen
        )
        selection_diagnostics[subset]['final_area_strata']=dict(
            Counter(r['area_stratum'] for r in chosen)
        )
    args.out_root.mkdir(parents=True);(args.out_root/'sources').mkdir()
    write_jsonl(args.out_root/'source_hash_cache.jsonl',[
        {'parquet_row_index':index,'source_pixel_sha256':sha}
        for index,sha in sorted(known_hashes.items())
        if index in excluded
    ])
    persist_candidate_hashes(args.out_root/'candidate_pixel_hash_cache.jsonl')
    records=[]
    for si,row in enumerate(selected):
        idx=row['parquet_row_index'];source_name=f"source_{row['source_subset']}_r{idx}.png"
        with Image.open(io.BytesIO(images[idx].as_py()[0]['bytes'])) as im:
            source=im.convert('RGB');size=qwen_canvas_size(*source.size)
            source=source.resize(size,Image.Resampling.LANCZOS)
            source.save(args.out_root/'sources'/source_name)
        for mi,mask in enumerate(row['masks']):
            case_id=len(records);task=task_types[case_id%len(task_types)]
            records.append(dict(image=f"{case_id:03d}_{row['source_subset']}_r{idx}_m{mi}_{task}.png",
                source_image=source_name,source_subset=row['source_subset'],parquet_row_index=idx,mask_index=mi,num_masks=row['num_masks'],
                source_pixel_sha256=row['source_pixel_sha256'],area_stratum=row['area_stratum'],
                mask=encode_rle(resize_mask(decode_rle(mask),size)),task_type=task,editing_instruction='',
                problem=row['problem'],answer=row['answer'],reference_binding=bind_reference(row['answer'],mi,row['num_masks'])))
    dev,holdout=split_rows(records,args.seed,.6)
    for name,subset in [('dev',dev),('holdout',holdout)]:
        root=args.out_root/name;root.mkdir();(root/'sources').symlink_to((args.out_root/'sources').resolve())
        write_jsonl(root/'annotations.jsonl',subset);write_jsonl(root/'input_annotations.jsonl',subset)
    write_jsonl(args.out_root/'annotations.jsonl',records)
    write_jsonl(args.out_root/'source_selection.jsonl',selected)
    summary=dict(seed=args.seed,task_types=task_types,cases=len(records),fresh_sources=len(selected),excluded_history_rows=len(excluded),
        history_hashes_reused=reused_hash_count,history_hashes_computed=len(missing_hashes),
        prior_unique_source_pixels=len(prior_hashes),duplicate_images_skipped=duplicates,
        selection_diagnostics=selection_diagnostics,
        history_manifests=manifests,bindings=dict(Counter(r['reference_binding']['status'] for r in records)),
        source_overlap_with_history=0,
        splits={name:dict(cases=len(v),types=dict(Counter(r['task_type'] for r in v))) for name,v in [('dev',dev),('holdout',holdout)]})
    (args.out_root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='history_manifests'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
