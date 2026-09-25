"""Reproducible paired editing ablations; reuse a model across a shard."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from tqdm import tqdm

from synthesis_pipeline.audit_edit_pairs import mask_array
from utils.context_edit import edit_context_crop, protected_neighbors, validate_refinement_execution

DEV_IDS = [0, 10, 20, 72, 1, 15, 25, 67, 34, 40, 44, 64, 13, 19, 23, 83]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--variant",
        choices=[
            "mirage_relaxed",
            "official_full",
            "context_edit",
            "context_removewide",
            "context_guided",
            "context_guided_attribute",
            "context_guided_attribute_v2",
            "context_guarded",
            "context_guarded_v2",
            "context_grounded_v3",
            "context_grounded_v4",
            "context_grounded_v3_qwen21",
            "context_grounded_v4_qwen21",
        ],
        required=True,
    )
    p.add_argument("--ids", default=",".join(map(str, DEV_IDS)))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument('--qwen21-prompt-policy', choices=['legacy', 'typed-v1','remove-shadow-v2','remove-evidence-v1','remove-parts-v1','remove-context-v1','relation-compact-v1','relation-spatial-v1','relation-located-v2','relation-action-v3'], default='legacy')
    p.add_argument('--qwen21-backend', choices=['diffusers', 'vllm-omni'], default='diffusers',
                   help='Qwen-Image-2.1 execution backend; vllm-omni uses its official Omni API')
    p.add_argument('--qwen21-target-guide',action='store_true',help='Add aligned boundary guide as second condition; clean image remains first')
    p.add_argument('--removal-conditioning',choices=['source','erase-neutral-v1','erase-prefill-v1','erase-neutral-v2'],default='source')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--remove-composition-policy',choices=['legacy','adaptive-remove-v1','adaptive-remove-v2','adaptive-remove-v3','adaptive-remove-v4','adaptive-remove-v5'],default='legacy')
    p.add_argument('--latent-protection-policy',choices=['legacy','guard-any-v1','guard-fraction-v1'],default='legacy')
    p.add_argument('--relation-geometry-policy',choices=['legacy','visible-v1'],default='legacy')
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="Export selected annotations without loading a model",
    )
    p.add_argument(
        "--model-id",
        default=None,
    )
    args = p.parse_args()
    wanted = {int(x) for x in args.ids.split(",")}
    rows = [
        json.loads(x)
        for x in (args.data_root / "annotations.jsonl").read_text().splitlines()
        if x.strip()
    ]
    rows = [r for r in rows if int(r["image"].split("_")[0]) in wanted]
    if len(rows) != len(wanted):
        raise ValueError("Missing requested cases")
    out = args.out_root / args.variant
    (out / "edited").mkdir(parents=True, exist_ok=True)
    if args.manifest_only and not (out / 'sources').exists():
        (out / 'sources').symlink_to((args.data_root / 'sources').resolve())
    if args.manifest_only:
        from synthesis_pipeline.prepare_samtok_data import write_jsonl
        write_jsonl(out/'annotations.jsonl',rows)
        print(out / "annotations.jsonl")
        return
    rows = rows[args.shard :: args.shards]
    from inference_mydemo_qwen2511 import collect_crop_inputs, load_crop_records
    from utils.qwen_pipeline_loader import load_qwen21_omni_pipeline, load_qwen_pipeline
    from utils.runner_qwen2511 import run_qwen_multi_branch
    crops = load_crop_records(str(args.data_root / "crops/crop_instruction.jsonl")) if args.variant == "mirage_relaxed" else {}
    from synthesis_pipeline.labeling_checkpoint import CaseCheckpoints, file_digest
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in {'resume','out_root','ids','shard','shards','manifest_only'}}
    checkpoints=CaseCheckpoints(out,settings)
    dependencies={r['image']:dict(source_sha256=file_digest(args.data_root/'sources'/r['source_image'])) for r in rows}
    pending = [r for r in rows if not args.resume or checkpoints.load(r,dependencies[r['image']]) is None]
    print(json.dumps(dict(stage='editing',total=len(rows),reused=len(rows)-len(pending),pending=len(pending))),flush=True)
    for row in pending:
        validate_refinement_execution(row,args.variant)
    if not pending:
        return
    t = time.perf_counter()
    qwen21=args.variant in {
        'context_grounded_v3_qwen21', 'context_grounded_v4_qwen21'}
    model_id=args.model_id or (
        '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1'
        if qwen21 else
        '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511')
    if args.qwen21_backend == 'vllm-omni':
        if not qwen21:
            raise ValueError('--qwen21-backend=vllm-omni requires a Qwen-Image-2.1 variant')
        pipe = load_qwen21_omni_pipeline(model_id)
    else:
        pipe = load_qwen_pipeline(
            model_id,"cuda",torch.bfloat16,"none",
            model_family='qwen21' if qwen21 else 'qwen2511')
    load_time = time.perf_counter() - t
    records = []
    for row in tqdm(pending, desc=args.variant):
        start = time.perf_counter()
        source = Image.open(args.data_root / "sources" / row["source_image"]).convert(
            "RGB"
        )
        mask = mask_array(source.size, row["mask"])
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        if args.variant == "mirage_relaxed":
            prompts, boxes, masks = collect_crop_inputs(
                crops[row["image"]], row["task_type"]
            )
            image = run_qwen_multi_branch(
                pipe,
                source,
                row["editing_instruction"],
                prompts,
                boxes,
                masks,
                num_inference_steps=args.steps,
                generator=generator,
                patch_ratio=0.5,
                write_margin_cells=3,
                mask_dilation_cells=2,
                branch_context_cells=6,
                correct_branch_schedule=True,
            )
        elif args.variant == "official_full":
            image = pipe(
                image=source,
                prompt=row["editing_instruction"],
                negative_prompt=" ",
                true_cfg_scale=4,
                guidance_scale=1,
                num_inference_steps=args.steps,
                generator=generator,
                height=source.height,
                width=source.width,
            ).images[0]
        else:
            image = edit_context_crop(
                pipe,
                source,
                row,
                mask,
                generator,
                args.steps,
                true_cfg_scale=1.0 if qwen21 else 4.0,
                negative_prompt=None if qwen21 else " ",
                remove_context_window=args.variant == "context_removewide",
                diagnostics_dir=out / "diagnostics" / Path(row["image"]).stem,
                target_guide=args.qwen21_target_guide or args.variant
                in {
                    "context_guided",
                    "context_guided_attribute",
                    "context_guided_attribute_v2",
                },
                attribute_mask_composition=args.variant
                in {"context_guided_attribute", "context_guided_attribute_v2"},
                preserve_attribute_texture=args.variant
                == "context_guided_attribute_v2",
                guarded_composition=args.variant in {"context_guarded", "context_guarded_v2"},
                protected_mask=protected_neighbors(args.data_root, row, source.size)
                if args.variant in {"context_guarded", "context_guarded_v2"} else None,
                replacement_context_window=args.variant == "context_guarded_v2",
                grounded_composition=args.variant in {
                    'context_grounded_v3','context_grounded_v4',
                    'context_grounded_v3_qwen21','context_grounded_v4_qwen21'},
                regional_denoising=args.variant in {
                    'context_grounded_v4','context_grounded_v4_qwen21'},
                qwen21_prompt_policy=args.qwen21_prompt_policy,
                remove_composition_policy=args.remove_composition_policy,
                latent_protection_policy=args.latent_protection_policy,
                relation_geometry_policy=args.relation_geometry_policy,
                removal_conditioning=args.removal_conditioning,
            )
        import os
        image_path=out/'edited'/row['image']
        temporary=image_path.with_suffix('.png.tmp')
        image.save(temporary,format='PNG');os.replace(temporary,image_path)
        record = {
            "image": row["image"],
            "variant": args.variant,
            "seconds": round(time.perf_counter() - start, 3),
            "steps": args.steps,
            "seed": args.seed,
            "latent_protection_policy":args.latent_protection_policy,
            "relation_geometry_policy":args.relation_geometry_policy,
            "remove_composition_policy": args.remove_composition_policy,
            "model_id": model_id,
            "pipeline_family": "qwen21" if qwen21 else "qwen2511",
            "qwen21_backend": args.qwen21_backend if qwen21 else None,
            "true_cfg_scale": 1.0 if qwen21 else 4.0,
            "qwen21_prompt_policy": args.qwen21_prompt_policy if qwen21 else None,
        }
        records.append(record)
        with (out / f"timing_shard{args.shard}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        checkpoints.save(row,record,artifacts=[image_path],dependencies=dependencies[row['image']])
        print(json.dumps(record), flush=True)
    (out / f"summary_shard{args.shard}.json").write_text(
        json.dumps({"model_load_seconds": load_time, "records": records}, indent=2)
    )


if __name__ == "__main__":
    main()
