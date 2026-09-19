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

from inference_mydemo_qwen2511 import (
    collect_crop_inputs,
    load_crop_records,
    load_qwen_pipeline,
)
from synthesis_pipeline.audit_edit_pairs import mask_array
from utils.runner_qwen2511 import run_qwen_multi_branch
from utils.context_edit import edit_context_crop

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
        ],
        required=True,
    )
    p.add_argument("--ids", default=",".join(map(str, DEV_IDS)))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="Export selected annotations without loading a model",
    )
    p.add_argument(
        "--model-id",
        default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511",
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
    if args.manifest_only:
        (out / "annotations.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        print(out / "annotations.jsonl")
        return
    rows = rows[args.shard :: args.shards]
    crops = load_crop_records(str(args.data_root / "crops/crop_instruction.jsonl"))
    pending = [r for r in rows if not (out / "edited" / r["image"]).exists()]
    if not pending:
        return
    t = time.perf_counter()
    pipe = load_qwen_pipeline(args.model_id, "cuda", torch.bfloat16, "none")
    load_time = time.perf_counter() - t
    records = []
    for row in tqdm(pending, desc=args.variant):
        start = time.perf_counter()
        source = Image.open(args.data_root / "sources" / row["source_image"]).convert(
            "RGB"
        )
        mask = mask_array(source.size, row["mask"])
        generator = torch.Generator(device="cuda").manual_seed(0)
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
                remove_context_window=args.variant == "context_removewide",
                diagnostics_dir=out / "diagnostics" / Path(row["image"]).stem,
                target_guide=args.variant
                in {
                    "context_guided",
                    "context_guided_attribute",
                    "context_guided_attribute_v2",
                },
                attribute_mask_composition=args.variant
                in {"context_guided_attribute", "context_guided_attribute_v2"},
                preserve_attribute_texture=args.variant
                == "context_guided_attribute_v2",
            )
        image.save(out / "edited" / row["image"])
        record = {
            "image": row["image"],
            "variant": args.variant,
            "seconds": round(time.perf_counter() - start, 3),
            "steps": args.steps,
            "seed": 0,
        }
        records.append(record)
        with (out / f"timing_shard{args.shard}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    (out / f"summary_shard{args.shard}.json").write_text(
        json.dumps({"model_load_seconds": load_time, "records": records}, indent=2)
    )


if __name__ == "__main__":
    main()
