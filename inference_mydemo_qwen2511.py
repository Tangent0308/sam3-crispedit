import argparse
import json
import os
import socket
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from utils.runner_qwen2511 import run_qwen_multi_branch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-branch Qwen-Image-Edit inference (single image or batch)."
    )

    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to a single image.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction string for single image mode.",
    )

    parser.add_argument(
        "--image-root",
        default=None,
        help="Folder containing original images (batch mode).",
    )
    parser.add_argument(
        "--instruction-jsonl",
        default=None,
        help="JSONL file with image->instruction mapping (batch mode only).",
    )
    parser.add_argument(
        "--crop-dir",
        default=None,
        help="Folder containing crop_instruction.jsonl.",
    )

    parser.add_argument(
        "--results-full-dir",
        default="results/qwen2511",
        help="Output folder for full images.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output file already exists.",
    )
    parser.add_argument(
        "--work-queue-dir",
        default=None,
        help=(
            "Optional shared directory for dynamic multi-worker scheduling. "
            "Every worker receives the same manifest and atomically claims one "
            "unfinished image at a time."
        ),
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Optional label recorded in dynamic queue claim files.",
    )
    parser.add_argument(
        "--queue-poll-seconds",
        type=float,
        default=0.5,
        help="Polling interval while all unfinished queue items are claimed.",
    )
    parser.add_argument(
        "--claim-timeout-seconds",
        type=float,
        default=3600.0,
        help=(
            "Reclaim queue entries older than this many seconds. Set to 0 to "
            "disable stale-claim recovery."
        ),
    )

    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen-Image-Edit-2511",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Pipeline device.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Torch dtype.",
    )
    parser.add_argument(
        "--cpu-offload",
        default="model",
        choices=["none", "model", "sequential"],
        help=(
            "CPU offload mode. "
            "'model' uses enable_model_cpu_offload (recommended), "
            "'sequential' uses enable_sequential_cpu_offload (slower, lower memory), "
            "'none' keeps full model on --device."
        ),
    )

    parser.add_argument(
        "--patch-ratio",
        type=float,
        default=0.2,
        help="Fraction of inference steps assigned to region branches.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=40,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="True CFG scale.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=" ",
        help="Negative prompt. Keep a blank string by default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    return parser.parse_args()


def dtype_from_str(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def load_instruction_map(jsonl_path: str):
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["image"]] = rec
    return mapping


def load_crop_records(jsonl_path: str):
    mapping = defaultdict(list)
    mask_root = Path(jsonl_path).resolve().parent.parent / "masks"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mask_name = rec.get("image")
            if mask_name:
                rec["_mask_path"] = str(mask_root / str(mask_name))
            mapping[rec["original_image"]].append(rec)
    return mapping


def load_qwen_pipeline(
    model_id: str,
    device: str,
    torch_dtype,
    cpu_offload: str,
):
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype
    )

    if cpu_offload != "none":
        if cpu_offload == "sequential":
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    return pipeline


def collect_crop_inputs(crop_records, case_task_type=None):
    crop_prompts = []
    bboxes = []
    region_masks = []
    for rec in crop_records:
        if rec.get("bbox") is None:
            continue
        instruction = str(rec["new_instruction"]).strip()
        if instruction and instruction[-1] not in ".!?":
            instruction += "."
        refer_object = str(rec.get("refer_object", "")).strip().rstrip(".")
        if refer_object:
            prompt = f"Target: {refer_object}. Instruction: {instruction}"
        else:
            prompt = instruction
        crop_prompts.append(prompt)
        bboxes.append(rec["bbox"])
        task_type = str(rec.get("task_type") or case_task_type or "").lower()
        mask_path = rec.get("_mask_path")
        if task_type in {"remove", "replace", "attribute"} and mask_path:
            with Image.open(mask_path) as mask_image:
                region_masks.append(np.asarray(mask_image.convert("L")) > 127)
        else:
            # Add uses the source mask as a placement anchor, so its generated
            # object may legitimately occupy nearby pixels inside the bbox.
            region_masks.append(None)

    return crop_prompts, bboxes, region_masks


def save_outputs(image_name: str, full_out, results_full_dir: str):
    os.makedirs(results_full_dir, exist_ok=True)
    full_save_path = os.path.join(results_full_dir, image_name)
    full_out.save(full_save_path)


def infer_one_image(
    pipe,
    img_name: str,
    image_root: str,
    inst_map: dict,
    crop_map: dict,
    results_full_dir: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
    generator_device: str,
    seed: int,
    patch_ratio: float,
):
    instruction_record = inst_map[img_name]
    if isinstance(instruction_record, str):
        source_image_name = img_name
        full_prompt = instruction_record
        case_task_type = None
    else:
        source_image_name = str(instruction_record.get("source_image") or img_name)
        full_prompt = str(instruction_record["editing_instruction"])
        case_task_type = instruction_record.get("task_type")
    full_image_path = os.path.join(image_root, source_image_name)
    crop_records = sorted(
        crop_map[img_name], key=lambda rec: str(rec.get("image") or "")
    )

    with Image.open(full_image_path) as image:
        full_image = image.convert("RGB")
    crop_prompts, bboxes, region_masks = collect_crop_inputs(
        crop_records, case_task_type=case_task_type
    )

    # A fresh per-image generator makes output independent of which worker
    # claims the image and of the order in which that worker receives jobs.
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    infer_start = time.perf_counter()
    full_out = run_qwen_multi_branch(
        pipe=pipe,
        full_image=full_image,
        full_prompt=full_prompt,
        crop_prompts=crop_prompts,
        bboxes=bboxes,
        region_masks=region_masks,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        patch_ratio=patch_ratio,
    )
    infer_elapsed = time.perf_counter() - infer_start

    print(f"[Runtime] {img_name} inference: {infer_elapsed:.3f}s")
    save_outputs(img_name, full_out, results_full_dir)


def run_inference_loop(
    pipe,
    image_names: list,
    image_root: str,
    inst_map: dict,
    crop_map: dict,
    results_full_dir: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
    generator_device: str,
    seed: int,
    patch_ratio: float,
    skip_existing: bool = False,
):
    for idx, img_name in enumerate(image_names):
        output_path = os.path.join(results_full_dir, img_name)
        if skip_existing and os.path.isfile(output_path):
            print(f"[Skip] Output exists: {output_path}")
            continue

        if len(image_names) > 1:
            print(f"\n=== [{idx + 1}/{len(image_names)}] Processing {img_name} ===")
        else:
            print(f"\n=== Processing {img_name} ===")

        infer_one_image(
            pipe=pipe,
            img_name=img_name,
            image_root=image_root,
            inst_map=inst_map,
            crop_map=crop_map,
            results_full_dir=results_full_dir,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator_device=generator_device,
            seed=seed,
            patch_ratio=patch_ratio,
        )


def _try_claim(claim_path: Path, worker_id: str, timeout_seconds: float) -> bool:
    if timeout_seconds > 0 and claim_path.exists():
        try:
            age = time.time() - claim_path.stat().st_mtime
            if age > timeout_seconds:
                stale_path = claim_path.with_name(
                    f"{claim_path.name}.stale-{os.getpid()}-{time.time_ns()}"
                )
                try:
                    os.replace(claim_path, stale_path)
                    stale_path.unlink(missing_ok=True)
                    print(f"[Queue] Reclaimed stale claim: {claim_path.name}")
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass

    try:
        descriptor = os.open(
            claim_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        return False

    payload = json.dumps(
        {
            "worker_id": worker_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "claimed_at": time.time(),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    return True


def run_dynamic_queue(
    pipe,
    image_names: list,
    image_root: str,
    inst_map: dict,
    crop_map: dict,
    results_full_dir: str,
    work_queue_dir: str,
    worker_id: str,
    queue_poll_seconds: float,
    claim_timeout_seconds: float,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
    generator_device: str,
    seed: int,
    patch_ratio: float,
):
    queue_dir = Path(work_queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    Path(results_full_dir).mkdir(parents=True, exist_ok=True)
    poll_seconds = max(0.05, float(queue_poll_seconds))
    completed_by_worker = 0

    while True:
        claimed = None
        pending = 0
        for img_name in image_names:
            if (Path(results_full_dir) / img_name).is_file():
                continue
            pending += 1
            claim_path = queue_dir / f"{Path(img_name).name}.claim"
            if _try_claim(claim_path, worker_id, claim_timeout_seconds):
                claimed = (img_name, claim_path)
                break

        if claimed is None:
            if pending == 0:
                print(
                    f"[Queue] Worker {worker_id} finished after "
                    f"{completed_by_worker} image(s)."
                )
                return
            time.sleep(poll_seconds)
            continue

        img_name, claim_path = claimed
        print(f"\n=== [Queue:{worker_id}] Processing {img_name} ===")
        try:
            infer_one_image(
                pipe=pipe,
                img_name=img_name,
                image_root=image_root,
                inst_map=inst_map,
                crop_map=crop_map,
                results_full_dir=results_full_dir,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator_device=generator_device,
                seed=seed,
                patch_ratio=patch_ratio,
            )
            completed_by_worker += 1
        finally:
            claim_path.unlink(missing_ok=True)


def main():
    args = parse_args()

    if args.work_queue_dir and args.image_path:
        raise ValueError("--work-queue-dir is only supported in batch mode")

    if args.image_path:
        inst_map = {os.path.basename(args.image_path): args.instruction}
    else:
        inst_map = load_instruction_map(args.instruction_jsonl)
    crop_jsonl_path = os.path.join(args.crop_dir, "crop_instruction.jsonl")
    crop_map = load_crop_records(crop_jsonl_path)

    torch_dtype = dtype_from_str(args.dtype)
    pipeline = load_qwen_pipeline(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=torch_dtype,
        cpu_offload=args.cpu_offload,
    )

    generator_device = "cuda" if args.device == "cuda" else "cpu"

    if args.image_path:
        image_root = os.path.dirname(args.image_path)
        image_names = [os.path.basename(args.image_path)]
    else:
        image_root = args.image_root
        missing_crop_records = sorted(set(inst_map) - set(crop_map))
        if missing_crop_records:
            raise ValueError(
                "Missing crop records for instruction images: "
                + ", ".join(missing_crop_records[:10])
            )
        # The instruction manifest is authoritative.  crop_map can be a shared
        # full-dataset file while instruction_jsonl is a GPU shard; iterating
        # crop_map here would make every shard process the full dataset and
        # eventually index a prompt that is absent from its inst_map.
        image_names = sorted(inst_map.keys())
        print(f"Found {len(image_names)} instruction images with crop records.")

    common_kwargs = {
        "pipe": pipeline,
        "image_names": image_names,
        "image_root": image_root,
        "inst_map": inst_map,
        "crop_map": crop_map,
        "results_full_dir": args.results_full_dir,
        "num_inference_steps": args.num_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "generator_device": generator_device,
        "seed": args.seed,
        "patch_ratio": args.patch_ratio,
    }
    if args.work_queue_dir:
        worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
        run_dynamic_queue(
            **common_kwargs,
            work_queue_dir=args.work_queue_dir,
            worker_id=worker_id,
            queue_poll_seconds=args.queue_poll_seconds,
            claim_timeout_seconds=args.claim_timeout_seconds,
        )
    else:
        run_inference_loop(
            **common_kwargs,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
