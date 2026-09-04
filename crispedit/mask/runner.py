"""Stage 2: turn Qwen grounding boxes into source-coordinate SAM3 masks."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from tqdm import tqdm

from crispedit.mask.pipeline import (
    MASK_POLICY_VERSION,
    annotate_grounded_sample,
    decode_image,
    encode_mask_png,
    normalized_box_to_pixels,
)


INSTANCE_TYPE = pa.struct(
    [
        ("instance_id", pa.string()),
        ("role", pa.string()),
        ("grounding_image", pa.string()),
        ("ref", pa.string()),
        ("bbox_2d", pa.list_(pa.float64())),
        ("bbox_xyxy", pa.list_(pa.float64())),
        ("mask_source", pa.string()),
        ("semantic_mask_source", pa.string()),
        ("predicted_iou", pa.float64()),
        ("box_iou", pa.float64()),
        ("inside_ratio", pa.float64()),
        ("candidate_count", pa.int64()),
        ("selected_count", pa.int64()),
        ("pcs_query_mode", pa.string()),
        ("pcs_text_candidate_count", pa.int64()),
        ("pcs_joint_candidate_count", pa.int64()),
        ("pcs_fusion", pa.string()),
        ("pcs_pair_iou", pa.float64()),
        ("pcs_text_fill", pa.float64()),
        ("pcs_joint_fill", pa.float64()),
        ("selection_reason", pa.string()),
        ("sam_prompt", pa.string()),
        ("region_mode", pa.string()),
        ("mask_density", pa.string()),
        ("semantic_detail_target", pa.bool_()),
        ("mapped_from_target", pa.bool_()),
        ("coverage_box_union", pa.bool_()),
        ("coverage_suppressed_for_sparse_semantics", pa.bool_()),
        ("directional_coverage", pa.list_(pa.float64())),
        ("errors", pa.list_(pa.string())),
        ("area", pa.int64()),
        ("rle_size", pa.list_(pa.int32())),
        ("rle_counts", pa.string()),
    ]
)

MASK_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("raw_type", pa.string()),
        ("canonical_type", pa.string()),
        ("instruction", pa.string()),
        ("ground_json", pa.string()),
        ("mask_png", pa.binary()),
        ("instance_masks", pa.list_(INSTANCE_TYPE)),
        ("mask_source", pa.string()),
        ("area_frac", pa.float64()),
        ("qc_flag", pa.string()),
        ("qc_flags_json", pa.string()),
        ("mask_height", pa.int32()),
        ("mask_width", pa.int32()),
        ("mask_sum", pa.int64()),
        ("ar_delta", pa.float64()),
        ("grounding_status", pa.string()),
        ("mllm_model", pa.string()),
        ("prompt_version", pa.string()),
        ("sam_version", pa.string()),
        ("mask_seconds", pa.float64()),
        ("prefilter_verdict", pa.string()),
        ("prefilter_confidence", pa.float64()),
        ("prefilter_method", pa.string()),
        ("prefilter_evidence_schema", pa.string()),
        ("prefilter_model_name", pa.string()),
        ("prefilter_run_id", pa.string()),
        ("filter_decision", pa.string()),
        ("prefilter_reason", pa.string()),
        ("filter_reason_codes", pa.string()),
        ("filter_mismatch_score", pa.float64()),
    ]
)


@dataclass
class MaskJob:
    input_path: str
    grounding_path: str
    output_path: str
    num_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 PVS mask generation from Qwen boxes")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", default=os.environ.get("CRISPEDIT_SAM3_CHECKPOINT_PATH"))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--include-types", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--preview-rows-per-shard", type=int, default=1)
    parser.add_argument("--collage-limit", type=int, default=64)
    parser.add_argument("--progress-mininterval", type=float, default=2.0)
    return parser.parse_args()


def parse_devices(spec: str) -> List[int]:
    devices = [int(part.strip().removeprefix("cuda:")) for part in spec.split(",") if part.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    return devices


def build_jobs(args: argparse.Namespace) -> List[MaskJob]:
    include = {part.strip() for part in args.include_types.split(",")} if args.include_types else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for grounding_path in sorted(args.grounding_dir.glob("*.parquet")):
        input_path = args.input_dir / grounding_path.name
        if not input_path.exists():
            raise FileNotFoundError(f"raw shard for grounding output is missing: {input_path}")
        ground_file = pq.ParquetFile(grounding_path)
        if ground_file.metadata.num_rows == 0:
            continue
        if include is not None:
            raw_types = set(pq.read_table(grounding_path, columns=["raw_type"])["raw_type"].to_pylist())
            if raw_types.isdisjoint(include):
                continue
        jobs.append(
            MaskJob(
                input_path=str(input_path),
                grounding_path=str(grounding_path),
                output_path=str(args.output_dir / grounding_path.name),
                num_rows=ground_file.metadata.num_rows,
            )
        )
    return jobs


def assign_jobs(jobs: Sequence[MaskJob], devices: Sequence[int]) -> List[Tuple[int, List[MaskJob]]]:
    buckets = [{"device": device, "rows": 0, "jobs": []} for device in devices]
    for job in sorted(jobs, key=lambda item: item.num_rows, reverse=True):
        bucket = min(buckets, key=lambda item: item["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += job.num_rows
    return [(item["device"], item["jobs"]) for item in buckets if item["jobs"]]


def _copy_metadata(ground_row: Dict) -> Dict:
    return {
        "grounding_status": str(ground_row.get("grounding_status", "")),
        "mllm_model": str(ground_row.get("mllm_model", "")),
        "prompt_version": str(ground_row.get("prompt_version", "")),
        "prefilter_verdict": str(ground_row.get("prefilter_verdict", "")),
        "prefilter_confidence": float(ground_row.get("prefilter_confidence", math.nan)),
        "prefilter_method": str(ground_row.get("prefilter_method", "")),
        "prefilter_evidence_schema": str(
            ground_row.get("prefilter_evidence_schema", "")
        ),
        "prefilter_model_name": str(ground_row.get("prefilter_model_name", "")),
        "prefilter_run_id": str(ground_row.get("prefilter_run_id", "")),
        "filter_decision": str(ground_row.get("filter_decision", "")),
        "prefilter_reason": str(ground_row.get("prefilter_reason", "")),
        "filter_reason_codes": str(ground_row.get("filter_reason_codes", "")),
        "filter_mismatch_score": float(
            ground_row.get("filter_mismatch_score", 0.0)
        ),
    }


def _empty_row(ground_row: Dict, qc_flag: str, sam_version: str, error: str = "") -> Dict:
    flags = [qc_flag]
    if error:
        flags.append(error)
    return {
        "row_idx": int(ground_row["row_idx"]),
        "raw_type": str(ground_row.get("raw_type", "")),
        "canonical_type": str(ground_row.get("canonical_type", qc_flag)),
        "instruction": str(ground_row.get("instruction", "")),
        "ground_json": str(ground_row.get("ground_json", "{}")),
        "mask_png": b"",
        "instance_masks": [],
        "mask_source": "box",
        "area_frac": math.nan if qc_flag in {"PREFILTER_SKIP", "ERROR"} else 0.0,
        "qc_flag": qc_flag,
        "qc_flags_json": json.dumps(flags, ensure_ascii=False),
        "mask_height": 0,
        "mask_width": 0,
        "mask_sum": 0,
        "ar_delta": math.nan,
        "sam_version": sam_version,
        "mask_seconds": 0.0,
        **_copy_metadata(ground_row),
    }


def _success_row(ground_row: Dict, result: Dict, seconds: float) -> Dict:
    mask = result["mask"].astype(np.uint8)
    return {
        "row_idx": int(ground_row["row_idx"]),
        "raw_type": str(ground_row["raw_type"]),
        "canonical_type": str(ground_row["canonical_type"]),
        "instruction": str(ground_row["instruction"]),
        "ground_json": str(ground_row["ground_json"]),
        "mask_png": encode_mask_png(mask),
        "instance_masks": result["instances"],
        "mask_source": str(result["mask_source"]),
        "area_frac": float(mask.mean()),
        "qc_flag": str(result["qc_flag"]),
        "qc_flags_json": json.dumps(result["qc_flags"], ensure_ascii=False),
        "mask_height": int(mask.shape[0]),
        "mask_width": int(mask.shape[1]),
        "mask_sum": int(mask.sum()),
        "ar_delta": float(result["ar_delta"]),
        "sam_version": str(result["sam_version"]),
        "mask_seconds": float(seconds),
        **_copy_metadata(ground_row),
    }


def _raw_rows_for_grounding(input_path: Path, ground_rows: Sequence[Dict]) -> Dict[int, Dict]:
    wanted = sorted(int(row["row_idx"]) for row in ground_rows)
    pf = pq.ParquetFile(input_path)
    result: Dict[int, Dict] = {}
    cursor = 0
    global_start = 0
    for group_index in range(pf.num_row_groups):
        count = pf.metadata.row_group(group_index).num_rows
        group_end = global_start + count
        local_indices = []
        while cursor < len(wanted) and wanted[cursor] < group_end:
            if wanted[cursor] >= global_start:
                local_indices.append(wanted[cursor])
            cursor += 1
        if local_indices:
            records = pf.read_row_group(group_index).to_pylist()
            for row_idx in local_indices:
                result[row_idx] = records[row_idx - global_start]
        global_start = group_end
    missing = sorted(set(wanted) - set(result))
    if missing:
        raise IndexError(f"grounding row_idx out of range for {input_path.name}: {missing}")
    return result


def _draw_ground_boxes(image: Image.Image, items: Sequence[Dict], color: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    shape = (canvas.height, canvas.width)
    for item in items:
        box = normalized_box_to_pixels(item["bbox_2d"], shape)
        draw.rectangle(tuple(float(value) for value in box), outline=color, width=max(2, min(canvas.size) // 300))
        draw.text((float(box[0]) + 3, float(box[1]) + 3), str(item.get("ref", "")), fill=color)
    return canvas


def _overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    if mask.shape != (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask * 255).resize(image.size, Image.Resampling.NEAREST)) > 0
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = mask.astype(np.uint8) * 110
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def save_preview(path: Path, sample: Dict, ground_row: Dict, result: Dict) -> None:
    source, target = sample["input_img"], sample["output_img"]
    payload = json.loads(ground_row["ground_json"])
    boxes = payload.get("boxes", {})
    panels = [
        ("source + boxes", _draw_ground_boxes(source, boxes.get("source", []), "#00ff66")),
        ("target + boxes", _draw_ground_boxes(target, boxes.get("target", []), "#00c8ff")),
        ("source mask overlay", _overlay(source, result["mask"])),
        ("union mask", Image.fromarray(result["mask"] * 255, mode="L").convert("RGB")),
    ]
    tile_width, tile_height = 520, 360
    header = 72
    canvas = Image.new("RGB", (tile_width * len(panels), tile_height + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 6),
        f"{Path(path).stem} row={ground_row['row_idx']} | {ground_row['raw_type']} | "
        f"{result['mask_source']} | {result['qc_flag']} | area={result['mask'].mean():.3f}",
        fill="black",
    )
    draw.text((8, 30), str(ground_row["instruction"])[:220], fill="black")
    for index, (label, panel) in enumerate(panels):
        panel.thumbnail((tile_width, tile_height - 24), Image.Resampling.LANCZOS)
        x = index * tile_width + (tile_width - panel.width) // 2
        y = header + 24 + (tile_height - 24 - panel.height) // 2
        canvas.paste(panel, (x, y))
        draw.text((index * tile_width + 8, header + 3), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def process_job(job: MaskJob, processor, args: argparse.Namespace, progress_queue, worker_index: int, sam_version: str) -> Dict:
    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    ground_rows = pq.read_table(job.grounding_path).to_pylist()
    raw_rows = _raw_rows_for_grounding(input_path, ground_rows)
    output_rows = []
    summary = {"rows": 0, "errors": 0, "flags": {}, "sources": {}, "types": {}}
    preview_count = 0
    for ground_row in ground_rows:
        row_idx = int(ground_row["row_idx"])
        if ground_row.get("qc_flag") == "PREFILTER_SKIP":
            out_row = _empty_row(ground_row, "PREFILTER_SKIP", sam_version)
        elif ground_row.get("qc_flag") == "GROUND_FAIL":
            out_row = _empty_row(ground_row, "GROUND_FAIL", sam_version)
        else:
            record = raw_rows[row_idx]
            sample = {
                "input_img": decode_image(record["input_img"]),
                "output_img": decode_image(record["output_img"]),
                "instruction": record["instruction"],
                "type": record["type"],
            }
            started = time.monotonic()
            try:
                import torch

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = annotate_grounded_sample(processor, sample, ground_row, sam_version)
                seconds = time.monotonic() - started
                out_row = _success_row(ground_row, result, seconds)
                if args.preview_dir and preview_count < args.preview_rows_per_shard:
                    preview_path = Path(args.preview_dir) / f"{input_path.stem}__row{row_idx}.png"
                    save_preview(preview_path, sample, ground_row, result)
                    preview_count += 1
            except Exception as exc:
                summary["errors"] += 1
                out_row = _empty_row(ground_row, "ERROR", sam_version, repr(exc))
                if args.fail_fast:
                    raise
        output_rows.append(out_row)
        flag, source, raw_type = out_row["qc_flag"], out_row["mask_source"], out_row["raw_type"]
        summary["flags"][flag] = summary["flags"].get(flag, 0) + 1
        summary["sources"][source] = summary["sources"].get(source, 0) + 1
        summary["types"][raw_type] = summary["types"].get(raw_type, 0) + 1
        summary["rows"] += 1
        progress_queue.put(
            {
                "kind": "rows",
                "count": 1,
                "worker": worker_index,
                "shard": input_path.name,
                "done_rows": summary["rows"],
                "total_rows": job.num_rows,
            }
        )
    table = pa.Table.from_pylist(output_rows, schema=MASK_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp_path, compression=args.compression)
    tmp_path.replace(output_path)
    return summary


def worker_main(worker_index: int, physical_device: int, jobs: List[MaskJob], args_dict: Dict, progress_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    args = argparse.Namespace(**args_dict)
    try:
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"SAM worker expected one visible GPU, found {torch.cuda.device_count()}")
        torch.cuda.set_device(0)
        progress_queue.put({"kind": "log", "message": f"mask-worker-{worker_index} loading SAM3 PVS on physical GPU {physical_device}"})
        model = build_sam3_image_model(
            device="cuda",
            checkpoint_path=args.checkpoint_path,
            load_from_HF=args.checkpoint_path is None,
            enable_inst_interactivity=True,
        )
        # Small edit elements (piercings, petals, stars) are frequently below
        # 0.5 even when their masks are accurate.  Stage 2 spatially filters all
        # PCS detections with the MLLM box, making 0.3 a safer recall-first gate.
        processor = Sam3Processor(model, device="cuda", confidence_threshold=0.3)
        sam_version = (
            f"{MASK_POLICY_VERSION}:"
            f"{Path(args.checkpoint_path).name if args.checkpoint_path else 'facebook/sam3'}"
        )
        for job in jobs:
            output_path = Path(job.output_path)
            if output_path.exists() and not args.overwrite:
                rows = pq.ParquetFile(output_path).metadata.num_rows
                summary = {"rows": rows, "errors": 0, "flags": {}, "sources": {}, "types": {}, "skipped_existing": True}
                progress_queue.put({"kind": "rows", "count": rows})
            else:
                summary = process_job(job, processor, args, progress_queue, worker_index, sam_version)
            progress_queue.put({"kind": "shard_done", "worker": worker_index, "shard": job.input_path, "summary": summary})
    except Exception as exc:
        progress_queue.put({"kind": "worker_error", "worker": worker_index, "device": physical_device, "error": repr(exc)})
        if args.fail_fast:
            raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_index})


def aggregate(messages: Sequence[Dict]) -> Dict:
    result = {"rows": 0, "errors": 0, "flags": {}, "sources": {}, "types": {}}
    for message in messages:
        summary = message.get("summary", {})
        result["rows"] += int(summary.get("rows", 0))
        result["errors"] += int(summary.get("errors", 0))
        for field in ("flags", "sources", "types"):
            for key, count in summary.get(field, {}).items():
                result[field][key] = result[field].get(key, 0) + int(count)
    return result


def build_collage(preview_dir: Path, output_path: Path, limit: int) -> Optional[Path]:
    paths = sorted(preview_dir.glob("*.png"))[:limit]
    if not paths:
        return None
    panels = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((900, 240), Image.Resampling.LANCZOS)
        panels.append(image.copy())
    width = 900
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, y))
        y += panel.height
    canvas.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.grounding_dir = args.grounding_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint_path:
        args.checkpoint_path = str(Path(args.checkpoint_path).resolve())
    if args.preview_dir:
        args.preview_dir = args.preview_dir.resolve()
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    if not jobs:
        raise SystemExit("no grounded mask jobs matched")
    devices = parse_devices(args.devices)
    assignments = assign_jobs(jobs, devices)
    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config = {
        "stage": "sam3_grounded_mask",
        "devices": devices,
        "total_rows": sum(job.num_rows for job in jobs),
        "args": args_dict,
        "jobs": [asdict(job) for job in jobs],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for worker_index, (device, worker_jobs) in enumerate(assignments):
        process = ctx.Process(
            target=worker_main,
            args=(worker_index, device, worker_jobs, args_dict, progress_queue),
            daemon=False,
        )
        process.start()
        processes.append(process)

    summaries = []
    errors = []
    done_workers = set()
    pbar = tqdm(total=sum(job.num_rows for job in jobs), desc="SAM3 mask rows", dynamic_ncols=True, mininterval=args.progress_mininterval)
    try:
        while len(done_workers) < len(processes):
            try:
                message = progress_queue.get(timeout=0.5)
            except queue.Empty:
                for index, process in enumerate(processes):
                    if index not in done_workers and not process.is_alive() and process.exitcode is not None:
                        done_workers.add(index)
                        if process.exitcode != 0:
                            errors.append(f"worker-{index} exited with code {process.exitcode}")
                continue
            kind = message.get("kind")
            if kind == "rows":
                pbar.update(int(message.get("count", 0)))
                if message.get("shard"):
                    pbar.set_postfix_str(f"{message['shard']} {message.get('done_rows')}/{message.get('total_rows')}")
            elif kind == "log":
                tqdm.write(message.get("message", ""))
            elif kind == "shard_done":
                summaries.append(message)
                summary = message.get("summary", {})
                tqdm.write(f"done {Path(message['shard']).name}: rows={summary.get('rows', 0)} errors={summary.get('errors', 0)} flags={summary.get('flags', {})}")
            elif kind == "worker_error":
                error = f"worker-{message.get('worker')} GPU {message.get('device')}: {message.get('error')}"
                errors.append(error)
                tqdm.write(error)
            elif kind == "worker_done":
                done_workers.add(int(message["worker"]))
    finally:
        pbar.close()
        for process in processes:
            process.join()

    summary = aggregate(summaries)
    summary["worker_errors"] = errors
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.preview_dir:
        collage = build_collage(args.preview_dir, args.output_dir / "preview_collage.png", args.collage_limit)
        if collage:
            print(f"preview collage: {collage}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors or summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
