"""Stage 2: render ScaleEdit grounding contracts as source-coordinate masks."""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from scaleedit import MASK_POLICY_VERSION
from scaleedit.io import decode_image, discover_shards
from scaleedit.mask_pipeline import annotate_sample


INSTANCE_TYPE = pa.struct(
    [
        ("instance_id", pa.string()),
        ("role", pa.string()),
        ("grounding_image", pa.string()),
        ("ref", pa.string()),
        ("bbox_2d", pa.list_(pa.float64())),
        ("bbox_xyxy", pa.list_(pa.float64())),
        ("mask_method", pa.string()),
        ("mask_source", pa.string()),
        ("semantic_mask_source", pa.string()),
        ("predicted_iou", pa.float64()),
        ("box_iou", pa.float64()),
        ("inside_ratio", pa.float64()),
        ("selection_reason", pa.string()),
        ("sam_prompt", pa.string()),
        ("region_mode", pa.string()),
        ("mask_density", pa.string()),
        ("mapped_from_target", pa.bool_()),
        ("errors", pa.list_(pa.string())),
        ("area", pa.int64()),
        ("rle_size", pa.list_(pa.int32())),
        ("rle_counts", pa.string()),
    ]
)

MASK_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("sample_id", pa.string()),
        ("source_relative_path", pa.string()),
        ("edit_task", pa.string()),
        ("final_task", pa.string()),
        ("original_instruction", pa.string()),
        ("final_instruction", pa.string()),
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
        ("mask_policy_version", pa.string()),
        ("mask_seconds", pa.float64()),
    ]
)


@dataclass(frozen=True)
class MaskJob:
    input_path: str
    grounding_path: str
    output_path: str
    num_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScaleEdit SAM3 hybrid mask labeling")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get(
            "SCALEEDIT_SAM3_CHECKPOINT_PATH",
            "/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt",
        ),
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--progress-mininterval", type=float, default=2.0)
    return parser.parse_args()


def parse_devices(spec: str) -> List[int]:
    devices = [
        int(part.strip().removeprefix("cuda:"))
        for part in spec.split(",")
        if part.strip()
    ]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    return devices


def build_jobs(args: argparse.Namespace) -> List[MaskJob]:
    raw_by_name = {path.name: path for path in discover_shards(args.input_dir)}
    jobs = []
    for grounding_path in sorted(args.grounding_dir.glob("part-*.parquet")):
        input_path = raw_by_name.get(grounding_path.name)
        if input_path is None:
            raise FileNotFoundError(f"missing source shard for {grounding_path.name}")
        ground_rows = pq.ParquetFile(grounding_path).metadata.num_rows
        raw_rows = pq.ParquetFile(input_path).metadata.num_rows
        if ground_rows > raw_rows:
            raise ValueError(
                f"grounding has more rows than source: {grounding_path.name} {ground_rows}>{raw_rows}"
            )
        jobs.append(
            MaskJob(
                input_path=str(input_path),
                grounding_path=str(grounding_path),
                output_path=str(args.output_dir / grounding_path.name),
                num_rows=ground_rows,
            )
        )
    if not jobs:
        raise FileNotFoundError(f"no part-*.parquet grounding shards under {args.grounding_dir}")
    return jobs


def assign_jobs(
    jobs: Sequence[MaskJob], devices: Sequence[int]
) -> List[Tuple[int, List[MaskJob]]]:
    buckets = [{"device": device, "rows": 0, "jobs": []} for device in devices]
    for job in sorted(jobs, key=lambda value: value.num_rows, reverse=True):
        bucket = min(buckets, key=lambda value: value["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += job.num_rows
    return [
        (item["device"], item["jobs"])
        for item in buckets
        if item["jobs"]
    ]


def _encode_mask(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def _metadata(ground_row: Dict) -> Dict:
    return {
        "row_idx": int(ground_row["row_idx"]),
        "sample_id": str(ground_row.get("sample_id", "")),
        "source_relative_path": str(ground_row.get("source_relative_path", "")),
        "edit_task": str(ground_row.get("edit_task", "")),
        "final_task": str(ground_row.get("final_task", "")),
        "original_instruction": str(ground_row.get("original_instruction", "")),
        "final_instruction": str(ground_row.get("final_instruction", "")),
        "ground_json": str(ground_row.get("ground_json", "{}")),
        "grounding_status": str(ground_row.get("grounding_status", "")),
        "mllm_model": str(ground_row.get("mllm_model", "")),
        "prompt_version": str(ground_row.get("prompt_version", "")),
    }


def _empty_row(ground_row: Dict, flag: str, sam_version: str, error: str = "") -> Dict:
    flags = [flag] + ([error] if error else [])
    return {
        **_metadata(ground_row),
        "mask_png": b"",
        "instance_masks": [],
        "mask_source": "none",
        "area_frac": math.nan,
        "qc_flag": flag,
        "qc_flags_json": json.dumps(flags, ensure_ascii=False),
        "mask_height": 0,
        "mask_width": 0,
        "mask_sum": 0,
        "ar_delta": math.nan,
        "sam_version": sam_version,
        "mask_policy_version": MASK_POLICY_VERSION,
        "mask_seconds": 0.0,
    }


def _success_row(ground_row: Dict, result: Dict, seconds: float) -> Dict:
    mask = result["mask"].astype(np.uint8)
    return {
        **_metadata(ground_row),
        "mask_png": _encode_mask(mask),
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
        "mask_policy_version": MASK_POLICY_VERSION,
        "mask_seconds": float(seconds),
    }


def process_job(
    job: MaskJob,
    processor,
    args: argparse.Namespace,
    progress_queue,
    worker_index: int,
    sam_version: str,
) -> Dict:
    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    raw_rows = pq.read_table(input_path).slice(0, job.num_rows).to_pylist()
    ground_rows = pq.read_table(job.grounding_path).to_pylist()
    if len(raw_rows) != len(ground_rows):
        raise ValueError(f"row mismatch for {input_path.name}")
    output_rows = []
    summary = {"rows": 0, "errors": 0, "flags": {}, "sources": {}, "modes": {}, "tasks": {}}
    for raw_row, ground_row in zip(raw_rows, ground_rows):
        row_idx = int(ground_row["row_idx"])
        if row_idx >= len(raw_rows) or str(raw_row.get("sample_id", "")) != str(
            ground_row.get("sample_id", "")
        ):
            raise ValueError(f"row identity mismatch in {input_path.name}:{row_idx}")
        mode = str(json.loads(ground_row["ground_json"]).get("mask_mode", "unresolved"))
        if ground_row.get("qc_flag") == "GROUND_FAIL":
            out_row = _empty_row(ground_row, "GROUND_FAIL", sam_version)
        else:
            sample = {
                "source": decode_image(raw_row["source_image"]),
                "target": decode_image(raw_row["edited_image"]),
            }
            started = time.monotonic()
            try:
                import torch

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = annotate_sample(processor, sample, ground_row, sam_version)
                out_row = _success_row(ground_row, result, time.monotonic() - started)
            except Exception as exc:
                summary["errors"] += 1
                out_row = _empty_row(ground_row, "ERROR", sam_version, repr(exc))
                if args.fail_fast:
                    raise
        output_rows.append(out_row)
        for field, key in (
            ("flags", out_row["qc_flag"]),
            ("sources", out_row["mask_source"]),
            ("modes", mode),
            ("tasks", out_row["final_task"]),
        ):
            summary[field][key] = summary[field].get(key, 0) + 1
        summary["rows"] += 1
        progress_queue.put(
            {
                "kind": "rows",
                "count": 1,
                "worker": worker_index,
                "shard": input_path.name,
            }
        )
    table = pa.Table.from_pylist(output_rows, schema=MASK_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp_path, compression=args.compression)
    tmp_path.replace(output_path)
    return summary


def worker_main(
    worker_index: int,
    physical_device: int,
    jobs: List[MaskJob],
    args_dict: Dict,
    progress_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    args = argparse.Namespace(**args_dict)
    try:
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"mask worker sees {torch.cuda.device_count()} GPUs, expected 1")
        torch.cuda.set_device(0)
        progress_queue.put(
            {
                "kind": "log",
                "message": f"mask-worker-{worker_index} loading SAM3 on GPU {physical_device}",
            }
        )
        model = build_sam3_image_model(
            device="cuda",
            checkpoint_path=args.checkpoint_path,
            load_from_HF=args.checkpoint_path is None,
            enable_inst_interactivity=True,
        )
        processor = Sam3Processor(model, device="cuda", confidence_threshold=0.3)
        checkpoint_name = (
            Path(args.checkpoint_path).name if args.checkpoint_path else "facebook/sam3"
        )
        sam_version = f"{MASK_POLICY_VERSION}:{checkpoint_name}"
        for job in jobs:
            output_path = Path(job.output_path)
            if output_path.exists() and not args.overwrite:
                rows = pq.ParquetFile(output_path).metadata.num_rows
                summary = {
                    "rows": rows,
                    "errors": 0,
                    "flags": {},
                    "sources": {},
                    "modes": {},
                    "tasks": {},
                    "skipped_existing": True,
                }
                progress_queue.put({"kind": "rows", "count": rows})
            else:
                summary = process_job(
                    job, processor, args, progress_queue, worker_index, sam_version
                )
            progress_queue.put(
                {
                    "kind": "shard_done",
                    "worker": worker_index,
                    "shard": job.input_path,
                    "summary": summary,
                }
            )
    except Exception as exc:
        progress_queue.put(
            {
                "kind": "worker_error",
                "worker": worker_index,
                "device": physical_device,
                "error": repr(exc),
            }
        )
        if args.fail_fast:
            raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_index})


def _merge_summaries(messages: Sequence[Dict]) -> Dict:
    result = {
        "rows": 0,
        "errors": 0,
        "flags": {},
        "sources": {},
        "modes": {},
        "tasks": {},
    }
    for message in messages:
        summary = message.get("summary", {})
        result["rows"] += int(summary.get("rows", 0))
        result["errors"] += int(summary.get("errors", 0))
        for field in ("flags", "sources", "modes", "tasks"):
            for key, count in summary.get(field, {}).items():
                result[field][key] = result[field].get(key, 0) + int(count)
    return result


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.grounding_dir = args.grounding_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint_path:
        args.checkpoint_path = str(Path(args.checkpoint_path).resolve())
        if not Path(args.checkpoint_path).is_file():
            raise FileNotFoundError(args.checkpoint_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    assignments = assign_jobs(jobs, parse_devices(args.devices))
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        "stage": "scaleedit_mask",
        "mask_policy_version": MASK_POLICY_VERSION,
        "total_rows": sum(job.num_rows for job in jobs),
        "jobs": [asdict(job) for job in jobs],
        "args": args_dict,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=worker_main,
            args=(index, device, worker_jobs, args_dict, progress_queue),
        )
        for index, (device, worker_jobs) in enumerate(assignments)
    ]
    for process in processes:
        process.start()
    messages, done = [], 0
    with tqdm(
        total=sum(job.num_rows for job in jobs),
        desc="ScaleEdit mask rows",
        dynamic_ncols=True,
        mininterval=args.progress_mininterval,
    ) as progress:
        while done < len(processes):
            try:
                message = progress_queue.get(timeout=1.0)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue
            kind = message.get("kind")
            if kind == "rows":
                progress.update(int(message.get("count", 0)))
            elif kind == "log":
                progress.write(message["message"])
            elif kind == "worker_error":
                progress.write(f"WORKER_ERROR {message}")
            elif kind == "worker_done":
                done += 1
            elif kind == "shard_done":
                messages.append(message)
    for process in processes:
        process.join()
    summary = _merge_summaries(messages)
    summary["worker_exit_codes"] = [process.exitcode for process in processes]
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise SystemExit(f"mask workers failed: {failed}")


if __name__ == "__main__":
    main()
