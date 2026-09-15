"""Render native RefEdit grounding rows as source-coordinate SAM3 masks."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from refedit import MASK_POLICY_VERSION
from refedit.io import discover_shards, sample_id, validate_schema
from scaleedit.io import decode_image
from scaleedit.mask_pipeline import annotate_sample
from scaleedit.mask_runner import (
    MASK_SCHEMA,
    MaskJob,
    _empty_row,
    _merge_summaries,
    _success_row,
    assign_jobs,
    parse_devices,
)


DEFAULT_DATASET_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get(
            "REFEDIT_SAM3_CHECKPOINT_PATH",
            os.environ.get(
                "SCALEEDIT_SAM3_CHECKPOINT_PATH",
                "/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt",
            ),
        ),
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--progress-mininterval", type=float, default=2.0)
    return parser.parse_args(argv)


def build_jobs(args: argparse.Namespace) -> List[MaskJob]:
    raw_by_name = {path.name: path for path in discover_shards(args.input_dir)}
    grounding_paths = sorted(args.grounding_dir.glob("train-*.parquet"))
    if not grounding_paths:
        raise FileNotFoundError(
            f"no train-*.parquet grounding shards under {args.grounding_dir}"
        )
    jobs = []
    for grounding_path in grounding_paths:
        input_path = raw_by_name.get(grounding_path.name)
        if input_path is None:
            raise FileNotFoundError(f"missing RefEdit source shard {grounding_path.name}")
        validate_schema(input_path)
        ground_rows = pq.ParquetFile(grounding_path).metadata.num_rows
        raw_rows = pq.ParquetFile(input_path).metadata.num_rows
        if ground_rows > raw_rows:
            raise ValueError(
                f"grounding has more rows than source: "
                f"{grounding_path.name} {ground_rows}>{raw_rows}"
            )
        jobs.append(
            MaskJob(
                input_path=str(input_path),
                grounding_path=str(grounding_path),
                output_path=str(args.output_dir / grounding_path.name),
                num_rows=ground_rows,
            )
        )
    return jobs


def _refedit_empty_row(ground_row: Dict, flag: str, sam_version: str, error: str = "") -> Dict:
    row = _empty_row(ground_row, flag, sam_version, error)
    row["mask_policy_version"] = MASK_POLICY_VERSION
    return row


def _refedit_success_row(ground_row: Dict, result: Dict, seconds: float) -> Dict:
    row = _success_row(ground_row, result, seconds)
    row["mask_policy_version"] = MASK_POLICY_VERSION
    return row


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
    ground_rows = pq.read_table(job.grounding_path).to_pylist()
    source_rows = pq.read_table(input_path).to_pylist()
    output_rows = []
    summary = {
        "rows": 0,
        "errors": 0,
        "flags": {},
        "sources": {},
        "modes": {},
        "tasks": {},
    }
    for ground_row in ground_rows:
        row_idx = int(ground_row["row_idx"])
        if row_idx < 0 or row_idx >= len(source_rows):
            raise IndexError(
                f"grounding row_idx out of range in {input_path.name}: {row_idx}"
            )
        raw_row = source_rows[row_idx]
        expected_id = sample_id(raw_row["img_id"])
        actual_id = str(ground_row.get("sample_id", ""))
        if expected_id != actual_id:
            raise ValueError(
                f"row identity mismatch in {input_path.name}:{row_idx}: "
                f"{expected_id!r}!={actual_id!r}"
            )
        payload = json.loads(ground_row["ground_json"])
        mode = str(payload.get("mask_mode", "unresolved"))
        if ground_row.get("qc_flag") == "GROUND_FAIL":
            out_row = _refedit_empty_row(ground_row, "GROUND_FAIL", sam_version)
        else:
            try:
                sample = {
                    "source": decode_image(raw_row["source_img"]),
                    "target": decode_image(raw_row["target_img"]),
                }
            except Exception as exc:
                summary["errors"] += 1
                out_row = _refedit_empty_row(
                    ground_row, "CORRUPT_IMAGE", sam_version, repr(exc)
                )
                if args.fail_fast:
                    raise
            else:
                started = time.monotonic()
                try:
                    import torch

                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        result = annotate_sample(
                            processor, sample, ground_row, sam_version
                        )
                    out_row = _refedit_success_row(
                        ground_row, result, time.monotonic() - started
                    )
                except Exception as exc:
                    summary["errors"] += 1
                    out_row = _refedit_empty_row(
                        ground_row, "ERROR", sam_version, repr(exc)
                    )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(output_rows, schema=MASK_SCHEMA)
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
            raise RuntimeError(
                f"mask worker sees {torch.cuda.device_count()} GPUs, expected 1"
            )
        torch.cuda.set_device(0)
        progress_queue.put(
            {
                "kind": "log",
                "message": (
                    f"RefEdit mask worker {worker_index} loading SAM3 "
                    f"on GPU {physical_device}"
                ),
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
            Path(args.checkpoint_path).name
            if args.checkpoint_path
            else "facebook/sam3"
        )
        sam_version = f"{MASK_POLICY_VERSION}:{checkpoint_name}"
        for job in jobs:
            output_path = Path(job.output_path)
            current = False
            if output_path.exists() and not args.overwrite:
                mask_table = pq.read_table(
                    output_path,
                    columns=["row_idx", "sample_id", "mask_policy_version"],
                )
                ground_table = pq.read_table(
                    job.grounding_path, columns=["row_idx", "sample_id"]
                )
                rows = mask_table.num_rows
                versions = {
                    str(value)
                    for value in mask_table["mask_policy_version"].to_pylist()
                }
                expected_versions = {MASK_POLICY_VERSION} if rows else set()
                mask_keys = list(
                    zip(
                        mask_table["row_idx"].to_pylist(),
                        mask_table["sample_id"].to_pylist(),
                    )
                )
                ground_keys = list(
                    zip(
                        ground_table["row_idx"].to_pylist(),
                        ground_table["sample_id"].to_pylist(),
                    )
                )
                current = (
                    rows == job.num_rows
                    and versions == expected_versions
                    and mask_keys == ground_keys
                )
            if current:
                summary = {
                    "rows": job.num_rows,
                    "errors": 0,
                    "flags": {},
                    "sources": {},
                    "modes": {},
                    "tasks": {},
                    "skipped_existing": True,
                }
                progress_queue.put({"kind": "rows", "count": job.num_rows})
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
        raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_index})


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
        "stage": "refedit_mask",
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
    messages, worker_errors, done = [], [], 0
    with tqdm(
        total=sum(job.num_rows for job in jobs),
        desc="RefEdit mask rows",
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
                progress.write(str(message["message"]))
            elif kind == "worker_error":
                worker_errors.append(message)
                progress.write(f"WORKER_ERROR {message}")
            elif kind == "worker_done":
                done += 1
            elif kind == "shard_done":
                messages.append(message)
    for process in processes:
        process.join()
    summary = _merge_summaries(messages)
    summary["worker_errors"] = worker_errors
    summary["worker_exit_codes"] = [process.exitcode for process in processes]
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if worker_errors or any(process.exitcode != 0 for process in processes):
        raise SystemExit("RefEdit mask workers failed")


if __name__ == "__main__":
    main()
