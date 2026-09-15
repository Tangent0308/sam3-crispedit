"""Native RefEdit two-pass grounding with Qwen3.5 served by vLLM."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from refedit import GROUND_PROMPT_VERSION
from refedit.io import discover_shards, iter_row_batches, sample_id, validate_schema
from refedit.policy import (
    TaskInference,
    apply_refedit_contract,
    infer_task,
    refedit_planner_prompt,
)
from refedit.selection import (
    load_prefilter_manifest_dir,
    validate_prefilter_rows,
)
from scaleedit.grounding_runner import (
    GROUND_SCHEMA,
    GroundingJob,
    Qwen35ScaleEditGrounder,
    _merge_summaries,
    _row_from_payload,
    assign_jobs,
    parse_device_groups,
)
from scaleedit.io import decode_image
from scaleedit.policy import build_observation_prompt


DEFAULT_DATASET_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit"
)
DEFAULT_MODEL_PATH = Path(
    "/mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B"
)


class Qwen35RefEditGrounder(Qwen35ScaleEditGrounder):
    """Reuse ScaleEdit v18 generation/parsing with a RefEdit preamble."""

    def _build_observation_prompt(self, final_task: object, instruction: object) -> str:
        return refedit_planner_prompt(
            build_observation_prompt(final_task, instruction)
        )


def _selection_ids(args: argparse.Namespace) -> Set[int]:
    selected = {int(value) for value in args.img_id}
    if args.selection_file:
        raw = json.loads(args.selection_file.read_text(encoding="utf-8"))
        values = raw.get("img_ids", []) if isinstance(raw, dict) else raw
        if not isinstance(values, list):
            raise ValueError("selection file must be a JSON list or {'img_ids': [...]} object")
        selected.update(int(value) for value in values)
    return selected


def build_jobs(args: argparse.Namespace) -> List[GroundingJob]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = _selection_ids(args)
    manifest_dir = getattr(args, "prefilter_manifest_dir", None)
    if manifest_dir and requested:
        raise ValueError(
            "--prefilter-manifest-dir is mutually exclusive with --img-id/--selection-file"
        )
    if manifest_dir and args.limit_rows_per_shard is not None:
        raise ValueError(
            "--prefilter-manifest-dir is mutually exclusive with --limit-rows-per-shard"
        )
    if requested and args.limit_rows_per_shard is not None:
        raise ValueError("selection and --limit-rows-per-shard are mutually exclusive")
    selected_by_shard = (
        load_prefilter_manifest_dir(Path(manifest_dir)) if manifest_dir else None
    )
    paths = discover_shards(args.input_dir)
    if selected_by_shard is not None:
        source_names = {path.name for path in paths}
        if set(selected_by_shard) != source_names:
            missing = sorted(source_names - set(selected_by_shard))
            extra = sorted(set(selected_by_shard) - source_names)
            raise ValueError(
                f"prefilter/source shard mismatch: missing={missing} extra={extra}"
            )
    if args.limit_shards is not None:
        paths = paths[: args.limit_shards]
    found: Set[int] = set()
    jobs: List[GroundingJob] = []
    for path in paths:
        validate_schema(path)
        selected_sample_ids: Tuple[str, ...] = ()
        if selected_by_shard is not None:
            selected_rows = selected_by_shard.get(path.name)
            if selected_rows is None:
                raise FileNotFoundError(
                    f"missing prefilter manifest shard for {path.name}"
                )
            validate_prefilter_rows(path, selected_rows)
            count = len(selected_rows)
            selected_sample_ids = tuple(
                str(row["sample_id"]) for row in selected_rows
            )
        elif requested:
            ids = [
                int(value)
                for value in pq.read_table(path, columns=["img_id"])[0].to_pylist()
            ]
            shard_ids = sorted(requested.intersection(ids))
            found.update(shard_ids)
            if not shard_ids:
                continue
            count = len(shard_ids)
            selected_sample_ids = tuple(sample_id(value) for value in shard_ids)
        else:
            count = pq.ParquetFile(path).metadata.num_rows
            if args.limit_rows_per_shard is not None:
                count = min(count, args.limit_rows_per_shard)
        jobs.append(
            GroundingJob(
                input_path=str(path),
                output_path=str(args.output_dir / path.name),
                num_rows=count,
                selected_sample_ids=selected_sample_ids,
            )
        )
    missing = requested - found
    if missing:
        preview = sorted(missing)[:20]
        raise KeyError(f"requested img_id values not found: {preview}")
    if not jobs:
        raise ValueError("RefEdit selection produced no grounding jobs")
    return jobs


def _canonical_record(path: Path, record: Dict, task: TaskInference) -> Dict:
    instruction = str(record["instruction"])
    return {
        "sample_id": sample_id(record["img_id"]),
        "source_relative_path": f"data/{path.name}",
        "edit_task": task.final_task,
        "final_task": task.final_task,
        "original_instruction": instruction,
        "final_instruction": instruction,
        "source_image": record["source_img"],
        "edited_image": record["target_img"],
    }


def process_job(
    job: GroundingJob,
    grounder: Qwen35RefEditGrounder,
    args: argparse.Namespace,
    progress_queue,
    worker_index: int,
) -> Dict:
    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = None
    completed = False
    summary = {
        "input_rows": 0,
        "rows": 0,
        "errors": 0,
        "skipped_images": 0,
        "skipped_samples": [],
        "statuses": {},
        "tasks": {},
    }
    selected_ids = set(job.selected_sample_ids)
    try:
        for indexed_records in iter_row_batches(input_path, args.batch_size):
            if selected_ids:
                indexed_records = [
                    item
                    for item in indexed_records
                    if sample_id(item[1]["img_id"]) in selected_ids
                ]
            else:
                indexed_records = [item for item in indexed_records if item[0] < job.num_rows]
            if not indexed_records:
                if selected_ids:
                    continue
                break
            summary["input_rows"] += len(indexed_records)
            valid: List[Tuple[int, Dict, Dict, TaskInference]] = []
            samples = []
            for row_idx, record in indexed_records:
                task = infer_task(record["instruction"])
                canonical = _canonical_record(input_path, record, task)
                try:
                    source = decode_image(record["source_img"])
                    target = decode_image(record["target_img"])
                except Exception as exc:
                    skipped = {
                        "shard": input_path.name,
                        "row_idx": row_idx,
                        "sample_id": canonical["sample_id"],
                        "error": repr(exc),
                    }
                    summary["skipped_images"] += 1
                    summary["skipped_samples"].append(skipped)
                    progress_queue.put({"kind": "log", "message": f"SKIP_CORRUPT_IMAGE {skipped}"})
                    progress_queue.put({"kind": "rows", "count": 1})
                    continue
                valid.append((row_idx, record, canonical, task))
                samples.append(
                    {
                        "source": source,
                        "target": target,
                        "instruction": canonical["final_instruction"],
                        "final_task": task.final_task,
                    }
                )
            if not samples:
                continue
            started = time.monotonic()
            try:
                payloads = grounder.infer(samples)
                payloads = [
                    apply_refedit_contract(payload, item[3])
                    for payload, item in zip(payloads, valid)
                ]
            except Exception as exc:
                if args.fail_fast:
                    raise
                summary["errors"] += len(samples)
                payloads = [
                    apply_refedit_contract(
                        {
                            "schema_version": 1,
                            "mask_mode": "unresolved",
                            "source": [],
                            "target": [],
                            "protected_foreground": [],
                            "ground_parse_ok": False,
                            "runtime_error": repr(exc),
                        },
                        item[3],
                    )
                    for item in valid
                ]
            elapsed = (time.monotonic() - started) / max(len(samples), 1)
            output_rows = []
            for (row_idx, _record, canonical, task), payload in zip(valid, payloads):
                payload["refedit_source"] = {
                    "img_id": int(canonical["sample_id"].split(":", 1)[1]),
                    "source_shard": input_path.name,
                    "source_row_idx": row_idx,
                }
                row = _row_from_payload(
                    row_idx,
                    canonical,
                    payload,
                    Path(args.model_path).name,
                    elapsed,
                )
                row["prompt_version"] = GROUND_PROMPT_VERSION
                row["ground_json"] = json.dumps(payload, ensure_ascii=False)
                output_rows.append(row)
                status = row["grounding_status"]
                summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
                summary["tasks"][task.final_task] = (
                    summary["tasks"].get(task.final_task, 0) + 1
                )
            table = pa.Table.from_pylist(output_rows, schema=GROUND_SCHEMA)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(tmp_path, GROUND_SCHEMA, compression=args.compression)
            writer.write_table(table)
            summary["rows"] += len(output_rows)
            progress_queue.put({"kind": "rows", "count": len(output_rows)})
        if summary["input_rows"] != job.num_rows:
            raise ValueError(
                f"selected row mismatch for {input_path.name}: "
                f"expected={job.num_rows} actual={summary['input_rows']}"
            )
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(tmp_path, GROUND_SCHEMA, compression=args.compression)
            writer.write_table(pa.Table.from_pylist([], schema=GROUND_SCHEMA))
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if completed and tmp_path.exists():
            tmp_path.replace(output_path)
    return summary


def worker_main(
    worker_index: int,
    physical_devices: List[int],
    jobs: List[GroundingJob],
    args_dict: Dict,
    progress_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in physical_devices)
    args = argparse.Namespace(**args_dict)
    try:
        progress_queue.put(
            {
                "kind": "log",
                "message": (
                    f"RefEdit ground worker {worker_index} loading vLLM "
                    f"on {physical_devices}"
                ),
            }
        )
        grounder = Qwen35RefEditGrounder(args)
        for job in jobs:
            output_path = Path(job.output_path)
            current = False
            if output_path.exists() and not args.overwrite:
                table = pq.read_table(
                    output_path, columns=["sample_id", "prompt_version"]
                )
                rows = table.num_rows
                versions = {
                    str(value) for value in table["prompt_version"].to_pylist()
                }
                expected_versions = {GROUND_PROMPT_VERSION} if rows else set()
                identities_match = True
                if job.selected_sample_ids:
                    identities_match = tuple(
                        str(value) for value in table["sample_id"].to_pylist()
                    ) == tuple(job.selected_sample_ids)
                current = (
                    rows == job.num_rows
                    and versions == expected_versions
                    and identities_match
                )
            if current:
                summary = {
                    "input_rows": job.num_rows,
                    "rows": job.num_rows,
                    "errors": 0,
                    "skipped_images": 0,
                    "skipped_samples": [],
                    "statuses": {},
                    "tasks": {},
                    "skipped_existing": True,
                }
                progress_queue.put({"kind": "rows", "count": job.num_rows})
            else:
                summary = process_job(job, grounder, args, progress_queue, worker_index)
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
                "devices": physical_devices,
                "error": repr(exc),
            }
        )
        raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_index})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--inference-backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-batch-size", type=int, default=4)
    parser.add_argument("--max-images-per-generate", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--planner-max-new-tokens", type=int, default=2048)
    parser.add_argument("--locator-max-new-tokens", type=int)
    parser.add_argument("--max-pixels", type=int, default=1_310_720)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--gpu-memory-gib", type=int, default=74)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--limit-rows-per-shard", type=int)
    parser.add_argument("--img-id", action="append", default=[])
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument(
        "--prefilter-manifest-dir",
        type=Path,
        help=(
            "directory of PASS-only RefEdit quality-prefilter manifest shards; "
            "when set, grounding runs only those source rows"
        ),
    )
    parser.add_argument("--progress-mininterval", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_path = str(args.model_path.resolve())
    if args.prefilter_manifest_dir:
        args.prefilter_manifest_dir = args.prefilter_manifest_dir.resolve()
    jobs = build_jobs(args)
    groups = parse_device_groups(args.devices, args.tensor_parallel_size)
    assignments = assign_jobs(jobs, groups)
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        "stage": "refedit_grounding",
        "prompt_version": GROUND_PROMPT_VERSION,
        "total_rows": sum(job.num_rows for job in jobs),
        "device_groups": groups,
        "jobs": [asdict(job) for job in jobs],
        "args": args_dict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=worker_main,
            args=(index, devices, worker_jobs, args_dict, progress_queue),
        )
        for index, (devices, worker_jobs) in enumerate(assignments)
    ]
    for process in processes:
        process.start()
    messages, worker_errors, done = [], [], 0
    with tqdm(
        total=sum(job.num_rows for job in jobs),
        desc="RefEdit grounding rows",
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
        raise SystemExit("RefEdit grounding workers failed")


if __name__ == "__main__":
    main()
