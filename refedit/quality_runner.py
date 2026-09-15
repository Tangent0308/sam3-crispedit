"""Production RefEdit pair-quality prefilter with sharded vLLM workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from refedit.io import discover_shards, iter_row_batches, sample_id, validate_schema
from refedit.policy import infer_task
from refedit.quality_prefilter import (
    JSON_CORRECTION_PROMPT,
    QUALITY_PROMPT_VERSION,
    build_quality_conversation,
    extract_json_object,
    failed_dimensions,
    normalize_quality_assessment,
)
from scaleedit.grounding_runner import (
    Qwen35ScaleEditGrounder,
    assign_jobs,
    parse_device_groups,
)
from scaleedit.io import decode_image


DEFAULT_INPUT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit"
)
DEFAULT_MODEL = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
)


AUDIT_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("sample_id", pa.string()),
        ("img_id", pa.int64()),
        ("source_relative_path", pa.string()),
        ("task", pa.string()),
        ("instruction", pa.string()),
        ("verdict", pa.string()),
        ("keep", pa.bool_()),
        ("failed_dimensions_json", pa.string()),
        ("reason_codes_json", pa.string()),
        ("summary", pa.string()),
        ("assessment_json", pa.string()),
        ("raw_response", pa.string()),
        ("attempts_json", pa.string()),
        ("parse_ok", pa.bool_()),
        ("error", pa.string()),
        ("confidence", pa.float32()),
        ("model_name", pa.string()),
        ("prompt_version", pa.string()),
        ("inference_seconds", pa.float32()),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("sample_id", pa.string()),
        ("img_id", pa.int64()),
        ("source_relative_path", pa.string()),
        ("task", pa.string()),
        ("instruction", pa.string()),
        ("prefilter_verdict", pa.string()),
        ("prefilter_confidence", pa.float32()),
        ("prefilter_reason_codes_json", pa.string()),
        ("prefilter_model_name", pa.string()),
        ("prefilter_prompt_version", pa.string()),
    ]
)


@dataclass
class QualityJob:
    input_path: str
    audit_path: str
    manifest_path: str
    num_rows: int
    selected_sample_ids: Tuple[str, ...] = ()


class Qwen38QualityAuditor(Qwen35ScaleEditGrounder):
    """Reuse the tested paired-image vLLM transport without grounding calls."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs per worker; TP=1 creates eight data-parallel workers on eight devices",
    )
    parser.add_argument("--img-id", action="append", default=[])
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--limit-rows-per-shard", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1_310_720)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=4,
        help="Bound CUDA-graph profiling to the real per-worker batch size",
    )
    parser.add_argument("--progress-mininterval", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _selection_ids(args: argparse.Namespace) -> Set[int]:
    return {int(value) for value in args.img_id}


def build_jobs(args: argparse.Namespace) -> List[QualityJob]:
    requested = _selection_ids(args)
    if requested and args.limit_rows_per_shard is not None:
        raise ValueError("--img-id and --limit-rows-per-shard are mutually exclusive")
    paths = discover_shards(args.input_dir)
    if args.limit_shards is not None:
        paths = paths[: args.limit_shards]
    found: Set[int] = set()
    jobs = []
    for path in paths:
        validate_schema(path)
        selected: Tuple[str, ...] = ()
        if requested:
            ids = [
                int(value)
                for value in pq.read_table(path, columns=["img_id"])[0].to_pylist()
            ]
            shard_ids = sorted(requested.intersection(ids))
            found.update(shard_ids)
            if not shard_ids:
                continue
            num_rows = len(shard_ids)
            selected = tuple(sample_id(value) for value in shard_ids)
        else:
            num_rows = pq.ParquetFile(path).metadata.num_rows
            if args.limit_rows_per_shard is not None:
                num_rows = min(num_rows, args.limit_rows_per_shard)
        jobs.append(
            QualityJob(
                input_path=str(path),
                audit_path=str(args.output_dir / "audit" / path.name),
                manifest_path=str(args.output_dir / "manifest" / path.name),
                num_rows=num_rows,
                selected_sample_ids=selected,
            )
        )
    missing = requested - found
    if missing:
        raise KeyError(f"requested img_id values not found: {sorted(missing)[:20]}")
    if not jobs:
        raise ValueError("quality-prefilter selection produced no jobs")
    return jobs


def _grounder_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        inference_backend="vllm",
        model_path=str(args.model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        max_pixels=args.max_pixels,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        max_images_per_generate=max(2, args.batch_size * 2),
        request_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        locator_max_new_tokens=args.max_new_tokens,
        planner_max_new_tokens=args.max_new_tokens,
        parse_retries=args.parse_retries,
        gpu_memory_gib=74,
    )


def _parse_with_retry(
    auditor: Qwen38QualityAuditor,
    conversation: List[Dict],
    raw_text: str,
    retries: int,
    max_tokens: int,
) -> Tuple[Dict, str, List[Dict]]:
    attempts = []
    text = raw_text
    for attempt in range(retries + 1):
        try:
            assessment = normalize_quality_assessment(extract_json_object(text))
        except Exception as exc:
            error = repr(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "parse_ok": False,
                    "error": error,
                    "raw_text": text,
                }
            )
            if attempt >= retries:
                raise
            retry = list(conversation) + [
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": JSON_CORRECTION_PROMPT + "\nParser error: " + error[:300],
                        }
                    ],
                },
            ]
            text = auditor.generate([retry], max_tokens=max_tokens)[0]
            continue
        attempts.append(
            {
                "attempt": attempt,
                "parse_ok": True,
                "error": "",
                "raw_text": text,
            }
        )
        return assessment, text, attempts
    raise AssertionError("unreachable")


def _error_row(
    path: Path,
    row_idx: int,
    record: Dict,
    model_name: str,
    error: object,
) -> Dict:
    instruction = str(record.get("instruction", ""))
    return {
        "row_idx": row_idx,
        "sample_id": sample_id(record["img_id"]),
        "img_id": int(record["img_id"]),
        "source_relative_path": f"data/{path.name}",
        "task": infer_task(instruction).final_task,
        "instruction": instruction,
        "verdict": "ERROR",
        "keep": False,
        "failed_dimensions_json": "[]",
        "reason_codes_json": '["OTHER"]',
        "summary": "The sample could not be audited.",
        "assessment_json": "{}",
        "raw_response": "",
        "attempts_json": "[]",
        "parse_ok": False,
        "error": repr(error),
        "confidence": 0.0,
        "model_name": model_name,
        "prompt_version": QUALITY_PROMPT_VERSION,
        "inference_seconds": 0.0,
    }


def _audit_row(
    path: Path,
    row_idx: int,
    record: Dict,
    task: str,
    assessment: Dict,
    raw_text: str,
    attempts: List[Dict],
    model_name: str,
    seconds: float,
) -> Dict:
    return {
        "row_idx": row_idx,
        "sample_id": sample_id(record["img_id"]),
        "img_id": int(record["img_id"]),
        "source_relative_path": f"data/{path.name}",
        "task": task,
        "instruction": str(record["instruction"]),
        "verdict": str(assessment["verdict"]),
        "keep": bool(assessment["keep"]),
        "failed_dimensions_json": json.dumps(
            failed_dimensions(assessment), ensure_ascii=False
        ),
        "reason_codes_json": json.dumps(
            assessment.get("reason_codes", []), ensure_ascii=False
        ),
        "summary": str(assessment.get("summary", "")),
        "assessment_json": json.dumps(assessment, ensure_ascii=False),
        "raw_response": raw_text,
        "attempts_json": json.dumps(attempts, ensure_ascii=False),
        "parse_ok": True,
        "error": "",
        "confidence": float(assessment.get("confidence", 0.0)),
        "model_name": model_name,
        "prompt_version": QUALITY_PROMPT_VERSION,
        "inference_seconds": float(seconds),
    }


def _manifest_row(row: Dict) -> Dict:
    return {
        "row_idx": row["row_idx"],
        "sample_id": row["sample_id"],
        "img_id": row["img_id"],
        "source_relative_path": row["source_relative_path"],
        "task": row["task"],
        "instruction": row["instruction"],
        "prefilter_verdict": row["verdict"],
        "prefilter_confidence": row["confidence"],
        "prefilter_reason_codes_json": row["reason_codes_json"],
        "prefilter_model_name": row["model_name"],
        "prefilter_prompt_version": row["prompt_version"],
    }


def _current_output(job: QualityJob) -> bool:
    audit_path = Path(job.audit_path)
    manifest_path = Path(job.manifest_path)
    if not audit_path.is_file() or not manifest_path.is_file():
        return False
    if pq.ParquetFile(audit_path).metadata.num_rows != job.num_rows:
        return False
    versions = set(
        str(value)
        for value in pq.read_table(audit_path, columns=["prompt_version"])[0].to_pylist()
    )
    return versions == {QUALITY_PROMPT_VERSION}


def _existing_summary(job: QualityJob) -> Dict:
    table = pq.read_table(job.audit_path, columns=["verdict", "parse_ok"])
    rows = table.to_pylist()
    return {
        "rows": len(rows),
        "verdicts": dict(Counter(str(row["verdict"]) for row in rows)),
        "parse_errors": sum(not bool(row["parse_ok"]) for row in rows),
        "skipped_existing": True,
    }


def process_job(
    job: QualityJob,
    auditor: Qwen38QualityAuditor,
    args: argparse.Namespace,
    progress_queue,
) -> Dict:
    input_path = Path(job.input_path)
    audit_path = Path(job.audit_path)
    manifest_path = Path(job.manifest_path)
    audit_tmp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_writer = pq.ParquetWriter(audit_tmp, AUDIT_SCHEMA, compression="zstd")
    manifest_writer = pq.ParquetWriter(
        manifest_tmp, MANIFEST_SCHEMA, compression="zstd"
    )
    selected = set(job.selected_sample_ids)
    summary = {
        "rows": 0,
        "verdicts": Counter(),
        "parse_errors": 0,
        "corrupt_images": 0,
        "skipped_existing": False,
    }
    completed = False
    try:
        for indexed in iter_row_batches(input_path, args.batch_size):
            if selected:
                indexed = [
                    item for item in indexed if sample_id(item[1]["img_id"]) in selected
                ]
            else:
                indexed = [item for item in indexed if item[0] < job.num_rows]
            if not indexed:
                if selected:
                    continue
                break
            prepared = []
            output_rows = []
            for row_idx, record in indexed:
                task = infer_task(record["instruction"]).final_task
                try:
                    source = decode_image(record["source_img"])
                    target = decode_image(record["target_img"])
                    conversation = build_quality_conversation(
                        source, target, task, record["instruction"]
                    )
                except Exception as exc:
                    output_rows.append(
                        _error_row(
                            input_path,
                            row_idx,
                            record,
                            args.model_path.name,
                            exc,
                        )
                    )
                    summary["corrupt_images"] += 1
                    continue
                prepared.append((row_idx, record, task, conversation))

            if prepared:
                started = time.monotonic()
                raw_outputs = auditor.generate(
                    [item[3] for item in prepared], max_tokens=args.max_new_tokens
                )
                seconds = (time.monotonic() - started) / len(prepared)
                for (row_idx, record, task, conversation), raw_text in zip(
                    prepared, raw_outputs
                ):
                    try:
                        assessment, final_text, attempts = _parse_with_retry(
                            auditor,
                            conversation,
                            raw_text,
                            args.parse_retries,
                            args.max_new_tokens,
                        )
                        row = _audit_row(
                            input_path,
                            row_idx,
                            record,
                            task,
                            assessment,
                            final_text,
                            attempts,
                            args.model_path.name,
                            seconds,
                        )
                    except Exception as exc:
                        row = _error_row(
                            input_path,
                            row_idx,
                            record,
                            args.model_path.name,
                            exc,
                        )
                        row["raw_response"] = raw_text
                    output_rows.append(row)

            output_rows.sort(key=lambda row: int(row["row_idx"]))
            audit_writer.write_table(pa.Table.from_pylist(output_rows, schema=AUDIT_SCHEMA))
            keep_rows = [_manifest_row(row) for row in output_rows if row["keep"]]
            if keep_rows:
                manifest_writer.write_table(
                    pa.Table.from_pylist(keep_rows, schema=MANIFEST_SCHEMA)
                )
            for row in output_rows:
                summary["rows"] += 1
                summary["verdicts"][row["verdict"]] += 1
                summary["parse_errors"] += not row["parse_ok"]
            progress_queue.put({"kind": "rows", "count": len(output_rows)})
        if summary["rows"] != job.num_rows:
            raise ValueError(
                f"row count mismatch for {input_path.name}: "
                f"expected={job.num_rows} actual={summary['rows']}"
            )
        completed = True
    finally:
        audit_writer.close()
        manifest_writer.close()
        if completed:
            audit_tmp.replace(audit_path)
            manifest_tmp.replace(manifest_path)
    summary["verdicts"] = dict(summary["verdicts"])
    return summary


def worker_main(
    worker_index: int,
    devices: List[int],
    jobs: List[QualityJob],
    args_dict: Dict,
    progress_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    args = argparse.Namespace(**args_dict)
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.model_path = Path(args.model_path)
    try:
        progress_queue.put(
            {
                "kind": "log",
                "message": (
                    f"quality worker {worker_index} loading Qwen3.8 vLLM on GPUs "
                    f"{devices} ({len(jobs)} shards)"
                ),
            }
        )
        auditor = Qwen38QualityAuditor(_grounder_args(args))
        for job in jobs:
            if not args.overwrite and _current_output(job):
                summary = _existing_summary(job)
                progress_queue.put({"kind": "rows", "count": job.num_rows})
            else:
                summary = process_job(job, auditor, args, progress_queue)
            progress_queue.put(
                {
                    "kind": "shard_done",
                    "worker": worker_index,
                    "shard": Path(job.input_path).name,
                    "summary": summary,
                }
            )
    except Exception as exc:
        progress_queue.put(
            {
                "kind": "worker_error",
                "worker": worker_index,
                "devices": devices,
                "error": repr(exc),
            }
        )
        raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_index})


def _aggregate(messages: List[Dict]) -> Dict:
    verdicts = Counter()
    result = {
        "shards": len(messages),
        "rows": 0,
        "parse_errors": 0,
        "corrupt_images": 0,
        "skipped_existing_shards": 0,
    }
    for message in messages:
        summary = message["summary"]
        result["rows"] += int(summary.get("rows", 0))
        result["parse_errors"] += int(summary.get("parse_errors", 0))
        result["corrupt_images"] += int(summary.get("corrupt_images", 0))
        result["skipped_existing_shards"] += bool(summary.get("skipped_existing"))
        verdicts.update(summary.get("verdicts", {}))
    result["verdicts"] = dict(verdicts)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_path = args.model_path.resolve()
    if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
        raise ValueError("quality-prefilter output must not be inside RefEdit source data")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    jobs = build_jobs(args)
    groups = parse_device_groups(args.devices, args.tensor_parallel_size)
    assignments = assign_jobs(jobs, groups)
    total = sum(job.num_rows for job in jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        "stage": "refedit_pair_quality_prefilter",
        "prompt_version": QUALITY_PROMPT_VERSION,
        "model_path": str(args.model_path),
        "total_rows": total,
        "device_groups": groups,
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
            args=(index, devices, worker_jobs, args_dict, progress_queue),
        )
        for index, (devices, worker_jobs) in enumerate(assignments)
    ]
    for process in processes:
        process.start()
    messages, worker_errors, done = [], [], 0
    with tqdm(
        total=total,
        desc="RefEdit quality-prefilter rows",
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
            elif kind == "shard_done":
                messages.append(message)
                summary = message["summary"]
                progress.write(
                    f"DONE worker={message['worker']} shard={message['shard']} "
                    f"rows={summary.get('rows')} verdicts={summary.get('verdicts')}"
                )
            elif kind == "worker_error":
                worker_errors.append(message)
                progress.write(f"WORKER_ERROR {message}")
            elif kind == "worker_done":
                done += 1
    for process in processes:
        process.join()
    summary = _aggregate(messages)
    summary.update(
        {
            "expected_shards": len(jobs),
            "expected_rows": total,
            "worker_errors": worker_errors,
            "worker_exit_codes": [process.exitcode for process in processes],
            "model_name": args.model_path.name,
            "prompt_version": QUALITY_PROMPT_VERSION,
        }
    )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if (
        worker_errors
        or any(process.exitcode != 0 for process in processes)
        or summary["shards"] != len(jobs)
        or summary["rows"] != total
    ):
        raise SystemExit("RefEdit quality-prefilter workers failed")


if __name__ == "__main__":
    main()
