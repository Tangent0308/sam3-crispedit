"""Eight-worker vLLM runner for the CrispEdit pair-quality prefilter."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from crispedit.prefilter.pair_quality import (
    EVIDENCE_SCHEMA,
    JSON_CORRECTION_PROMPT,
    PREFILTER_METHOD,
    PROMPT_VERSION,
    build_quality_conversation,
    extract_json_object,
    failed_dimensions,
    normalize_quality_assessment,
    unresolved_dimensions,
)
from crispedit.common import canonical_edit_type, decode_image, raw_type_from_filename
from crispedit.inference import (
    Qwen38FilterEngine,
    assign_jobs,
    parse_device_groups,
)


DEFAULT_MODEL = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
)


AUDIT_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("raw_type", pa.string()),
        ("canonical_type", pa.string()),
        ("instruction", pa.string()),
        ("prefilter_verdict", pa.string()),
        ("prefilter_decision", pa.string()),
        ("prefilter_confidence", pa.float64()),
        ("prefilter_reason", pa.string()),
        ("prefilter_failure_mode", pa.string()),
        ("prefilter_parse_ok", pa.bool_()),
        ("prefilter_model_name", pa.string()),
        ("prefilter_method", pa.string()),
        ("prefilter_evidence_schema", pa.string()),
        ("prefilter_run_id", pa.string()),
        ("prefilter_prompt_version", pa.string()),
        ("prefilter_assessment_json", pa.string()),
        ("prefilter_failed_dimensions_json", pa.string()),
        ("prefilter_unresolved_dimensions_json", pa.string()),
        ("prefilter_reason_codes_json", pa.string()),
        ("prefilter_raw_response", pa.string()),
        ("prefilter_attempts_json", pa.string()),
        ("prefilter_error", pa.string()),
        ("prefilter_inference_seconds", pa.float64()),
        ("filter_decision", pa.string()),
        ("filter_reason_codes", pa.string()),
        ("filter_mismatch_score", pa.float64()),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("prefilter_verdict", pa.string()),
        ("prefilter_decision", pa.string()),
        ("prefilter_confidence", pa.float64()),
        ("prefilter_reason", pa.string()),
        ("prefilter_failure_mode", pa.string()),
        ("prefilter_parse_ok", pa.bool_()),
        ("prefilter_model_name", pa.string()),
        ("prefilter_method", pa.string()),
        ("prefilter_evidence_schema", pa.string()),
        ("prefilter_run_id", pa.string()),
        ("prefilter_prompt_version", pa.string()),
        ("prefilter_failed_dimensions_json", pa.string()),
        ("prefilter_unresolved_dimensions_json", pa.string()),
        ("prefilter_reason_codes_json", pa.string()),
        ("filter_decision", pa.string()),
        ("filter_reason_codes", pa.string()),
        ("filter_mismatch_score", pa.float64()),
    ]
)


@dataclass
class PairQualityJob:
    input_path: str
    audit_path: str
    manifest_path: str
    num_rows: int
    row_indices: Tuple[int, ...] = ()


class Qwen38PairAuditor(Qwen38FilterEngine):
    """Reuse the deterministic paired-image vLLM transport."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--include-types", default="")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Process one SHARD.parquet:ROW_IDX case; repeat for a regression slice",
    )
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--shard-list-file", type=Path,
                        help="Text file with exact parquet basenames to process on this node")
    parser.add_argument("--limit-rows-per-shard", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1_310_720)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=4)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--progress-mininterval", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_cases(values: Iterable[str]) -> Dict[str, Tuple[int, ...]]:
    selected: Dict[str, Set[int]] = {}
    for value in values:
        try:
            shard, raw_index = str(value).rsplit(":", 1)
            row_idx = int(raw_index)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid --case {value!r}; expected SHARD.parquet:ROW_IDX") from exc
        if not shard.endswith(".parquet") or row_idx < 0:
            raise ValueError(f"invalid --case {value!r}; expected SHARD.parquet:ROW_IDX")
        selected.setdefault(shard, set()).add(row_idx)
    return {name: tuple(sorted(indices)) for name, indices in selected.items()}


def build_jobs(args: argparse.Namespace) -> List[PairQualityJob]:
    selected = parse_cases(args.case)
    if selected and args.limit_rows_per_shard is not None:
        raise ValueError("--case and --limit-rows-per-shard are mutually exclusive")
    include_types = {
        canonical_edit_type(value)
        for value in str(args.include_types or "").split(",")
        if value.strip()
    }
    paths = sorted(args.input_dir.glob("*.parquet"))
    shard_list_file = getattr(args, "shard_list_file", None)
    if shard_list_file is not None:
        names = [line.strip() for line in shard_list_file.read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)) or any(Path(name).name != name or not name.endswith('.parquet') for name in names):
            raise ValueError("--shard-list-file must contain unique parquet basenames")
        available = {path.name for path in paths}
        missing = sorted(set(names) - available)
        if missing:
            raise FileNotFoundError(f"shard list contains missing input shards: {missing[:10]}")
        paths = [path for path in paths if path.name in set(names)]
    if selected:
        available = {path.name for path in paths}
        missing = sorted(set(selected) - available)
        if missing:
            raise FileNotFoundError(f"selected shard(s) not found: {missing}")
        paths = [path for path in paths if path.name in selected]
    if include_types:
        paths = [
            path
            for path in paths
            if canonical_edit_type(raw_type_from_filename(path)) in include_types
        ]
    if args.limit_shards is not None:
        paths = paths[: args.limit_shards]

    jobs = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = {"input_img", "output_img", "instruction", "type"} - set(
            parquet.schema_arrow.names
        )
        if missing:
            raise ValueError(f"{path.name} misses required columns: {sorted(missing)}")
        row_indices = selected.get(path.name, ())
        if row_indices and row_indices[-1] >= parquet.metadata.num_rows:
            raise IndexError(
                f"selected row out of range for {path.name}: {row_indices[-1]}"
            )
        num_rows = len(row_indices) if row_indices else parquet.metadata.num_rows
        if args.limit_rows_per_shard is not None:
            num_rows = min(num_rows, args.limit_rows_per_shard)
        jobs.append(
            PairQualityJob(
                input_path=str(path),
                audit_path=str(args.output_dir / "audit" / path.name),
                manifest_path=str(args.output_dir / "manifest" / path.name),
                num_rows=num_rows,
                row_indices=row_indices,
            )
        )
    if not jobs:
        raise ValueError("pair-quality prefilter selection produced no jobs")
    return jobs


def _auditor_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        inference_backend="vllm",
        model_path=str(args.model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        max_pixels=args.max_pixels,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_enforce_eager=args.vllm_enforce_eager,
        max_images_per_generate=max(2, args.batch_size * 2),
        request_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        locator_max_new_tokens=args.max_new_tokens,
        planner_max_new_tokens=args.max_new_tokens,
        parse_retries=args.parse_retries,
        gpu_memory_gib=74,
    )


def _parse_with_retry(
    auditor: Qwen38PairAuditor,
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
                {"attempt": attempt, "parse_ok": False, "error": error, "raw_text": text}
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
            {"attempt": attempt, "parse_ok": True, "error": "", "raw_text": text}
        )
        return assessment, text, attempts
    raise AssertionError("unreachable")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _filter_score(verdict: str, confidence: float) -> float:
    if verdict == "PASS":
        return 0.0
    if verdict == "UNSURE":
        return max(0.5, confidence)
    return confidence if verdict == "FAIL" else 1.0


def _reason_fields(assessment: Dict) -> Tuple[List[str], str]:
    failed = list(failed_dimensions(assessment))
    unresolved = list(unresolved_dimensions(assessment))
    codes = list(assessment.get("reason_codes") or [])
    if not codes:
        codes = [f"{name.upper()}_FAIL" for name in failed]
        codes.extend(f"{name.upper()}_UNSURE" for name in unresolved)
    failure_mode = codes[0] if codes else "OTHER"
    return codes, failure_mode


def _audit_row(
    job: PairQualityJob,
    row_idx: int,
    record: Dict,
    assessment: Dict,
    raw_text: str,
    attempts: List[Dict],
    model_name: str,
    run_id: str,
    seconds: float,
) -> Dict:
    verdict = str(assessment["verdict"])
    decision = "keep" if verdict == "PASS" else "drop"
    confidence = float(assessment.get("confidence", 0.0))
    codes, failure_mode = _reason_fields(assessment)
    return {
        "row_idx": row_idx,
        "raw_type": str(record.get("type") or raw_type_from_filename(Path(job.input_path))),
        "canonical_type": canonical_edit_type(record.get("type")),
        "instruction": str(record.get("instruction", "")),
        "prefilter_verdict": verdict,
        "prefilter_decision": decision,
        "prefilter_confidence": confidence,
        "prefilter_reason": str(assessment.get("summary", "")),
        "prefilter_failure_mode": failure_mode,
        "prefilter_parse_ok": True,
        "prefilter_model_name": model_name,
        "prefilter_method": PREFILTER_METHOD,
        "prefilter_evidence_schema": EVIDENCE_SCHEMA,
        "prefilter_run_id": run_id,
        "prefilter_prompt_version": PROMPT_VERSION,
        "prefilter_assessment_json": _json(assessment),
        "prefilter_failed_dimensions_json": _json(list(failed_dimensions(assessment))),
        "prefilter_unresolved_dimensions_json": _json(list(unresolved_dimensions(assessment))),
        "prefilter_reason_codes_json": _json(codes),
        "prefilter_raw_response": raw_text,
        "prefilter_attempts_json": _json(attempts),
        "prefilter_error": "",
        "prefilter_inference_seconds": float(seconds),
        "filter_decision": decision,
        "filter_reason_codes": "|".join(codes),
        "filter_mismatch_score": _filter_score(verdict, confidence),
    }


def _error_row(
    job: PairQualityJob,
    row_idx: int,
    record: Dict,
    model_name: str,
    run_id: str,
    error: object,
    raw_text: str = "",
) -> Dict:
    message = repr(error)
    return {
        "row_idx": row_idx,
        "raw_type": str(record.get("type") or raw_type_from_filename(Path(job.input_path))),
        "canonical_type": canonical_edit_type(record.get("type")),
        "instruction": str(record.get("instruction", "")),
        "prefilter_verdict": "ERROR",
        "prefilter_decision": "drop",
        "prefilter_confidence": 0.0,
        "prefilter_reason": message,
        "prefilter_failure_mode": "PREFILTER_ERROR",
        "prefilter_parse_ok": False,
        "prefilter_model_name": model_name,
        "prefilter_method": PREFILTER_METHOD,
        "prefilter_evidence_schema": EVIDENCE_SCHEMA,
        "prefilter_run_id": run_id,
        "prefilter_prompt_version": PROMPT_VERSION,
        "prefilter_assessment_json": "{}",
        "prefilter_failed_dimensions_json": "[]",
        "prefilter_unresolved_dimensions_json": "[]",
        "prefilter_reason_codes_json": '["PREFILTER_ERROR"]',
        "prefilter_raw_response": raw_text,
        "prefilter_attempts_json": "[]",
        "prefilter_error": message,
        "prefilter_inference_seconds": 0.0,
        "filter_decision": "drop",
        "filter_reason_codes": "PREFILTER_ERROR",
        "filter_mismatch_score": 1.0,
    }


def _manifest_row(row: Dict) -> Dict:
    return {name: row[name] for name in MANIFEST_SCHEMA.names}


def _iter_batches(job: PairQualityJob, batch_size: int):
    path = Path(job.input_path)
    if job.row_indices:
        table = pq.read_table(path)
        pending = []
        for row_idx in job.row_indices:
            pending.append((row_idx, table.slice(row_idx, 1).to_pylist()[0]))
            if len(pending) == batch_size:
                yield pending
                pending = []
        if pending:
            yield pending
        return
    row_idx = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        records = batch.to_pylist()
        if row_idx + len(records) > job.num_rows:
            records = records[: job.num_rows - row_idx]
        if records:
            yield [(row_idx + offset, record) for offset, record in enumerate(records)]
        row_idx += len(records)
        if row_idx >= job.num_rows:
            break


def _current_output(job: PairQualityJob) -> bool:
    audit_path, manifest_path = Path(job.audit_path), Path(job.manifest_path)
    if not audit_path.is_file() or not manifest_path.is_file():
        return False
    audit = pq.read_table(audit_path, columns=["row_idx", "prefilter_prompt_version"])
    manifest = pq.read_table(manifest_path, columns=["row_idx"])
    if audit.num_rows != job.num_rows or manifest.num_rows != job.num_rows:
        return False
    if set(audit["prefilter_prompt_version"].to_pylist()) != {PROMPT_VERSION}:
        return False
    expected = list(job.row_indices) if job.row_indices else list(range(job.num_rows))
    return audit["row_idx"].to_pylist() == expected == manifest["row_idx"].to_pylist()


def _existing_summary(job: PairQualityJob) -> Dict:
    rows = pq.read_table(job.audit_path, columns=["prefilter_verdict"]).to_pylist()
    return {
        "rows": len(rows),
        "verdicts": dict(Counter(row["prefilter_verdict"] for row in rows)),
        "parse_errors": sum(row["prefilter_verdict"] == "ERROR" for row in rows),
        "skipped_existing": True,
    }


def process_job(
    job: PairQualityJob,
    auditor: Qwen38PairAuditor,
    args: argparse.Namespace,
    progress_queue,
) -> Dict:
    audit_path, manifest_path = Path(job.audit_path), Path(job.manifest_path)
    audit_tmp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": 0,
        "verdicts": Counter(),
        "parse_errors": 0,
        "skipped_existing": False,
    }
    completed = False
    audit_writer = pq.ParquetWriter(audit_tmp, AUDIT_SCHEMA, compression="zstd")
    manifest_writer = pq.ParquetWriter(manifest_tmp, MANIFEST_SCHEMA, compression="zstd")
    try:
        for indexed in _iter_batches(job, args.batch_size):
            prepared = []
            output_rows = []
            for row_idx, record in indexed:
                try:
                    source = decode_image(record["input_img"])
                    target = decode_image(record["output_img"])
                    conversation = build_quality_conversation(
                        source, target, record.get("type"), record.get("instruction")
                    )
                    prepared.append((row_idx, record, conversation))
                except Exception as exc:
                    output_rows.append(
                        _error_row(
                            job,
                            row_idx,
                            record,
                            args.model_path.name,
                            args.run_id,
                            exc,
                        )
                    )
            if prepared:
                started = time.monotonic()
                raw_outputs = auditor.generate(
                    [item[2] for item in prepared], max_tokens=args.max_new_tokens
                )
                seconds = (time.monotonic() - started) / len(prepared)
                for (row_idx, record, conversation), raw_text in zip(
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
                            job,
                            row_idx,
                            record,
                            assessment,
                            final_text,
                            attempts,
                            args.model_path.name,
                            args.run_id,
                            seconds,
                        )
                    except Exception as exc:
                        row = _error_row(
                            job,
                            row_idx,
                            record,
                            args.model_path.name,
                            args.run_id,
                            exc,
                            raw_text=raw_text,
                        )
                    output_rows.append(row)
            output_rows.sort(key=lambda row: int(row["row_idx"]))
            audit_writer.write_table(pa.Table.from_pylist(output_rows, schema=AUDIT_SCHEMA))
            manifest_writer.write_table(
                pa.Table.from_pylist(
                    [_manifest_row(row) for row in output_rows], schema=MANIFEST_SCHEMA
                )
            )
            for row in output_rows:
                summary["rows"] += 1
                summary["verdicts"][row["prefilter_verdict"]] += 1
                summary["parse_errors"] += not row["prefilter_parse_ok"]
            progress_queue.put({"kind": "rows", "count": len(output_rows)})
        if summary["rows"] != job.num_rows:
            raise ValueError(
                f"row count mismatch {Path(job.input_path).name}: "
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
    jobs: List[PairQualityJob],
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
                    f"pair-quality worker {worker_index} ready (lazy Qwen3.8 vLLM loading) "
                    f"on GPUs {devices} ({len(jobs)} shards)"
                ),
            }
        )
        auditor = None
        for job in jobs:
            if not args.overwrite and _current_output(job):
                summary = _existing_summary(job)
                progress_queue.put({"kind": "rows", "count": job.num_rows})
            else:
                if auditor is None:
                    auditor = Qwen38PairAuditor(_auditor_args(args))
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
        "skipped_existing_shards": 0,
    }
    for message in messages:
        summary = message["summary"]
        result["rows"] += int(summary.get("rows", 0))
        result["parse_errors"] += int(summary.get("parse_errors", 0))
        result["skipped_existing_shards"] += bool(summary.get("skipped_existing"))
        verdicts.update(summary.get("verdicts", {}))
    result["verdicts"] = dict(verdicts)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.run_id = "pair_prefilter_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
        raise ValueError("pair-quality output must not be inside the source dataset")
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
        "stage": "crispedit_pair_quality_prefilter",
        "method": PREFILTER_METHOD,
        "evidence_schema": EVIDENCE_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "model_path": str(args.model_path),
        "run_id": args.run_id,
        "total_rows": total,
        "device_groups": groups,
        "jobs": [asdict(job) for job in jobs],
        "args": args_dict,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
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
        desc="CrispEdit pair-quality rows",
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
                progress.write(
                    f"DONE worker={message['worker']} shard={message['shard']} "
                    f"summary={message['summary']}"
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
            "method": PREFILTER_METHOD,
            "prompt_version": PROMPT_VERSION,
            "run_id": args.run_id,
        }
    )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if (
        worker_errors
        or any(process.exitcode != 0 for process in processes)
        or summary["shards"] != len(jobs)
        or summary["rows"] != total
    ):
        raise SystemExit("CrispEdit pair-quality workers failed")


if __name__ == "__main__":
    main()
