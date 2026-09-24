"""Eight-worker vLLM runner for the CrispEdit benchmark-scene filter."""

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
from typing import Dict, Iterable, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from crispedit.difficulty.benchmark_scene import (
    EVIDENCE_SCHEMA,
    FILTER_METHOD,
    JSON_CORRECTION_PROMPT,
    PROMPT_VERSION,
    build_scene_conversation,
    deterministic_screen,
    parse_scene_response,
)
from crispedit.prefilter.pair_runner import DEFAULT_MODEL, parse_cases
from crispedit.common import canonical_edit_type, decode_image, raw_type_from_filename
from crispedit.inference import (
    Qwen38FilterEngine,
    assign_jobs,
    parse_device_groups,
)


AUDIT_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("raw_type", pa.string()),
        ("canonical_type", pa.string()),
        ("instruction", pa.string()),
        ("source_prefilter_verdict", pa.string()),
        ("source_prefilter_run_id", pa.string()),
        ("scene_decision", pa.string()),
        ("scene_pass", pa.bool_()),
        ("scene_target", pa.string()),
        ("scene_reference", pa.string()),
        ("scene_reason", pa.string()),
        ("scene_model_called", pa.bool_()),
        ("scene_parse_ok", pa.bool_()),
        ("scene_model_name", pa.string()),
        ("scene_filter_method", pa.string()),
        ("scene_evidence_schema", pa.string()),
        ("scene_prompt_version", pa.string()),
        ("scene_run_id", pa.string()),
        ("scene_assessment_json", pa.string()),
        ("scene_raw_response", pa.string()),
        ("scene_attempts_json", pa.string()),
        ("scene_error", pa.string()),
        ("scene_inference_seconds", pa.float64()),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("source_prefilter_verdict", pa.string()),
        ("source_prefilter_run_id", pa.string()),
        ("scene_decision", pa.string()),
        ("scene_pass", pa.bool_()),
        ("scene_target", pa.string()),
        ("scene_reference", pa.string()),
        ("scene_reason", pa.string()),
        ("scene_model_called", pa.bool_()),
        ("scene_parse_ok", pa.bool_()),
        ("scene_model_name", pa.string()),
        ("scene_filter_method", pa.string()),
        ("scene_evidence_schema", pa.string()),
        ("scene_prompt_version", pa.string()),
        ("scene_run_id", pa.string()),
    ]
)


@dataclass
class SceneFilterJob:
    input_path: str
    prefilter_path: str
    audit_path: str
    manifest_path: str
    num_rows: int
    row_indices: Tuple[int, ...]


class Qwen38SceneAuditor(Qwen38FilterEngine):
    """Reuse the deterministic Qwen3.8 vLLM transport."""

    def shutdown(self, timeout: float = 30.0) -> None:
        """Explicitly stop vLLM's EngineCore subprocess before worker exit."""

        if self.inference_backend != "vllm":
            return
        llm_engine = getattr(self.model, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown(timeout=timeout)


class SceneResponseError(ValueError):
    """Carry every failed parse attempt into the audit output."""

    def __init__(self, error: Exception, raw_text: str, attempts: List[Dict]):
        super().__init__(str(error))
        self.original_error = error
        self.raw_text = raw_text
        self.attempts = attempts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prefilter-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--include-types", default="")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Process one prefilter-PASS SHARD.parquet:ROW_IDX case; repeatable",
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        help="JSON array/object or text file containing SHARD.parquet:ROW_IDX cases",
    )
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--shard-list-file", type=Path,
                        help="Text file with exact parquet basenames to process on this node")
    parser.add_argument("--limit-rows-per-shard", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=1_310_720)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=4)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--progress-mininterval", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_case_file(path: Path | None) -> List[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("cases", payload.get("selection", []))
    if not isinstance(payload, list):
        raise ValueError("case file must contain a JSON array, a cases object, or text lines")
    output = []
    for item in payload:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, dict) and "shard" in item and "row_idx" in item:
            output.append(f"{item['shard']}:{item['row_idx']}")
        else:
            raise ValueError(f"invalid case file item: {item!r}")
    return output


def _pass_indices(path: Path, expected_rows: int) -> Tuple[int, ...]:
    table = pq.read_table(
        path,
        columns=["row_idx", "prefilter_verdict"],
    )
    if table.num_rows != expected_rows:
        raise ValueError(
            f"prefilter row count mismatch for {path.name}: "
            f"expected={expected_rows} actual={table.num_rows}"
        )
    row_indices = [int(value) for value in table["row_idx"].to_pylist()]
    if row_indices != list(range(expected_rows)):
        raise ValueError(f"prefilter row_idx is not aligned for {path.name}")
    verdicts = table["prefilter_verdict"].to_pylist()
    return tuple(index for index, verdict in enumerate(verdicts) if verdict == "PASS")


def build_jobs(args: argparse.Namespace) -> List[SceneFilterJob]:
    selected = parse_cases(list(args.case) + _read_case_file(args.case_file))
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
        expected_rows = parquet.metadata.num_rows
        missing_columns = {"input_img", "instruction", "type"} - set(
            parquet.schema_arrow.names
        )
        if missing_columns:
            raise ValueError(f"{path.name} misses columns: {sorted(missing_columns)}")
        prefilter_path = args.prefilter_manifest_dir / path.name
        if not prefilter_path.is_file():
            raise FileNotFoundError(prefilter_path)
        pass_indices = set(_pass_indices(prefilter_path, expected_rows))
        if selected:
            requested = selected.get(path.name, ())
            rejected = sorted(set(requested) - pass_indices)
            if rejected:
                raise ValueError(
                    f"selected cases are not Qwen3.8 prefilter PASS in {path.name}: {rejected}"
                )
            row_indices = tuple(requested)
        else:
            row_indices = tuple(sorted(pass_indices))
        if args.limit_rows_per_shard is not None:
            row_indices = row_indices[: args.limit_rows_per_shard]
        if not row_indices:
            continue
        jobs.append(
            SceneFilterJob(
                input_path=str(path),
                prefilter_path=str(prefilter_path),
                audit_path=str(args.output_dir / "audit" / path.name),
                manifest_path=str(args.output_dir / "manifest" / path.name),
                num_rows=len(row_indices),
                row_indices=row_indices,
            )
        )
    if not jobs:
        raise ValueError("benchmark-scene filter selection produced no prefilter-PASS jobs")
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
        max_images_per_generate=max(1, args.batch_size),
        request_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        locator_max_new_tokens=args.max_new_tokens,
        planner_max_new_tokens=args.max_new_tokens,
        parse_retries=args.parse_retries,
        gpu_memory_gib=74,
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_with_retry(
    auditor: Qwen38SceneAuditor,
    conversation: List[Dict],
    raw_text: str,
    retries: int,
    max_tokens: int,
) -> Tuple[Dict, str, List[Dict]]:
    attempts = []
    text = raw_text
    for attempt in range(retries + 1):
        try:
            assessment = parse_scene_response(text)
        except Exception as exc:
            error = repr(exc)
            attempts.append(
                {"attempt": attempt, "parse_ok": False, "error": error, "raw_text": text}
            )
            if attempt >= retries:
                raise SceneResponseError(exc, text, attempts) from exc
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
        attempts.append({"attempt": attempt, "parse_ok": True, "error": "", "raw_text": text})
        return assessment, text, attempts
    raise AssertionError("unreachable")


def _base_row(
    *,
    row_idx: int,
    record: Dict,
    prefilter: Dict,
    model_name: str,
    run_id: str,
) -> Dict:
    return {
        "row_idx": row_idx,
        "raw_type": str(record.get("type") or ""),
        "canonical_type": canonical_edit_type(record.get("type")),
        "instruction": str(record.get("instruction") or ""),
        "source_prefilter_verdict": str(prefilter.get("prefilter_verdict") or ""),
        "source_prefilter_run_id": str(prefilter.get("prefilter_run_id") or ""),
        "scene_model_name": model_name,
        "scene_filter_method": FILTER_METHOD,
        "scene_evidence_schema": EVIDENCE_SCHEMA,
        "scene_prompt_version": PROMPT_VERSION,
        "scene_run_id": run_id,
    }


def _assessment_row(
    *,
    row_idx: int,
    record: Dict,
    prefilter: Dict,
    assessment: Dict,
    raw_text: str,
    attempts: List[Dict],
    model_name: str,
    run_id: str,
    seconds: float,
) -> Dict:
    row = _base_row(
        row_idx=row_idx,
        record=record,
        prefilter=prefilter,
        model_name=model_name,
        run_id=run_id,
    )
    row.update(
        {
            "scene_decision": assessment["verdict"],
            "scene_pass": assessment["verdict"] == "PASS",
            "scene_target": assessment["target"],
            "scene_reference": assessment["reference"],
            "scene_reason": assessment["reason"],
            "scene_model_called": True,
            "scene_parse_ok": True,
            "scene_assessment_json": _json(assessment),
            "scene_raw_response": raw_text,
            "scene_attempts_json": _json(attempts),
            "scene_error": "",
            "scene_inference_seconds": float(seconds),
        }
    )
    return row


def _screened_row(
    *,
    row_idx: int,
    record: Dict,
    prefilter: Dict,
    model_name: str,
    run_id: str,
    reason: str,
) -> Dict:
    row = _base_row(
        row_idx=row_idx,
        record=record,
        prefilter=prefilter,
        model_name=model_name,
        run_id=run_id,
    )
    assessment = {
        "verdict": "DROP",
        "target": "ineligible edit type",
        "reference": "NONE",
        "reason": reason,
    }
    row.update(
        {
            "scene_decision": "DROP",
            "scene_pass": False,
            "scene_target": "ineligible edit type",
            "scene_reference": "NONE",
            "scene_reason": reason,
            "scene_model_called": False,
            "scene_parse_ok": True,
            "scene_assessment_json": _json(assessment),
            "scene_raw_response": "",
            "scene_attempts_json": "[]",
            "scene_error": "",
            "scene_inference_seconds": 0.0,
        }
    )
    return row


def _error_row(
    *,
    row_idx: int,
    record: Dict,
    prefilter: Dict,
    model_name: str,
    run_id: str,
    error: object,
    raw_text: str = "",
    attempts: List[Dict] | None = None,
    seconds: float = 0.0,
) -> Dict:
    message = repr(error)
    row = _base_row(
        row_idx=row_idx,
        record=record,
        prefilter=prefilter,
        model_name=model_name,
        run_id=run_id,
    )
    row.update(
        {
            "scene_decision": "DROP",
            "scene_pass": False,
            "scene_target": "unavailable",
            "scene_reference": "NONE",
            "scene_reason": message,
            "scene_model_called": True,
            "scene_parse_ok": False,
            "scene_assessment_json": "{}",
            "scene_raw_response": raw_text,
            "scene_attempts_json": _json(attempts or []),
            "scene_error": message,
            "scene_inference_seconds": float(seconds),
        }
    )
    return row


def _manifest_row(row: Dict) -> Dict:
    return {name: row[name] for name in MANIFEST_SCHEMA.names}


def _iter_batches(job: SceneFilterJob, batch_size: int):
    raw_table = pq.read_table(job.input_path)
    prefilter_table = pq.read_table(job.prefilter_path)
    pending = []
    for row_idx in job.row_indices:
        record = raw_table.slice(row_idx, 1).to_pylist()[0]
        prefilter = prefilter_table.slice(row_idx, 1).to_pylist()[0]
        pending.append((row_idx, record, prefilter))
        if len(pending) == batch_size:
            yield pending
            pending = []
    if pending:
        yield pending


def _current_output(job: SceneFilterJob) -> bool:
    audit_path, manifest_path = Path(job.audit_path), Path(job.manifest_path)
    if not audit_path.is_file() or not manifest_path.is_file():
        return False
    audit_columns = set(pq.ParquetFile(audit_path).schema_arrow.names)
    manifest_columns = set(pq.ParquetFile(manifest_path).schema_arrow.names)
    if not set(AUDIT_SCHEMA.names).issubset(audit_columns):
        return False
    if not set(MANIFEST_SCHEMA.names).issubset(manifest_columns):
        return False
    audit = pq.read_table(
        audit_path,
        columns=["row_idx", "scene_prompt_version", "scene_parse_ok"],
    )
    manifest = pq.read_table(manifest_path, columns=["row_idx"])
    if audit.num_rows != job.num_rows or manifest.num_rows != job.num_rows:
        return False
    if set(audit["scene_prompt_version"].to_pylist()) != {PROMPT_VERSION}:
        return False
    if not all(audit["scene_parse_ok"].to_pylist()):
        return False
    expected = list(job.row_indices)
    return audit["row_idx"].to_pylist() == expected == manifest["row_idx"].to_pylist()


def _existing_summary(job: SceneFilterJob) -> Dict:
    rows = pq.read_table(
        job.audit_path,
        columns=["scene_decision", "scene_model_called", "scene_parse_ok"],
    ).to_pylist()
    return {
        "rows": len(rows),
        "decisions": dict(Counter(row["scene_decision"] for row in rows)),
        "model_calls": sum(bool(row["scene_model_called"]) for row in rows),
        "parse_errors": sum(not bool(row["scene_parse_ok"]) for row in rows),
        "skipped_existing": True,
    }


def process_job(
    job: SceneFilterJob,
    auditor: Qwen38SceneAuditor,
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
        "decisions": Counter(),
        "model_calls": 0,
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
            for row_idx, record, prefilter in indexed:
                screen = deterministic_screen(record.get("type"))
                if not screen["eligible"]:
                    output_rows.append(
                        _screened_row(
                            row_idx=row_idx,
                            record=record,
                            prefilter=prefilter,
                            model_name=args.model_path.name,
                            run_id=args.run_id,
                            reason=screen["reason"],
                        )
                    )
                    continue
                try:
                    source = decode_image(record["input_img"])
                    conversation = build_scene_conversation(
                        source, record.get("type"), record.get("instruction")
                    )
                    prepared.append((row_idx, record, prefilter, conversation))
                except Exception as exc:
                    output_rows.append(
                        _error_row(
                            row_idx=row_idx,
                            record=record,
                            prefilter=prefilter,
                            model_name=args.model_path.name,
                            run_id=args.run_id,
                            error=exc,
                        )
                    )
            if prepared:
                started = time.monotonic()
                raw_outputs = auditor.generate(
                    [item[3] for item in prepared], max_tokens=args.max_new_tokens
                )
                seconds = (time.monotonic() - started) / len(prepared)
                for (row_idx, record, prefilter, conversation), raw_text in zip(
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
                        row = _assessment_row(
                            row_idx=row_idx,
                            record=record,
                            prefilter=prefilter,
                            assessment=assessment,
                            raw_text=final_text,
                            attempts=attempts,
                            model_name=args.model_path.name,
                            run_id=args.run_id,
                            seconds=seconds,
                        )
                    except Exception as exc:
                        parse_error = exc if isinstance(exc, SceneResponseError) else None
                        row = _error_row(
                            row_idx=row_idx,
                            record=record,
                            prefilter=prefilter,
                            model_name=args.model_path.name,
                            run_id=args.run_id,
                            error=(
                                parse_error.original_error if parse_error else exc
                            ),
                            raw_text=(parse_error.raw_text if parse_error else raw_text),
                            attempts=(parse_error.attempts if parse_error else None),
                            seconds=seconds,
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
                summary["decisions"][row["scene_decision"]] += 1
                summary["model_calls"] += bool(row["scene_model_called"])
                summary["parse_errors"] += not bool(row["scene_parse_ok"])
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
    summary["decisions"] = dict(summary["decisions"])
    return summary


def worker_main(
    worker_index: int,
    devices: List[int],
    jobs: List[SceneFilterJob],
    args_dict: Dict,
    progress_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    args = argparse.Namespace(**args_dict)
    args.input_dir = Path(args.input_dir)
    args.prefilter_manifest_dir = Path(args.prefilter_manifest_dir)
    args.output_dir = Path(args.output_dir)
    args.model_path = Path(args.model_path)
    auditor = None
    try:
        progress_queue.put(
            {
                "kind": "log",
                "message": (
                    f"benchmark-scene worker {worker_index} ready (lazy Qwen3.8 vLLM loading) "
                    f"on GPUs {devices} ({len(jobs)} shards)"
                ),
            }
        )
        for job in jobs:
            if not args.overwrite and _current_output(job):
                summary = _existing_summary(job)
                progress_queue.put({"kind": "rows", "count": job.num_rows})
            else:
                if auditor is None:
                    auditor = Qwen38SceneAuditor(_auditor_args(args))
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
        shutdown_error = None
        if auditor is not None:
            try:
                auditor.shutdown()
            except Exception as exc:
                shutdown_error = exc
                progress_queue.put(
                    {
                        "kind": "worker_error",
                        "worker": worker_index,
                        "devices": devices,
                        "error": f"vLLM shutdown failed: {exc!r}",
                    }
                )
        progress_queue.put({"kind": "worker_done", "worker": worker_index})
        if shutdown_error is not None:
            raise shutdown_error


def _aggregate(messages: Iterable[Dict]) -> Dict:
    decisions = Counter()
    result = {
        "shards": 0,
        "rows": 0,
        "model_calls": 0,
        "parse_errors": 0,
        "skipped_existing_shards": 0,
    }
    for message in messages:
        summary = message["summary"]
        result["shards"] += 1
        result["rows"] += int(summary.get("rows", 0))
        result["model_calls"] += int(summary.get("model_calls", 0))
        result["parse_errors"] += int(summary.get("parse_errors", 0))
        result["skipped_existing_shards"] += bool(summary.get("skipped_existing"))
        decisions.update(summary.get("decisions", {}))
    result["decisions"] = dict(decisions)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.prefilter_manifest_dir = args.prefilter_manifest_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_path = args.model_path.resolve()
    args.run_id = "benchmark_scene_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
        raise ValueError("scene-filter output must not be inside the source dataset")
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
        "stage": "crispedit_benchmark_scene_filter",
        "method": FILTER_METHOD,
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
        desc="CrispEdit benchmark-scene rows",
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
    join_deadline = time.monotonic() + 90.0
    for process in processes:
        process.join(timeout=max(0.0, join_deadline - time.monotonic()))
    lingering = [index for index, process in enumerate(processes) if process.is_alive()]
    if lingering:
        worker_errors.append(
            {
                "kind": "worker_error",
                "workers": lingering,
                "error": "worker exit exceeded 90 seconds after inference",
            }
        )
        for index in lingering:
            processes[index].terminate()
        terminate_deadline = time.monotonic() + 10.0
        for index in lingering:
            processes[index].join(
                timeout=max(0.0, terminate_deadline - time.monotonic())
            )
        for index in lingering:
            if processes[index].is_alive():
                processes[index].kill()
                processes[index].join(timeout=5.0)
    summary = _aggregate(messages)
    summary.update(
        {
            "expected_shards": len(jobs),
            "expected_rows": total,
            "worker_errors": worker_errors,
            "worker_exit_codes": [process.exitcode for process in processes],
            "model_name": args.model_path.name,
            "method": FILTER_METHOD,
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
        raise SystemExit("CrispEdit benchmark-scene workers failed")


if __name__ == "__main__":
    main()
