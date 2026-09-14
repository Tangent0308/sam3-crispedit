#!/usr/bin/env python3
"""Run the referential-edit difficulty filter over all unified dataset shards.

The supervisor uses external data parallelism: four independent vLLM engines
with tensor parallelism two consume all eight GPUs.  After they exit, eight
single-GPU SAM3 workers reuse the same devices.  Every source parquet produces
one small evidence parquet, so the run is resumable at source-shard granularity
without copying source or edited images into the result dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from difficulty_filter.referential import (  # noqa: E402
    FILTER_POLICY_VERSION,
    MLLM_PROMPT_VERSION,
    SAM_COUNT_POLICY_VERSION,
    build_mllm_prompt,
    deterministic_screen,
    extract_grounding_refs,
    fuse_evidence,
    instruction_cues,
    parse_mllm_response,
)
from filter_referential_edits import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_QWEN_PATH,
    DEFAULT_SAM_PATH,
    METADATA_COLUMNS,
    _decode_image,
    _device_list,
    _run_sam_count,
    _write_json,
    _write_parquet,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "ScaleEdit-CrispEdit-mask-referential-filter"
)

MLLM_SOURCE_COLUMNS = [
    "sample_id",
    "edit_type",
    "instruction",
    "mask_mode",
    "ground_json",
    "source_image",
]
SAM_SOURCE_COLUMNS = ["sample_id", "source_image", "mask_png"]

MLLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "edited_subject_phrase": {"type": "string", "maxLength": 200},
        "object_category": {"type": "string", "maxLength": 100},
        "visible_same_class_count": {"type": "integer", "minimum": 0},
        "selected_instance_count": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "subset_relation": {
            "type": "string",
            "enum": ["yes", "likely", "no", "uncertain"],
        },
        "reference_cues": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "spatial",
                    "ordinal",
                    "cardinality",
                    "relation",
                    "appearance",
                    "identity",
                    "none",
                ],
            },
        },
        "fine_grained_referential": {
            "type": "string",
            "enum": ["yes", "likely", "no"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 240},
    },
    "required": [
        "edited_subject_phrase",
        "object_category",
        "visible_same_class_count",
        "selected_instance_count",
        "subset_relation",
        "reference_cues",
        "fine_grained_referential",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dataset_names(spec: str) -> Tuple[str, ...]:
    values = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("at least one dataset name is required")
    return values


def discover_shards(
    dataset_root: Path,
    datasets: Sequence[str],
    shard_glob: str = "*.parquet",
) -> List[Tuple[str, Path]]:
    shards: List[Tuple[str, Path]] = []
    for dataset in datasets:
        paths = sorted((dataset_root / dataset / "data").glob(shard_glob))
        if not paths:
            raise FileNotFoundError(
                f"no source shards match {shard_glob!r} for {dataset}"
            )
        shards.extend((dataset, path) for path in paths)
    return shards


def _assigned_shards(
    shards: Sequence[Tuple[str, Path]], worker_index: int, num_workers: int
) -> List[Tuple[str, Path]]:
    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError("worker-index must satisfy 0 <= index < num-workers")
    return list(shards[worker_index::num_workers])


def _effective_rows(path: Path, max_rows_per_shard: int) -> int:
    rows = pq.ParquetFile(path).metadata.num_rows
    return min(rows, max_rows_per_shard) if max_rows_per_shard > 0 else rows


def _evidence_path(
    output_root: Path, stage: str, dataset: str, source_path: Path
) -> Path:
    return output_root / stage / dataset / source_path.name


def _valid_evidence(
    output: Path,
    expected_rows: int,
    version_column: str,
    expected_version: str,
) -> bool:
    if not output.is_file():
        return False
    try:
        parquet = pq.ParquetFile(output)
        if parquet.metadata.num_rows != expected_rows:
            return False
        # The unified source contains legitimate zero-row parquet shards. The
        # generic empty evidence writer has no columns, so readability plus the
        # expected zero count is the complete validation contract for them.
        if expected_rows == 0:
            return True
        values = pq.read_table(output, columns=[version_column]).column(0).to_pylist()
        return bool(values) and all(value == expected_version for value in values)
    except Exception:
        return False


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body["updated_at"] = _utc_now()
    _write_json(body, path)


def _mllm_skip_record(row: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "mllm_status": "skipped_deterministic",
        "prompt_version": MLLM_PROMPT_VERSION,
        "prompt": "",
        "raw_response": "",
        "parse_error": "",
        "edited_subject_phrase": "",
        "object_category": "",
        "visible_same_class_count": None,
        "selected_instance_count": None,
        "subset_relation": "",
        "reference_cues": [],
        "fine_grained_referential": "",
        "mllm_confidence": None,
        "mllm_reason": reason,
    }


def _mllm_error_record(
    row: Mapping[str, Any], status: str, prompt: str, error: str
) -> Dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "mllm_status": status,
        "prompt_version": MLLM_PROMPT_VERSION,
        "prompt": prompt,
        "raw_response": "",
        "parse_error": error,
        "edited_subject_phrase": "",
        "object_category": "",
        "visible_same_class_count": None,
        "selected_instance_count": None,
        "subset_relation": "",
        "reference_cues": [],
        "fine_grained_referential": "",
        "mllm_confidence": None,
        "mllm_reason": "",
    }


def _load_mllm_engine(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    devices = _device_list(args.devices)
    if len(devices) != args.tensor_parallel_size:
        raise ValueError(
            "--devices must contain exactly tensor-parallel-size devices for "
            "each external data-parallel worker"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model = LLM(
        model=str(args.model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(1, args.batch_size),
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": args.max_pixels},
        mm_encoder_tp_mode="data",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        performance_mode="throughput",
        generation_config="vllm",
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        n=1,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        seed=0,
        skip_special_tokens=True,
        structured_outputs=(
            StructuredOutputsParams(json=MLLM_JSON_SCHEMA)
            if getattr(args, "structured_output", False)
            else None
        ),
    )
    return processor, model, sampling


def _generate_mllm_batch(
    rows: Sequence[Mapping[str, Any]],
    processor: Any,
    model: Any,
    sampling: Any,
    max_pixels: int,
    prompt_suffix: str = "",
) -> List[Dict[str, Any]]:
    requests = []
    valid_rows = []
    prompts = []
    records: List[Dict[str, Any]] = []
    for row in rows:
        prompt = build_mllm_prompt(
            str(row["instruction"]),
            str(row["edit_type"]),
            extract_grounding_refs(str(row.get("ground_json") or "")),
        )
        prompt += prompt_suffix
        try:
            image = _decode_image(row["source_image"])
        except Exception as exc:
            records.append(
                _mllm_error_record(row, "input_error", prompt, repr(exc))
            )
            continue
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        requests.append(
            {
                "prompt": rendered,
                "multi_modal_data": {"image": [image]},
                "mm_processor_kwargs": {"max_pixels": max_pixels},
            }
        )
        valid_rows.append(row)
        prompts.append(prompt)

    if requests:
        results = model.generate(requests, sampling, use_tqdm=False)
        if len(results) != len(valid_rows):
            raise RuntimeError("vLLM output count does not match request count")
        for row, prompt, result in zip(valid_rows, prompts, results):
            raw = result.outputs[0].text
            try:
                parsed = parse_mllm_response(raw)
                status = "ok"
                error = ""
            except Exception as exc:
                parsed = {
                    "edited_subject_phrase": "",
                    "object_category": "",
                    "visible_same_class_count": None,
                    "selected_instance_count": None,
                    "subset_relation": "",
                    "reference_cues": [],
                    "fine_grained_referential": "",
                    "confidence": None,
                    "reason": "",
                }
                status = "parse_error"
                error = repr(exc)
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "mllm_status": status,
                    "prompt_version": MLLM_PROMPT_VERSION,
                    "prompt": prompt,
                    "raw_response": raw,
                    "parse_error": error,
                    "edited_subject_phrase": parsed["edited_subject_phrase"],
                    "object_category": parsed["object_category"],
                    "visible_same_class_count": parsed[
                        "visible_same_class_count"
                    ],
                    "selected_instance_count": parsed["selected_instance_count"],
                    "subset_relation": parsed["subset_relation"],
                    "reference_cues": parsed["reference_cues"],
                    "fine_grained_referential": parsed[
                        "fine_grained_referential"
                    ],
                    "mllm_confidence": parsed["confidence"],
                    "mllm_reason": parsed["reason"],
                }
            )
    return records


def stage_repair_mllm(args: argparse.Namespace) -> None:
    """Retry only parse-error rows with constrained JSON and a short reason."""

    datasets = _dataset_names(args.datasets)
    evidence_paths = []
    for dataset in datasets:
        evidence_paths.extend(
            (dataset, path)
            for path in sorted(
                (args.output_root / "mllm" / dataset).glob("*.parquet")
            )
        )
    if not evidence_paths:
        raise FileNotFoundError(f"no MLLM evidence under {args.output_root / 'mllm'}")

    targets: List[Dict[str, Any]] = []
    locations: Dict[str, Tuple[Path, int]] = {}
    evidence_cache: Dict[Path, List[Dict[str, Any]]] = {}
    for dataset, evidence_path in evidence_paths:
        if pq.ParquetFile(evidence_path).metadata.num_rows == 0:
            continue
        evidence_rows = pq.read_table(evidence_path).to_pylist()
        failed_indices = [
            index
            for index, row in enumerate(evidence_rows)
            if row["mllm_status"] == "parse_error"
        ]
        if not failed_indices:
            continue
        source_path = args.dataset_root / dataset / "data" / evidence_path.name
        source_rows = pq.read_table(
            source_path, columns=MLLM_SOURCE_COLUMNS
        ).to_pylist()
        source_by_id = {str(row["sample_id"]): row for row in source_rows}
        evidence_cache[evidence_path] = evidence_rows
        for index in failed_indices:
            sample_id = str(evidence_rows[index]["sample_id"])
            source = source_by_id.get(sample_id)
            if source is None:
                raise RuntimeError(
                    f"repair target {sample_id} not found in {source_path}"
                )
            targets.append(source)
            locations[sample_id] = (evidence_path, index)

    if not targets:
        print("no MLLM parse errors require repair", flush=True)
        return
    print(
        f"repairing {len(targets)} MLLM parse errors with structured JSON",
        flush=True,
    )
    args.structured_output = True
    processor, model, sampling = _load_mllm_engine(args)
    repaired: List[Dict[str, Any]] = []
    suffix = (
        "\nIMPORTANT REPAIR CONSTRAINT: Return the JSON immediately. The reason "
        "must contain at most 20 words. Do not analyze, reconsider, or explain "
        "inside the reason."
    )
    for start in range(0, len(targets), args.batch_size):
        repaired.extend(
            _generate_mllm_batch(
                targets[start : start + args.batch_size],
                processor,
                model,
                sampling,
                args.max_pixels,
                prompt_suffix=suffix,
            )
        )
    failures = [row for row in repaired if row["mllm_status"] != "ok"]
    if failures:
        raise RuntimeError(
            "structured MLLM repair still failed for "
            f"{[row['sample_id'] for row in failures]}"
        )

    for record in repaired:
        evidence_path, index = locations[str(record["sample_id"])]
        evidence_cache[evidence_path][index] = record
    for evidence_path, rows in evidence_cache.items():
        _write_parquet(rows, evidence_path)
    print(
        f"successfully repaired {len(repaired)} MLLM rows across "
        f"{len(evidence_cache)} shards",
        flush=True,
    )


def stage_mllm_worker(args: argparse.Namespace) -> None:
    datasets = _dataset_names(args.datasets)
    shards = discover_shards(args.dataset_root, datasets, args.shard_glob)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    assigned = _assigned_shards(shards, args.worker_index, args.num_workers)
    total_rows = sum(
        _effective_rows(path, args.max_rows_per_shard) for _, path in assigned
    )
    progress_path = (
        args.output_root
        / "progress"
        / f"mllm-worker-{args.worker_index:02d}.json"
    )
    base_progress = {
        "stage": "mllm",
        "worker_index": args.worker_index,
        "pid": os.getpid(),
        "total_shards": len(assigned),
        "total_rows": total_rows,
        "completed_shards": 0,
        "processed_rows": 0,
        "model_calls": 0,
        "parse_errors": 0,
        "input_errors": 0,
        "current_shard": "",
        "state": "starting",
    }
    _write_progress(progress_path, base_progress)

    completed_shards = 0
    processed_rows = 0
    model_calls = 0
    parse_errors = 0
    input_errors = 0
    pending: List[Tuple[str, Path]] = []
    for dataset, source_path in assigned:
        expected = _effective_rows(source_path, args.max_rows_per_shard)
        output = _evidence_path(args.output_root, "mllm", dataset, source_path)
        if args.resume and _valid_evidence(
            output, expected, "prompt_version", MLLM_PROMPT_VERSION
        ):
            completed_shards += 1
            processed_rows += expected
        else:
            pending.append((dataset, source_path))

    progress = dict(base_progress)
    progress.update(
        {
            "completed_shards": completed_shards,
            "processed_rows": processed_rows,
            "state": "complete" if not pending else "loading_model",
        }
    )
    _write_progress(progress_path, progress)
    if not pending:
        print(f"worker={args.worker_index} all MLLM shards already complete", flush=True)
        return

    processor, model, sampling = _load_mllm_engine(args)
    progress["state"] = "running"
    _write_progress(progress_path, progress)
    print(
        f"worker={args.worker_index} vLLM ready devices={args.devices} "
        f"pending_shards={len(pending)} rows={total_rows - processed_rows}",
        flush=True,
    )

    for dataset, source_path in pending:
        progress["current_shard"] = f"{dataset}/{source_path.name}"
        _write_progress(progress_path, progress)
        table = pq.read_table(source_path, columns=MLLM_SOURCE_COLUMNS)
        if args.max_rows_per_shard > 0:
            table = table.slice(0, args.max_rows_per_shard)
        source_rows = table.to_pylist()
        by_id: Dict[str, Dict[str, Any]] = {}
        eligible = []
        for row in source_rows:
            screen = deterministic_screen(str(row["edit_type"]), str(row["mask_mode"]))
            if screen.eligible:
                eligible.append(row)
            else:
                by_id[str(row["sample_id"])] = _mllm_skip_record(
                    row, screen.reason
                )
        processed_rows += len(source_rows) - len(eligible)
        progress["processed_rows"] = processed_rows
        _write_progress(progress_path, progress)

        for start in range(0, len(eligible), args.batch_size):
            batch = eligible[start : start + args.batch_size]
            batch_records = _generate_mllm_batch(
                batch, processor, model, sampling, args.max_pixels
            )
            for record in batch_records:
                by_id[str(record["sample_id"])] = record
                parse_errors += record["mllm_status"] == "parse_error"
                input_errors += record["mllm_status"] == "input_error"
            model_calls += sum(
                record["mllm_status"] != "input_error" for record in batch_records
            )
            processed_rows += len(batch)
            progress.update(
                {
                    "processed_rows": processed_rows,
                    "model_calls": model_calls,
                    "parse_errors": parse_errors,
                    "input_errors": input_errors,
                }
            )
            _write_progress(progress_path, progress)

        ordered = [by_id[str(row["sample_id"])] for row in source_rows]
        output = _evidence_path(args.output_root, "mllm", dataset, source_path)
        _write_parquet(ordered, output)
        completed_shards += 1
        progress.update(
            {
                "completed_shards": completed_shards,
                "current_shard": "",
            }
        )
        _write_progress(progress_path, progress)
        print(
            f"worker={args.worker_index} completed={completed_shards}/{len(assigned)} "
            f"rows={processed_rows}/{total_rows} calls={model_calls} "
            f"parse_errors={parse_errors} input_errors={input_errors}",
            flush=True,
        )

    progress["state"] = "complete"
    _write_progress(progress_path, progress)


def _sam_skip_result(sample_id: Any, prompt: str, status: str) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "sam_policy_version": SAM_COUNT_POLICY_VERSION,
        "sam_prompt": prompt,
        "sam_confidence_threshold": None,
        "sam_status": status,
        "sam_count": None,
        "sam_selected_count": None,
        "sam_unselected_count": None,
        "sam_high_confidence_count": None,
        "sam_raw_count": None,
        "sam_instances": [],
        "sam_duplicate_rejects": [],
        "sam_area_rejects": [],
        "sam_instance_masks_png": [],
        "image_width": None,
        "image_height": None,
        "sam_error": "",
    }


def stage_sam_worker(args: argparse.Namespace) -> None:
    devices = _device_list(args.devices)
    if len(devices) != 1:
        raise ValueError("each full SAM worker requires exactly one device")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0])

    datasets = _dataset_names(args.datasets)
    shards = discover_shards(args.dataset_root, datasets, args.shard_glob)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    assigned = _assigned_shards(shards, args.worker_index, args.num_workers)
    total_rows = sum(
        _effective_rows(path, args.max_rows_per_shard) for _, path in assigned
    )
    progress_path = (
        args.output_root / "progress" / f"sam-worker-{args.worker_index:02d}.json"
    )
    progress = {
        "stage": "sam",
        "worker_index": args.worker_index,
        "pid": os.getpid(),
        "total_shards": len(assigned),
        "total_rows": total_rows,
        "completed_shards": 0,
        "processed_rows": 0,
        "sam_calls": 0,
        "sam_errors": 0,
        "current_shard": "",
        "state": "starting",
    }
    _write_progress(progress_path, progress)

    completed_shards = 0
    processed_rows = 0
    sam_calls = 0
    sam_errors = 0
    pending: List[Tuple[str, Path]] = []
    for dataset, source_path in assigned:
        expected = _effective_rows(source_path, args.max_rows_per_shard)
        mllm_path = _evidence_path(args.output_root, "mllm", dataset, source_path)
        if not _valid_evidence(
            mllm_path, expected, "prompt_version", MLLM_PROMPT_VERSION
        ):
            raise RuntimeError(f"missing or invalid MLLM evidence: {mllm_path}")
        output = _evidence_path(args.output_root, "sam", dataset, source_path)
        if args.resume and _valid_evidence(
            output, expected, "sam_policy_version", SAM_COUNT_POLICY_VERSION
        ):
            completed_shards += 1
            processed_rows += expected
        else:
            pending.append((dataset, source_path))

    progress.update(
        {
            "completed_shards": completed_shards,
            "processed_rows": processed_rows,
            "state": "complete" if not pending else "loading_model",
        }
    )
    _write_progress(progress_path, progress)
    if not pending:
        print(f"worker={args.worker_index} all SAM shards already complete", flush=True)
        return

    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.cuda.set_device(0)
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=str(args.checkpoint_path),
        load_from_HF=False,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(
        model, device="cuda", confidence_threshold=args.confidence_threshold
    )
    progress["state"] = "running"
    _write_progress(progress_path, progress)
    print(
        f"worker={args.worker_index} SAM3 ready device={args.devices} "
        f"pending_shards={len(pending)} rows={total_rows - processed_rows}",
        flush=True,
    )

    for dataset, source_path in pending:
        progress["current_shard"] = f"{dataset}/{source_path.name}"
        _write_progress(progress_path, progress)
        source_table = pq.read_table(source_path, columns=SAM_SOURCE_COLUMNS)
        if args.max_rows_per_shard > 0:
            source_table = source_table.slice(0, args.max_rows_per_shard)
        source_rows = source_table.to_pylist()
        mllm_path = _evidence_path(args.output_root, "mllm", dataset, source_path)
        mllm_rows = pq.read_table(mllm_path).to_pylist()
        if len(source_rows) != len(mllm_rows):
            raise RuntimeError(f"MLLM/source row mismatch for {source_path}")

        records: List[Dict[str, Any]] = []
        for row_index, (row, evidence) in enumerate(zip(source_rows, mllm_rows)):
            if str(row["sample_id"]) != str(evidence["sample_id"]):
                raise RuntimeError(
                    f"MLLM/source sample mismatch at {source_path}:{row_index}"
                )
            category = str(evidence.get("object_category") or "")
            if evidence["mllm_status"] != "ok" or not category:
                result = _sam_skip_result(
                    row["sample_id"], category, "skipped_no_mllm_category"
                )
            else:
                base = {
                    "sample_id": row["sample_id"],
                    "sam_policy_version": SAM_COUNT_POLICY_VERSION,
                    "sam_prompt": category,
                    "sam_confidence_threshold": args.confidence_threshold,
                }
                try:
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        counted = _run_sam_count(
                            processor,
                            _decode_image(row["source_image"]),
                            row["mask_png"],
                            category,
                            min_area_fraction=args.min_area_fraction,
                            max_area_fraction=args.max_area_fraction,
                            selected_min_instance_fraction=(
                                args.selected_min_instance_fraction
                            ),
                            selected_min_containment=args.selected_min_containment,
                            selected_min_image_fraction=(
                                args.selected_min_image_fraction
                            ),
                            store_masks=args.store_detection_masks,
                        )
                    base.update(counted)
                    base["sam_error"] = ""
                    sam_calls += 1
                except Exception as exc:
                    base.update(
                        _sam_skip_result(
                            row["sample_id"], category, "error"
                        )
                    )
                    base["sam_confidence_threshold"] = args.confidence_threshold
                    base["sam_error"] = repr(exc)
                    sam_errors += 1
                result = base
            records.append(result)
            processed_rows += 1
            if (row_index + 1) % args.progress_every == 0:
                progress.update(
                    {
                        "processed_rows": processed_rows,
                        "sam_calls": sam_calls,
                        "sam_errors": sam_errors,
                    }
                )
                _write_progress(progress_path, progress)

        output = _evidence_path(args.output_root, "sam", dataset, source_path)
        _write_parquet(records, output)
        completed_shards += 1
        progress.update(
            {
                "completed_shards": completed_shards,
                "processed_rows": processed_rows,
                "sam_calls": sam_calls,
                "sam_errors": sam_errors,
                "current_shard": "",
            }
        )
        _write_progress(progress_path, progress)
        print(
            f"worker={args.worker_index} completed={completed_shards}/{len(assigned)} "
            f"rows={processed_rows}/{total_rows} calls={sam_calls} "
            f"errors={sam_errors}",
            flush=True,
        )

    progress["state"] = "complete"
    _write_progress(progress_path, progress)


def _parsed_mllm(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None or row["mllm_status"] != "ok":
        return None
    return {
        "edited_subject_phrase": row["edited_subject_phrase"],
        "object_category": row["object_category"],
        "visible_same_class_count": row["visible_same_class_count"],
        "selected_instance_count": row["selected_instance_count"],
        "subset_relation": row["subset_relation"],
        "reference_cues": row["reference_cues"],
        "fine_grained_referential": row["fine_grained_referential"],
        "confidence": row["mllm_confidence"],
        "reason": row["mllm_reason"],
    }


def stage_fuse_full(args: argparse.Namespace) -> None:
    datasets = _dataset_names(args.datasets)
    shards = discover_shards(args.dataset_root, datasets, args.shard_glob)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    decision_counts: Counter[str] = Counter()
    by_dataset: Dict[str, Counter[str]] = {
        dataset: Counter() for dataset in datasets
    }
    strict_manifest: List[Dict[str, Any]] = []
    loose_manifest: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    input_rows = 0

    for shard_index, (dataset, source_path) in enumerate(shards, 1):
        expected = _effective_rows(source_path, args.max_rows_per_shard)
        mllm_path = _evidence_path(args.output_root, "mllm", dataset, source_path)
        sam_path = _evidence_path(args.output_root, "sam", dataset, source_path)
        if not _valid_evidence(
            mllm_path, expected, "prompt_version", MLLM_PROMPT_VERSION
        ):
            raise RuntimeError(f"missing or invalid MLLM evidence: {mllm_path}")
        if not _valid_evidence(
            sam_path, expected, "sam_policy_version", SAM_COUNT_POLICY_VERSION
        ):
            raise RuntimeError(f"missing or invalid SAM evidence: {sam_path}")

        source_table = pq.read_table(source_path, columns=METADATA_COLUMNS)
        if args.max_rows_per_shard > 0:
            source_table = source_table.slice(0, args.max_rows_per_shard)
        source_rows = source_table.to_pylist()
        mllm_rows = pq.read_table(mllm_path).to_pylist()
        sam_rows = pq.read_table(sam_path).to_pylist()
        if not len(source_rows) == len(mllm_rows) == len(sam_rows):
            raise RuntimeError(f"evidence/source row mismatch for {source_path}")

        audit_rows = []
        for row_index, (source, mllm, sam) in enumerate(
            zip(source_rows, mllm_rows, sam_rows)
        ):
            sample_id = str(source["sample_id"])
            if sample_id != str(mllm["sample_id"]) or sample_id != str(
                sam["sample_id"]
            ):
                raise RuntimeError(
                    f"evidence/source sample mismatch at {source_path}:{row_index}"
                )
            screen = deterministic_screen(
                str(source["edit_type"]), str(source["mask_mode"])
            )
            parsed = _parsed_mllm(mllm)
            sam_count = int(sam["sam_count"]) if sam["sam_status"] == "ok" else None
            sam_selected = (
                int(sam["sam_selected_count"])
                if sam["sam_status"] == "ok"
                else None
            )
            fusion = fuse_evidence(
                deterministic_eligible=screen.eligible,
                deterministic_reason=screen.reason,
                mllm=parsed,
                sam_count=sam_count,
                sam_selected_count=sam_selected,
                all_selected_fraction=args.all_selected_fraction,
            )
            audit = {
                "sample_id": sample_id,
                "source_dataset": source["source_dataset"],
                "source_shard": source["source_shard"],
                "source_row_idx": source["source_row_idx"],
                "unified_parquet": str(source_path),
                "unified_parquet_row_index": row_index,
                "edit_type": source["edit_type"],
                "raw_edit_type": source["raw_edit_type"],
                "instruction": source["instruction"],
                "mask_mode": source["mask_mode"],
                "mask_area_fraction": source["mask_area_fraction"],
                "deterministic_eligible": screen.eligible,
                "deterministic_reason": screen.reason,
                "lexical_cues": instruction_cues(str(source["instruction"])),
                "edited_subject_phrase": mllm.get("edited_subject_phrase", ""),
                "object_category": mllm.get("object_category", ""),
                "mllm_status": mllm["mllm_status"],
                "mllm_visible_count": mllm["visible_same_class_count"],
                "mllm_selected_count": mllm["selected_instance_count"],
                "mllm_subset_relation": mllm["subset_relation"],
                "mllm_judgment": mllm["fine_grained_referential"],
                "mllm_confidence": mllm["mllm_confidence"],
                "mllm_reason": mllm["mllm_reason"],
                "sam_status": sam["sam_status"],
                "sam_count": sam_count,
                "sam_selected_count": sam_selected,
                "sam_unselected_count": sam["sam_unselected_count"],
                "sam_high_confidence_count": sam["sam_high_confidence_count"],
                "filter_policy_version": FILTER_POLICY_VERSION,
                "decision": fusion["decision"],
                "loose_keep": fusion["loose_keep"],
                "decision_reason": fusion["reason"],
            }
            audit_rows.append(audit)
            decision_counts[fusion["decision"]] += 1
            by_dataset[dataset][fusion["decision"]] += 1
            status_counts[f"mllm:{mllm['mllm_status']}"] += 1
            status_counts[f"sam:{sam['sam_status']}"] += 1
            manifest_row = {
                "sample_id": sample_id,
                "source_dataset": source["source_dataset"],
                "source_shard": source["source_shard"],
                "source_row_idx": source["source_row_idx"],
                "unified_parquet": str(source_path),
                "unified_parquet_row_index": row_index,
                "decision": fusion["decision"],
                "loose_keep": fusion["loose_keep"],
                "decision_reason": fusion["reason"],
            }
            if fusion["decision"] == "keep":
                strict_manifest.append(manifest_row)
            if fusion["loose_keep"]:
                loose_manifest.append(manifest_row)
        audit_path = _evidence_path(
            args.output_root, "audit", dataset, source_path
        )
        _write_parquet(audit_rows, audit_path)
        input_rows += len(audit_rows)
        if shard_index % args.progress_every_shards == 0 or shard_index == len(shards):
            print(
                f"fuse shards={shard_index}/{len(shards)} rows={input_rows} "
                f"keep={decision_counts['keep']} review={decision_counts['review']} "
                f"drop={decision_counts['drop']}",
                flush=True,
            )

    final_dir = args.output_root / "final"
    _write_parquet(strict_manifest, final_dir / "selected_manifest.parquet")
    _write_parquet(loose_manifest, final_dir / "loose_selected_manifest.parquet")
    summary = {
        "stage": "fuse_full",
        "completed_at": _utc_now(),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "prompt_version": MLLM_PROMPT_VERSION,
        "sam_policy_version": SAM_COUNT_POLICY_VERSION,
        "filter_policy_version": FILTER_POLICY_VERSION,
        "all_selected_fraction": args.all_selected_fraction,
        "input_shards": len(shards),
        "input_rows": input_rows,
        "decision_counts": dict(decision_counts),
        "strict_keep_rows": len(strict_manifest),
        "loose_keep_rows": len(loose_manifest),
        "by_dataset": {
            dataset: dict(counts) for dataset, counts in by_dataset.items()
        },
        "status_counts": dict(status_counts),
    }
    _write_json(summary, final_dir / "summary.json")
    _write_json(summary, args.output_root / "_SUCCESS")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def _read_progress_files(output_root: Path, stage: str) -> Dict[str, Any]:
    reports = []
    for path in sorted((output_root / "progress").glob(f"{stage}-worker-*.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return {
        "workers_reporting": len(reports),
        "workers_complete": sum(row.get("state") == "complete" for row in reports),
        "processed_rows": sum(int(row.get("processed_rows", 0)) for row in reports),
        "total_rows": sum(int(row.get("total_rows", 0)) for row in reports),
        "completed_shards": sum(
            int(row.get("completed_shards", 0)) for row in reports
        ),
        "total_shards": sum(int(row.get("total_shards", 0)) for row in reports),
        "model_calls": sum(int(row.get("model_calls", 0)) for row in reports),
        "parse_errors": sum(int(row.get("parse_errors", 0)) for row in reports),
        "input_errors": sum(int(row.get("input_errors", 0)) for row in reports),
        "sam_calls": sum(int(row.get("sam_calls", 0)) for row in reports),
        "sam_errors": sum(int(row.get("sam_errors", 0)) for row in reports),
        "states": Counter(str(row.get("state", "unknown")) for row in reports),
    }


def _progress_line(stage: str, report: Mapping[str, Any]) -> str:
    processed = int(report["processed_rows"])
    total = int(report["total_rows"])
    percent = 100.0 * processed / total if total else 0.0
    extra = (
        f"calls={report['model_calls']} parse_errors={report['parse_errors']} "
        f"input_errors={report['input_errors']}"
        if stage == "mllm"
        else f"calls={report['sam_calls']} errors={report['sam_errors']}"
    )
    return (
        f"[{_utc_now()}] stage={stage} rows={processed}/{total} ({percent:.2f}%) "
        f"shards={report['completed_shards']}/{report['total_shards']} "
        f"workers={report['workers_reporting']} states={dict(report['states'])} {extra}"
    )


def _run_workers(
    stage: str,
    commands: Sequence[Sequence[str]],
    output_root: Path,
    poll_seconds: int,
) -> None:
    processes = []
    handles = []
    try:
        for index, command in enumerate(commands):
            log_path = output_root / "logs" / f"{stage}-worker-{index:02d}.log"
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            handles.append(handle)
        print(
            f"[{_utc_now()}] launched {len(processes)} {stage} workers "
            f"pids={[process.pid for process in processes]}",
            flush=True,
        )
        while True:
            return_codes = [process.poll() for process in processes]
            report = _read_progress_files(output_root, stage)
            print(_progress_line(stage, report), flush=True)
            failed = [
                (index, code)
                for index, code in enumerate(return_codes)
                if code not in (None, 0)
            ]
            if failed:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError(f"{stage} workers failed: {failed}")
            if all(code == 0 for code in return_codes):
                break
            time.sleep(poll_seconds)
    finally:
        for handle in handles:
            handle.close()


def _worker_base_args(args: argparse.Namespace, command: str) -> List[str]:
    result = [
        str(args.python),
        str(Path(__file__).resolve()),
        command,
        "--dataset-root",
        str(args.dataset_root),
        "--output-root",
        str(args.output_root),
        "--datasets",
        args.datasets,
        "--shard-glob",
        args.shard_glob,
        "--max-shards",
        str(args.max_shards),
        "--max-rows-per-shard",
        str(args.max_rows_per_shard),
        "--resume",
    ]
    return result


def stage_supervise(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_root / "progress").mkdir(parents=True, exist_ok=True)
    datasets = _dataset_names(args.datasets)
    shards = discover_shards(args.dataset_root, datasets, args.shard_glob)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for _, path in shards)
    if args.max_rows_per_shard > 0:
        total_rows = sum(
            min(pq.ParquetFile(path).metadata.num_rows, args.max_rows_per_shard)
            for _, path in shards
        )
    config = {
        "started_at": _utc_now(),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "datasets": list(datasets),
        "source_shards": len(shards),
        "source_rows": total_rows,
        "mllm_model": str(args.model_path),
        "sam_checkpoint": str(args.checkpoint_path),
        "mllm_prompt_version": MLLM_PROMPT_VERSION,
        "sam_policy_version": SAM_COUNT_POLICY_VERSION,
        "filter_policy_version": FILTER_POLICY_VERSION,
        "gpu_layout": "4 external DP workers x TP2, then 8 SAM workers x 1 GPU",
        "mllm_batch_size_per_worker": args.batch_size,
        "max_pixels": args.max_pixels,
    }
    _write_json(config, args.output_root / "run_config.json")
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)

    mllm_commands = []
    for worker in range(4):
        command = _worker_base_args(args, "mllm-worker")
        command.extend(
            [
                "--worker-index",
                str(worker),
                "--num-workers",
                "4",
                "--devices",
                f"{2 * worker},{2 * worker + 1}",
                "--tensor-parallel-size",
                "2",
                "--batch-size",
                str(args.batch_size),
                "--model-path",
                str(args.model_path),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--max-pixels",
                str(args.max_pixels),
                "--max-model-len",
                str(args.max_model_len),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
        )
        mllm_commands.append(command)
    _run_workers("mllm", mllm_commands, args.output_root, args.poll_seconds)

    sam_commands = []
    for worker in range(8):
        command = _worker_base_args(args, "sam-worker")
        command.extend(
            [
                "--worker-index",
                str(worker),
                "--num-workers",
                "8",
                "--devices",
                str(worker),
                "--checkpoint-path",
                str(args.checkpoint_path),
            ]
        )
        sam_commands.append(command)
    _run_workers("sam", sam_commands, args.output_root, args.poll_seconds)

    fuse_command = _worker_base_args(args, "fuse-full")
    print(f"[{_utc_now()}] starting full fusion", flush=True)
    subprocess.run(fuse_command, cwd=REPO_ROOT, check=True)
    print(
        f"[{_utc_now()}] full referential filter complete: {args.output_root}",
        flush=True,
    )


def _add_common_worker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", default="crispedit,scaleedit")
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--shard-glob", default="*.parquet")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-rows-per-shard", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    mllm = subparsers.add_parser("mllm-worker")
    _add_common_worker_args(mllm)
    mllm.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_PATH)
    mllm.add_argument("--tensor-parallel-size", type=int, default=2)
    mllm.add_argument("--batch-size", type=int, default=8)
    mllm.add_argument("--max-new-tokens", type=int, default=384)
    mllm.add_argument("--max-pixels", type=int, default=1_310_720)
    mllm.add_argument("--max-model-len", type=int, default=8192)
    mllm.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    mllm.set_defaults(structured_output=False, function=stage_mllm_worker)

    repair = subparsers.add_parser("repair-mllm")
    repair.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    repair.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    repair.add_argument("--datasets", default="crispedit,scaleedit")
    repair.add_argument("--devices", default="0,1")
    repair.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_PATH)
    repair.add_argument("--tensor-parallel-size", type=int, default=2)
    repair.add_argument("--batch-size", type=int, default=8)
    repair.add_argument("--max-new-tokens", type=int, default=768)
    repair.add_argument("--max-pixels", type=int, default=1_310_720)
    repair.add_argument("--max-model-len", type=int, default=8192)
    repair.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    repair.set_defaults(structured_output=True, function=stage_repair_mllm)

    sam = subparsers.add_parser("sam-worker")
    _add_common_worker_args(sam)
    sam.add_argument("--checkpoint-path", type=Path, default=DEFAULT_SAM_PATH)
    sam.add_argument("--confidence-threshold", type=float, default=0.30)
    sam.add_argument("--min-area-fraction", type=float, default=0.00003)
    sam.add_argument("--max-area-fraction", type=float, default=0.90)
    sam.add_argument("--selected-min-instance-fraction", type=float, default=0.02)
    sam.add_argument("--selected-min-containment", type=float, default=0.10)
    sam.add_argument("--selected-min-image-fraction", type=float, default=0.00001)
    sam.add_argument("--store-detection-masks", action="store_true")
    sam.add_argument("--progress-every", type=int, default=10)
    sam.set_defaults(function=stage_sam_worker)

    fuse = subparsers.add_parser("fuse-full")
    fuse.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    fuse.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    fuse.add_argument("--datasets", default="crispedit,scaleedit")
    fuse.add_argument("--shard-glob", default="*.parquet")
    fuse.add_argument("--max-shards", type=int, default=0)
    fuse.add_argument("--max-rows-per-shard", type=int, default=0)
    fuse.add_argument("--all-selected-fraction", type=float, default=0.90)
    fuse.add_argument("--progress-every-shards", type=int, default=25)
    fuse.add_argument("--resume", action="store_true")
    fuse.set_defaults(function=stage_fuse_full)

    supervise = subparsers.add_parser("supervise")
    supervise.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    supervise.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    supervise.add_argument("--datasets", default="crispedit,scaleedit")
    supervise.add_argument("--shard-glob", default="*.parquet")
    supervise.add_argument("--max-shards", type=int, default=0)
    supervise.add_argument("--max-rows-per-shard", type=int, default=0)
    supervise.add_argument("--python", type=Path, default=Path(sys.executable))
    supervise.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_PATH)
    supervise.add_argument("--checkpoint-path", type=Path, default=DEFAULT_SAM_PATH)
    supervise.add_argument("--batch-size", type=int, default=8)
    supervise.add_argument("--max-new-tokens", type=int, default=384)
    supervise.add_argument("--max-pixels", type=int, default=1_310_720)
    supervise.add_argument("--max-model-len", type=int, default=8192)
    supervise.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    supervise.add_argument("--poll-seconds", type=int, default=30)
    supervise.set_defaults(function=stage_supervise)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
