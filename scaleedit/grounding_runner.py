"""Stage 1: ScaleEdit paired-image reasoning and task-aware region grounding."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from scaleedit import PROMPT_VERSION
from scaleedit.io import decode_image, discover_shards, image_size, iter_row_batches
from scaleedit.policy import (
    apply_task_post_policy,
    build_grounding_prompt,
    build_object_viewpoint_retry_prompt,
    build_observation_prompt,
    canonical_task,
    grounding_status,
    object_viewpoint_ref,
    parse_grounding,
    parse_observation,
)


GROUND_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("sample_id", pa.string()),
        ("source_relative_path", pa.string()),
        ("edit_task", pa.string()),
        ("final_task", pa.string()),
        ("original_instruction", pa.string()),
        ("final_instruction", pa.string()),
        ("ground_json", pa.string()),
        ("ground_parse_ok", pa.bool_()),
        ("grounding_status", pa.string()),
        ("qc_flag", pa.string()),
        ("source_width", pa.int32()),
        ("source_height", pa.int32()),
        ("target_width", pa.int32()),
        ("target_height", pa.int32()),
        ("mllm_model", pa.string()),
        ("prompt_version", pa.string()),
        ("grounding_seconds", pa.float64()),
    ]
)


@dataclass(frozen=True)
class GroundingJob:
    input_path: str
    output_path: str
    num_rows: int
    selected_sample_ids: Tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ScaleEdit two-pass paired-image grounding with Qwen3.5"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get(
                "SCALEEDIT_QWEN_MODEL_PATH",
                "/mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B",
            )
        ),
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-batch-size", type=int, default=4)
    parser.add_argument("--max-images-per-generate", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=1_310_720)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--gpu-memory-gib", type=int, default=74)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--limit-rows-per-shard", type=int, default=None)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Process only this exact sample_id; repeat for targeted auditable repairs",
    )
    parser.add_argument("--progress-mininterval", type=float, default=2.0)
    return parser.parse_args()


def parse_device_groups(spec: str, tensor_parallel_size: int) -> List[List[int]]:
    devices = [
        int(part.strip().removeprefix("cuda:"))
        for part in spec.split(",")
        if part.strip()
    ]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    if tensor_parallel_size <= 0 or len(devices) % tensor_parallel_size:
        raise ValueError(
            f"{len(devices)} devices cannot be divided into TP={tensor_parallel_size} groups"
        )
    return [
        devices[index : index + tensor_parallel_size]
        for index in range(0, len(devices), tensor_parallel_size)
    ]


def build_jobs(args: argparse.Namespace) -> List[GroundingJob]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = {str(value) for value in args.sample_id}
    if requested and args.limit_rows_per_shard is not None:
        raise ValueError("--sample-id and --limit-rows-per-shard are mutually exclusive")
    found: Set[str] = set()
    jobs = []
    for path in discover_shards(args.input_dir):
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        required = {"sample_id", "final_task", "final_instruction", "source_image", "edited_image"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{path.name} misses required columns: {missing}")
        selected: Tuple[str, ...] = ()
        if requested:
            shard_ids = [str(value) for value in pq.read_table(path, columns=["sample_id"])[0].to_pylist()]
            selected = tuple(value for value in shard_ids if value in requested)
            duplicate = found.intersection(selected)
            if duplicate:
                raise ValueError(f"duplicate requested sample_id(s): {sorted(duplicate)}")
            found.update(selected)
            count = len(selected)
            if not count:
                continue
        else:
            count = parquet.metadata.num_rows
            if args.limit_rows_per_shard is not None:
                count = min(count, args.limit_rows_per_shard)
        jobs.append(
            GroundingJob(
                input_path=str(path),
                output_path=str(args.output_dir / path.name),
                num_rows=count,
                selected_sample_ids=selected,
            )
        )
    missing = requested - found
    if missing:
        raise KeyError(f"requested sample_id(s) not found: {sorted(missing)}")
    return jobs


def assign_jobs(
    jobs: Sequence[GroundingJob], groups: Sequence[Sequence[int]]
) -> List[Tuple[List[int], List[GroundingJob]]]:
    buckets = [{"devices": list(group), "rows": 0, "jobs": []} for group in groups]
    for job in sorted(jobs, key=lambda item: item.num_rows, reverse=True):
        bucket = min(buckets, key=lambda item: item["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += job.num_rows
    return [
        (item["devices"], item["jobs"])
        for item in buckets
        if item["jobs"]
    ]


def _chunks_by_image_budget(
    conversations: Sequence[List[Dict]], max_images: int
) -> List[List[List[Dict]]]:
    result: List[List[List[Dict]]] = []
    pending: List[List[Dict]] = []
    image_count = 0
    for conversation in conversations:
        count = sum(
            1
            for message in conversation
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image"
        )
        if pending and image_count + count > max_images:
            result.append(pending)
            pending, image_count = [], 0
        pending.append(conversation)
        image_count += count
    if pending:
        result.append(pending)
    return result


class Qwen35ScaleEditGrounder:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        self.torch = torch
        self.args = args
        visible_count = torch.cuda.device_count()
        if visible_count != args.tensor_parallel_size:
            raise RuntimeError(
                f"worker sees {visible_count} GPUs, expected TP={args.tensor_parallel_size}"
            )
        max_memory = {index: f"{args.gpu_memory_gib}GiB" for index in range(visible_count)}
        self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            image_processor.size.longest_edge = int(args.max_pixels)
        self.input_device = next(
            parameter.device
            for parameter in self.model.parameters()
            if parameter.device.type != "meta"
        )

    @staticmethod
    def _conversation(source: Image.Image, target: Image.Image, prompt: str) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1 (source, full image):"},
                    {"type": "image", "image": source},
                    {"type": "text", "text": "Image 2 (edited result, full image):"},
                    {"type": "image", "image": target},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    @classmethod
    def _followup(
        cls,
        source: Image.Image,
        target: Image.Image,
        observation_prompt: str,
        observation_text: str,
        grounding_prompt: str,
    ) -> List[Dict]:
        return [
            cls._conversation(source, target, observation_prompt)[0],
            {
                "role": "assistant",
                "content": [{"type": "text", "text": observation_text}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": grounding_prompt}],
            },
        ]

    def _generate_once(self, conversations: Sequence[List[Dict]]) -> List[str]:
        inputs = self.processor.apply_chat_template(
            list(conversations),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(self.input_device)
        prompt_width = int(inputs.input_ids.shape[1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        return self.processor.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _generate_backoff(self, conversations: Sequence[List[Dict]]) -> List[str]:
        try:
            return self._generate_once(conversations)
        except self.torch.cuda.OutOfMemoryError:
            if len(conversations) <= 1:
                raise
            for index in range(self.torch.cuda.device_count()):
                with self.torch.cuda.device(index):
                    self.torch.cuda.empty_cache()
            midpoint = len(conversations) // 2
            return self._generate_backoff(conversations[:midpoint]) + self._generate_backoff(
                conversations[midpoint:]
            )

    def generate(self, conversations: Sequence[List[Dict]]) -> List[str]:
        outputs: List[str] = []
        for chunk in _chunks_by_image_budget(
            conversations, self.args.max_images_per_generate
        ):
            outputs.extend(self._generate_backoff(chunk))
        return outputs

    def _parse_with_retry(self, conversation: List[Dict], text: str, parser):
        error = ""
        parsed = None
        for attempt in range(self.args.parse_retries + 1):
            try:
                parsed = parser(text)
                error = ""
                break
            except Exception as exc:
                error = repr(exc)
                if attempt < self.args.parse_retries:
                    text = self.generate([conversation])[0]
        return text, parsed, error

    def infer(self, samples: Sequence[Dict]) -> List[Dict]:
        observation_jobs = []
        for sample in samples:
            prompt = build_observation_prompt(sample["final_task"], sample["instruction"])
            conversation = self._conversation(sample["source"], sample["target"], prompt)
            observation_jobs.append((prompt, conversation))

        observation_texts: List[str] = []
        for start in range(0, len(observation_jobs), self.args.request_batch_size):
            chunk = observation_jobs[start : start + self.args.request_batch_size]
            observation_texts.extend(self.generate([item[1] for item in chunk]))

        observations = []
        grounding_jobs = []
        for sample, (prompt, conversation), initial_text in zip(
            samples, observation_jobs, observation_texts
        ):
            text, parsed, error = self._parse_with_retry(
                conversation, initial_text, parse_observation
            )
            observation = {
                "prompt": prompt,
                "raw_text": text,
                "parsed": parsed or {},
                "parse_ok": not error,
                "error": error,
            }
            observations.append(observation)
            context = parsed if parsed is not None else {"unparsed_observation": text}
            grounding_prompt = build_grounding_prompt(
                sample["final_task"], sample["instruction"], context
            )
            grounding_conversation = self._followup(
                sample["source"],
                sample["target"],
                prompt,
                text,
                grounding_prompt,
            )
            grounding_jobs.append((grounding_prompt, grounding_conversation))

        grounding_texts: List[str] = []
        for start in range(0, len(grounding_jobs), self.args.request_batch_size):
            chunk = grounding_jobs[start : start + self.args.request_batch_size]
            grounding_texts.extend(self.generate([item[1] for item in chunk]))

        payloads = []
        for sample_index, (observation, (prompt, conversation), initial_text) in enumerate(
            zip(observations, grounding_jobs, grounding_texts)
        ):
            text, parsed, error = self._parse_with_retry(
                conversation, initial_text, parse_grounding
            )
            route_retry = None
            route_ref = object_viewpoint_ref(
                samples[sample_index]["final_task"], samples[sample_index]["instruction"]
            )
            if not error and parsed is not None and parsed.get("mask_mode") == "full_image" and route_ref:
                retry_prompt = build_object_viewpoint_retry_prompt(
                    samples[sample_index]["final_task"],
                    samples[sample_index]["instruction"],
                    observation["parsed"],
                    route_ref,
                )
                retry_conversation = conversation + [
                    {"role": "assistant", "content": [{"type": "text", "text": text}]},
                    {"role": "user", "content": [{"type": "text", "text": retry_prompt}]},
                ]
                retry_text, retry_parsed, retry_error = self._parse_with_retry(
                    retry_conversation,
                    self.generate([retry_conversation])[0],
                    parse_grounding,
                )
                accepted = bool(
                    not retry_error
                    and retry_parsed is not None
                    and retry_parsed.get("mask_mode") == "regions"
                    and (retry_parsed.get("source") or retry_parsed.get("target"))
                )
                route_retry = {
                    "object_ref": route_ref,
                    "prompt": retry_prompt,
                    "raw_text": retry_text,
                    "parse_ok": not retry_error,
                    "accepted": accepted,
                    "error": retry_error or ("retry did not return regions" if not accepted else ""),
                }
                if accepted:
                    parsed = retry_parsed
            parsed = parsed or {
                "prompt_version": PROMPT_VERSION,
                "mask_mode": "unresolved",
                "source": [],
                "target": [],
                "protected_foreground": [],
            }
            payload = {
                "schema_version": 1,
                **parsed,
                "ground_parse_ok": not error,
                "observation": observation,
                "grounding": {
                    "prompt": prompt,
                    "raw_text": text,
                    "parse_ok": not error,
                    "error": error,
                },
            }
            if route_retry is not None:
                payload["grounding"]["route_retry"] = route_retry
            payload = apply_task_post_policy(
                samples[sample_index]["final_task"],
                samples[sample_index]["instruction"],
                payload,
            )
            payloads.append(payload)
        return payloads


def _row_from_payload(
    row_idx: int,
    record: Dict,
    payload: Dict,
    model_name: str,
    seconds: float,
) -> Dict:
    source_width, source_height = image_size(record["source_image"])
    target_width, target_height = image_size(record["edited_image"])
    status = grounding_status(payload)
    parse_ok = bool(payload.get("ground_parse_ok")) and status not in {
        "PARSE_ERROR",
        "RUNTIME_ERROR",
        "GROUND_FAIL",
    }
    return {
        "row_idx": row_idx,
        "sample_id": str(record.get("sample_id", "")),
        "source_relative_path": str(record.get("source_relative_path", "")),
        "edit_task": str(record.get("edit_task", "")),
        "final_task": canonical_task(record.get("final_task")),
        "original_instruction": str(record.get("original_instruction", "")),
        "final_instruction": str(record.get("final_instruction", "")),
        "ground_json": json.dumps(payload, ensure_ascii=False),
        "ground_parse_ok": parse_ok,
        "grounding_status": status,
        "qc_flag": "OK" if parse_ok else "GROUND_FAIL",
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "mllm_model": model_name,
        "prompt_version": PROMPT_VERSION,
        "grounding_seconds": float(seconds),
    }


def process_job(
    job: GroundingJob,
    grounder: Qwen35ScaleEditGrounder,
    args: argparse.Namespace,
    progress_queue,
    worker_index: int,
) -> Dict:
    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = None
    completed = False
    summary = {"rows": 0, "errors": 0, "statuses": {}, "tasks": {}}
    selected_ids = set(job.selected_sample_ids)
    try:
        for indexed_records in iter_row_batches(input_path, args.batch_size):
            if selected_ids:
                indexed_records = [
                    item
                    for item in indexed_records
                    if str(item[1].get("sample_id", "")) in selected_ids
                ]
            else:
                indexed_records = [item for item in indexed_records if item[0] < job.num_rows]
            if not indexed_records:
                if selected_ids:
                    continue
                break
            samples = [
                {
                    "source": decode_image(record["source_image"]),
                    "target": decode_image(record["edited_image"]),
                    "instruction": str(record["final_instruction"]),
                    "final_task": canonical_task(record["final_task"]),
                }
                for _, record in indexed_records
            ]
            started = time.monotonic()
            try:
                payloads = grounder.infer(samples)
            except Exception as exc:
                if args.fail_fast:
                    raise
                summary["errors"] += len(samples)
                payloads = [
                    {
                        "schema_version": 1,
                        "prompt_version": PROMPT_VERSION,
                        "mask_mode": "unresolved",
                        "source": [],
                        "target": [],
                        "protected_foreground": [],
                        "ground_parse_ok": False,
                        "runtime_error": repr(exc),
                    }
                    for _ in samples
                ]
            elapsed = (time.monotonic() - started) / max(len(samples), 1)
            rows = [
                _row_from_payload(row_idx, record, payload, Path(args.model_path).name, elapsed)
                for (row_idx, record), payload in zip(indexed_records, payloads)
            ]
            for row in rows:
                status, task = row["grounding_status"], row["final_task"]
                summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
                summary["tasks"][task] = summary["tasks"].get(task, 0) + 1
            table = pa.Table.from_pylist(rows, schema=GROUND_SCHEMA)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(tmp_path, GROUND_SCHEMA, compression=args.compression)
            writer.write_table(table)
            summary["rows"] += len(rows)
            progress_queue.put(
                {
                    "kind": "rows",
                    "count": len(rows),
                    "worker": worker_index,
                    "shard": input_path.name,
                }
            )
        if summary["rows"] != job.num_rows:
            raise ValueError(
                f"selected row mismatch for {input_path.name}: "
                f"expected={job.num_rows} actual={summary['rows']}"
            )
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
                "message": f"ground-worker-{worker_index} loading Qwen on GPUs {physical_devices}",
            }
        )
        grounder = Qwen35ScaleEditGrounder(args)
        for job in jobs:
            output_path = Path(job.output_path)
            if output_path.exists() and not args.overwrite:
                rows = pq.ParquetFile(output_path).metadata.num_rows
                summary = {
                    "rows": rows,
                    "errors": 0,
                    "statuses": {},
                    "tasks": {},
                    "skipped_existing": True,
                }
                progress_queue.put({"kind": "rows", "count": rows})
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
        if args.fail_fast:
            raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_index})


def _merge_summaries(messages: Sequence[Dict]) -> Dict:
    total = {"rows": 0, "errors": 0, "statuses": {}, "tasks": {}}
    for message in messages:
        summary = message.get("summary", {})
        total["rows"] += int(summary.get("rows", 0))
        total["errors"] += int(summary.get("errors", 0))
        for field in ("statuses", "tasks"):
            for key, count in summary.get(field, {}).items():
                total[field][key] = total[field].get(key, 0) + int(count)
    return total


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_path = str(args.model_path.resolve())
    jobs = build_jobs(args)
    groups = parse_device_groups(args.devices, args.tensor_parallel_size)
    assignments = assign_jobs(jobs, groups)
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        "stage": "scaleedit_grounding",
        "prompt_version": PROMPT_VERSION,
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
    messages, done = [], 0
    with tqdm(
        total=sum(job.num_rows for job in jobs),
        desc="ScaleEdit grounding rows",
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
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    summary = _merge_summaries(messages)
    summary["worker_exit_codes"] = [process.exitcode for process in processes]
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(f"grounding workers failed: {failed}")


if __name__ == "__main__":
    main()
