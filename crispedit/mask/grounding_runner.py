"""Stage 1: Qwen grounding for the CrispEdit mask pipeline.

The default 8-GPU layout is four independent BF16 replicas with tensor/model
parallel size 2.  The production backend uses vLLM continuous batching while
using the Qwen3.8 vLLM backend.  Each worker writes all raw responses and
validated boxes before SAM3 is loaded.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from crispedit.common import supported_shard

from crispedit.mask.selection import SCENE_FIELDS, apply_scene, load_filters
from crispedit.mask.artifacts import check_output_location, reusable_table, signature, signed_schema
from crispedit.mask.checklist import strict_json, parse_checklist_grounding, grounding_checklist

from crispedit.mask.grounding import (OBSERVATION_PROMPT_VERSION, PROMPT_VERSION,
    build_change_observation_prompt, build_grounding_requests, canonicalize_type,
    grounding_is_complete, parse_change_observation, prompt_version_for_mode)


DEFAULT_MODEL_PATH = "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"

# Pass 1 still makes one source/result comparison, but color/material edits get
# paired overlapping views of those same two images.  This gives small faces,
# hands, fur, and other subject surfaces enough vision tokens without adding a
# third model turn or introducing pixel differences.



GROUND_SCHEMA = pa.schema(
    [
        ("row_idx", pa.int64()),
        ("raw_type", pa.string()),
        ("canonical_type", pa.string()),
        ("instruction", pa.string()),
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
        *SCENE_FIELDS,
    ]
)


@dataclass
class GroundingJob:
    raw_type: str
    input_path: str
    output_path: str
    manifest_path: Optional[str]
    row_indices: Optional[List[int]]
    num_rows: int
    difficulty_path: Optional[str] = None
    selected_rows: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.8-27B edit-unit grounding for CrispEdit")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=os.environ.get("CRISPEDIT_GROUNDING_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--keep-manifest-dir", type=Path, default=None)
    parser.add_argument("--difficulty-manifest-dir", type=Path, default=None,
                        help="Sparse scene manifest; requires --keep-manifest-dir. Only double PASS is labeled.")
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Optional JSON bad-case selection; emits sparse rows with original row_idx values",
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--inference-backend",
        choices=("vllm",),
        default="vllm",
        help="Qwen inference engine; production full runs should use vllm",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("two-pass",),
        default="two-pass",
        help="observe realized edits, then ground their segmentation units",
    )
    parser.add_argument("--include-types", default=None)
    parser.add_argument("--max-shards-per-type", type=int, default=None)
    parser.add_argument("--limit-rows-per-shard", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Samples grouped before request flattening")
    parser.add_argument("--request-batch-size", type=int, default=2, help="Maximum image-pair requests per generate call")
    parser.add_argument(
        "--max-images-per-generate",
        type=int,
        default=20,
        help=(
            "Maximum total image inputs in one generate call. Requests are "
            "greedily split below this visual-load limit; a single request is "
            "never split. Set <=0 to disable the limit."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--observation-max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-pixels", type=int, default=1_310_720, help="Per-image Qwen preprocessing pixel cap")
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-max-images-per-prompt", type=int, default=16)
    parser.add_argument(
        "--vllm-mm-encoder-tp-mode",
        choices=("weights", "data"),
        default="data",
        help="Replicate the vision encoder across TP ranks to process image batches in parallel",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (debugging only; slower)",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--progress-mininterval", type=float, default=2.0)
    return parser.parse_args()


def raw_type_from_filename(path: Path) -> str:
    match = re.match(r"(.+)_\d+\.parquet$", path.name)
    return match.group(1) if match else path.stem


def parse_device_groups(spec: str, tensor_parallel_size: int) -> List[List[int]]:
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    ids = [int(part.strip().removeprefix("cuda:")) for part in spec.split(",") if part.strip()]
    if not ids:
        raise ValueError("at least one CUDA device is required")
    if len(ids) % tensor_parallel_size:
        raise ValueError(
            f"{len(ids)} devices cannot be divided into TP={tensor_parallel_size} groups"
        )
    return [ids[index : index + tensor_parallel_size] for index in range(0, len(ids), tensor_parallel_size)]


def load_selection(path: Optional[Path]) -> Optional[Dict[str, List[int]]]:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("selection JSON must be a list or an object with a cases list")
    selected: Dict[str, List[int]] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("each selection item must be an object")
        shard = str(item.get("shard", item.get("parquet", ""))).strip()
        row_idx = int(item["row_idx"])
        if not shard or row_idx < 0:
            raise ValueError(f"invalid selection item: {item}")
        selected.setdefault(shard, []).append(row_idx)
    return {name: sorted(set(indices)) for name, indices in selected.items()}


def build_jobs(args: argparse.Namespace) -> List[GroundingJob]:
    difficulty_dir = getattr(args, "difficulty_manifest_dir", None)
    if difficulty_dir and not args.keep_manifest_dir:
        raise ValueError("--difficulty-manifest-dir requires --keep-manifest-dir")
    selection = load_selection(args.selection_file)
    include = {part.strip() for part in args.include_types.split(",")} if args.include_types else None
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(args.input_dir.glob("*.parquet")):
        if selection is not None and path.name not in selection:
            continue
        raw_type = raw_type_from_filename(path)
        if not supported_shard(path):
            continue
        if include is not None and raw_type not in include:
            continue
        grouped.setdefault(raw_type, []).append(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[GroundingJob] = []
    for raw_type, paths in sorted(grouped.items()):
        if args.max_shards_per_type is not None:
            paths = paths[: args.max_shards_per_type]
        for input_path in paths:
            total = pq.ParquetFile(input_path).metadata.num_rows
            indices = selection.get(input_path.name) if selection is not None else None
            if indices is not None:
                invalid = [index for index in indices if index >= total]
                if invalid:
                    raise IndexError(f"selection rows out of range for {input_path.name}: {invalid}")
                if args.limit_rows_per_shard is not None:
                    indices = indices[: args.limit_rows_per_shard]
                num_rows = len(indices)
            else:
                num_rows = min(total, args.limit_rows_per_shard) if args.limit_rows_per_shard else total
            if num_rows == 0:
                continue
            manifest_path = None
            if args.keep_manifest_dir is not None:
                candidate = args.keep_manifest_dir / input_path.name
                if not candidate.exists():
                    raise FileNotFoundError(f"keep manifest missing: {candidate}")
                manifest_path = str(candidate)
            difficulty_path = str(Path(difficulty_dir) / input_path.name) if difficulty_dir else None
            quality, scene = load_filters(manifest_path, difficulty_path, total)
            wanted = indices if indices is not None else range(num_rows)
            selected_rows = sum(
                apply_scene(prefilter_fields(quality.get(index)), scene.get(index), bool(difficulty_path))["filter_decision"] == "keep"
                for index in wanted
            )
            jobs.append(
                GroundingJob(
                    raw_type=raw_type,
                    input_path=str(input_path),
                    output_path=str(args.output_dir / input_path.name),
                    manifest_path=manifest_path,
                    row_indices=indices,
                    num_rows=num_rows,
                    difficulty_path=difficulty_path,
                    selected_rows=selected_rows,
                )
            )
    if selection is not None:
        matched = {Path(job.input_path).name for job in jobs}
        missing = sorted(set(selection) - matched)
        if missing:
            raise FileNotFoundError(f"selected shards not found under input-dir: {missing}")
    return jobs


def assign_jobs(jobs: Sequence[GroundingJob], groups: Sequence[Sequence[int]]) -> List[Tuple[List[int], List[GroundingJob]]]:
    buckets = [{"group": list(group), "rows": 0, "jobs": []} for group in groups]
    for job in sorted(jobs, key=lambda item: item.selected_rows if item.selected_rows is not None else item.num_rows, reverse=True):
        bucket = min(buckets, key=lambda item: item["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += max(1, job.selected_rows if job.selected_rows is not None else job.num_rows)
    return [(item["group"], item["jobs"]) for item in buckets if item["jobs"]]


def decode_image(cell: Dict) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def image_size(cell: Dict) -> Tuple[int, int]:
    """Read image dimensions without decoding its full pixel payload."""

    with Image.open(io.BytesIO(cell["bytes"])) as image:
        return image.size


def conversation_image_count(conversation: Sequence[Dict]) -> int:
    return sum(
        1
        for message in conversation
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "image"
    )


def split_conversations_by_image_budget(
    conversations: Sequence[List[Dict]], max_images: int
) -> List[List[List[Dict]]]:
    """Keep large color/detail prompts from making an unsafe GPU batch."""

    if not conversations:
        return []
    if max_images <= 0:
        return [list(conversations)]
    chunks: List[List[List[Dict]]] = []
    current: List[List[Dict]] = []
    current_images = 0
    for conversation in conversations:
        image_count = conversation_image_count(conversation)
        if current and current_images + image_count > max_images:
            chunks.append(current)
            current = []
            current_images = 0
        current.append(conversation)
        current_images += image_count
    if current:
        chunks.append(current)
    return chunks


def conversations_for_vllm(
    conversations: Sequence[List[Dict]],
) -> List[List[Dict]]:
    """Convert internal PIL chat parts to vLLM's in-memory image schema.

    Keeping this conversion at the engine boundary means both backends use the
    exact same messages, prompt versions, image ordering, and multi-turn
    observation context.
    """

    converted: List[List[Dict]] = []
    for conversation in conversations:
        converted_messages = []
        for message in conversation:
            content = message.get("content", "")
            if not isinstance(content, list):
                converted_messages.append(dict(message))
                continue
            converted_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    converted_content.append(
                        {"type": "image_pil", "image_pil": part["image"]}
                    )
                else:
                    converted_content.append(dict(part) if isinstance(part, dict) else part)
            converted_messages.append({**message, "content": converted_content})
        converted.append(converted_messages)
    return converted


def _selected_batches(path: Path, indices: Sequence[int], batch_size: int) -> Iterable[List[Tuple[int, Dict]]]:
    pf = pq.ParquetFile(path)
    wanted = sorted(indices)
    cursor = 0
    global_start = 0
    pending: List[Tuple[int, Dict]] = []
    for group_index in range(pf.num_row_groups):
        count = pf.metadata.row_group(group_index).num_rows
        group_end = global_start + count
        local_indices = []
        while cursor < len(wanted) and wanted[cursor] < group_end:
            if wanted[cursor] >= global_start:
                local_indices.append(wanted[cursor])
            cursor += 1
        if local_indices:
            rows = pf.read_row_group(group_index).to_pylist()
            for row_idx in local_indices:
                pending.append((row_idx, rows[row_idx - global_start]))
                if len(pending) >= batch_size:
                    yield pending
                    pending = []
        global_start = group_end
    if pending:
        yield pending


def iter_record_batches(job: GroundingJob, batch_size: int) -> Iterable[List[Tuple[int, Dict]]]:
    path = Path(job.input_path)
    if job.row_indices is not None:
        yield from _selected_batches(path, job.row_indices, batch_size)
        return
    row_idx = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        records = batch.to_pylist()
        if job.num_rows < row_idx + len(records):
            records = records[: job.num_rows - row_idx]
        if records:
            yield [(row_idx + offset, record) for offset, record in enumerate(records)]
        row_idx += len(records)
        if row_idx >= job.num_rows:
            break


def prefilter_fields(row: Optional[Dict]) -> Dict:
    if row is None:
        return {
            "prefilter_verdict": "NOT_RUN",
            "prefilter_confidence": math.nan,
            "prefilter_method": "",
            "prefilter_evidence_schema": "",
            "prefilter_model_name": "",
            "prefilter_run_id": "",
            "filter_decision": "keep",
            "prefilter_reason": "",
            "filter_reason_codes": "",
            "filter_mismatch_score": 0.0,
        }
    return {
        "prefilter_verdict": str(row.get("prefilter_verdict", "")),
        "prefilter_confidence": float(row.get("prefilter_confidence", math.nan)),
        "prefilter_method": str(row.get("prefilter_method", "")),
        "prefilter_evidence_schema": str(row.get("prefilter_evidence_schema", "")),
        "prefilter_model_name": str(row.get("prefilter_model_name", "")),
        "prefilter_run_id": str(row.get("prefilter_run_id", "")),
        "filter_decision": str(
            row.get("filter_decision", row.get("prefilter_decision", "drop"))
        ),
        "prefilter_reason": str(row.get("prefilter_reason", "")),
        "filter_reason_codes": str(row.get("filter_reason_codes", "")),
        "filter_mismatch_score": float(row.get("filter_mismatch_score", 0.0)),
    }


class Qwen38Grounder:
    @staticmethod
    def correction_conversation(conversation, response, error):
        return list(conversation) + [
            {"role":"assistant", "content":[{"type":"text", "text":response}]},
            {"role":"user", "content":[{"type":"text", "text":
                f"Correct the JSON response. Validation error: {error}. "
                "Include every requested candidate ID exactly once. Use the original images and schema. "
                "Return valid JSON only, without commentary."}]},
        ]

    def shutdown(self):
        if self.backend == "vllm":
            engine = getattr(self.model, "llm_engine", None)
            core = getattr(engine, "engine_core", None)
            if core is not None:
                core.shutdown(timeout=30.0)

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend = args.inference_backend
        self.prompt_version = prompt_version_for_mode(args.grounding_mode)
        self._init_vllm()


    def _init_vllm(self) -> None:
        # vLLM/FlashInfer may JIT-compile optimized kernels in child processes.
        # A direct ``.venv/bin/python`` invocation does not necessarily place
        # companion tools such as ``ninja`` on PATH, so propagate that location
        # before the engine forks its workers.
        # Do not resolve the venv's Python symlink: resolving it would collapse
        # back to /usr/bin and hide the venv-local ``ninja`` executable.
        executable_dir = str(Path(sys.executable).parent)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if executable_dir not in path_entries:
            os.environ["PATH"] = os.pathsep.join([executable_dir, *path_entries])
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM backend requested but vllm is not installed; run "
                "scripts/setup_crispedit_env.sh"
            ) from exc

        if not 0.0 < self.args.vllm_gpu_memory_utilization < 1.0:
            raise ValueError("vllm_gpu_memory_utilization must be between 0 and 1")
        if self.args.vllm_max_images_per_prompt < 1:
            raise ValueError("vllm_max_images_per_prompt must be positive")

        # Each outer worker sees exactly one physical TP group through
        # CUDA_VISIBLE_DEVICES. vLLM then owns scheduling and KV-cache management
        # within that isolated group.
        self.model = LLM(
            model=self.args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.args.tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.vllm_max_model_len,
            max_num_seqs=max(1, self.args.request_batch_size),
            limit_mm_per_prompt={
                "image": self.args.vllm_max_images_per_prompt,
            },
            mm_processor_kwargs={"max_pixels": int(self.args.max_pixels)},
            mm_encoder_tp_mode=self.args.vllm_mm_encoder_tp_mode,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            performance_mode="throughput",
            enforce_eager=self.args.vllm_enforce_eager,
            generation_config="vllm",
            disable_log_stats=True,
            seed=0,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.args.max_new_tokens,
        )

    @staticmethod
    def _conversation(source, target, prompt):
        return [{'role': 'user', 'content': [
            {'type':'text', 'text':'Image 1 (source, full image):'},
            {'type':'image', 'image':source},
            {'type':'text', 'text':'Image 2 (result, full image):'},
            {'type':'image', 'image':target}, {'type':'text', 'text':prompt}]}]




    def _generate_once(self, conversations: Sequence[List[Dict]], max_tokens=None) -> List[str]:
        for conversation in conversations:
            image_count = conversation_image_count(conversation)
            if image_count > self.args.vllm_max_images_per_prompt:
                raise ValueError(
                    f"request has {image_count} images, exceeding "
                    f"--vllm-max-images-per-prompt="
                    f"{self.args.vllm_max_images_per_prompt}"
                )
        params = copy.copy(self.sampling_params)
        params.max_tokens = max_tokens or self.args.max_new_tokens
        outputs = self.model.chat(
            messages=conversations_for_vllm(conversations),
            sampling_params=params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        self._last_batch_stats = [{"finish_reason": output.outputs[0].finish_reason,
                                   "output_tokens": len(output.outputs[0].token_ids)} for output in outputs]
        return [output.outputs[0].text for output in outputs]



    def _generate(self, conversations: Sequence[List[Dict]], max_tokens=None) -> List[str]:
        outputs: List[str] = []
        self.last_generation_stats = []
        for chunk in split_conversations_by_image_budget(
            conversations, self.args.max_images_per_generate
        ):
            outputs.extend(self._generate_once(chunk, max_tokens))
            self.last_generation_stats.extend(self._last_batch_stats)
        return outputs


    def infer(self, samples: Sequence[Dict]) -> List[Dict]:
        payloads = [{"schema_version": 2, "prompt_version": self.prompt_version,
                     "grounding_mode": "two-pass", "requests": [],
                     "boxes": {"source": [], "target": []}} for _ in samples]
        jobs = []
        for index, sample in enumerate(samples):
            prompt = build_change_observation_prompt(sample['type'], sample['instruction'])
            conversation = self._conversation(sample['input_img'], sample['output_img'], prompt)
            jobs.append((index, prompt, conversation))
        for start in range(0, len(jobs), self.args.request_batch_size):
            chunk = jobs[start:start + self.args.request_batch_size]
            budget = self.args.observation_max_new_tokens
            texts = self._generate([item[2] for item in chunk], max_tokens=budget)
            stats = list(self.last_generation_stats)
            for (index, prompt, conversation), text, initial_stats in zip(chunk, texts, stats):
                parsed, error = {}, ''
                attempts = [{"raw_text": text, "max_tokens": budget, **initial_stats}]
                for attempt in range(self.args.parse_retries + 1):
                    try:
                        strict_json(text)
                        parsed = parse_change_observation(text)
                        for identity, change in enumerate(parsed['changes']):
                            change['change_id'] = identity
                        error = ''
                        break
                    except (ValueError, TypeError, KeyError) as exc:
                        error = repr(exc)
                        if attempt < self.args.parse_retries:
                            if attempts[-1].get('finish_reason') == 'length':
                                retry = [{**conversation[0], 'content': [
                                    *conversation[0]['content'], {'type': 'text', 'text':
                                    'Return a COMPLETE concise JSON object. Keep descriptions short; '
                                    'group nearby tiny elements. Do not repeat an item.'}]}]
                            else:
                                retry = self.correction_conversation(conversation, text, error)
                            text = self._generate([retry], max_tokens=budget * 2)[0]
                            attempts.append({'raw_text': text, 'max_tokens': budget * 2,
                                             **self.last_generation_stats[0]})
                payloads[index]['observation'] = {
                    'prompt_version': OBSERVATION_PROMPT_VERSION, 'prompt': prompt,
                    'raw_text': text, 'parsed': parsed, 'parse_ok': not error,
                    'error': error, 'attempts': attempts}
        requests = []
        for index, sample in enumerate(samples):
            payload = payloads[index]
            observation = payload['observation']
            if not observation['parse_ok']:
                payload['observation_failed'] = True
                continue
            context = observation['parsed']
            if not context['changes']:
                payload['no_realized_changes'] = True
                continue
            if canonicalize_type(sample['type']) != 'add':
                payload['canvas_issues'] = [
                    {'change_id': c['change_id'], 'reason': 'TARGET_ONLY_EDIT_IN_SOURCE_ONLY_TYPE'}
                    for c in context['changes'] if not c['source_ref'] and c['target_ref']]
            for request in build_grounding_requests(sample['type'], sample['instruction'], context):
                if not grounding_checklist(context, request.grounding_image):
                    continue
                selected = sample['input_img'] if request.grounding_image == 'source' else sample['output_img']
                conversation = [{'role': 'user', 'content': [
                    {'type': 'text', 'text': f'Full {request.grounding_image} image:'},
                    {'type': 'image', 'image': selected}, {'type': 'text', 'text': request.prompt}]}]
                requests.append((index, request.grounding_image, conversation))
        for start in range(0, len(requests), self.args.request_batch_size):
            chunk = requests[start:start + self.args.request_batch_size]
            texts = self._generate([item[2] for item in chunk])
            for (index, side, conversation), text in zip(chunk, texts):
                boxes, unresolved, attempts, error = [], [], [], ''
                for attempt in range(self.args.parse_retries + 1):
                    attempts.append({'raw_text': text})
                    try:
                        boxes, unresolved = parse_checklist_grounding(
                            text, payloads[index]['observation']['parsed'], side)
                        error = ''
                        break
                    except (ValueError, TypeError, KeyError) as exc:
                        error = repr(exc)
                        if attempt < self.args.parse_retries:
                            text = self._generate([self.correction_conversation(conversation, text, error)],
                                                  max_tokens=self.args.max_new_tokens * 2)[0]
                payloads[index]['requests'].append({'grounding_image': side, 'raw_text': text,
                    'boxes': boxes, 'parse_ok': not error, 'error': error,
                    'unresolved': unresolved, 'attempts': attempts})
                payloads[index]['boxes'][side] = boxes
        return payloads


def _skip_row(
    row_idx: int,
    record: Dict,
    pre: Dict,
    model_name: str,
    prompt_version: str,
    grounding_mode: str,
) -> Dict:
    source_width, source_height = image_size(record["input_img"])
    target_width, target_height = image_size(record["output_img"])
    payload = {
        "schema_version": 2 if grounding_mode == "two-pass" else 1,
        "prompt_version": prompt_version,
        "grounding_mode": grounding_mode,
        "requests": [],
        "boxes": {},
    }
    return {
        "row_idx": row_idx,
        "raw_type": str(record.get("type", "")),
        "canonical_type": "PREFILTER_SKIP",
        "instruction": str(record.get("instruction", "")),
        "ground_json": json.dumps(payload, ensure_ascii=False),
        "ground_parse_ok": True,
        "grounding_status": "PREFILTER_SKIP",
        "qc_flag": "PREFILTER_SKIP",
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "mllm_model": model_name,
        "prompt_version": prompt_version,
        "grounding_seconds": 0.0,
        **pre,
    }




def _result_row(row_idx: int, sample: Dict, payload: Dict, pre: Dict, model_name: str, seconds: float) -> Dict:
    etype = canonicalize_type(sample["type"])
    parse_ok = not payload.get("runtime_error") and not payload.get("observation_failed") and all(
        bool(item["parse_ok"]) for item in payload["requests"]
    )
    complete = parse_ok and not any(item.get("unresolved") for item in payload["requests"]) and grounding_is_complete(etype, payload["boxes"])
    if not parse_ok:
        status = "PARSE_ERROR"
    elif complete and (payload.get('canvas_issues') or any(
            item.get('evidence_issues') or item.get('refinement_identity_issues') for item in payload['requests'])
            or payload.get('coverage_review_failed')):
        status = 'GROUND_REVIEW'
    elif complete:
        status = "OK"
    else:
        status = "GROUND_FAIL"
    return {
        "row_idx": row_idx,
        "raw_type": str(sample["type"]),
        "canonical_type": etype,
        "instruction": str(sample["instruction"]),
        "ground_json": json.dumps(payload, ensure_ascii=False),
        "ground_parse_ok": parse_ok,
        "grounding_status": status,
        "qc_flag": ('GROUND_REVIEW' if status == 'GROUND_REVIEW' else 'OK') if complete else "GROUND_FAIL",
        "source_width": sample["input_img"].width,
        "source_height": sample["input_img"].height,
        "target_width": sample["output_img"].width,
        "target_height": sample["output_img"].height,
        "mllm_model": model_name,
        "prompt_version": str(payload.get("prompt_version", PROMPT_VERSION)),
        "grounding_seconds": float(seconds),
        **pre,
    }


def process_job(
    job: GroundingJob,
    grounder: Qwen38Grounder,
    args: argparse.Namespace,
    progress_queue,
    worker_index: int,
) -> Dict:
    output_path = Path(job.output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    manifest, scene = load_filters(job.manifest_path, job.difficulty_path,
                                  pq.ParquetFile(job.input_path).metadata.num_rows)
    output_schema = signed_schema(GROUND_SCHEMA, grounding_signature(job, args))
    model_name = Path(args.model_path).name
    writer = None
    summary = {
        "rows": 0,
        "errors": 0,
        "ground_fail": 0,
        "observation_parse_fail": 0,
        "prefilter_skipped": 0,
        "statuses": {},
    }
    completed = False
    try:
        for indexed_records in iter_record_batches(job, args.batch_size):
            out_rows: List[Optional[Dict]] = [None] * len(indexed_records)
            infer_samples: List[Dict] = []
            infer_slots: List[Tuple[int, int, Dict]] = []
            for slot, (row_idx, record) in enumerate(indexed_records):
                manifest_row = manifest.get(row_idx) if job.manifest_path else None
                if job.manifest_path and manifest_row is None:
                    raise KeyError(f"missing manifest row {Path(job.input_path).name}:{row_idx}")
                pre = apply_scene(prefilter_fields(manifest_row), scene.get(row_idx), bool(job.difficulty_path))
                if pre["filter_decision"] != "keep":
                    out_rows[slot] = _skip_row(
                        row_idx,
                        record,
                        pre,
                        model_name,
                        prompt_version_for_mode(args.grounding_mode),
                        args.grounding_mode,
                    )
                    summary["prefilter_skipped"] += 1
                else:
                    sample = {
                        "input_img": decode_image(record["input_img"]),
                        "output_img": decode_image(record["output_img"]),
                        "instruction": record["instruction"],
                        "type": record["type"],
                    }
                    infer_slots.append((slot, row_idx, pre))
                    infer_samples.append(sample)
            if infer_samples:
                started = time.monotonic()
                try:
                    payloads = grounder.infer(infer_samples)
                    elapsed = (time.monotonic() - started) / len(infer_samples)
                    for sample, payload, (slot, row_idx, pre) in zip(infer_samples, payloads, infer_slots):
                        out_rows[slot] = _result_row(row_idx, sample, payload, pre, model_name, elapsed)
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    elapsed = (time.monotonic() - started) / len(infer_samples)
                    for sample, (slot, row_idx, pre) in zip(infer_samples, infer_slots):
                        payload = {
                            "schema_version": 2 if args.grounding_mode == "two-pass" else 1,
                            "prompt_version": grounder.prompt_version,
                            "grounding_mode": args.grounding_mode,
                            "requests": [],
                            "boxes": {"source": [], "target": []},
                            "runtime_error": repr(exc),
                        }
                        out_rows[slot] = _result_row(row_idx, sample, payload, pre, model_name, elapsed)
                        summary["errors"] += 1

            rows = [row for row in out_rows if row is not None]
            for row in rows:
                status = row["grounding_status"]
                summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
                summary["ground_fail"] += int(row["qc_flag"] == "GROUND_FAIL")
                payload = json.loads(row["ground_json"])
                observation = payload.get("observation")
                summary["observation_parse_fail"] += int(
                    bool(observation) and not bool(observation.get("parse_ok"))
                )
            table = pa.Table.from_pylist(rows, schema=output_schema)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(tmp_path, output_schema, compression=args.compression)
            writer.write_table(table)
            summary["rows"] += len(rows)
            progress_queue.put(
                {
                    "kind": "rows",
                    "count": len(rows),
                    "worker": worker_index,
                    "shard": Path(job.input_path).name,
                    "done_rows": summary["rows"],
                    "total_rows": job.num_rows,
                }
            )
        if summary["rows"] != job.num_rows:
            raise ValueError(f"incomplete grounding shard: {job.input_path}")
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if completed and tmp_path.exists():
            tmp_path.replace(output_path)
    return summary


def grounding_signature(job, args):
    excluded = {"output_dir", "input_dir", "keep_manifest_dir", "difficulty_manifest_dir",
                "devices", "overwrite", "progress_mininterval", "selection_file", "fail_fast"}
    settings = {key: value for key, value in vars(args).items() if key not in excluded}
    settings["row_indices"] = job.row_indices
    return signature(job.input_path, [job.manifest_path, job.difficulty_path, __file__,
                     Path(__file__).with_name("grounding.py"), Path(__file__).with_name("checklist.py"),
                     Path(__file__).with_name("selection.py")], settings)


def summarize_grounding(table):
    rows = table.to_pylist()
    statuses = {}
    errors = observation_errors = 0
    for row in rows:
        status = row["grounding_status"]
        statuses[status] = statuses.get(status, 0) + 1
        payload = json.loads(row["ground_json"])
        errors += bool(payload.get("runtime_error"))
        observation = payload.get("observation")
        observation_errors += bool(observation) and not bool(observation.get("parse_ok"))
    return {"rows": len(rows), "errors": errors, "statuses": statuses,
            "prefilter_skipped": statuses.get("PREFILTER_SKIP", 0),
            "ground_fail": sum(row["qc_flag"] == "GROUND_FAIL" for row in rows),
            "observation_parse_fail": observation_errors}


def worker_main(worker_index: int, physical_devices: List[int], jobs: List[GroundingJob], args_dict: Dict, progress_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in physical_devices)
    args = argparse.Namespace(**args_dict)
    grounder = None
    try:
        progress_queue.put(
            {
                "kind": "log",
                "message": f"ground-worker-{worker_index} loading TP={len(physical_devices)} on physical GPUs {physical_devices}",
            }
        )
        for job in jobs:
            output_path = Path(job.output_path)
            expected = job.row_indices if job.row_indices is not None else range(job.num_rows)
            existing = None if args.overwrite else reusable_table(output_path, grounding_signature(job, args), GROUND_SCHEMA, expected)
            if existing is not None and summarize_grounding(existing)["errors"]:
                existing = None
            if existing is not None:
                summary = {**summarize_grounding(existing), "skipped_existing": True}
                progress_queue.put({"kind": "rows", "count": summary["rows"]})
            else:
                if job.selected_rows != 0 and grounder is None:
                    grounder = Qwen38Grounder(args)
                summary = process_job(job, grounder, args, progress_queue, worker_index)
            progress_queue.put({"kind": "shard_done", "worker": worker_index, "shard": job.input_path, "summary": summary})
    except Exception as exc:
        progress_queue.put({"kind": "worker_error", "worker": worker_index, "devices": physical_devices, "error": repr(exc)})
        raise
    finally:
        try:
            if grounder is not None:
                grounder.shutdown()
        finally:
            progress_queue.put({"kind": "worker_done", "worker": worker_index})


def aggregate(summaries: Sequence[Dict]) -> Dict:
    result = {
        "rows": 0,
        "errors": 0,
        "ground_fail": 0,
        "observation_parse_fail": 0,
        "prefilter_skipped": 0,
        "statuses": {},
    }
    for message in summaries:
        summary = message.get("summary", {})
        for key in ("rows", "errors", "ground_fail", "observation_parse_fail", "prefilter_skipped"):
            result[key] += int(summary.get(key, 0))
        for status, count in summary.get("statuses", {}).items():
            result["statuses"][status] = result["statuses"].get(status, 0) + int(count)
    return result


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_path = str(Path(args.model_path).resolve())
    if args.keep_manifest_dir is not None:
        args.keep_manifest_dir = args.keep_manifest_dir.resolve()
    if args.difficulty_manifest_dir is not None:
        args.difficulty_manifest_dir = args.difficulty_manifest_dir.resolve()
    check_output_location(args.output_dir, args.input_dir, args.keep_manifest_dir, args.difficulty_manifest_dir)
    if args.selection_file is not None:
        args.selection_file = args.selection_file.resolve()
    jobs = build_jobs(args)
    if not jobs:
        raise SystemExit("no grounding jobs matched")
    groups = parse_device_groups(args.devices, args.tensor_parallel_size)
    assignments = assign_jobs(jobs, groups)
    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config = {
        "stage": "mllm_grounding",
        "prompt_version": prompt_version_for_mode(args.grounding_mode),
        "device_groups": groups,
        "total_rows": sum(job.num_rows for job in jobs),
        "selected_rows": sum(job.selected_rows for job in jobs),
        "args": args_dict,
        "jobs": [asdict(job) for job in jobs],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for worker_index, (physical_devices, worker_jobs) in enumerate(assignments):
        process = ctx.Process(
            target=worker_main,
            args=(worker_index, physical_devices, worker_jobs, args_dict, progress_queue),
            daemon=False,
        )
        process.start()
        processes.append(process)

    total_rows = sum(job.num_rows for job in jobs)
    summaries: List[Dict] = []
    done_workers = set()
    worker_errors = []
    pbar = tqdm(total=total_rows, desc="grounding rows", dynamic_ncols=True, mininterval=args.progress_mininterval)
    try:
        while len(done_workers) < len(processes):
            try:
                message = progress_queue.get(timeout=0.5)
            except queue.Empty:
                for index, process in enumerate(processes):
                    if index not in done_workers and not process.is_alive() and process.exitcode is not None:
                        done_workers.add(index)
                        if process.exitcode != 0:
                            worker_errors.append(f"worker-{index} exited with code {process.exitcode}")
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
                tqdm.write(f"done {Path(message['shard']).name}: rows={summary.get('rows', 0)} ground_fail={summary.get('ground_fail', 0)} errors={summary.get('errors', 0)}")
            elif kind == "worker_error":
                error = f"worker-{message.get('worker')} {message.get('devices')}: {message.get('error')}"
                worker_errors.append(error)
                tqdm.write(error)
            elif kind == "worker_done":
                done_workers.add(int(message["worker"]))
    finally:
        pbar.close()
        for process in processes:
            process.join()

    summary = aggregate(summaries)
    summary["expected_rows"] = total_rows
    summary["expected_shards"] = len(jobs)
    summary["shards"] = len(summaries)
    summary["selected_rows"] = sum(job.selected_rows for job in jobs)
    summary["skipped_existing_shards"] = sum(bool(item["summary"].get("skipped_existing")) for item in summaries)
    summary["worker_exit_codes"] = [process.exitcode for process in processes]
    summary["worker_errors"] = worker_errors
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if worker_errors or summary["errors"] or summary["rows"] != total_rows or len(summaries) != len(jobs) or any(process.exitcode != 0 for process in processes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
