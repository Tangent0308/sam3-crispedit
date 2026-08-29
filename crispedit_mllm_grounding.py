#!/usr/bin/env python3
"""Stage 1: Qwen3.5 grounding for the CrispEdit mask pipeline.

The default 8-GPU layout is four independent BF16 replicas with tensor/model
parallel size 2.  Each worker makes one request per routed grounding image and
writes all raw responses and validated boxes before SAM3 is loaded.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import queue
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from crispedit_grounding import (
    BBOX_REFINEMENT_PROMPT_VERSION,
    OBSERVATION_PROMPT_VERSION,
    PROMPT_VERSION,
    bbox_refinement_crop,
    box_needs_local_refinement,
    build_change_observation_prompt,
    build_bbox_refinement_prompt,
    build_grounding_requests,
    canonicalize_type,
    conservative_refined_bbox,
    grounding_images,
    grounding_is_complete,
    local_bbox_refinement_enabled,
    map_crop_bbox_to_full,
    parse_bbox_refinement_output,
    parse_change_observation,
    parse_grounding_output,
    prompt_version_for_mode,
)


DEFAULT_MODEL_PATH = "/mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B"

# Pass 1 still makes one source/result comparison, but color/material edits get
# paired overlapping views of those same two images.  This gives small faces,
# hands, fur, and other subject surfaces enough vision tokens without adding a
# third model turn or introducing pixel differences.
LOCAL_DETAIL_TILES = (
    ("upper-left", (0.0, 0.0, 0.55, 0.55)),
    ("upper-right", (0.45, 0.0, 1.0, 0.55)),
    ("lower-left", (0.0, 0.45, 0.55, 1.0)),
    ("lower-right", (0.45, 0.45, 1.0, 1.0)),
)


def build_local_detail_views(
    raw_type: object,
    source: Image.Image,
    target: Image.Image,
) -> List[Dict]:
    if canonicalize_type(raw_type) != "color":
        return []
    views = []
    for label, normalized_box in LOCAL_DETAIL_TILES:
        paired = []
        for image in (source, target):
            width, height = image.size
            x1, y1, x2, y2 = normalized_box
            crop_box = (
                round(x1 * width),
                round(y1 * height),
                round(x2 * width),
                round(y2 * height),
            )
            paired.append(image.crop(crop_box).resize(image.size, Image.Resampling.LANCZOS))
        views.append(
            {
                "label": label,
                "normalized_box": list(normalized_box),
                "source": paired[0],
                "target": paired[1],
            }
        )
    return views

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 bbox grounding for CrispEdit")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=os.environ.get("CRISPEDIT_GROUNDING_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--keep-manifest-dir", type=Path, default=None)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Optional JSON bad-case selection; emits sparse rows with original row_idx values",
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--grounding-mode",
        choices=("two-pass", "single"),
        default="two-pass",
        help="two-pass first observes realized changes, then grounds them; single reproduces prompt v2",
    )
    parser.add_argument("--include-types", default=None)
    parser.add_argument("--max-shards-per-type", type=int, default=None)
    parser.add_argument("--limit-rows-per-shard", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Samples grouped before request flattening")
    parser.add_argument("--request-batch-size", type=int, default=2, help="Maximum image-pair requests per generate call")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=1_310_720, help="Per-image Qwen preprocessing pixel cap")
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument(
        "--bbox-refinement",
        choices=("off", "small", "all"),
        default="small",
        help="Re-ground small/all proposal boxes in enlarged single-candidate crops",
    )
    parser.add_argument("--bbox-refine-threshold", type=float, default=220.0)
    parser.add_argument("--bbox-refine-min-context", type=float, default=120.0)
    parser.add_argument("--bbox-refine-context-scale", type=float, default=1.0)
    parser.add_argument("--bbox-safety-padding-min", type=float, default=12.0)
    parser.add_argument("--bbox-safety-padding-max", type=float, default=24.0)
    parser.add_argument("--gpu-memory-gib", type=int, default=74)
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
    selection = load_selection(args.selection_file)
    include = {part.strip() for part in args.include_types.split(",")} if args.include_types else None
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(args.input_dir.glob("*.parquet")):
        if selection is not None and path.name not in selection:
            continue
        raw_type = raw_type_from_filename(path)
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
            jobs.append(
                GroundingJob(
                    raw_type=raw_type,
                    input_path=str(input_path),
                    output_path=str(args.output_dir / input_path.name),
                    manifest_path=manifest_path,
                    row_indices=indices,
                    num_rows=num_rows,
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
    for job in sorted(jobs, key=lambda item: item.num_rows, reverse=True):
        bucket = min(buckets, key=lambda item: item["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += job.num_rows
    return [(item["group"], item["jobs"]) for item in buckets if item["jobs"]]


def decode_image(cell: Dict) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


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


def load_manifest(path: Optional[str]) -> Dict[int, Dict]:
    if path is None:
        return {}
    return {int(row["row_idx"]): row for row in pq.read_table(path).to_pylist()}


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


class Qwen35Grounder:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        self.torch = torch
        self.args = args
        self.prompt_version = prompt_version_for_mode(args.grounding_mode)
        visible_count = torch.cuda.device_count()
        if visible_count != args.tensor_parallel_size:
            raise RuntimeError(
                f"worker sees {visible_count} GPUs, expected TP={args.tensor_parallel_size}; "
                "CUDA_VISIBLE_DEVICES must be set before importing torch"
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
    def _conversation(
        source: Image.Image,
        target: Image.Image,
        prompt: str,
        detail_views: Optional[Sequence[Dict]] = None,
    ) -> List[Dict]:
        content = [
            {"type": "text", "text": "Image 1 (source, full image):"},
            {"type": "image", "image": source},
            {"type": "text", "text": "Image 2 (result, full image):"},
            {"type": "image", "image": target},
        ]
        for view in detail_views or []:
            label = view["label"]
            normalized_box = view["normalized_box"]
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Image 1 detail tile {label} (enlarged crop of full-image "
                            f"normalized region {normalized_box}):"
                        ),
                    },
                    {"type": "image", "image": view["source"]},
                    {"type": "text", "text": f"Image 2 matching detail tile {label}:"},
                    {"type": "image", "image": view["target"]},
                ]
            )
        content.append({"type": "text", "text": prompt})
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    @staticmethod
    def _followup_conversation(
        source: Image.Image,
        target: Image.Image,
        observation_prompt: str,
        observation_response: str,
        grounding_prompt: str,
        detail_views: Optional[Sequence[Dict]] = None,
    ) -> List[Dict]:
        """Rebuild the first turn plus its answer, then ask the grounding follow-up."""

        first_turn = Qwen35Grounder._conversation(
            source, target, observation_prompt, detail_views
        )[0]
        return [
            first_turn,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": observation_response}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": grounding_prompt}],
            },
        ]

    @staticmethod
    def _bbox_refinement_conversation(views: Sequence[Dict], prompt: str) -> List[Dict]:
        content = []
        for view in views:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Candidate {view['candidate_id']} enlarged crop for ref "
                            f"{json.dumps(view['ref'], ensure_ascii=False)}:"
                        ),
                    },
                    {"type": "image", "image": view["image"]},
                ]
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _bbox_refinement_views(
        self, image: Image.Image, boxes: Sequence[Dict]
    ) -> List[Dict]:
        views = []
        for candidate_id, box in enumerate(boxes):
            should_refine = self.args.bbox_refinement == "all" or (
                self.args.bbox_refinement == "small"
                and box_needs_local_refinement(box, self.args.bbox_refine_threshold)
            )
            if not should_refine:
                continue
            crop_bbox = bbox_refinement_crop(
                box["bbox_2d"],
                min_context=self.args.bbox_refine_min_context,
                context_scale=self.args.bbox_refine_context_scale,
            )
            width, height = image.size
            pixel_crop = (
                round(crop_bbox[0] * width / 1000.0),
                round(crop_bbox[1] * height / 1000.0),
                round(crop_bbox[2] * width / 1000.0),
                round(crop_bbox[3] * height / 1000.0),
            )
            views.append(
                {
                    "candidate_id": candidate_id,
                    "ref": box["ref"],
                    "initial_bbox": list(box["bbox_2d"]),
                    "crop_bbox": crop_bbox,
                    "image": image.crop(pixel_crop),
                }
            )
        return views

    def _generate(self, conversations: Sequence[List[Dict]]) -> List[str]:
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

    def infer(self, samples: Sequence[Dict]) -> List[Dict]:
        requests: List[Tuple[int, str, List[Dict]]] = []
        payloads = [
            {
                "schema_version": 2 if self.args.grounding_mode == "two-pass" else 1,
                "prompt_version": self.prompt_version,
                "grounding_mode": self.args.grounding_mode,
                "requests": [],
                "boxes": {"source": [], "target": []},
            }
            for _ in samples
        ]

        observations: List[Optional[Dict]] = [None] * len(samples)
        if self.args.grounding_mode == "two-pass":
            observation_jobs = []
            for sample_index, sample in enumerate(samples):
                prompt = build_change_observation_prompt(sample["type"], sample["instruction"])
                detail_views = build_local_detail_views(
                    sample["type"], sample["input_img"], sample["output_img"]
                )
                conversation = self._conversation(
                    sample["input_img"], sample["output_img"], prompt, detail_views
                )
                observation_jobs.append((sample_index, prompt, conversation, detail_views))
            for start in range(0, len(observation_jobs), self.args.request_batch_size):
                chunk = observation_jobs[start : start + self.args.request_batch_size]
                texts = self._generate([item[2] for item in chunk])
                for (sample_index, prompt, conversation, detail_views), initial_text in zip(chunk, texts):
                    text = initial_text
                    error = ""
                    parsed: Dict = {}
                    for attempt in range(self.args.parse_retries + 1):
                        try:
                            parsed = parse_change_observation(text)
                            error = ""
                            break
                        except Exception as exc:
                            error = repr(exc)
                            if attempt < self.args.parse_retries:
                                text = self._generate([conversation])[0]
                    observation = {
                        "prompt_version": OBSERVATION_PROMPT_VERSION,
                        "prompt": prompt,
                        "raw_text": text,
                        "parsed": parsed,
                        "parse_ok": not error,
                        "error": error,
                    }
                    if detail_views:
                        observation["detail_views"] = [
                            {
                                "label": view["label"],
                                "normalized_box": view["normalized_box"],
                            }
                            for view in detail_views
                        ]
                    observations[sample_index] = observation
                    payloads[sample_index]["observation"] = observation

        for sample_index, sample in enumerate(samples):
            observation = observations[sample_index]
            observation_context = observation["parsed"] if observation and observation["parse_ok"] else None
            # A malformed observation remains useful natural-language evidence
            # for the follow-up, but does not become a trusted JSON checklist.
            if observation and not observation["parse_ok"] and observation["raw_text"].strip():
                observation_context = {"unparsed_observation": observation["raw_text"].strip()}
            for request in build_grounding_requests(
                sample["type"], sample["instruction"], observation_context
            ):
                if observation is not None:
                    detail_views = build_local_detail_views(
                        sample["type"], sample["input_img"], sample["output_img"]
                    )
                    is_supplemental_side = request.grounding_image not in grounding_images(
                        sample["type"]
                    )
                    # For color/material verification and opposite-side
                    # collateral edits, the raw first answer can anchor pass 2
                    # to an erroneous `changed=false` or wrong-side entity.
                    # The normalized first-pass specification is already
                    # embedded in request.prompt, so use a clean second turn.
                    if canonicalize_type(sample["type"]) == "color" or is_supplemental_side:
                        conversation = self._conversation(
                            sample["input_img"],
                            sample["output_img"],
                            request.prompt,
                            detail_views,
                        )
                    else:
                        conversation = self._followup_conversation(
                            sample["input_img"],
                            sample["output_img"],
                            observation["prompt"],
                            observation["raw_text"],
                            request.prompt,
                            detail_views,
                        )
                else:
                    conversation = self._conversation(
                        sample["input_img"], sample["output_img"], request.prompt
                    )
                requests.append((sample_index, request.grounding_image, conversation))

        parsed_results: List[Dict] = []
        for start in range(0, len(requests), self.args.request_batch_size):
            chunk = requests[start : start + self.args.request_batch_size]
            conversations = [item[2] for item in chunk]
            texts = self._generate(conversations)
            for (_, _, conversation), text in zip(chunk, texts):
                error = ""
                boxes: List[Dict] = []
                for attempt in range(self.args.parse_retries + 1):
                    try:
                        boxes = parse_grounding_output(text)
                        error = ""
                        break
                    except Exception as exc:
                        error = repr(exc)
                        if attempt < self.args.parse_retries:
                            text = self._generate([conversation])[0]
                parsed_results.append(
                    {"raw_text": text, "boxes": boxes, "parse_ok": not error, "error": error}
                )

        refinement_jobs = []
        if self.args.bbox_refinement != "off":
            for result_index, ((sample_index, grounding_image, _), result) in enumerate(
                zip(requests, parsed_results)
            ):
                if not result["parse_ok"] or not result["boxes"]:
                    continue
                sample = samples[sample_index]
                # Background boxes identify foreground content to protect, not
                # the edit region itself.  Enlarging/refining those boxes can
                # unnecessarily subtract editable background, so preserve the
                # established background route exactly.
                if not local_bbox_refinement_enabled(sample["type"]):
                    continue
                selected_image = (
                    sample["input_img"] if grounding_image == "source" else sample["output_img"]
                )
                views = self._bbox_refinement_views(selected_image, result["boxes"])
                if not views:
                    continue
                candidates = [
                    {
                        "candidate_id": view["candidate_id"],
                        "ref": view["ref"],
                        "initial_bbox": view["initial_bbox"],
                        "crop_bbox": view["crop_bbox"],
                    }
                    for view in views
                ]
                prompt = build_bbox_refinement_prompt(candidates)
                conversation = self._bbox_refinement_conversation(views, prompt)
                refinement_jobs.append(
                    {
                        "result_index": result_index,
                        "views": views,
                        "prompt": prompt,
                        "conversation": conversation,
                    }
                )

        for start in range(0, len(refinement_jobs), self.args.request_batch_size):
            chunk = refinement_jobs[start : start + self.args.request_batch_size]
            texts = self._generate([job["conversation"] for job in chunk])
            for job, initial_text in zip(chunk, texts):
                text = initial_text
                error = ""
                refinements: List[Dict] = []
                for attempt in range(self.args.parse_retries + 1):
                    try:
                        refinements = parse_bbox_refinement_output(text)
                        error = ""
                        break
                    except Exception as exc:
                        error = repr(exc)
                        if attempt < self.args.parse_retries:
                            text = self._generate([job["conversation"]])[0]

                result = parsed_results[job["result_index"]]
                initial_boxes = [dict(box) for box in result["boxes"]]
                final_boxes = [dict(box) for box in initial_boxes]
                refined_by_id = {
                    item["candidate_id"]: item for item in refinements
                } if not error else {}
                audit_candidates = []
                for view in job["views"]:
                    candidate_id = view["candidate_id"]
                    refined = refined_by_id.get(candidate_id)
                    mapped_bbox = None
                    if refined is not None:
                        mapped_bbox = map_crop_bbox_to_full(
                            view["crop_bbox"], refined["bbox_2d"]
                        )
                    final_bbox = conservative_refined_bbox(
                        view["initial_bbox"],
                        mapped_bbox,
                        min_padding=self.args.bbox_safety_padding_min,
                        max_padding=self.args.bbox_safety_padding_max,
                    )
                    final_boxes[candidate_id]["bbox_2d"] = final_bbox
                    audit_candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "ref": view["ref"],
                            "initial_bbox": view["initial_bbox"],
                            "crop_bbox": view["crop_bbox"],
                            "refined_crop_bbox": refined["bbox_2d"] if refined else None,
                            "refined_full_bbox": mapped_bbox,
                            "final_bbox": final_bbox,
                        }
                    )
                result["initial_boxes"] = initial_boxes
                result["boxes"] = final_boxes
                result["bbox_refinement"] = {
                    "prompt_version": BBOX_REFINEMENT_PROMPT_VERSION,
                    "prompt": job["prompt"],
                    "raw_text": text,
                    "parse_ok": not error,
                    "error": error,
                    "candidates": audit_candidates,
                }

        for request, result in zip(requests, parsed_results):
            sample_index, grounding_image, _ = request
            entry = {"grounding_image": grounding_image, **result}
            payloads[sample_index]["requests"].append(entry)
            payloads[sample_index]["boxes"][grounding_image] = result["boxes"]
        return payloads


def _skip_row(
    row_idx: int,
    record: Dict,
    pre: Dict,
    model_name: str,
    prompt_version: str,
    grounding_mode: str,
) -> Dict:
    source = decode_image(record["input_img"])
    target = decode_image(record["output_img"])
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
        "source_width": source.width,
        "source_height": source.height,
        "target_width": target.width,
        "target_height": target.height,
        "mllm_model": model_name,
        "prompt_version": prompt_version,
        "grounding_seconds": 0.0,
        **pre,
    }


def _result_row(row_idx: int, sample: Dict, payload: Dict, pre: Dict, model_name: str, seconds: float) -> Dict:
    etype = canonicalize_type(sample["type"])
    parse_ok = not payload.get("runtime_error") and all(
        bool(item["parse_ok"]) for item in payload["requests"]
    )
    complete = parse_ok and grounding_is_complete(etype, payload["boxes"])
    if etype == "style":
        status = "STYLE_FULL_IMAGE"
    elif not parse_ok:
        status = "PARSE_ERROR"
    elif complete and etype == "replace" and not all(payload["boxes"].get(side) for side in ("source", "target")):
        status = "PARTIAL_OK"
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
        "qc_flag": "OK" if complete else "GROUND_FAIL",
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
    grounder: Qwen35Grounder,
    args: argparse.Namespace,
    progress_queue,
    worker_index: int,
) -> Dict:
    output_path = Path(job.output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    manifest = load_manifest(job.manifest_path)
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
                pre = prefilter_fields(manifest_row)
                if pre["filter_decision"] != "keep":
                    out_rows[slot] = _skip_row(
                        row_idx,
                        record,
                        pre,
                        model_name,
                        grounder.prompt_version,
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
                    "shard": Path(job.input_path).name,
                    "done_rows": summary["rows"],
                    "total_rows": job.num_rows,
                }
            )
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if completed and tmp_path.exists():
            tmp_path.replace(output_path)
    return summary


def worker_main(worker_index: int, physical_devices: List[int], jobs: List[GroundingJob], args_dict: Dict, progress_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in physical_devices)
    args = argparse.Namespace(**args_dict)
    try:
        progress_queue.put(
            {
                "kind": "log",
                "message": f"ground-worker-{worker_index} loading TP={len(physical_devices)} on physical GPUs {physical_devices}",
            }
        )
        grounder = Qwen35Grounder(args)
        for job in jobs:
            output_path = Path(job.output_path)
            if output_path.exists() and not args.overwrite:
                rows = pq.ParquetFile(output_path).metadata.num_rows
                summary = {
                    "rows": rows,
                    "errors": 0,
                    "ground_fail": 0,
                    "observation_parse_fail": 0,
                    "prefilter_skipped": 0,
                    "statuses": {},
                    "skipped_existing": True,
                }
                progress_queue.put({"kind": "rows", "count": rows})
            else:
                summary = process_job(job, grounder, args, progress_queue, worker_index)
            progress_queue.put({"kind": "shard_done", "worker": worker_index, "shard": job.input_path, "summary": summary})
    except Exception as exc:
        progress_queue.put({"kind": "worker_error", "worker": worker_index, "devices": physical_devices, "error": repr(exc)})
        if args.fail_fast:
            raise
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
    summary["worker_errors"] = worker_errors
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if worker_errors or summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
