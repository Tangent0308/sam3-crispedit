#!/usr/bin/env python3
"""Select same-class, subset-referential image-edit training examples.

The stages are deliberately separate processes.  vLLM releases its tensor-
parallel workers before SAM3 is loaded, which makes GPU ownership predictable and
also gives every expensive stage a resumable parquet checkpoint.
"""

from __future__ import annotations

import argparse
import heapq
import io
import json
import os
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from difficulty_filter.referential import (  # noqa: E402
    FILTER_POLICY_VERSION,
    MLLM_PROMPT_VERSION,
    SAM_COUNT_POLICY_VERSION,
    build_mllm_prompt,
    deduplicate_sam_instances,
    deterministic_screen,
    extract_grounding_refs,
    fuse_evidence,
    instruction_cues,
    parse_mllm_response,
    stable_priority,
)


DEFAULT_DATASET_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "ScaleEdit-CrispEdit-mask-train"
)
DEFAULT_QWEN_PATH = Path(
    "/mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B"
)
DEFAULT_SAM_PATH = Path(
    "/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"
)

METADATA_COLUMNS = [
    "sample_id",
    "source_dataset",
    "source_shard",
    "source_row_idx",
    "edit_type",
    "raw_edit_type",
    "instruction",
    "mask_mode",
    "mask_area_fraction",
]

CANDIDATE_COLUMNS = METADATA_COLUMNS + [
    "ground_json",
    "metadata_json",
    "source_image",
    "edited_image",
    "mask_png",
]


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression="zstd")
    temporary.replace(path)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_writable_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {path}")


def _stratum(row: Mapping[str, Any]) -> Tuple[str, bool, str, List[str]]:
    screen = deterministic_screen(str(row["edit_type"]), str(row["mask_mode"]))
    cues = instruction_cues(str(row["instruction"]))
    if not screen.eligible:
        return "excluded", False, screen.reason, cues
    if cues:
        return "cue_candidate", True, screen.reason, cues
    return "local_control", True, screen.reason, cues


def _quotas(sample_size: int, datasets: Sequence[str]) -> Dict[Tuple[str, str], int]:
    if sample_size < len(datasets) * 3:
        raise ValueError("sample-size is too small for the three sampling strata")
    per_dataset = [sample_size // len(datasets)] * len(datasets)
    for index in range(sample_size % len(datasets)):
        per_dataset[index] += 1
    result: Dict[Tuple[str, str], int] = {}
    for dataset, total in zip(datasets, per_dataset):
        excluded = max(1, round(total * 0.20))
        controls = max(1, round(total * 0.24))
        cues = total - excluded - controls
        result[(dataset, "cue_candidate")] = cues
        result[(dataset, "local_control")] = controls
        result[(dataset, "excluded")] = excluded
    return result


def _heap_offer(
    heap: List[Tuple[int, int, Dict[str, Any]]],
    row: Dict[str, Any],
    quota: int,
    seed: int,
    serial: int,
) -> None:
    if quota <= 0:
        return
    priority = int(stable_priority(str(row["sample_id"]), seed), 16)
    item = (-priority, serial, row)
    if len(heap) < quota:
        heapq.heappush(heap, item)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _selected_rows_with_images(
    selected: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    requests: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        requests[str(row["_parquet_path"])][int(row["_parquet_row_index"])] = row

    output: List[Dict[str, Any]] = []
    for path_text, wanted in tqdm(sorted(requests.items()), desc="load selected images"):
        table = pq.read_table(path_text, columns=CANDIDATE_COLUMNS)
        for row_index, selection in wanted.items():
            source = table.slice(row_index, 1).to_pylist()[0]
            if str(source["sample_id"]) != str(selection["sample_id"]):
                raise RuntimeError(f"row locator mismatch in {path_text} at {row_index}")
            source.update(
                {
                    "selection_stratum": selection["selection_stratum"],
                    "deterministic_eligible": selection["deterministic_eligible"],
                    "deterministic_reason": selection["deterministic_reason"],
                    "lexical_cues": selection["lexical_cues"],
                    "metadata_refs": extract_grounding_refs(source["ground_json"]),
                    "unified_parquet": path_text,
                    "unified_parquet_row_index": row_index,
                }
            )
            output.append(source)
    return sorted(output, key=lambda row: str(row["sample_id"]))


def stage_prepare(args: argparse.Namespace) -> None:
    _require_writable_output(args.output, args.overwrite)
    datasets = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
    quotas = _quotas(args.sample_size, datasets)
    heaps: Dict[Tuple[str, str], List[Tuple[int, int, Dict[str, Any]]]] = {
        key: [] for key in quotas
    }
    population = Counter()
    serial = 0
    for dataset in datasets:
        data_dir = args.dataset_root / dataset / "data"
        paths = sorted(data_dir.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no parquet shards under {data_dir}")
        for path in tqdm(paths, desc=f"scan {dataset}"):
            table = pq.read_table(path, columns=METADATA_COLUMNS)
            for row_index, row in enumerate(table.to_pylist()):
                stratum, eligible, reason, cues = _stratum(row)
                key = (dataset, stratum)
                population[key] += 1
                selection = dict(row)
                selection.update(
                    {
                        "selection_stratum": stratum,
                        "deterministic_eligible": eligible,
                        "deterministic_reason": reason,
                        "lexical_cues": cues,
                        "_parquet_path": str(path),
                        "_parquet_row_index": row_index,
                    }
                )
                _heap_offer(heaps[key], selection, quotas[key], args.seed, serial)
                serial += 1

    selected = []
    for key, requested in quotas.items():
        rows = [item[2] for item in heaps[key]]
        if len(rows) != requested:
            raise RuntimeError(f"stratum {key} has {len(rows)} rows, expected {requested}")
        selected.extend(rows)
    output_rows = _selected_rows_with_images(selected)
    _write_parquet(output_rows, args.output)
    summary = {
        "stage": "prepare",
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "sample_size": len(output_rows),
        "seed": args.seed,
        "quotas": {f"{key[0]}:{key[1]}": value for key, value in quotas.items()},
        "population": {
            f"{key[0]}:{key[1]}": value for key, value in sorted(population.items())
        },
        "source_dataset_counts": dict(Counter(row["source_dataset"] for row in output_rows)),
        "stratum_counts": dict(Counter(row["selection_stratum"] for row in output_rows)),
    }
    _write_json(summary, args.output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _device_list(spec: str) -> List[int]:
    devices = [int(part.strip().removeprefix("cuda:")) for part in spec.split(",") if part.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    return devices


def stage_mllm(args: argparse.Namespace) -> None:
    _require_writable_output(args.output, args.overwrite)
    devices = _device_list(args.devices)
    if len(devices) != args.tensor_parallel_size:
        raise ValueError(
            "this stage launches one vLLM engine, so --devices must contain exactly "
            f"TP={args.tensor_parallel_size} devices"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    environment_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    candidates = pq.read_table(args.input).to_pylist()
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
    )

    records: List[Dict[str, Any]] = []
    eligible = [row for row in candidates if row["deterministic_eligible"]]
    for row in candidates:
        if not row["deterministic_eligible"]:
            records.append(
                {
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
                    "mllm_reason": row["deterministic_reason"],
                }
            )

    for start in tqdm(range(0, len(eligible), args.batch_size), desc="Qwen3.5 audit"):
        chunk = eligible[start : start + args.batch_size]
        requests = []
        prompts = []
        for row in chunk:
            prompt = build_mllm_prompt(
                row["instruction"], row["edit_type"], row.get("metadata_refs") or []
            )
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": _decode_image(row["source_image"])},
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
            image = conversation[0]["content"][0]["image"]
            requests.append(
                {
                    "prompt": rendered,
                    "multi_modal_data": {"image": [image]},
                    "mm_processor_kwargs": {"max_pixels": args.max_pixels},
                }
            )
            prompts.append(prompt)
        results = model.generate(requests, sampling, use_tqdm=False)
        if len(results) != len(chunk):
            raise RuntimeError("vLLM output count does not match request count")
        for row, prompt, result in zip(chunk, prompts, results):
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
                    "visible_same_class_count": parsed["visible_same_class_count"],
                    "selected_instance_count": parsed["selected_instance_count"],
                    "subset_relation": parsed["subset_relation"],
                    "reference_cues": parsed["reference_cues"],
                    "fine_grained_referential": parsed["fine_grained_referential"],
                    "mllm_confidence": parsed["confidence"],
                    "mllm_reason": parsed["reason"],
                }
            )
    by_id = {str(row["sample_id"]): row for row in records}
    ordered = [by_id[str(row["sample_id"])] for row in candidates]
    _write_parquet(ordered, args.output)
    summary = {
        "stage": "mllm",
        "input": str(args.input),
        "output": str(args.output),
        "model": str(args.model_path),
        "prompt_version": MLLM_PROMPT_VERSION,
        "calls": len(eligible),
        "status_counts": dict(Counter(row["mllm_status"] for row in ordered)),
        "judgment_counts": dict(
            Counter(row["fine_grained_referential"] or "skipped" for row in ordered)
        ),
        "object_categories": dict(
            Counter(row["object_category"] for row in ordered if row["object_category"])
        ),
    }
    _write_json(summary, args.output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _encode_binary_mask(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def _run_sam_count(
    processor: Any,
    image: Image.Image,
    edit_mask_png: bytes,
    category: str,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
    selected_min_instance_fraction: float,
    selected_min_containment: float,
    selected_min_image_fraction: float,
    store_masks: bool,
) -> Dict[str, Any]:
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=category, state=state)
    masks = output["masks"].detach().cpu().numpy()
    # SAM3 runs under the production bfloat16 autocast context. NumPy has no
    # native bfloat16 dtype, so cast the small metadata tensors before export.
    boxes = output["boxes"].detach().float().cpu().numpy()
    scores = output["scores"].detach().float().cpu().numpy()
    height, width = image.height, image.width
    edit_mask = np.asarray(
        Image.open(io.BytesIO(edit_mask_png))
        .convert("L")
        .resize((width, height), Image.Resampling.NEAREST)
    ) > 0
    edit_area = int(edit_mask.sum())
    candidates = []
    area_rejects = []
    proposal_order = np.argsort(-scores)[:100]
    for index in proposal_order:
        mask_raw, box_raw, score_raw = masks[index], boxes[index], scores[index]
        mask = np.asarray(mask_raw).squeeze().astype(bool)
        area_fraction = float(mask.mean())
        record = {
            "proposal_index": index,
            "score": float(score_raw),
            "bbox_xyxy": [float(value) for value in np.asarray(box_raw).tolist()],
            "area_fraction": area_fraction,
            "_mask": mask,
        }
        if area_fraction < min_area_fraction or area_fraction > max_area_fraction:
            public = dict(record)
            public.pop("_mask")
            public["reject_reason"] = "implausible_area"
            area_rejects.append(public)
        else:
            candidates.append(record)
    kept, duplicates = deduplicate_sam_instances(candidates)
    public_kept = []
    masks_png = []
    for item in kept:
        mask = item.pop("_mask")
        instance_area = int(mask.sum())
        intersection = int(np.logical_and(mask, edit_mask).sum())
        instance_fraction = intersection / max(instance_area, 1)
        containment = intersection / max(min(instance_area, edit_area), 1)
        image_fraction = intersection / max(width * height, 1)
        item["edit_intersection_area"] = intersection
        item["edit_overlap_instance_fraction"] = instance_fraction
        item["edit_overlap_containment"] = containment
        item["edit_overlap_image_fraction"] = image_fraction
        item["selected_by_edit_mask"] = bool(
            image_fraction >= selected_min_image_fraction
            and (
                instance_fraction >= selected_min_instance_fraction
                or containment >= selected_min_containment
            )
        )
        public_kept.append(item)
        if store_masks:
            masks_png.append(_encode_binary_mask(mask))
    for item in duplicates:
        item.pop("_mask", None)
    return {
        "sam_status": "ok",
        "sam_count": len(public_kept),
        "sam_selected_count": sum(
            item["selected_by_edit_mask"] for item in public_kept
        ),
        "sam_unselected_count": sum(
            not item["selected_by_edit_mask"] for item in public_kept
        ),
        "sam_high_confidence_count": sum(item["score"] >= 0.50 for item in public_kept),
        "sam_raw_count": len(masks),
        "sam_instances": public_kept,
        "sam_duplicate_rejects": duplicates,
        "sam_area_rejects": area_rejects,
        "sam_instance_masks_png": masks_png,
        "image_width": width,
        "image_height": height,
    }


def stage_sam(args: argparse.Namespace) -> None:
    _require_writable_output(args.output, args.overwrite)
    devices = _device_list(args.devices)
    if len(devices) != 1:
        raise ValueError("the smoke-test SAM stage accepts exactly one device")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0])

    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    candidates = pq.read_table(args.input).to_pylist()
    mllm_rows = {
        str(row["sample_id"]): row for row in pq.read_table(args.mllm).to_pylist()
    }
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
    records = []
    for row in tqdm(candidates, desc="SAM3 same-class count"):
        evidence = mllm_rows.get(str(row["sample_id"]))
        base = {
            "sample_id": row["sample_id"],
            "sam_policy_version": SAM_COUNT_POLICY_VERSION,
            "sam_prompt": "" if evidence is None else evidence["object_category"],
            "sam_confidence_threshold": args.confidence_threshold,
        }
        if evidence is None or evidence["mllm_status"] != "ok":
            base.update(
                {
                    "sam_status": "skipped_no_mllm_category",
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
            )
        else:
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = _run_sam_count(
                        processor,
                        _decode_image(row["source_image"]),
                        row["mask_png"],
                        evidence["object_category"],
                        min_area_fraction=args.min_area_fraction,
                        max_area_fraction=args.max_area_fraction,
                        selected_min_instance_fraction=args.selected_min_instance_fraction,
                        selected_min_containment=args.selected_min_containment,
                        selected_min_image_fraction=args.selected_min_image_fraction,
                        store_masks=args.store_detection_masks,
                    )
                base.update(result)
                base["sam_error"] = ""
            except Exception as exc:
                base.update(
                    {
                        "sam_status": "error",
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
                        "sam_error": repr(exc),
                    }
                )
        records.append(base)
    _write_parquet(records, args.output)
    summary = {
        "stage": "sam",
        "input": str(args.input),
        "mllm": str(args.mllm),
        "output": str(args.output),
        "checkpoint": str(args.checkpoint_path),
        "policy_version": SAM_COUNT_POLICY_VERSION,
        "status_counts": dict(Counter(row["sam_status"] for row in records)),
        "count_histogram": dict(
            sorted(
                Counter(
                    row["sam_count"]
                    for row in records
                    if row["sam_count"] is not None
                ).items()
            )
        ),
    }
    _write_json(summary, args.output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def stage_fuse(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "audit.parquet"
    _require_writable_output(audit_path, args.overwrite)
    candidates = pq.read_table(args.input).to_pylist()
    mllm_by_id = {
        str(row["sample_id"]): row for row in pq.read_table(args.mllm).to_pylist()
    }
    sam_by_id = {str(row["sample_id"]): row for row in pq.read_table(args.sam).to_pylist()}
    audit = []
    for row in candidates:
        sample_id = str(row["sample_id"])
        mllm_row = mllm_by_id.get(sample_id)
        sam_row = sam_by_id.get(sample_id)
        parsed_mllm = None
        if mllm_row is not None and mllm_row["mllm_status"] == "ok":
            parsed_mllm = {
                "object_category": mllm_row["object_category"],
                "visible_same_class_count": mllm_row["visible_same_class_count"],
                "selected_instance_count": mllm_row["selected_instance_count"],
                "subset_relation": mllm_row["subset_relation"],
                "reference_cues": mllm_row["reference_cues"],
                "fine_grained_referential": mllm_row["fine_grained_referential"],
                "confidence": mllm_row["mllm_confidence"],
                "reason": mllm_row["mllm_reason"],
            }
        sam_count = None
        sam_selected_count = None
        if sam_row is not None and sam_row["sam_status"] == "ok":
            sam_count = int(sam_row["sam_count"])
            sam_selected_count = int(sam_row["sam_selected_count"])
        fusion = fuse_evidence(
            deterministic_eligible=bool(row["deterministic_eligible"]),
            deterministic_reason=str(row["deterministic_reason"]),
            mllm=parsed_mllm,
            sam_count=sam_count,
            sam_selected_count=sam_selected_count,
            all_selected_fraction=args.all_selected_fraction,
        )
        audit.append(
            {
                "sample_id": sample_id,
                "source_dataset": row["source_dataset"],
                "source_shard": row["source_shard"],
                "source_row_idx": row["source_row_idx"],
                "edit_type": row["edit_type"],
                "instruction": row["instruction"],
                "mask_mode": row["mask_mode"],
                "selection_stratum": row["selection_stratum"],
                "deterministic_eligible": row["deterministic_eligible"],
                "deterministic_reason": row["deterministic_reason"],
                "lexical_cues": row["lexical_cues"],
                "metadata_refs": row["metadata_refs"],
                "edited_subject_phrase": (
                    "" if mllm_row is None else mllm_row["edited_subject_phrase"]
                ),
                "object_category": "" if mllm_row is None else mllm_row["object_category"],
                "mllm_status": "missing" if mllm_row is None else mllm_row["mllm_status"],
                "mllm_visible_count": (
                    None if mllm_row is None else mllm_row["visible_same_class_count"]
                ),
                "mllm_selected_count": (
                    None if mllm_row is None else mllm_row["selected_instance_count"]
                ),
                "mllm_subset_relation": "" if mllm_row is None else mllm_row["subset_relation"],
                "mllm_judgment": "" if mllm_row is None else mllm_row["fine_grained_referential"],
                "mllm_confidence": None if mllm_row is None else mllm_row["mllm_confidence"],
                "mllm_reason": "" if mllm_row is None else mllm_row["mllm_reason"],
                "sam_status": "missing" if sam_row is None else sam_row["sam_status"],
                "sam_count": sam_count,
                "sam_selected_count": sam_selected_count,
                "sam_unselected_count": (
                    None if sam_row is None else sam_row["sam_unselected_count"]
                ),
                "sam_high_confidence_count": (
                    None if sam_row is None else sam_row["sam_high_confidence_count"]
                ),
                "filter_policy_version": FILTER_POLICY_VERSION,
                "decision": fusion["decision"],
                "loose_keep": fusion["loose_keep"],
                "decision_reason": fusion["reason"],
            }
        )
    _write_parquet(audit, audit_path)
    manifest_fields = [
        "sample_id",
        "source_dataset",
        "source_shard",
        "source_row_idx",
        "decision",
        "loose_keep",
        "decision_reason",
    ]
    selected = [
        {key: row[key] for key in manifest_fields}
        for row in audit
        if row["decision"] == "keep"
    ]
    loose_selected = [
        {key: row[key] for key in manifest_fields} for row in audit if row["loose_keep"]
    ]
    _write_parquet(selected, args.output_dir / "selected_manifest.parquet")
    _write_parquet(
        loose_selected, args.output_dir / "loose_selected_manifest.parquet"
    )
    summary = {
        "stage": "fuse",
        "policy_version": FILTER_POLICY_VERSION,
        "all_selected_fraction": args.all_selected_fraction,
        "input_rows": len(audit),
        "decision_counts": dict(Counter(row["decision"] for row in audit)),
        "strict_keep_rows": len(selected),
        "loose_keep_rows": len(loose_selected),
        "by_dataset": {
            dataset: dict(
                Counter(
                    row["decision"]
                    for row in audit
                    if row["source_dataset"] == dataset
                )
            )
            for dataset in sorted({row["source_dataset"] for row in audit})
        },
        "by_stratum": {
            stratum: dict(
                Counter(
                    row["decision"]
                    for row in audit
                    if row["selection_stratum"] == stratum
                )
            )
            for stratum in sorted({row["selection_stratum"] for row in audit})
        },
    }
    _write_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _fit_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _overlay_mask(image: Image.Image, mask_png: bytes, color: Tuple[int, int, int]) -> Image.Image:
    result = image.convert("RGBA")
    mask = (
        Image.open(io.BytesIO(mask_png))
        .convert("L")
        .resize(image.size, Image.Resampling.NEAREST)
    )
    tint = Image.new("RGBA", image.size, color + (105,))
    result.alpha_composite(Image.composite(tint, Image.new("RGBA", image.size), mask))
    return result.convert("RGB")


def _sam_panel(
    row: Mapping[str, Any], sam: Mapping[str, Any], size: Tuple[int, int]
) -> Image.Image:
    original = _decode_image(row["source_image"])
    rendered = original.copy()
    masks = sam.get("sam_instance_masks_png") or []
    palette = [(255, 70, 70), (50, 190, 90), (55, 120, 255), (235, 170, 30), (185, 70, 220)]
    for index, mask in enumerate(masks):
        rendered = _overlay_mask(rendered, mask, palette[index % len(palette)])
    draw = ImageDraw.Draw(rendered)
    for index, instance in enumerate(sam.get("sam_instances") or []):
        box = [round(float(value)) for value in instance["bbox_xyxy"]]
        color = palette[index % len(palette)]
        draw.rectangle(box, outline=color, width=max(2, min(original.size) // 250))
        draw.text((box[0] + 2, box[1] + 2), f"{index + 1}:{instance['score']:.2f}", fill=color)
    return _fit_image(rendered, size)


def stage_visualize(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = {str(row["sample_id"]): row for row in pq.read_table(args.input).to_pylist()}
    audit = pq.read_table(args.audit).to_pylist()
    sam = {str(row["sample_id"]): row for row in pq.read_table(args.sam).to_pylist()}
    cell_size = (args.cell_width, args.cell_height)
    row_width = cell_size[0] * 3
    font = ImageFont.load_default()
    pages = []
    for page_index, start in enumerate(range(0, len(audit), args.rows_per_page), 1):
        chunk = audit[start : start + args.rows_per_page]
        page = Image.new("RGB", (row_width, len(chunk) * (cell_size[1] + 132)), "white")
        draw = ImageDraw.Draw(page)
        for local_index, result in enumerate(chunk):
            sample_id = str(result["sample_id"])
            row = candidates[sample_id]
            sam_row = sam.get(sample_id, {"sam_instances": [], "sam_instance_masks_png": []})
            y = local_index * (cell_size[1] + 132)
            header = (
                f"[{start + local_index:02d}] {result['decision'].upper()} "
                f"loose={result['loose_keep']} | {result['source_dataset']} | "
                f"{result['edit_type']}\n"
                f"{result['instruction']}\n"
                f"category={result['object_category'] or '-'} | "
                f"MLLM={result['mllm_judgment'] or '-'} "
                f"visible={result['mllm_visible_count']} "
                f"subset={result['mllm_subset_relation'] or '-'} "
                f"| SAM total/selected={result['sam_count']}/{result['sam_selected_count']} "
                f"| {result['decision_reason']}"
            )
            wrapped = "\n".join(
                line
                for paragraph in header.splitlines()
                for line in textwrap.wrap(paragraph, width=155) or [""]
            )
            draw.multiline_text((6, y + 4), wrapped, fill="black", font=font, spacing=3)
            panel_y = y + 132
            source = _decode_image(row["source_image"])
            labeled = _overlay_mask(source, row["mask_png"], (255, 55, 55))
            source_panel = _fit_image(labeled, cell_size)
            sam_panel = _sam_panel(row, sam_row, cell_size)
            target_panel = _fit_image(_decode_image(row["edited_image"]), cell_size)
            page.paste(source_panel, (0, panel_y))
            page.paste(sam_panel, (cell_size[0], panel_y))
            page.paste(target_panel, (cell_size[0] * 2, panel_y))
            draw.text(
                (5, panel_y + 5),
                "source + edit mask",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            draw.text(
                (cell_size[0] + 5, panel_y + 5),
                "source + SAM peers",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            draw.text(
                (cell_size[0] * 2 + 5, panel_y + 5),
                "edited",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
        path = args.output_dir / f"page-{page_index:02d}.jpg"
        page.save(path, quality=92)
        pages.append(str(path))
    index = {
        "pages": pages,
        "legend": {
            "left": "source image with original edit-mask label in red",
            "middle": "source image with SAM3 same-class instance masks/boxes",
            "right": "edited image (not shown to the filter MLLM)",
        },
    }
    _write_json(index, args.output_dir / "index.json")
    print(json.dumps(index, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser("prepare", help="screen and stratify a small audit sample")
    prepare.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    prepare.add_argument("--datasets", default="crispedit,scaleedit")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--sample-size", type=int, default=50)
    prepare.add_argument("--seed", type=int, default=20260913)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(function=stage_prepare)

    mllm = subparsers.add_parser("mllm", help="run one source-image Qwen judgment per eligible row")
    mllm.add_argument("--input", type=Path, required=True)
    mllm.add_argument("--output", type=Path, required=True)
    mllm.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_PATH)
    mllm.add_argument("--devices", default="0,1")
    mllm.add_argument("--tensor-parallel-size", type=int, default=2)
    mllm.add_argument("--batch-size", type=int, default=8)
    mllm.add_argument("--max-new-tokens", type=int, default=384)
    mllm.add_argument("--max-pixels", type=int, default=1_310_720)
    mllm.add_argument("--max-model-len", type=int, default=8192)
    mllm.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    mllm.add_argument("--overwrite", action="store_true")
    mllm.set_defaults(function=stage_mllm)

    sam = subparsers.add_parser("sam", help="count all source-image peers with SAM3")
    sam.add_argument("--input", type=Path, required=True)
    sam.add_argument("--mllm", type=Path, required=True)
    sam.add_argument("--output", type=Path, required=True)
    sam.add_argument("--checkpoint-path", type=Path, default=DEFAULT_SAM_PATH)
    sam.add_argument("--devices", default="0")
    sam.add_argument("--confidence-threshold", type=float, default=0.30)
    sam.add_argument("--min-area-fraction", type=float, default=0.00003)
    sam.add_argument("--max-area-fraction", type=float, default=0.90)
    sam.add_argument("--selected-min-instance-fraction", type=float, default=0.02)
    sam.add_argument("--selected-min-containment", type=float, default=0.10)
    sam.add_argument("--selected-min-image-fraction", type=float, default=0.00001)
    sam.add_argument("--store-detection-masks", action="store_true")
    sam.add_argument("--overwrite", action="store_true")
    sam.set_defaults(function=stage_sam)

    fuse = subparsers.add_parser("fuse", help="combine deterministic, MLLM, and SAM evidence")
    fuse.add_argument("--input", type=Path, required=True)
    fuse.add_argument("--mllm", type=Path, required=True)
    fuse.add_argument("--sam", type=Path, required=True)
    fuse.add_argument("--output-dir", type=Path, required=True)
    fuse.add_argument("--all-selected-fraction", type=float, default=0.90)
    fuse.add_argument("--overwrite", action="store_true")
    fuse.set_defaults(function=stage_fuse)

    visualize = subparsers.add_parser(
        "visualize", help="render an auditable comparison contact sheet"
    )
    visualize.add_argument("--input", type=Path, required=True)
    visualize.add_argument("--sam", type=Path, required=True)
    visualize.add_argument("--audit", type=Path, required=True)
    visualize.add_argument("--output-dir", type=Path, required=True)
    visualize.add_argument("--rows-per-page", type=int, default=6)
    visualize.add_argument("--cell-width", type=int, default=420)
    visualize.add_argument("--cell-height", type=int, default=300)
    visualize.set_defaults(function=stage_visualize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
