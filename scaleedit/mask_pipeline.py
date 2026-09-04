"""ScaleEdit hybrid masks: full canvas, inverse foreground, SAM3, and direct boxes."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from crispedit.mask.pipeline import (
    aspect_ratio_delta,
    dilate_mask,
    encode_rle,
    expand_box,
    map_target_mask_to_source,
    mask_from_box,
    mask_to_box,
    normalized_box_to_pixels,
    segment_grounded_box,
)
from scaleedit import MASK_POLICY_VERSION


def _union(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        result |= (mask > 0).astype(np.uint8)
    return result


def _direct_box_mask(item: Dict, shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
    raw_box = normalized_box_to_pixels(item["bbox_2d"], shape)
    # Grounding boxes already include a safety margin. Add only a tiny raster
    # margin for antialiased glyph/line boundaries, never the large SAM prompt margin.
    pixel_box = expand_box(raw_box, shape, frac=0.01, min_image_frac=0.002)
    mask = mask_from_box(pixel_box, shape)
    return mask, {
        "mask_source": "direct_box",
        "semantic_mask_source": "direct_box",
        "bbox_xyxy": [round(float(value), 2) for value in pixel_box],
        "selection_reason": "SCALEEDIT_SPARSE_OR_TEXT_REGION",
        "sam_prompt": "",
        "errors": [],
    }


def _segment_items(
    processor,
    state: Dict | None,
    items: Sequence[Dict],
    shape: Tuple[int, int],
    grounding_image: str,
    role: str,
    final_task: str,
) -> Tuple[List[np.ndarray], List[Dict]]:
    masks: List[np.ndarray] = []
    metadata_rows: List[Dict] = []
    sam_edit_type = "color" if final_task in {"color_change", "material_change"} else "replace"
    for index, item in enumerate(items):
        method = str(item.get("mask_method", "sam"))
        if method == "box":
            mask, metadata = _direct_box_mask(item, shape)
        else:
            if state is None:
                raise RuntimeError("SAM image state is unavailable for a semantic item")
            mask, metadata = segment_grounded_box(
                processor,
                state,
                str(item["ref"]),
                item["bbox_2d"],
                shape,
                edit_type=sam_edit_type,
                region_mode=str(item.get("region_mode", "object")),
                mask_density=str(item.get("mask_density", "object")),
            )
        metadata_rows.append(
            {
                "instance_id": f"{grounding_image}_{index}",
                "role": role,
                "grounding_image": grounding_image,
                "ref": str(item["ref"]),
                "bbox_2d": [float(value) for value in item["bbox_2d"]],
                "mask_method": method,
                "region_mode": str(item.get("region_mode", "object")),
                "mask_density": str(item.get("mask_density", "object")),
                "mapped_from_target": False,
                **metadata,
            }
        )
        masks.append(mask.astype(np.uint8))
    return masks, metadata_rows


def _finalize_instances(masks: Sequence[np.ndarray], instances: List[Dict]) -> None:
    for mask, instance in zip(masks, instances):
        rle = encode_rle(mask)
        instance["rle_size"] = rle["size"]
        instance["rle_counts"] = rle["counts"]
        instance["area"] = int(mask.sum())
        instance["predicted_iou"] = float(instance.get("predicted_iou", math.nan))
        instance["box_iou"] = float(instance.get("box_iou", math.nan))
        instance["inside_ratio"] = float(instance.get("inside_ratio", math.nan))
        instance["mapped_from_target"] = bool(instance.get("mapped_from_target", False))


def annotate_sample(processor, sample: Dict, ground_row: Dict, sam_version: str) -> Dict:
    source = sample["source"].convert("RGB")
    target = sample["target"].convert("RGB")
    source_shape = (source.height, source.width)
    target_shape = (target.height, target.width)
    payload = json.loads(ground_row["ground_json"])
    mode = str(payload.get("mask_mode", "unresolved"))
    final_task = str(ground_row.get("final_task", ""))
    ar_delta = aspect_ratio_delta(source.size, target.size)
    ar_mismatch = ar_delta > 0.02

    if ground_row.get("qc_flag") == "GROUND_FAIL":
        return {
            "mask": np.zeros(source_shape, dtype=np.uint8),
            "instances": [],
            "mask_source": "none",
            "qc_flag": "GROUND_FAIL",
            "qc_flags": ["GROUND_FAIL"],
            "ar_delta": ar_delta,
            "sam_version": sam_version,
        }

    if mode == "full_image":
        return {
            "mask": np.ones(source_shape, dtype=np.uint8),
            "instances": [],
            "mask_source": "full_image",
            "qc_flag": "OK",
            "qc_flags": ["FULL_IMAGE"],
            "ar_delta": ar_delta,
            "sam_version": sam_version,
        }

    source_items = (
        payload.get("protected_foreground", [])
        if mode == "protect_foreground"
        else payload.get("source", [])
    )
    target_items = [] if mode == "protect_foreground" else payload.get("target", [])
    source_needs_sam = any(item.get("mask_method", "sam") == "sam" for item in source_items)
    target_needs_sam = any(item.get("mask_method", "sam") == "sam" for item in target_items)
    source_state = processor.set_image(source) if source_needs_sam else None
    target_state = processor.set_image(target) if target_needs_sam else None

    source_masks, source_instances = _segment_items(
        processor,
        source_state,
        source_items,
        source_shape,
        "source",
        "preserve_foreground" if mode == "protect_foreground" else "edit_region",
        final_task,
    )
    target_masks, target_instances = _segment_items(
        processor,
        target_state,
        target_items,
        target_shape,
        "target",
        "edit_region",
        final_task,
    )

    mapped_target_masks: List[np.ndarray] = []
    for mask, metadata in zip(target_masks, target_instances):
        mapped = map_target_mask_to_source(mask, source_shape, ar_mismatch)
        metadata["mapped_from_target"] = True
        mapped_box = mask_to_box(mapped)
        metadata["bbox_xyxy"] = (
            [round(float(value), 2) for value in mapped_box]
            if mapped_box is not None
            else [0.0, 0.0, 0.0, 0.0]
        )
        mapped_target_masks.append(mapped)

    all_masks = source_masks + mapped_target_masks
    all_instances = source_instances + target_instances
    if mode == "protect_foreground":
        protected = _union(source_masks, source_shape)
        radius = max(1, round(min(source_shape) * 0.015))
        mask = (1 - dilate_mask(protected, radius)).astype(np.uint8)
    else:
        mask = _union(all_masks, source_shape)

    _finalize_instances(all_masks, all_instances)
    sources = {str(item.get("mask_source", "")) for item in all_instances}
    flags: List[str] = []
    if any(item.get("mask_source") == "box" for item in all_instances):
        flags.append("BOX_FALLBACK")
    if any(item.get("mask_source") == "direct_box" for item in all_instances):
        flags.append("DIRECT_BOX")
    if mode == "protect_foreground":
        flags.append("INVERSE_FOREGROUND")
    if ar_mismatch:
        flags.append("AR_MISMATCH")
    if not mask.any():
        flags.append("EMPTY_MASK")

    if mode == "protect_foreground":
        mask_source = "inverse_foreground"
    elif len(sources) == 1:
        mask_source = next(iter(sources))
    else:
        mask_source = "hybrid"
    qc_flag = "OK"
    if "EMPTY_MASK" in flags:
        qc_flag = "EMPTY_MASK"
    elif "BOX_FALLBACK" in flags:
        qc_flag = "BOX_FALLBACK"
    elif "AR_MISMATCH" in flags:
        qc_flag = "AR_MISMATCH"

    return {
        "mask": mask,
        "instances": all_instances,
        "mask_source": mask_source,
        "qc_flag": qc_flag,
        "qc_flags": flags or ["OK"],
        "ar_delta": ar_delta,
        "sam_version": sam_version,
        "mask_policy_version": MASK_POLICY_VERSION,
    }
