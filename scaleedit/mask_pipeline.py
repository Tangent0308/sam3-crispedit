"""ScaleEdit hybrid masks: full canvas, inverse foreground, SAM3, and direct boxes."""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from crispedit.mask.pipeline import (
    _pcs_mask,
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


_OBJECT_OUTPUT_GUARD_FRAC = 0.12
_OBJECT_OUTPUT_GUARD_MIN_IMAGE_FRAC = 0.004
_OBJECT_COMPONENT_MIN_LARGEST_FRAC = 0.025
_NEGATIVE_SPACE_SEARCH_EXPAND_FRAC = 0.50
_NEGATIVE_SPACE_BOUNDARY_DILATE_FRAC = 0.003
_COMPACT_INTERACTION_RE = re.compile(
    r"\b(?:hand|hands|finger|fingers|phone|smartphone|cellphone|cell phone|"
    r"mobile phone)\b",
    re.IGNORECASE,
)


def _component_count(mask: np.ndarray) -> int:
    return max(
        0,
        cv2.connectedComponents(
            (mask > 0).astype(np.uint8), connectivity=8
        )[0]
        - 1,
    )


def _compact_completion_audit(component_count: int) -> Dict:
    return {
        "completion_applied": False,
        "completion_rule": "",
        "completion_radius": 0,
        "completion_added_area": 0,
        "completion_filled_hole_area": 0,
        "completion_component_count_before": int(component_count),
        "completion_component_count_after": int(component_count),
    }


def _complete_compact_interaction_mask(
    mask: np.ndarray,
    item: Dict,
    shape: Tuple[int, int],
    metadata: Dict,
    output_guard_bbox_2d: Sequence[float],
) -> Tuple[np.ndarray, Dict]:
    """Conservatively close PCS holes in hands and handheld phones.

    Text-conditioned PCS masks can contain small holes or adjacent fragments
    where fingers occlude a phone, or where a steering wheel occludes a hand.
    Closing every SAM object would damage legitimate negative space such as a
    mug handle or mirror frame, so this completion is restricted to an explicit
    compact-interaction vocabulary and PCS-selected masks.  Growth is clipped
    to the already trusted output guard and the radius is capped at six pixels.
    """

    original = (mask > 0).astype(np.uint8)
    before_components = _component_count(original)
    audit = _compact_completion_audit(before_components)
    ref = str(item.get("ref", ""))
    if (
        not original.any()
        or str(metadata.get("mask_source", "")) != "pcs"
        or str(item.get("region_mode", "object"))
        not in {"object", "multi_instance"}
        or str(item.get("mask_density", "object")) != "object"
        or not _COMPACT_INTERACTION_RE.search(ref)
    ):
        return original, audit

    anchor = normalized_box_to_pixels(item["bbox_2d"], shape)
    anchor_width = max(float(anchor[2] - anchor[0]), 1.0)
    anchor_height = max(float(anchor[3] - anchor[1]), 1.0)
    radius = max(2, min(6, int(round(min(anchor_width, anchor_height) * 0.025))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    completed = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)

    # Fill only small enclosed holes. The exterior background is connected to
    # an image edge and is never filled; the cap prevents filling a meaningful
    # opening even if SAM accidentally encloses one.
    inverse = (1 - completed).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    hole_cap = min(
        max(64, int(math.ceil(float(completed.sum()) * 0.03))),
        max(64, int(math.ceil(float(shape[0] * shape[1]) * 0.001))),
    )
    filled_hole_area = 0
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        enclosed = (
            x > 0
            and y > 0
            and x + width < shape[1]
            and y + height < shape[0]
        )
        if enclosed and area <= hole_cap:
            completed[labels == label] = 1
            filled_hole_area += area

    guard = mask_from_box(
        normalized_box_to_pixels(output_guard_bbox_2d, shape), shape
    )
    completed &= guard
    after_components = _component_count(completed)
    return completed.astype(np.uint8), {
        **audit,
        "completion_applied": True,
        "completion_rule": "compact_interaction_pcs_closing_v1",
        "completion_radius": int(radius),
        "completion_added_area": int(completed.sum() - original.sum()),
        "completion_filled_hole_area": int(filled_hole_area),
        "completion_component_count_after": int(after_components),
    }


def _pixel_box_to_normalized(
    box: Sequence[float] | None, shape: Tuple[int, int]
) -> List[float]:
    if box is None:
        return [0.0, 0.0, 0.0, 0.0]
    height, width = shape
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        round(1000.0 * x1 / max(width, 1), 3),
        round(1000.0 * y1 / max(height, 1), 3),
        round(1000.0 * x2 / max(width, 1), 3),
        round(1000.0 * y2 / max(height, 1), 3),
    ]


def _clean_semantic_object_mask(
    mask: np.ndarray,
    item: Dict,
    shape: Tuple[int, int],
    metadata: Dict,
) -> Tuple[np.ndarray, Dict]:
    """Keep ordinary object masks near their trusted MLLM anchor.

    The 35% object-search rectangle is deliberately generous so SAM can
    recover a clipped handle, limb, or toy.  It is not an appropriate output
    envelope: shelf edges and isolated semantic fragments found near the
    object must not survive just because they lie in that search rectangle.
    Dense/aggregate/sparse regions are excluded because disconnected pixels
    can be the intended edit for those contracts.
    """

    raw = (mask > 0).astype(np.uint8)
    raw_box = mask_to_box(raw)
    raw_components = max(0, cv2.connectedComponents(raw, connectivity=8)[0] - 1)
    cleanup = {
        "raw_semantic_bbox_2d": _pixel_box_to_normalized(raw_box, shape),
        "output_guard_bbox_2d": [0.0, 0.0, 1000.0, 1000.0],
        "raw_component_count": int(raw_components),
        "kept_component_count": int(raw_components),
        "removed_component_count": 0,
        "removed_area": 0,
    }
    ordinary_object = (
        str(item.get("mask_method", "sam")) == "sam"
        and str(item.get("region_mode", "object")) in {"object", "multi_instance"}
        and str(item.get("mask_density", "object")) == "object"
    )
    if not ordinary_object or not raw.any():
        return raw, {**cleanup, **_compact_completion_audit(raw_components)}

    anchor = normalized_box_to_pixels(item["bbox_2d"], shape)
    guard = expand_box(
        anchor,
        shape,
        frac=_OBJECT_OUTPUT_GUARD_FRAC,
        min_image_frac=_OBJECT_OUTPUT_GUARD_MIN_IMAGE_FRAC,
        min_margin_max_dimension_frac=0.08,
    )
    cleanup["output_guard_bbox_2d"] = _pixel_box_to_normalized(guard, shape)
    x1, y1, x2, y2 = [int(round(value)) for value in guard]
    guarded = raw.copy()
    guarded[:y1, :] = 0
    guarded[y2:, :] = 0
    guarded[:, :x1] = 0
    guarded[:, x2:] = 0
    if not guarded.any():
        return raw, {**cleanup, **_compact_completion_audit(raw_components)}

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        guarded, connectivity=8
    )
    if count <= 1:
        cleaned_audit = {
            **cleanup,
            "kept_component_count": int(guarded.any()),
            "removed_component_count": int(raw_components - int(guarded.any())),
            "removed_area": int(raw.sum() - guarded.sum()),
        }
        completed, completion = _complete_compact_interaction_mask(
            guarded,
            item,
            shape,
            metadata,
            cleaned_audit["output_guard_bbox_2d"],
        )
        return completed, {**cleaned_audit, **completion}

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest_label = int(np.argmax(areas)) + 1
    min_area = max(
        8,
        int(math.ceil(float(areas.max()) * _OBJECT_COMPONENT_MIN_LARGEST_FRAC)),
    )
    ax1, ay1, ax2, ay2 = [int(round(value)) for value in anchor]
    anchor_labels = set(
        int(value)
        for value in np.unique(labels[ay1:ay2, ax1:ax2])
        if int(value) > 0
    )
    keep = {
        label
        for label in anchor_labels
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    }
    if not keep:
        keep = {largest_label}
    cleaned = np.isin(labels, list(keep)).astype(np.uint8)
    cleaned_box = mask_to_box(cleaned)
    removed_components = max(0, raw_components - len(keep))
    cleaned_audit = {
        **cleanup,
        "semantic_bbox_2d": _pixel_box_to_normalized(cleaned_box, shape),
        "kept_component_count": len(keep),
        "removed_component_count": int(removed_components),
        "removed_area": int(raw.sum() - cleaned.sum()),
    }
    completed, completion = _complete_compact_interaction_mask(
        cleaned,
        item,
        shape,
        metadata,
        cleaned_audit["output_guard_bbox_2d"],
    )
    completed_box = mask_to_box(completed)
    cleaned_audit["semantic_bbox_2d"] = _pixel_box_to_normalized(
        completed_box, shape
    )
    return completed, {**cleaned_audit, **completion}


def _target_map_dilate_frac(item: Dict, mask: np.ndarray) -> float:
    """Use a small geometry-aware mapping margin instead of a fixed 1.5%."""

    if bool(item.get("negative_space", False)):
        return 0.002
    if str(item.get("mask_method", "sam")) == "box":
        return 0.002
    region_mode = str(item.get("region_mode", "object"))
    density = str(item.get("mask_density", "object"))
    if density == "sparse":
        return 0.002
    if region_mode == "aggregate_region" or density == "dense":
        return 0.004
    semantic_box = mask_to_box(mask)
    if semantic_box is None:
        return 0.002
    x1, y1, x2, y2 = semantic_box
    box_area = max(float(x2 - x1) * float(y2 - y1), 1.0)
    fill_ratio = float((mask > 0).sum()) / box_area
    # Thin contours such as mirror frames grow by several times under the old
    # 15px radius.  Three pixels on a 1024px image is enough antialias/alignment
    # tolerance without turning a frame into a thick wall patch.
    return 0.0025 if fill_ratio < 0.25 else 0.005


def _select_carrier_component(
    carrier_mask: np.ndarray,
    anchor_mask: np.ndarray,
    search_mask: np.ndarray,
) -> np.ndarray:
    """Select the carrier instance adjacent to the localized empty region."""

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (carrier_mask > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(carrier_mask, dtype=np.uint8)
    anchor_y, anchor_x = np.where(anchor_mask > 0)
    anchor_center = (
        float(anchor_x.mean()) if anchor_x.size else carrier_mask.shape[1] / 2.0,
        float(anchor_y.mean()) if anchor_y.size else carrier_mask.shape[0] / 2.0,
    )
    candidates = []
    for label in range(1, count):
        component = labels == label
        anchor_overlap = int((component & (anchor_mask > 0)).sum())
        search_overlap = int((component & (search_mask > 0)).sum())
        cx, cy = centroids[label]
        distance = (float(cx) - anchor_center[0]) ** 2 + (
            float(cy) - anchor_center[1]
        ) ** 2
        candidates.append(
            (
                int(anchor_overlap > 0),
                anchor_overlap,
                search_overlap,
                -distance,
                int(stats[label, cv2.CC_STAT_AREA]),
                label,
            )
        )
    selected_label = max(candidates)[-1]
    return (labels == selected_label).astype(np.uint8)


def _negative_space_from_carrier_mask(
    carrier_mask: np.ndarray,
    bbox_2d: Sequence[float],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict]:
    """Recover a localized hole/concavity by inverting a carrier silhouette.

    A plain inverse of the SAM result would select the entire image.  Instead,
    complete the selected carrier instance with a convex silhouette, subtract
    its observed foreground, and retain only the concavity that overlaps the
    MLLM negative-space anchor.  The expanded search recovers an open mouth
    that a tight bbox may stop at where it blends into the background.
    """

    anchor_box = normalized_box_to_pixels(bbox_2d, shape)
    search_box = expand_box(
        anchor_box,
        shape,
        frac=_NEGATIVE_SPACE_SEARCH_EXPAND_FRAC,
        min_image_frac=0.01,
    )
    anchor_mask = mask_from_box(anchor_box, shape)
    search_mask = mask_from_box(search_box, shape)
    carrier = _select_carrier_component(carrier_mask, anchor_mask, search_mask)
    carrier_box = mask_to_box(carrier)
    audit = {
        "negative_space_search_bbox_2d": _pixel_box_to_normalized(
            search_box, shape
        ),
        "carrier_semantic_bbox_2d": _pixel_box_to_normalized(carrier_box, shape),
        "negative_space_strategy": "carrier_convex_inverse",
    }
    if not carrier.any():
        return np.zeros(shape, dtype=np.uint8), audit

    ys, xs = np.where(carrier > 0)
    points = np.column_stack([xs, ys]).astype(np.int32)
    if len(points) < 3:
        return np.zeros(shape, dtype=np.uint8), audit
    completed = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(completed, cv2.convexHull(points), 1)
    inverse_candidates = (
        (completed > 0) & (carrier == 0) & (search_mask > 0)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse_candidates, connectivity=8
    )
    if count <= 1:
        return np.zeros(shape, dtype=np.uint8), audit

    candidate_scores = []
    for label in range(1, count):
        component = labels == label
        anchor_overlap = int((component & (anchor_mask > 0)).sum())
        candidate_scores.append(
            (
                int(anchor_overlap > 0),
                anchor_overlap,
                int(stats[label, cv2.CC_STAT_AREA]),
                label,
            )
        )
    selected_label = max(candidate_scores)[-1]
    negative = (labels == selected_label).astype(np.uint8)
    radius = max(
        1, round(min(shape) * _NEGATIVE_SPACE_BOUNDARY_DILATE_FRAC)
    )
    negative = dilate_mask(negative, radius) & search_mask
    audit.update(
        {
            "negative_space_component_count": int(count - 1),
            "negative_space_boundary_radius": int(radius),
        }
    )
    return negative.astype(np.uint8), audit


def _segment_negative_space(
    processor,
    state: Dict,
    item: Dict,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict]:
    """Segment a foreground carrier, then invert only its localized concavity."""

    carrier_ref = str(item.get("carrier_ref", "")).strip()
    if not carrier_ref:
        raise ValueError("negative-space item requires carrier_ref")
    height, width = shape
    full_box = np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float32)
    carrier_mask, carrier_metadata = _pcs_mask(
        processor,
        state,
        carrier_ref,
        full_box,
        shape,
        use_geometric_prompt=False,
    )
    errors: List[str] = []
    if carrier_mask is None:
        errors.append("negative_space:carrier_pcs_unavailable")
        mask = mask_from_box(normalized_box_to_pixels(item["bbox_2d"], shape), shape)
        inverse_audit = {
            "negative_space_search_bbox_2d": list(item["bbox_2d"]),
            "carrier_semantic_bbox_2d": [0.0, 0.0, 0.0, 0.0],
            "negative_space_strategy": "localized_box_fallback",
            "negative_space_component_count": 0,
            "negative_space_boundary_radius": 0,
        }
        source = "negative_space_box"
        selection_reason = "NEGATIVE_SPACE_CARRIER_UNAVAILABLE"
    else:
        mask, inverse_audit = _negative_space_from_carrier_mask(
            carrier_mask, item["bbox_2d"], shape
        )
        if not mask.any():
            errors.append("negative_space:no_localized_inverse_component")
            mask = mask_from_box(
                normalized_box_to_pixels(item["bbox_2d"], shape), shape
            )
            inverse_audit["negative_space_strategy"] = "localized_box_fallback"
            source = "negative_space_box"
            selection_reason = "NEGATIVE_SPACE_INVERSE_UNAVAILABLE"
        else:
            source = "negative_space_inverse"
            selection_reason = "NEGATIVE_SPACE_CARRIER_CONVEX_INVERSE"

    semantic_box = mask_to_box(mask)
    carrier_box = mask_to_box(carrier_mask) if carrier_mask is not None else None
    component_count = max(
        0,
        cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)[0]
        - 1,
    )
    return mask.astype(np.uint8), {
        **carrier_metadata,
        **inverse_audit,
        "mask_source": source,
        "semantic_mask_source": "carrier_pcs_inverse",
        "bbox_xyxy": [
            round(float(value), 2)
            for value in normalized_box_to_pixels(item["bbox_2d"], shape)
        ],
        "sam_anchor_bbox_2d": [float(value) for value in item["bbox_2d"]],
        "sam_positive_bbox_2d": [0.0, 0.0, 1000.0, 1000.0],
        "semantic_bbox_2d": _pixel_box_to_normalized(semantic_box, shape),
        "raw_semantic_bbox_2d": _pixel_box_to_normalized(semantic_box, shape),
        "output_guard_bbox_2d": inverse_audit["negative_space_search_bbox_2d"],
        "sam_search_expand_frac": _NEGATIVE_SPACE_SEARCH_EXPAND_FRAC,
        "pvs_area": 0,
        "pcs_area": int(carrier_mask.sum()) if carrier_mask is not None else 0,
        "pvs_component_count": 0,
        "pcs_component_count": (
            max(
                0,
                cv2.connectedComponents(
                    (carrier_mask > 0).astype(np.uint8), connectivity=8
                )[0]
                - 1,
            )
            if carrier_mask is not None
            else 0
        ),
        "pvs_pcs_iou": 0.0,
        "pvs_boundary_contacts": [False, False, False, False],
        "pcs_boundary_contacts": [False, False, False, False],
        "selection_reason": selection_reason,
        "sam_prompt": carrier_ref,
        "raw_component_count": int(component_count),
        "kept_component_count": int(component_count),
        "removed_component_count": 0,
        "removed_area": 0,
        "errors": errors,
        "carrier_semantic_bbox_2d": _pixel_box_to_normalized(carrier_box, shape),
    }


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
        if bool(item.get("negative_space", False)):
            if state is None:
                raise RuntimeError("SAM image state is unavailable for negative space")
            mask, metadata = _segment_negative_space(
                processor, state, item, shape
            )
        elif method == "box":
            mask, metadata = _direct_box_mask(item, shape)
        else:
            if state is None:
                raise RuntimeError("SAM image state is unavailable for a semantic item")
            item_region_mode = str(item.get("region_mode", "object"))
            # Multi-instance candidates are expanded to one item per member
            # before this point. Segment each member as an ordinary anchored
            # object instead of asking SAM to interpret an unsupported mode.
            sam_region_mode = (
                "object" if item_region_mode == "multi_instance" else item_region_mode
            )
            mask, metadata = segment_grounded_box(
                processor,
                state,
                str(item["ref"]),
                item["bbox_2d"],
                shape,
                edit_type=sam_edit_type,
                region_mode=sam_region_mode,
                mask_density=str(item.get("mask_density", "object")),
                anchor_box_2d=item.get("sam_anchor_bbox_2d"),
            )
            mask, cleanup = _clean_semantic_object_mask(
                mask, item, shape, metadata
            )
            metadata.update(cleanup)
            metadata["region_mode"] = item_region_mode
        metadata_rows.append(
            {
                "instance_id": (
                    f"{grounding_image}_c{int(item.get('candidate_id', index))}"
                    f"_m{int(item.get('member_index', 0))}"
                ),
                "candidate_id": int(item.get("candidate_id", index)),
                "member_index": int(item.get("member_index", 0)),
                "role": role,
                "grounding_image": grounding_image,
                "ref": str(item["ref"]),
                "bbox_2d": [float(value) for value in item["bbox_2d"]],
                "mask_method": method,
                "region_mode": str(item.get("region_mode", "object")),
                "selection_mode": str(item.get("selection_mode", "single")),
                "mask_extent": str(item.get("mask_extent", "whole_object")),
                "expected_count": item.get("expected_count"),
                "mask_density": str(item.get("mask_density", "object")),
                "negative_space": bool(item.get("negative_space", False)),
                "negative_space_carrier_ref": str(item.get("carrier_ref", "")),
                "mapped_from_target": False,
                "target_map_dilate_frac": 0.0,
                "target_map_dilate_radius": 0,
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
        [dict(item) for item in payload.get("protected_foreground", [])]
        if mode == "protect_foreground"
        else [dict(item) for item in payload.get("source", [])]
    )
    target_items = (
        []
        if mode == "protect_foreground"
        else [dict(item) for item in payload.get("target", [])]
    )
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
    for mask, metadata, item in zip(target_masks, target_instances, target_items):
        dilate_frac = _target_map_dilate_frac(item, mask)
        mapped = map_target_mask_to_source(
            mask,
            source_shape,
            ar_mismatch,
            dilate_frac=dilate_frac,
        )
        metadata["mapped_from_target"] = True
        metadata["target_map_dilate_frac"] = float(dilate_frac)
        metadata["target_map_dilate_radius"] = max(
            1,
            round(
                min(source_shape)
                * (dilate_frac + (0.02 if ar_mismatch else 0.0))
            ),
        )
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
    flags.extend(str(value) for value in payload.get("semantic_qc_flags", []))
    if any(
        item.get("mask_source") in {"box", "negative_space_box"}
        for item in all_instances
    ):
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
    elif any(
        value.startswith(
            (
                "COUNT_MISMATCH",
                "MULTI_INSTANCE_SINGLETON",
                "SPATIAL_ORDER_MISMATCH",
                "BBOX_PARTIAL_JSON_RECOVERY",
                "BBOX_SINGLE_QUERY_UNION_RECOVERY",
            )
        )
        for value in flags
    ):
        qc_flag = "SEMANTIC_QC"

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
