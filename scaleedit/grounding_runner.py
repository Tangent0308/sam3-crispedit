"""Stage 1: ScaleEdit paired-image reasoning and task-aware region grounding."""

from __future__ import annotations

import argparse
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
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from scaleedit import PROMPT_VERSION
from scaleedit.io import decode_image, discover_shards, image_size, iter_row_batches
from scaleedit.policy import (
    BBOX_LOCALIZATION_PROMPT_VERSION,
    TEXT_TASKS,
    apply_task_post_policy,
    apply_observation_plan_policy,
    build_bbox_correction_prompt,
    build_grounding_prompt,
    build_observation_correction_prompt,
    build_object_viewpoint_retry_prompt,
    build_observation_prompt,
    canonical_task,
    grounding_from_localization,
    localization_requires_image_pair,
    grounding_status,
    object_viewpoint_ref,
    parse_bbox_localization,
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


class CorruptSampleImageError(ValueError):
    """A source-data image cell cannot be decoded as an RGB image."""


def _decode_sample_images(record: Dict) -> Tuple[Image.Image, Image.Image]:
    decoded = []
    for field in ("source_image", "edited_image"):
        try:
            decoded.append(decode_image(record[field]))
        except Exception as exc:
            raise CorruptSampleImageError(f"{field}: {exc!r}") from exc
    return decoded[0], decoded[1]


def _aligned_low_frequency_lab_difference(
    source: Image.Image,
    target: Image.Image,
) -> np.ndarray | None:
    """Return a coarse aligned Lab difference map, or ``None`` if unsafe."""

    source_ar = source.width / max(source.height, 1)
    target_ar = target.width / max(target.height, 1)
    if abs(source_ar - target_ar) / max(source_ar, 1e-6) > 0.02:
        return None
    grid_width = 160
    grid_height = max(48, int(round(grid_width / source_ar)))

    def low_frequency_lab(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        small = cv2.resize(
            lab, (grid_width, grid_height), interpolation=cv2.INTER_AREA
        )
        return cv2.GaussianBlur(small, (0, 0), 2.0)

    return np.linalg.norm(
        low_frequency_lab(source) - low_frequency_lab(target), axis=2
    )


def _normalized_box_to_grid(
    box: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return (
        max(0, min(width - 1, int(math.floor(x1 * width / 1000.0)))),
        max(0, min(height - 1, int(math.floor(y1 * height / 1000.0)))),
        max(1, min(width, int(math.ceil(x2 * width / 1000.0)))),
        max(1, min(height, int(math.ceil(y2 * height / 1000.0)))),
    )


def _box_center(box: Sequence[float]) -> Tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection_width = max(
        0.0, min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0]))
    )
    intersection_height = max(
        0.0, min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1]))
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    return intersection / max(first_area + second_area - intersection, 1e-6)


def _interval_gap(a1: int, a2: int, b1: int, b2: int) -> int:
    return max(0, max(a1, b1) - min(a2, b2))


def refine_text_localization_boxes(
    source: Image.Image,
    target: Image.Image,
    final_task: object,
    plan: Dict,
    localized_boxes: Sequence[Dict],
) -> List[Dict]:
    """Refine text boxes with high-confidence aligned pair evidence.

    A text mask is a filled box, so SAM cannot repair a shifted locator result.
    When the two images retain the same geometry and differ locally, the glyph
    replacement itself is much stronger coordinate evidence than a second
    coordinate guess.  The MLLM boxes remain the search anchors; this routine
    only activates for text tasks with a low global pair difference, extracts
    a compact collinear change component, and records every adjustment.  It
    makes no additional model request and leaves non-text boxes untouched.
    """

    result = [dict(item) for item in localized_boxes]
    if canonical_task(final_task) not in TEXT_TASKS or not result:
        return result
    source_ar = source.width / max(source.height, 1)
    target_ar = target.width / max(target.height, 1)
    if abs(source_ar - target_ar) / max(source_ar, 1e-6) > 0.02:
        return result

    scale = min(1.0, 1280.0 / max(source.width, source.height, 1))
    grid_width = max(64, int(round(source.width * scale)))
    grid_height = max(64, int(round(source.height * scale)))

    def lab(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (grid_width, grid_height), Image.Resampling.LANCZOS
            )
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    difference = np.linalg.norm(lab(source) - lab(target), axis=2)
    median_difference = float(np.median(difference))
    if median_difference > 18.0:
        return result
    # Keep this a high-precision verifier. Lower thresholds merge reflections,
    # poster texture, or compression drift into a nearby text component.
    threshold = max(40.0, min(60.0, float(np.percentile(difference, 95))))
    changed = (difference > threshold).astype(np.uint8)
    changed_fraction = float(changed.mean())
    if changed_fraction < 0.00001 or changed_fraction > 0.12:
        return result

    plan_by_id = {
        int(item["candidate_id"]): item
        for item in plan.get("localization_items", [])
    }
    localized_by_id = {
        int(item["candidate_id"]): item for item in result
    }
    text_ids = [
        candidate_id
        for candidate_id, item in plan_by_id.items()
        if candidate_id in localized_by_id
        and (
            str(item.get("geometry", "")) == "sparse_marks"
            or str(item.get("mask_method", "")) == "box"
        )
    ]
    source_ids = [
        value
        for value in text_ids
        if plan_by_id[value].get("image_side") == "source"
    ]
    unused_target_ids = {
        value
        for value in text_ids
        if plan_by_id[value].get("image_side") == "target"
    }

    groups: List[List[int]] = []
    for source_id in source_ids:
        source_center = _box_center(localized_by_id[source_id]["bbox_2d"])
        candidates = []
        for target_id in unused_target_ids:
            target_center = _box_center(localized_by_id[target_id]["bbox_2d"])
            distance = math.hypot(
                source_center[0] - target_center[0],
                source_center[1] - target_center[1],
            )
            candidates.append((distance, target_id))
        if candidates and min(candidates)[0] <= 250.0:
            _, target_id = min(candidates)
            unused_target_ids.remove(target_id)
            groups.append([source_id, target_id])
        else:
            groups.append([source_id])
    groups.extend([[value] for value in sorted(unused_target_ids)])

    for candidate_ids in groups:
        original_boxes = [
            [float(value) for value in localized_by_id[candidate_id]["bbox_2d"]]
            for candidate_id in candidate_ids
        ]
        search_boxes = original_boxes
        correspondence_anchor_override = None
        # Replacement text normally keeps the old glyph block's position. A
        # short replacement string (for example "ART") may already occur
        # elsewhere in the edited image, so a pure target-side text query can
        # legitimately select the wrong occurrence. When the planner assigns
        # the source and target strings the same spatial role but their boxes
        # are disjoint, use the less-ambiguous old-text box as the search
        # anchor for aligned pair evidence. This changes no MLLM calls and is
        # deliberately disabled for text that actually moves to a new place.
        if len(candidate_ids) == 2:
            source_candidate_ids = [
                candidate_id
                for candidate_id in candidate_ids
                if plan_by_id[candidate_id].get("image_side") == "source"
            ]
            target_candidate_ids = [
                candidate_id
                for candidate_id in candidate_ids
                if plan_by_id[candidate_id].get("image_side") == "target"
            ]
            if len(source_candidate_ids) == 1 and len(target_candidate_ids) == 1:
                source_id = source_candidate_ids[0]
                target_id = target_candidate_ids[0]
                source_index = candidate_ids.index(source_id)
                target_index = candidate_ids.index(target_id)
                source_hint = " ".join(
                    str(plan_by_id[source_id].get("spatial_hint", ""))
                    .casefold()
                    .split()
                )
                target_hint = " ".join(
                    str(plan_by_id[target_id].get("spatial_hint", ""))
                    .casefold()
                    .split()
                )
                source_box = original_boxes[source_index]
                target_box = original_boxes[target_index]
                generic_hint_tokens = {
                    "a",
                    "an",
                    "at",
                    "above",
                    "below",
                    "from",
                    "in",
                    "of",
                    "on",
                    "replacing",
                    "text",
                    "the",
                    "to",
                }
                source_hint_tokens = set(
                    re.findall(r"[a-z0-9]+", source_hint)
                ) - generic_hint_tokens
                target_hint_tokens = set(
                    re.findall(r"[a-z0-9]+", target_hint)
                ) - generic_hint_tokens
                hint_overlap = len(source_hint_tokens & target_hint_tokens) / max(
                    1, min(len(source_hint_tokens), len(target_hint_tokens))
                )
                separated = max(
                    _interval_gap(
                        int(source_box[0]),
                        int(source_box[2]),
                        int(target_box[0]),
                        int(target_box[2]),
                    ),
                    _interval_gap(
                        int(source_box[1]),
                        int(source_box[3]),
                        int(target_box[1]),
                        int(target_box[3]),
                    ),
                )
                source_width = source_box[2] - source_box[0]
                source_height = source_box[3] - source_box[1]
                target_width = target_box[2] - target_box[0]
                target_height = target_box[3] - target_box[1]
                source_cross_size = min(source_width, source_height)
                target_cross_size = min(target_width, target_height)
                center_distance = math.dist(
                    _box_center(source_box), _box_center(target_box)
                )
                meaningful_shift = center_distance >= 0.65 * max(
                    1.0, min(source_cross_size, target_cross_size)
                )
                if (
                    source_hint
                    and (source_hint == target_hint or hint_overlap >= 0.50)
                    and _box_iou(source_box, target_box) < 0.05
                    and (separated >= 15 or meaningful_shift)
                ):
                    search_boxes = [list(source_box), list(source_box)]
                    correspondence_anchor_override = {
                        "rule": "same_location_text_uses_source_anchor_v1",
                        "source_candidate_id": source_id,
                        "target_candidate_id": target_id,
                        "rejected_target_bbox_2d": target_box,
                        "spatial_hint_token_overlap": round(hint_overlap, 3),
                    }
        union = [
            min(box[0] for box in search_boxes),
            min(box[1] for box in search_boxes),
            max(box[2] for box in search_boxes),
            max(box[3] for box in search_boxes),
        ]
        ux1, uy1, ux2, uy2 = _normalized_box_to_grid(
            union, grid_width, grid_height
        )
        union_width, union_height = max(ux2 - ux1, 1), max(uy2 - uy1, 1)
        margin_x = max(8, int(math.ceil(0.22 * union_width)))
        margin_y = max(6, int(math.ceil(0.35 * union_height)))
        rx1, ry1 = max(0, ux1 - margin_x), max(0, uy1 - margin_y)
        rx2 = min(grid_width, ux2 + margin_x)
        ry2 = min(grid_height, uy2 + margin_y)
        roi = np.zeros_like(changed)
        roi[ry1:ry2, rx1:rx2] = changed[ry1:ry2, rx1:rx2]

        horizontal = union_width >= union_height
        long_kernel = max(
            3,
            int(round((grid_width if horizontal else grid_height) * 0.008)),
        )
        if long_kernel % 2 == 0:
            long_kernel += 1
        short_kernel = 3
        kernel_size = (
            (long_kernel, short_kernel)
            if horizontal
            else (short_kernel, long_kernel)
        )
        connected = cv2.morphologyEx(
            roi,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            connected, connectivity=8
        )
        components = []
        for label in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[label]]
            if area < 4:
                continue
            overlap = int(
                np.count_nonzero(labels[uy1:uy2, ux1:ux2] == label)
            )
            if overlap:
                components.append(
                    {
                        "label": label,
                        "box": [x, y, x + width, y + height],
                        "area": area,
                        "overlap": overlap,
                    }
                )
        if not components:
            continue
        seed = max(
            components,
            key=lambda item: (item["overlap"], item["area"]),
        )
        selected = [seed]
        selected_labels = {seed["label"]}
        changed_selection = True
        while changed_selection:
            changed_selection = False
            envelope = [
                min(item["box"][0] for item in selected),
                min(item["box"][1] for item in selected),
                max(item["box"][2] for item in selected),
                max(item["box"][3] for item in selected),
            ]
            envelope_width = max(envelope[2] - envelope[0], 1)
            envelope_height = max(envelope[3] - envelope[1], 1)
            for component in components:
                if component["label"] in selected_labels:
                    continue
                box = component["box"]
                if component["area"] < max(4, int(seed["area"] * 0.01)):
                    continue
                if horizontal:
                    cross_gap = _interval_gap(
                        envelope[1], envelope[3], box[1], box[3]
                    )
                    main_gap = _interval_gap(
                        envelope[0], envelope[2], box[0], box[2]
                    )
                    cross_limit = max(envelope_height, box[3] - box[1])
                    main_limit = max(10, 2 * cross_limit, int(0.12 * union_width))
                else:
                    cross_gap = _interval_gap(
                        envelope[0], envelope[2], box[0], box[2]
                    )
                    main_gap = _interval_gap(
                        envelope[1], envelope[3], box[1], box[3]
                    )
                    cross_limit = max(envelope_width, box[2] - box[0])
                    main_limit = max(10, 2 * cross_limit, int(0.12 * union_height))
                if cross_gap <= cross_limit and main_gap <= main_limit:
                    selected.append(component)
                    selected_labels.add(component["label"])
                    changed_selection = True

        component_box = [
            min(item["box"][0] for item in selected),
            min(item["box"][1] for item in selected),
            max(item["box"][2] for item in selected),
            max(item["box"][3] for item in selected),
        ]
        padding = 3
        component_box = [
            max(0, component_box[0] - padding),
            max(0, component_box[1] - padding),
            min(grid_width, component_box[2] + padding),
            min(grid_height, component_box[3] + padding),
        ]
        refined = [
            round(1000.0 * component_box[0] / grid_width, 3),
            round(1000.0 * component_box[1] / grid_height, 3),
            round(1000.0 * component_box[2] / grid_width, 3),
            round(1000.0 * component_box[3] / grid_height, 3),
        ]
        refined_area = (refined[2] - refined[0]) * (refined[3] - refined[1])
        union_area = max((union[2] - union[0]) * (union[3] - union[1]), 1.0)
        if refined_area > 180_000.0 or not (0.12 <= refined_area / union_area <= 3.5):
            continue
        # If both MLLM boxes already agree and the change component would more
        # than double their footprint, the extra evidence is usually nearby
        # illustration/texture drift rather than the requested glyphs.
        if (
            len(original_boxes) == 2
            and _box_iou(original_boxes[0], original_boxes[1]) >= 0.80
            and refined_area > 1.5 * union_area
        ):
            continue
        audit = {
            "rule": "aligned_pair_text_component_refinement_v1",
            "paired_candidate_ids": list(candidate_ids),
            "difference_threshold": round(threshold, 3),
            "global_changed_fraction": round(changed_fraction, 6),
            "component_count": len(selected),
            "refined_bbox_2d": refined,
        }
        if correspondence_anchor_override is not None:
            audit["correspondence_anchor_override"] = correspondence_anchor_override
        for candidate_id, original in zip(candidate_ids, original_boxes):
            localized = localized_by_id[candidate_id]
            localized["bbox_2d"] = refined
            localized["bbox_refinement"] = {
                **audit,
                "original_bbox_2d": original,
            }
    return result


def expand_dense_aggregate_localization_boxes(
    plan: Dict,
    localized_boxes: Sequence[Dict],
) -> List[Dict]:
    """Add a bounded recall margin to compact dense-content boxes.

    A locator box for touching small objects or material in a container often
    follows the central visible mass and clips an outer row. SAM cannot recover
    pixels excluded from its crop. A small asymmetric margin is therefore part
    of the generic dense/aggregate contract. It is intentionally disabled for
    paired surface-completion candidates, which have their own evidence-based
    expansion and can cover much larger coherent structures.
    """

    result = [dict(item) for item in localized_boxes]
    plan_by_id = {
        int(item["candidate_id"]): item
        for item in plan.get("localization_items", [])
    }
    paired_surface_ids = {
        int(override["candidate_id"])
        for override in plan.get("plan_policy_overrides", [])
        if override.get("rule") == "appearance_surface_uses_all_changed_sections_v1"
        and override.get("candidate_id") is not None
    }
    for localized in result:
        candidate_id = int(localized["candidate_id"])
        item = plan_by_id.get(candidate_id, {})
        if (
            candidate_id in paired_surface_ids
            or item.get("mask_extent") == "surface_region"
            or item.get("mask_method") != "sam"
            or item.get("geometry") != "dense_region"
            or item.get("region_mode") != "aggregate_region"
        ):
            continue
        original = [float(value) for value in localized["bbox_2d"]]
        width = original[2] - original[0]
        height = original[3] - original[1]
        margin_x = max(8.0, 0.08 * width)
        margin_y = max(8.0, 0.15 * height)
        expanded = [
            round(max(0.0, original[0] - margin_x), 3),
            round(max(0.0, original[1] - margin_y), 3),
            round(min(1000.0, original[2] + margin_x), 3),
            round(min(1000.0, original[3] + margin_y), 3),
        ]
        localized["bbox_2d"] = expanded
        localized["bbox_refinement"] = {
            "rule": "dense_aggregate_recall_margin_v1",
            "original_bbox_2d": original,
            "refined_bbox_2d": expanded,
            "horizontal_margin_fraction": 0.08,
            "vertical_margin_fraction": 0.15,
        }
    return result


def refine_surface_localization_boxes(
    source: Image.Image,
    target: Image.Image,
    plan: Dict,
    localized_boxes: Sequence[Dict],
) -> List[Dict]:
    """Expand a surface box with aligned, low-frequency appearance change.

    Qwen can ground one named color band yet treat an adjoining band changed by
    the same generation as another building. For source-side ``surface_region``
    items only, a heavily downsampled/blurred pair finds a coherent change
    component that overlaps the trusted Qwen box and extends beyond it. The
    component only expands the SAM search box; pixel difference never becomes
    the output mask. Alignment, area, and overlap guards keep this disabled for
    camera or whole-composition changes.
    """

    result = [dict(item) for item in localized_boxes]
    paired_surface_ids = {
        int(override["candidate_id"])
        for override in plan.get("plan_policy_overrides", [])
        if override.get("rule") == "appearance_surface_uses_all_changed_sections_v1"
        and override.get("candidate_id") is not None
    }
    surface_plans = {
        int(item["candidate_id"]): item
        for item in plan.get("localization_items", [])
        if int(item["candidate_id"]) in paired_surface_ids
        if item.get("image_side") == "source"
        and item.get("mask_extent") == "surface_region"
        and not item.get("optional")
    }
    if not surface_plans or not result:
        return result
    difference = _aligned_low_frequency_lab_difference(source, target)
    if difference is None:
        return result
    grid_height, grid_width = difference.shape
    if float(np.median(difference)) > 25.0:
        return result

    optional_surface_plans = {
        int(item["candidate_id"]): item
        for item in plan.get("localization_items", [])
        if item.get("mask_extent") == "surface_region" and item.get("optional")
    }
    optional_surface_ids = set(optional_surface_plans)
    if any(int(item["candidate_id"]) in optional_surface_ids for item in result):
        return result

    kernel_open = np.ones((3, 3), dtype=np.uint8)
    kernel_close = np.ones((5, 5), dtype=np.uint8)
    for localized in result:
        candidate_id = int(localized["candidate_id"])
        if candidate_id not in surface_plans:
            continue
        original = [float(value) for value in localized["bbox_2d"]]
        ax1 = max(
            0,
            min(
                grid_width - 1,
                int(math.floor(original[0] * grid_width / 1000)),
            ),
        )
        ay1 = max(
            0,
            min(
                grid_height - 1,
                int(math.floor(original[1] * grid_height / 1000)),
            ),
        )
        ax2 = max(
            ax1 + 1,
            min(grid_width, int(math.ceil(original[2] * grid_width / 1000))),
        )
        ay2 = max(
            ay1 + 1,
            min(grid_height, int(math.ceil(original[3] * grid_height / 1000))),
        )
        anchor_area = max((ax2 - ax1) * (ay2 - ay1), 1)
        chosen = None
        for threshold in (52.0, 48.0, 44.0, 40.0, 36.0, 32.0, 28.0):
            changed = (difference > threshold).astype(np.uint8)
            changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel_open)
            changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel_close)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                changed, connectivity=8
            )
            for label in range(1, count):
                inside = int(
                    np.count_nonzero(labels[ay1:ay2, ax1:ax2] == label)
                )
                if inside < max(20, int(math.ceil(anchor_area * 0.08))):
                    continue
                x, y, width, height, _ = [int(value) for value in stats[label]]
                component = [
                    1000.0 * x / grid_width,
                    1000.0 * y / grid_height,
                    1000.0 * (x + width) / grid_width,
                    1000.0 * (y + height) / grid_height,
                ]
                component_area_frac = (
                    (component[2] - component[0])
                    * (component[3] - component[1])
                    / 1_000_000.0
                )
                if component_area_frac > 0.55:
                    continue
                anchor_width = max(original[2] - original[0], 1.0)
                anchor_height = max(original[3] - original[1], 1.0)
                extension = max(
                    original[0] - component[0],
                    component[2] - original[2],
                    original[1] - component[1],
                    component[3] - original[3],
                )
                if extension < 0.12 * min(anchor_width, anchor_height):
                    continue
                chosen = (component, threshold, inside)
                break
            if chosen is not None:
                break
        if chosen is None:
            continue
        component, threshold, inside = chosen
        refinement = {
            "rule": "aligned_surface_low_frequency_completion_v1",
            "original_bbox_2d": original,
            "change_component_bbox_2d": [
                round(value, 3) for value in component
            ],
            "difference_threshold": threshold,
            "anchor_overlap_cells": inside,
        }
        missing_optional_ids = [
            candidate_id
            for candidate_id in sorted(optional_surface_ids)
            if not any(
                int(candidate["candidate_id"]) == candidate_id
                for candidate in result
            )
        ]
        if missing_optional_ids:
            # Split the largest component extension away from the primary box
            # into an independent SAM anchor. A single expanded box makes SAM
            # prefer the primary facade and still omit the adjoining section.
            overlap_x = 0.05 * max(original[2] - original[0], 1.0)
            overlap_y = 0.05 * max(original[3] - original[1], 1.0)
            extensions = {
                "left": original[0] - component[0],
                "right": component[2] - original[2],
                "top": original[1] - component[1],
                "bottom": component[3] - original[3],
            }
            direction = max(extensions, key=extensions.get)
            if direction == "left":
                residual = [
                    component[0],
                    component[1],
                    min(1000.0, original[0] + overlap_x),
                    component[3],
                ]
            elif direction == "right":
                residual = [
                    max(0.0, original[2] - overlap_x),
                    component[1],
                    component[2],
                    component[3],
                ]
            elif direction == "top":
                residual = [
                    component[0],
                    component[1],
                    component[2],
                    min(1000.0, original[1] + overlap_y),
                ]
            else:
                residual = [
                    component[0],
                    max(0.0, original[3] - overlap_y),
                    component[2],
                    component[3],
                ]
            optional_id = missing_optional_ids[0]
            result.append(
                {
                    "candidate_id": optional_id,
                    "member_index": 0,
                    "ref": "",
                    "bbox_2d": [round(value, 3) for value in residual],
                    "bbox_refinement": {
                        **refinement,
                        "output": "independent_residual_anchor",
                        "extension_direction": direction,
                    },
                }
            )
        else:
            refined = [
                max(0.0, min(original[0], component[0])),
                max(0.0, min(original[1], component[1])),
                min(1000.0, max(original[2], component[2])),
                min(1000.0, max(original[3], component[3])),
            ]
            localized["bbox_2d"] = [round(value, 3) for value in refined]
            localized["bbox_refinement"] = refinement
    return result


_GLOBAL_SURFACE_REF_RE = re.compile(
    r"\b(?:background|environment|wall|floor|ground|sky|scene|canvas)\b",
    re.IGNORECASE,
)


def upgrade_global_change_mask_mode(
    source: Image.Image,
    target: Image.Image,
    grounded: Dict,
) -> Dict:
    """Promote a clearly global paired-image change from regions to full image.

    This is intentionally a high-precision override. It requires all three of:
    near-universal low-frequency appearance change, an MLLM-grounded background
    surface spanning three canvas boundaries, and a separate foreground edit
    whose union with that surface covers almost the complete composition.
    Local recolors and ordinary large objects therefore retain their regional
    route, while cases such as a glowing object plus a wholly relit environment
    become a full-canvas mask without another MLLM call.
    """

    if str(grounded.get("mask_mode", "")) != "regions":
        return grounded
    difference = _aligned_low_frequency_lab_difference(source, target)
    if difference is None:
        return grounded

    items = [
        dict(item)
        for side in ("source", "target")
        for item in grounded.get(side, [])
    ]
    if not items:
        return grounded
    surfaces = []
    non_surfaces = []
    for item in items:
        box = [float(value) for value in item.get("bbox_2d", [])]
        if len(box) != 4:
            continue
        is_named_surface = (
            str(item.get("mask_extent", "")) == "surface_region"
            and _GLOBAL_SURFACE_REF_RE.search(str(item.get("ref", ""))) is not None
        )
        (surfaces if is_named_surface else non_surfaces).append((item, box))
    if not surfaces or not non_surfaces:
        return grounded

    spanning_surfaces = []
    for item, box in surfaces:
        width = box[2] - box[0]
        height = box[3] - box[1]
        boundary_contacts = sum(
            (box[0] <= 25.0, box[1] <= 25.0, box[2] >= 975.0, box[3] >= 975.0)
        )
        if width >= 850.0 and height >= 500.0 and boundary_contacts >= 3:
            spanning_surfaces.append((item, box))
    if not spanning_surfaces:
        return grounded

    boxes = [box for _, box in surfaces + non_surfaces]
    envelope = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    if envelope[2] - envelope[0] < 900.0 or envelope[3] - envelope[1] < 750.0:
        return grounded

    median_difference = float(np.median(difference))
    mean_difference = float(np.mean(difference))
    p90_difference = float(np.percentile(difference, 90))
    changed_fraction = float(np.mean(difference > 44.0))
    if median_difference < 52.0 or changed_fraction < 0.90:
        return grounded

    result = dict(grounded)
    result["global_route_override"] = {
        "rule": "grounded_background_plus_global_pair_change_v1",
        "original_mask_mode": "regions",
        "original_source": grounded.get("source", []),
        "original_target": grounded.get("target", []),
        "evidence_bbox_2d": [round(value, 3) for value in envelope],
        "median_low_frequency_lab_difference": round(median_difference, 3),
        "mean_low_frequency_lab_difference": round(mean_difference, 3),
        "p90_low_frequency_lab_difference": round(p90_difference, 3),
        "fraction_above_44": round(changed_fraction, 6),
    }
    result["mask_mode"] = "full_image"
    result["source"] = []
    result["target"] = []
    result["protected_foreground"] = []
    return result


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
    parser.add_argument(
        "--inference-backend",
        choices=("vllm", "transformers"),
        default=os.environ.get("SCALEEDIT_INFERENCE_BACKEND", "vllm"),
        help="Qwen generation backend; prompts and ScaleEdit policy are identical",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-batch-size", type=int, default=4)
    parser.add_argument("--max-images-per-generate", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Legacy/default locator generation limit",
    )
    parser.add_argument(
        "--planner-max-new-tokens",
        type=int,
        default=2048,
        help="Independent Round-1 limit; larger than locator output to avoid truncated plans",
    )
    parser.add_argument(
        "--locator-max-new-tokens",
        type=int,
        default=None,
        help="Round-2 limit; defaults to --max-new-tokens",
    )
    parser.add_argument("--max-pixels", type=int, default=1_310_720)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--gpu-memory-gib", type=int, default=74)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
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

        self.torch = torch
        self.args = args
        self.inference_backend = args.inference_backend
        visible_count = torch.cuda.device_count()
        if visible_count != args.tensor_parallel_size:
            raise RuntimeError(
                f"worker sees {visible_count} GPUs, expected TP={args.tensor_parallel_size}"
            )
        if self.inference_backend == "vllm":
            self._init_vllm()
        else:
            self._init_transformers(visible_count)

    def _init_transformers(self, visible_count: int) -> None:
        from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        max_memory = {
            index: f"{self.args.gpu_memory_gib}GiB" for index in range(visible_count)
        }
        self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            self.args.model_path,
            dtype=self.torch.bfloat16,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.args.model_path, trust_remote_code=True, local_files_only=True
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            image_processor.size.longest_edge = int(self.args.max_pixels)
        self.input_device = next(
            parameter.device
            for parameter in self.model.parameters()
            if parameter.device.type != "meta"
        )

    def _init_vllm(self) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # vLLM/FlashInfer may JIT-compile a kernel by invoking the environment's
        # `ninja` executable. Directly calling venv/bin/python does not prepend
        # venv/bin to PATH, so make that invocation mode self-contained.
        # Do not resolve the interpreter symlink: uv-managed virtualenvs point
        # at the base Python, while console scripts live beside sys.executable.
        environment_bin = str(Path(sys.executable).parent)
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(
            self.args.model_path, trust_remote_code=True, local_files_only=True
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            image_processor.size.longest_edge = int(self.args.max_pixels)
        self.model = LLM(
            model=self.args.model_path,
            tensor_parallel_size=self.args.tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            seed=0,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.vllm_max_model_len,
            limit_mm_per_prompt={"image": 2},
            mm_processor_kwargs={"max_pixels": int(self.args.max_pixels)},
            generation_config="vllm",
        )
        # This is the vLLM equivalent of transformers.generate with
        # do_sample=False and all sampling warpers disabled.
        self.SamplingParams = SamplingParams
        self.sampling_params_by_max_tokens = {}
        self.sampling_params = self._sampling_params(self._locator_max_tokens())

    def _planner_max_tokens(self) -> int:
        return int(getattr(self.args, "planner_max_new_tokens", 2048))

    def _locator_max_tokens(self) -> int:
        value = getattr(self.args, "locator_max_new_tokens", None)
        if value is None:
            value = getattr(self.args, "max_new_tokens", 1024)
        return int(value)

    def _sampling_params(self, max_tokens: int):
        max_tokens = int(max_tokens)
        cached = self.sampling_params_by_max_tokens.get(max_tokens)
        if cached is not None:
            return cached
        params = self.SamplingParams(
            n=1,
            max_tokens=max_tokens,
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
        self.sampling_params_by_max_tokens[max_tokens] = params
        return params

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

    @staticmethod
    def _localization_conversation(
        source: Image.Image,
        target: Image.Image,
        prompt: str,
        plan: Dict,
    ) -> List[Dict]:
        """Send one image when every locator query is on the same side.

        Official Qwen grounding examples use a single image followed by a short
        locate request. Mixed source/target plans still use one two-image model
        call so the pipeline's MLLM call count does not increase.
        """

        image_sides = [
            side
            for side in ("source", "target")
            if any(
                str(item.get("image_side")) == side
                for item in plan.get("localization_items", [])
            )
        ]
        pair_required = localization_requires_image_pair(plan)
        if image_sides == ["source"] and not pair_required:
            content = [
                {"type": "image", "image": source},
                {"type": "text", "text": prompt},
            ]
        elif image_sides == ["target"] and not pair_required:
            content = [
                {"type": "image", "image": target},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [
                {"type": "text", "text": "Image 1 (source):"},
                {"type": "image", "image": source},
                {"type": "text", "text": "Image 2 (edited result):"},
                {"type": "image", "image": target},
                {"type": "text", "text": prompt},
            ]
        return [{"role": "user", "content": content}]

    @staticmethod
    def _conversation_images(conversation: List[Dict]) -> List[Image.Image]:
        return [
            part["image"]
            for message in conversation
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image"
        ]

    def _generate_transformers(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
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
                max_new_tokens=int(max_tokens or self._locator_max_tokens()),
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

    def _generate_vllm(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        requests = []
        for conversation in conversations:
            images = self._conversation_images(conversation)
            if len(images) not in (1, 2):
                raise ValueError(
                    f"ScaleEdit vLLM request requires one or two images, got {len(images)}"
                )
            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            requests.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images},
                    "mm_processor_kwargs": {
                        "max_pixels": int(self.args.max_pixels),
                    },
                }
            )
        sampling_params = (
            self.sampling_params
            if max_tokens is None
            else self._sampling_params(int(max_tokens))
        )
        results = self.model.generate(
            requests,
            sampling_params,
            use_tqdm=False,
        )
        if len(results) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(results)} results for {len(requests)} requests"
            )
        outputs = []
        for result in results:
            if len(result.outputs) != 1:
                raise RuntimeError(
                    f"vLLM returned {len(result.outputs)} candidates; expected exactly one"
                )
            outputs.append(result.outputs[0].text)
        return outputs

    def _generate_once(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        if self.inference_backend == "vllm":
            return self._generate_vllm(conversations, max_tokens=max_tokens)
        return self._generate_transformers(conversations, max_tokens=max_tokens)

    def _generate_backoff(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        try:
            return self._generate_once(conversations, max_tokens=max_tokens)
        except self.torch.cuda.OutOfMemoryError:
            if len(conversations) <= 1:
                raise
            for index in range(self.torch.cuda.device_count()):
                with self.torch.cuda.device(index):
                    self.torch.cuda.empty_cache()
            midpoint = len(conversations) // 2
            return self._generate_backoff(
                conversations[:midpoint], max_tokens=max_tokens
            ) + self._generate_backoff(
                conversations[midpoint:], max_tokens=max_tokens
            )

    def generate(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        outputs: List[str] = []
        for chunk in _chunks_by_image_budget(
            conversations, self.args.max_images_per_generate
        ):
            outputs.extend(self._generate_backoff(chunk, max_tokens=max_tokens))
        return outputs

    @staticmethod
    def _corrective_conversation(
        conversation: List[Dict], invalid_text: str, correction_prompt: str
    ) -> List[Dict]:
        return list(conversation) + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(invalid_text)}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": correction_prompt}],
            },
        ]

    def _parse_with_retry(
        self,
        conversation: List[Dict],
        text: str,
        parser,
        correction_builder,
        max_tokens: int,
    ):
        error = ""
        parsed = None
        attempts = []
        for attempt in range(self.args.parse_retries + 1):
            try:
                parsed = parser(text)
                error = ""
            except Exception as exc:
                error = repr(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_text": text,
                    "parse_ok": not error,
                    "error": error,
                }
            )
            if not error:
                break
            if attempt < self.args.parse_retries:
                correction_prompt = correction_builder(error)
                attempts[-1]["correction_prompt"] = correction_prompt
                retry_conversation = self._corrective_conversation(
                    conversation, text, correction_prompt
                )
                text = self.generate(
                    [retry_conversation], max_tokens=max_tokens
                )[0]
        return text, parsed, error, attempts

    def infer(self, samples: Sequence[Dict]) -> List[Dict]:
        observation_jobs = []
        for sample in samples:
            prompt = build_observation_prompt(sample["final_task"], sample["instruction"])
            conversation = self._conversation(sample["source"], sample["target"], prompt)
            observation_jobs.append((prompt, conversation))

        observation_texts: List[str] = []
        for start in range(0, len(observation_jobs), self.args.request_batch_size):
            chunk = observation_jobs[start : start + self.args.request_batch_size]
            observation_texts.extend(
                self.generate(
                    [item[1] for item in chunk],
                    max_tokens=self._planner_max_tokens(),
                )
            )

        observations = []
        for sample, (prompt, conversation), initial_text in zip(
            samples, observation_jobs, observation_texts
        ):
            text, parsed, error, attempts = self._parse_with_retry(
                conversation,
                initial_text,
                parse_observation,
                build_observation_correction_prompt,
                self._planner_max_tokens(),
            )
            observation = {
                "prompt": prompt,
                "raw_text": text,
                "parsed": parsed or {},
                "parse_ok": not error,
                "error": error,
                "attempts": attempts,
            }
            observations.append(observation)

        # Keep the rare viewpoint correction semantic-only. It repairs the
        # first-pass plan and leaves all coordinate work to the bbox locator.
        route_retry_jobs = []
        for sample_index, observation in enumerate(observations):
            parsed = observation.get("parsed") if observation.get("parse_ok") else None
            route_ref = object_viewpoint_ref(
                samples[sample_index]["final_task"], samples[sample_index]["instruction"]
            )
            if parsed is not None and parsed.get("mask_mode") == "full_image" and route_ref:
                retry_prompt = build_object_viewpoint_retry_prompt(
                    samples[sample_index]["final_task"],
                    samples[sample_index]["instruction"],
                    parsed,
                    route_ref,
                )
                route_retry_jobs.append(
                    {
                        "sample_index": sample_index,
                        "object_ref": route_ref,
                        "prompt": retry_prompt,
                        "conversation": self._conversation(
                            samples[sample_index]["source"],
                            samples[sample_index]["target"],
                            retry_prompt,
                        ),
                    }
                )

        route_retry_texts: List[str] = []
        for start in range(0, len(route_retry_jobs), self.args.request_batch_size):
            chunk = route_retry_jobs[start : start + self.args.request_batch_size]
            route_retry_texts.extend(
                self.generate(
                    [job["conversation"] for job in chunk],
                    max_tokens=self._planner_max_tokens(),
                )
            )
        route_retries: Dict[int, Dict] = {}
        for job, initial_text in zip(route_retry_jobs, route_retry_texts):
            retry_text, retry_parsed, retry_error, retry_attempts = self._parse_with_retry(
                job["conversation"],
                initial_text,
                parse_observation,
                build_observation_correction_prompt,
                self._planner_max_tokens(),
            )
            accepted = bool(
                not retry_error
                and retry_parsed is not None
                and retry_parsed.get("mask_mode") == "regions"
                and retry_parsed.get("localization_items")
            )
            route_retries[job["sample_index"]] = {
                "object_ref": job["object_ref"],
                "prompt": job["prompt"],
                "raw_text": retry_text,
                "parse_ok": not retry_error,
                "accepted": accepted,
                "error": retry_error
                or ("retry did not return a regions plan" if not accepted else ""),
                "attempts": retry_attempts,
            }
            if accepted:
                observations[job["sample_index"]]["parsed"] = retry_parsed

        # Enforce instruction-grounded scope invariants before constructing the
        # bbox-only prompt. This is deterministic and adds no model request.
        for sample, observation in zip(samples, observations):
            if observation.get("parse_ok") and observation.get("parsed"):
                observation["parsed"] = apply_observation_plan_policy(
                    sample["final_task"], sample["instruction"], observation["parsed"]
                )

        # Round 2 starts a fresh conversation and sees only the canonical item
        # checklist. The first prompt and raw first-pass JSON are deliberately
        # absent, so the model has one task: candidate_id -> complete bbox.
        localization_jobs = []
        for sample_index, (sample, observation) in enumerate(zip(samples, observations)):
            plan = observation.get("parsed") if observation.get("parse_ok") else None
            if not plan or plan.get("mask_mode") == "full_image":
                continue
            locator_prompt = build_grounding_prompt(
                sample["final_task"], sample["instruction"], plan
            )
            localization_jobs.append(
                {
                    "sample_index": sample_index,
                    "plan": plan,
                    "source": sample["source"],
                    "target": sample["target"],
                    "prompt": locator_prompt,
                    "conversation": self._localization_conversation(
                        sample["source"], sample["target"], locator_prompt, plan
                    ),
                }
            )

        localization_texts: List[str] = []
        for start in range(0, len(localization_jobs), self.args.request_batch_size):
            chunk = localization_jobs[start : start + self.args.request_batch_size]
            localization_texts.extend(
                self.generate(
                    [job["conversation"] for job in chunk],
                    max_tokens=self._locator_max_tokens(),
                )
            )

        localization_results: Dict[int, Dict] = {}
        for job, initial_text in zip(localization_jobs, localization_texts):
            expected_ids = [
                int(item["candidate_id"])
                for item in job["plan"].get("localization_items", [])
            ]
            aggregate_ids = [
                int(item["candidate_id"])
                for item in job["plan"].get("localization_items", [])
                if item.get("region_mode") == "aggregate_region"
            ]
            multi_ids = [
                int(item["candidate_id"])
                for item in job["plan"].get("localization_items", [])
                if item.get("region_mode") == "multi_instance"
                or item.get("selection_mode") == "all_matching"
            ]
            optional_ids = [
                int(item["candidate_id"])
                for item in job["plan"].get("localization_items", [])
                if item.get("optional")
            ]

            def locator_parser(
                value: str,
                *,
                plan=job["plan"],
                ids=expected_ids,
                aggregate=aggregate_ids,
                multi=multi_ids,
                optional=optional_ids,
                source=job["source"],
                target=job["target"],
                final_task=samples[job["sample_index"]]["final_task"],
            ) -> Dict:
                boxes = parse_bbox_localization(
                    value,
                    ids,
                    aggregate_candidate_ids=aggregate,
                    multi_candidate_ids=multi,
                    optional_candidate_ids=optional,
                )
                boxes = refine_text_localization_boxes(
                    source,
                    target,
                    final_task,
                    plan,
                    boxes,
                )
                boxes = expand_dense_aggregate_localization_boxes(plan, boxes)
                boxes = refine_surface_localization_boxes(
                    source, target, plan, boxes
                )
                grounded = grounding_from_localization(plan, boxes)
                return upgrade_global_change_mask_mode(source, target, grounded)

            text, parsed, error, attempts = self._parse_with_retry(
                job["conversation"],
                initial_text,
                locator_parser,
                lambda parse_error, ids=expected_ids, multi=multi_ids, aggregate=aggregate_ids, optional=optional_ids: build_bbox_correction_prompt(
                    ids,
                    parse_error,
                    multi_candidate_ids=multi,
                    aggregate_candidate_ids=aggregate,
                    optional_candidate_ids=optional,
                ),
                self._locator_max_tokens(),
            )
            localization_results[job["sample_index"]] = {
                "prompt": job["prompt"],
                "raw_text": text,
                "parsed": parsed,
                "error": error,
                "attempts": attempts,
            }

        payloads = []
        for sample_index, observation in enumerate(observations):
            plan = observation.get("parsed") if observation.get("parse_ok") else None
            localization = localization_results.get(sample_index)
            if plan is None:
                parsed = None
                error = observation.get("error") or "observation plan unavailable"
                grounding_audit = {
                    "prompt_version": BBOX_LOCALIZATION_PROMPT_VERSION,
                    "prompt": "",
                    "raw_text": "",
                    "parse_ok": False,
                    "error": error,
                    "skipped_reason": "observation_parse_failed",
                }
            elif plan.get("mask_mode") == "full_image":
                parsed = grounding_from_localization(plan, [])
                error = ""
                grounding_audit = {
                    "prompt_version": BBOX_LOCALIZATION_PROMPT_VERSION,
                    "prompt": "",
                    "raw_text": "",
                    "parse_ok": True,
                    "error": "",
                    "skipped_reason": "full_image_has_no_boxes",
                }
            else:
                parsed = localization.get("parsed") if localization else None
                error = (
                    localization.get("error", "")
                    if localization
                    else "bbox localization result unavailable"
                )
                grounding_audit = {
                    "prompt_version": BBOX_LOCALIZATION_PROMPT_VERSION,
                    "prompt": localization.get("prompt", "") if localization else "",
                    "raw_text": localization.get("raw_text", "") if localization else "",
                    "parse_ok": not error,
                    "error": error,
                    "attempts": localization.get("attempts", []) if localization else [],
                }
            parse_ok = bool(parsed is not None and observation.get("parse_ok") and not error)
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
                "ground_parse_ok": parse_ok,
                "observation": observation,
                "grounding": grounding_audit,
            }
            if sample_index in route_retries:
                payload["grounding"]["route_retry"] = route_retries[sample_index]
            payloads.append(payload)
        payloads = [
            apply_task_post_policy(sample["final_task"], sample["instruction"], payload)
            for sample, payload in zip(samples, payloads)
        ]
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
    summary = {
        "input_rows": 0,
        "rows": 0,
        "errors": 0,
        "skipped_images": 0,
        "skipped_samples": [],
        "statuses": {},
        "tasks": {},
    }
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
            summary["input_rows"] += len(indexed_records)
            valid_records = []
            samples = []
            for row_idx, record in indexed_records:
                try:
                    source, target = _decode_sample_images(record)
                except CorruptSampleImageError as exc:
                    skipped = {
                        "shard": input_path.name,
                        "row_idx": row_idx,
                        "sample_id": str(record.get("sample_id", "")),
                        "error": str(exc),
                    }
                    summary["skipped_images"] += 1
                    summary["skipped_samples"].append(skipped)
                    progress_queue.put(
                        {
                            "kind": "log",
                            "message": f"SKIP_CORRUPT_IMAGE {json.dumps(skipped, ensure_ascii=False)}",
                        }
                    )
                    progress_queue.put(
                        {
                            "kind": "rows",
                            "count": 1,
                            "worker": worker_index,
                            "shard": input_path.name,
                            "skipped": True,
                        }
                    )
                    continue
                valid_records.append((row_idx, record))
                samples.append(
                    {
                        "source": source,
                        "target": target,
                        "instruction": str(record["final_instruction"]),
                        "final_task": canonical_task(record["final_task"]),
                    }
                )
            if not samples:
                continue
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
                for (row_idx, record), payload in zip(valid_records, payloads)
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
        if summary["input_rows"] != job.num_rows:
            raise ValueError(
                f"selected row mismatch for {input_path.name}: "
                f"expected={job.num_rows} actual={summary['input_rows']}"
            )
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(tmp_path, GROUND_SCHEMA, compression=args.compression)
            writer.write_table(pa.Table.from_pylist([], schema=GROUND_SCHEMA))
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
                "message": (
                    f"ground-worker-{worker_index} loading Qwen with "
                    f"{args.inference_backend} on GPUs {physical_devices}"
                ),
            }
        )
        grounder = Qwen35ScaleEditGrounder(args)
        for job in jobs:
            output_path = Path(job.output_path)
            output_is_current = False
            if output_path.exists() and not args.overwrite:
                rows = pq.ParquetFile(output_path).metadata.num_rows
                if rows:
                    versions = set(
                        str(value)
                        for value in pq.read_table(
                            output_path, columns=["prompt_version"]
                        )[0].to_pylist()
                    )
                    output_is_current = versions == {PROMPT_VERSION}
                if output_is_current:
                    summary = {
                        "input_rows": job.num_rows,
                        "rows": rows,
                        "errors": 0,
                        "skipped_images": max(0, job.num_rows - rows),
                        "skipped_samples": [],
                        "statuses": {},
                        "tasks": {},
                        "skipped_existing": True,
                    }
                    progress_queue.put({"kind": "rows", "count": job.num_rows})
                else:
                    progress_queue.put(
                        {
                            "kind": "log",
                            "message": (
                                f"ground-worker-{worker_index} recomputing stale output "
                                f"{output_path.name} for prompt_version={PROMPT_VERSION}"
                            ),
                        }
                    )
            if not output_is_current:
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
    total = {
        "input_rows": 0,
        "rows": 0,
        "errors": 0,
        "skipped_images": 0,
        "skipped_samples": [],
        "statuses": {},
        "tasks": {},
    }
    for message in messages:
        summary = message.get("summary", {})
        total["input_rows"] += int(summary.get("input_rows", summary.get("rows", 0)))
        total["rows"] += int(summary.get("rows", 0))
        total["errors"] += int(summary.get("errors", 0))
        total["skipped_images"] += int(summary.get("skipped_images", 0))
        total["skipped_samples"].extend(summary.get("skipped_samples", []))
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
