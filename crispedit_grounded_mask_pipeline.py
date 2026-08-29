"""Stage-2 box-prompted SAM3 mask generation without pixel differences."""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

from crispedit_grounding import canonicalize_type


@dataclass(frozen=True)
class MaskConfig:
    box_expand_frac: float = 0.025
    # A deterministic image-relative safety margin is applied only along a
    # genuinely tiny box dimension.  Applying it to normal face/limb boxes can
    # change SAM's semantic proposal, while tiny earrings, fingers and petals
    # still need more than a percentage of their own one-pixel-scale extent.
    min_box_expand_image_frac: float = 0.015
    min_box_expand_max_dimension_frac: float = 0.05
    pvs_box_iou: float = 0.60
    pvs_inside_ratio: float = 0.90
    pvs_containment_expand_frac: float = 0.05
    pcs_inside_ratio: float = 0.80
    pcs_containment_expand_frac: float = 0.05
    min_directional_box_coverage: float = 0.80
    pvs_sparse_fill_ratio: float = 0.35
    pvs_dense_fill_ratio: float = 0.70
    semantic_area_ratio: float = 0.70
    pcs_dense_fill_ratio: float = 0.65
    pcs_sparse_fill_ratio: float = 0.40
    pcs_dense_area_multiple: float = 1.75
    pcs_tiny_fill_ratio: float = 0.01
    pcs_tiny_recovery_max_fill_ratio: float = 0.50
    pcs_tiny_recovery_max_selected: int = 2
    pcs_tiny_recovery_max_confidence: float = 0.50
    target_map_dilate_frac: float = 0.015
    target_map_connected_region_dilate_frac: float = 0.003
    ar_mismatch_threshold: float = 0.02
    ar_mismatch_extra_dilate_frac: float = 0.02
    background_foreground_dilate_frac: float = 0.015
    connected_region_min_components: int = 6
    connected_region_min_selected_instances: int = 6
    connected_region_max_bbox_area_frac: float = 0.50
    connected_region_max_median_component_area_frac: float = 0.003
    connected_region_max_largest_component_ratio: float = 0.60
    connected_region_hull_dilate_frac: float = 0.004


CFG = MaskConfig()
MASK_POLICY_VERSION = "sam3-dual-prompt-region-fusion-v5-surface-aware"


# A box is not a sufficient semantic prompt for these targets: it commonly
# selects the enclosing face/person/background instead of the sparse edit.  PCS
# is still spatially restricted by the MLLM box, so this does not reintroduce
# the old pipeline's global same-class-instance ambiguity.
_SEMANTIC_DETAIL_RE = re.compile(
    r"\b(?:all relevant|multiple|scattered|array|various|surround(?:ing|ed)?|"
    r"throughout|among|decorat\w*|flowers?|petals?|piercings?|tattoos?|chains?|"
    r"tufts?|fungi|boats?|birds?|stars?|arms?|hands?|glasses|implants?|frames?|"
    r"containing)\b",
    flags=re.IGNORECASE,
)

_AGGREGATE_REF_RE = re.compile(
    r"\b(?:multiple|scattered|cluster(?:ed)?|group|array|row|band|halo|spray|"
    r"bunch|piercings|petals|flowers|stars|spots|dots|crumbs|debris|decorations)\b",
    flags=re.IGNORECASE,
)

_SPARSE_REF_RE = re.compile(
    r"\b(?:scattered|separate|individual|piercings|petals|stars|spots|dots|"
    r"crumbs|debris|birds|studs|gems|marks)\b",
    flags=re.IGNORECASE,
)

_CONNECTABLE_SMALL_GROUP_RE = re.compile(
    r"\b(?:flowers?|wildflowers?|blooms?|petals?|piercings?|tattoos?|stars?|"
    r"spots?|dots?|studs?|gems?|beads?|speckles?|confetti|marks?)\b",
    flags=re.IGNORECASE,
)

_HUMAN_SUBJECT_RE = re.compile(
    r"\b(?:human|person|people|man|man's|men|male|woman|woman's|women|female|"
    r"boy|boy's|girl|girl's|performer|subject|model)\b",
    flags=re.IGNORECASE,
)

_HUMAN_SKIN_PART_RE = re.compile(
    r"\b(?:face|head|neck|arms?|hands?|legs?|feet|foot)\b",
    flags=re.IGNORECASE,
)


def decode_image(cell: Dict) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def encode_mask_png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if radius <= 0 or mask.size == 0:
        return mask
    size = 2 * int(radius) + 1
    image = Image.fromarray(mask * 255, mode="L")
    return (np.asarray(image.filter(ImageFilter.MaxFilter(size=size))) > 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    return (np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def clip_box(box: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def normalized_box_to_pixels(box_2d: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [float(value) for value in box_2d]
    return clip_box([x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0], shape)


def expand_box(
    box: Sequence[float],
    shape: Tuple[int, int],
    frac: float,
    min_image_frac: float = 0.0,
    min_margin_max_dimension_frac: Optional[float] = None,
) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [float(value) for value in box]
    short_side = float(min(height, width))
    width = x2 - x1
    height = y2 - y1
    relative_margin_x = width * frac
    relative_margin_y = height * frac
    minimum_margin = short_side * min_image_frac
    tiny_dimension_limit = (
        math.inf
        if min_margin_max_dimension_frac is None
        else short_side * min_margin_max_dimension_frac
    )
    margin_x = (
        max(relative_margin_x, minimum_margin)
        if width <= tiny_dimension_limit
        else relative_margin_x
    )
    margin_y = (
        max(relative_margin_y, minimum_margin)
        if height <= tiny_dimension_limit
        else relative_margin_y
    )
    return clip_box([x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y], shape)


def mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def box_iou(first: Optional[Sequence[float]], second: Optional[Sequence[float]]) -> float:
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(intersection / max(area_a + area_b - intersection, 1e-8))


def mask_from_box(box: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = clip_box(box, shape)
    left, top = int(math.floor(x1)), int(math.floor(y1))
    right, bottom = int(math.ceil(x2)), int(math.ceil(y2))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, top) : min(height, bottom), max(0, left) : min(width, right)] = 1
    return mask


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _outside_ratio(mask: np.ndarray, box: Sequence[float], shape: Tuple[int, int]) -> float:
    area = int(mask.sum())
    if area == 0:
        return 1.0
    inside = int((mask & mask_from_box(box, shape)).sum())
    return float(1.0 - inside / area)


def _directional_coverage(mask: np.ndarray, box: Sequence[float]) -> Tuple[float, float]:
    mask_box = mask_to_box(mask)
    if mask_box is None:
        return 0.0, 0.0
    mx1, my1, mx2, my2 = mask_box
    bx1, by1, bx2, by2 = [float(value) for value in box]
    x_cover = max(0.0, min(mx2, bx2) - max(mx1, bx1)) / max(bx2 - bx1, 1e-8)
    y_cover = max(0.0, min(my2, by2) - max(my1, by1)) / max(by2 - by1, 1e-8)
    return float(x_cover), float(y_cover)


def _norm_cxcywh(box: Sequence[float], shape: Tuple[int, int]) -> List[float]:
    height, width = shape
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def _pvs_mask(processor, state: Dict, box: np.ndarray, shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Dict]:
    masks, predicted_ious, _ = processor.model.predict_inst(
        inference_state=state,
        box=np.asarray(box, dtype=np.float32),
        multimask_output=True,
    )
    candidates = []
    containment_box = expand_box(box, shape, CFG.pvs_containment_expand_frac)
    for index in range(len(masks)):
        mask = (np.asarray(masks[index]) > 0).astype(np.uint8)
        if mask.shape != shape:
            mask = resize_mask(mask, shape)
        candidate_box = mask_to_box(mask)
        candidate_box_iou = box_iou(candidate_box, box)
        inside_ratio = 1.0 - _outside_ratio(mask, containment_box, shape)
        predicted_iou = float(np.asarray(predicted_ious[index]).item())
        candidates.append(
            {
                "mask": mask,
                "predicted_iou": predicted_iou,
                "box_iou": candidate_box_iou,
                "inside_ratio": inside_ratio,
                "accepted": candidate_box_iou >= CFG.pvs_box_iou and inside_ratio >= CFG.pvs_inside_ratio,
            }
        )
    accepted = [item for item in candidates if item["accepted"]]
    if not accepted:
        return None, {"candidate_count": len(candidates), "reason": "PVS_INCONSISTENT"}
    best = max(accepted, key=lambda item: (item["predicted_iou"], item["box_iou"]))
    mask = best.pop("mask")
    best["candidate_count"] = len(candidates)
    best["selected_count"] = 1
    return mask, best


def _pcs_mask(
    processor,
    state: Dict,
    ref: str,
    box: np.ndarray,
    shape: Tuple[int, int],
    use_geometric_prompt: bool = False,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Return phrase-grounded instances spatially contained by the MLLM box."""

    processor.reset_all_prompts(state)
    output = processor.set_text_prompt(prompt=ref, state=state)
    if use_geometric_prompt:
        output = processor.add_geometric_prompt(_norm_cxcywh(box, shape), True, state=state)
    if "masks" not in output or int(output["masks"].shape[0]) == 0:
        return None, {
            "candidate_count": 0,
            "reason": "PCS_EMPTY",
            "pcs_query_mode": "text+box" if use_geometric_prompt else "text",
        }
    selected = []
    containment_box = expand_box(box, shape, CFG.pcs_containment_expand_frac)
    for index in range(int(output["masks"].shape[0])):
        mask = output["masks"][index, 0].detach().cpu().numpy().astype(np.uint8)
        if mask.shape != shape:
            mask = resize_mask(mask, shape)
        predicted_box = clip_box(
            output["boxes"][index].detach().cpu().numpy().astype(np.float32), shape
        )
        mask_box = mask_to_box(mask)
        overlap = max(box_iou(predicted_box, box), box_iou(mask_box, box))
        inside_ratio = 1.0 - _outside_ratio(mask, containment_box, shape)
        if mask_box is not None and inside_ratio >= CFG.pcs_inside_ratio:
            selected.append(
                {
                    "mask": mask,
                    "box_iou": overlap,
                    "inside_ratio": inside_ratio,
                    "predicted_iou": float(output["scores"][index].item()),
                }
            )
    if not selected:
        return None, {
            "candidate_count": int(output["masks"].shape[0]),
            "reason": "PCS_INCONSISTENT",
            "pcs_query_mode": "text+box" if use_geometric_prompt else "text",
        }
    union = np.zeros(shape, dtype=np.uint8)
    for item in selected:
        union |= item["mask"]
    return union, {
        "candidate_count": int(output["masks"].shape[0]),
        "selected_count": len(selected),
        "box_iou": max(item["box_iou"] for item in selected),
        "inside_ratio": min(item["inside_ratio"] for item in selected),
        "predicted_iou": max(item["predicted_iou"] for item in selected),
        "pcs_query_mode": "text+box" if use_geometric_prompt else "text",
    }


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(((first > 0) & (second > 0)).sum())
    union = int(((first > 0) | (second > 0)).sum())
    return float(intersection / max(union, 1))


def _fuse_pcs_prompts(
    text_mask: Optional[np.ndarray],
    text_metadata: Dict,
    joint_mask: Optional[np.ndarray],
    joint_metadata: Dict,
    prompt_box: np.ndarray,
    region_mode: str,
    mask_density: str = "object",
) -> Tuple[Optional[np.ndarray], Dict]:
    """Fuse text-only and text+box PCS without letting a search box fill the mask.

    For aggregate regions the text query is good at retaining separate tiny
    semantic instances, while the joint query can recover instances missed by
    the global text proposal.  Their union is recall-friendly unless one query
    degenerates into a dense enclosing-object mask.  For a normal object, the
    joint prompt is more spatially specific and is therefore preferred.
    """

    audit = {
        "pcs_text_candidate_count": int(text_metadata.get("candidate_count", 0)),
        "pcs_joint_candidate_count": int(joint_metadata.get("candidate_count", 0)),
    }
    if text_mask is None and joint_mask is None:
        return None, {
            **audit,
            "reason": "PCS_EMPTY_BOTH_PROMPTS",
            "pcs_query_mode": "text|text+box",
            "pcs_fusion": "none",
        }
    if joint_mask is None:
        return text_mask, {
            **text_metadata,
            **audit,
            "pcs_query_mode": "text",
            "pcs_fusion": "text_only",
        }
    if text_mask is None:
        return joint_mask, {
            **joint_metadata,
            **audit,
            "pcs_query_mode": "text+box",
            "pcs_fusion": "joint_only",
        }

    box_area = max(_box_area(prompt_box), 1.0)
    text_fill = float(text_mask.sum()) / box_area
    joint_fill = float(joint_mask.sum()) / box_area
    area_small = max(1.0, float(min(text_mask.sum(), joint_mask.sum())))
    area_multiple = float(max(text_mask.sum(), joint_mask.sum())) / area_small
    pair_iou = _mask_iou(text_mask, joint_mask)

    if region_mode == "aggregate_region":
        if mask_density == "sparse":
            # A highly specific phrase can occasionally return one or two
            # low-confidence specks while the same phrase plus the regional
            # box recovers the actual flower/texture bed.  Treat that as a
            # failed sparse proposal, not evidence that the joint result is an
            # enclosing object.  The candidate-count and confidence gates keep
            # legitimate sparse groups (petals, piercings, stars) on text PCS.
            if min(text_fill, joint_fill) < CFG.pcs_tiny_fill_ratio and max(
                text_fill, joint_fill
            ) <= CFG.pcs_tiny_recovery_max_fill_ratio:
                if text_fill <= joint_fill:
                    tiny_metadata, dense_mask, dense_metadata, chosen = (
                        text_metadata,
                        joint_mask,
                        joint_metadata,
                        "text+box",
                    )
                else:
                    tiny_metadata, dense_mask, dense_metadata, chosen = (
                        joint_metadata,
                        text_mask,
                        text_metadata,
                        "text",
                    )
                tiny_selected = int(tiny_metadata.get("selected_count", 0))
                tiny_confidence = float(tiny_metadata.get("predicted_iou", 0.0))
                if (
                    tiny_selected <= CFG.pcs_tiny_recovery_max_selected
                    and tiny_confidence < CFG.pcs_tiny_recovery_max_confidence
                ):
                    return dense_mask, {
                        **dense_metadata,
                        **audit,
                        "pcs_query_mode": "text|text+box",
                        "pcs_fusion": f"recover_tiny_low_confidence_choose_{chosen}",
                        "pcs_pair_iou": pair_iou,
                        "pcs_text_fill": text_fill,
                        "pcs_joint_fill": joint_fill,
                    }
            overly_dense_disagreement = (
                area_multiple >= CFG.pcs_dense_area_multiple and pair_iou < 0.65
            ) or (
                max(text_fill, joint_fill) > 0.25
                and max(text_fill, joint_fill) > min(text_fill, joint_fill) * 1.2
            )
            if overly_dense_disagreement:
                if text_fill <= joint_fill:
                    mask, metadata, chosen = text_mask, text_metadata, "text"
                else:
                    mask, metadata, chosen = joint_mask, joint_metadata, "text+box"
                return mask, {
                    **metadata,
                    **audit,
                    "pcs_query_mode": "text|text+box",
                    "pcs_fusion": f"sparse_reject_dense_choose_{chosen}",
                    "pcs_pair_iou": pair_iou,
                    "pcs_text_fill": text_fill,
                    "pcs_joint_fill": joint_fill,
                }
        one_dense_one_sparse = (
            max(text_fill, joint_fill) >= CFG.pcs_dense_fill_ratio
            and min(text_fill, joint_fill) <= CFG.pcs_sparse_fill_ratio
            and area_multiple >= CFG.pcs_dense_area_multiple
        )
        if one_dense_one_sparse:
            if text_fill <= joint_fill:
                mask, metadata, chosen = text_mask, text_metadata, "text"
            else:
                mask, metadata, chosen = joint_mask, joint_metadata, "text+box"
            fusion = f"reject_dense_choose_{chosen}"
        else:
            mask = (text_mask | joint_mask).astype(np.uint8)
            metadata = {
                "candidate_count": int(text_metadata.get("candidate_count", 0))
                + int(joint_metadata.get("candidate_count", 0)),
                "selected_count": int(text_metadata.get("selected_count", 0))
                + int(joint_metadata.get("selected_count", 0)),
                "box_iou": max(
                    float(text_metadata.get("box_iou", 0.0)),
                    float(joint_metadata.get("box_iou", 0.0)),
                ),
                "inside_ratio": min(
                    float(text_metadata.get("inside_ratio", 1.0)),
                    float(joint_metadata.get("inside_ratio", 1.0)),
                ),
                "predicted_iou": max(
                    float(text_metadata.get("predicted_iou", 0.0)),
                    float(joint_metadata.get("predicted_iou", 0.0)),
                ),
            }
            fusion = "aggregate_union"
        return mask, {
            **metadata,
            **audit,
            "pcs_query_mode": "text|text+box",
            "pcs_fusion": fusion,
            "pcs_pair_iou": pair_iou,
            "pcs_text_fill": text_fill,
            "pcs_joint_fill": joint_fill,
        }

    if area_multiple >= 2.0 and pair_iou < 0.55:
        if text_fill <= joint_fill:
            object_mask, object_metadata, chosen = text_mask, text_metadata, "text"
        else:
            object_mask, object_metadata, chosen = joint_mask, joint_metadata, "text+box"
        return object_mask, {
            **object_metadata,
            **audit,
            "pcs_query_mode": "text|text+box",
            "pcs_fusion": f"object_reject_dense_choose_{chosen}",
            "pcs_pair_iou": pair_iou,
            "pcs_text_fill": text_fill,
            "pcs_joint_fill": joint_fill,
        }

    return joint_mask, {
        **joint_metadata,
        **audit,
        "pcs_query_mode": "text+box",
        "pcs_fusion": "joint_preferred_for_object",
        "pcs_pair_iou": pair_iou,
        "pcs_text_fill": text_fill,
        "pcs_joint_fill": joint_fill,
    }


def _semantic_detail_target(ref: str) -> bool:
    return bool(_SEMANTIC_DETAIL_RE.search(str(ref)))


def _region_mode(item: Dict) -> str:
    raw_mode = str(item.get("region_mode", "")).strip().lower()
    if raw_mode in {"aggregate", "aggregate_region", "nearby_group", "group"}:
        return "aggregate_region"
    if raw_mode in {"object", "single", "single_object"}:
        return "object"
    # Robust fallback for a schema-omitting model response.  Keep this narrower
    # than _SEMANTIC_DETAIL_RE so singular hands/arms remain normal objects.
    return "aggregate_region" if _AGGREGATE_REF_RE.search(str(item.get("ref", ""))) else "object"


def _mask_density(item: Dict, region_mode: str) -> str:
    raw_density = str(item.get("mask_density", "")).strip().lower()
    if raw_density in {"sparse", "dense", "object"}:
        return raw_density
    if region_mode == "object":
        return "object"
    return "sparse" if _SPARSE_REF_RE.search(str(item.get("ref", ""))) else "dense"


def _sam_text_prompt(ref: str) -> str:
    """Normalize verbose MLLM labels without reducing them to a head noun."""

    prompt = re.sub(
        r"^all relevant instances of\s+", "", str(ref).strip(), flags=re.IGNORECASE
    )
    # Foliage is useful to the MLLM as a layout cue, but asking SAM for
    # "flowers and foliage" often selects an enclosing lawn or a large pampas
    # plume.  The regional box already supplies the layout; retain the concrete
    # flower category as the semantic prompt.
    if re.search(r"\b(?:flowers?|wildflowers?|blooms?|roses?)\b", prompt, re.I):
        prompt = re.sub(
            r"\s+(?:and|with)\s+(?:(?:dried|green|mixed|dense)\s+){0,2}"
            r"foliage\b.*$",
            "",
            prompt,
            flags=re.IGNORECASE,
        ).strip()
    words = prompt.split()
    if len(words) > 8:
        body_part = re.search(
            r"\b(?:his|her|their)\s+((?:right|left)\s+)?(hand|arm)\b(.*)$",
            prompt,
            flags=re.IGNORECASE,
        )
        if body_part:
            side = body_part.group(1) or ""
            prompt = f"{side}{body_part.group(2)}{body_part.group(3)}".strip()
    return prompt


def _color_surface_sam_prompt(ref: str, prompt: str) -> str:
    """Make human skin edits semantic without asking SAM for a whole person.

    MLLM boxes identify the specific subject.  Inside that spatial region,
    phrases such as ``man's arms`` can still resolve to a full person or shirt.
    For color/material edits on explicit human anatomy, retain only the visible
    anatomical categories and mark them as exposed skin.  Non-human and
    clothing/object references are deliberately untouched.
    """

    if not _HUMAN_SUBJECT_RE.search(str(ref)):
        return prompt
    parts = []
    for match in _HUMAN_SKIN_PART_RE.finditer(str(ref)):
        part = match.group(0).lower()
        if part in {"foot", "feet"}:
            part = "feet"
        elif part.endswith("s"):
            part = part
        elif part in {"arm", "hand", "leg"}:
            part = f"{part}s"
        if part not in parts:
            parts.append(part)
    if not parts:
        return prompt
    return f"exposed human {' and '.join(parts)} skin"


def _prefer_pcs_candidate(
    ref: str,
    prompt_box: np.ndarray,
    pvs_mask: Optional[np.ndarray],
    pcs_mask: Optional[np.ndarray],
    region_mode: str = "object",
) -> Tuple[bool, str]:
    if pcs_mask is None:
        return False, "PCS_UNAVAILABLE"
    if pvs_mask is None:
        return True, "PVS_UNAVAILABLE"
    box_area = max(_box_area(prompt_box), 1.0)
    pvs_fill = float(pvs_mask.sum()) / box_area
    pcs_fill = float(pcs_mask.sum()) / box_area
    if region_mode == "aggregate_region":
        return True, "AGGREGATE_REGION"
    if _semantic_detail_target(ref):
        return True, "SEMANTIC_DETAIL"
    if pvs_fill < CFG.pvs_sparse_fill_ratio and pcs_fill > pvs_fill / CFG.semantic_area_ratio:
        return True, "PVS_SPARSE"
    if pvs_fill > CFG.pvs_dense_fill_ratio and pcs_fill < pvs_fill * CFG.semantic_area_ratio:
        return True, "PVS_DENSE"
    return False, "PVS_PRIMARY"


def segment_grounded_box(
    processor,
    state: Dict,
    ref: str,
    box_2d: Sequence[float],
    shape: Tuple[int, int],
    edit_type: Optional[str] = None,
    region_mode: str = "object",
    mask_density: str = "object",
) -> Tuple[np.ndarray, Dict]:
    region_mode = "aggregate_region" if region_mode == "aggregate_region" else "object"
    mask_density = mask_density if mask_density in {"sparse", "dense", "object"} else "object"
    raw_box = normalized_box_to_pixels(box_2d, shape)
    prompt_box = expand_box(
        raw_box,
        shape,
        CFG.box_expand_frac,
        CFG.min_box_expand_image_frac,
        CFG.min_box_expand_max_dimension_frac,
    )
    errors: List[str] = []
    sam_prompt = _sam_text_prompt(ref)
    if edit_type == "color":
        sam_prompt = _color_surface_sam_prompt(ref, sam_prompt)
    pvs_mask = None
    pvs_metadata: Dict = {}
    try:
        pvs_mask, pvs_metadata = _pvs_mask(processor, state, prompt_box, shape)
    except Exception as exc:
        errors.append(f"pvs:{exc!r}")

    text_pcs_mask = None
    text_pcs_metadata: Dict = {}
    try:
        text_pcs_mask, text_pcs_metadata = _pcs_mask(
            processor,
            state,
            sam_prompt,
            prompt_box,
            shape,
            use_geometric_prompt=False,
        )
    except Exception as exc:
        errors.append(f"pcs_text:{exc!r}")
    joint_pcs_mask = None
    joint_pcs_metadata: Dict = {}
    try:
        joint_pcs_mask, joint_pcs_metadata = _pcs_mask(
            processor,
            state,
            sam_prompt,
            prompt_box,
            shape,
            use_geometric_prompt=True,
        )
    except Exception as exc:
        errors.append(f"pcs_text_box:{exc!r}")
    pcs_mask, pcs_metadata = _fuse_pcs_prompts(
        text_pcs_mask,
        text_pcs_metadata,
        joint_pcs_mask,
        joint_pcs_metadata,
        prompt_box,
        region_mode,
        mask_density,
    )

    use_pcs, selection_reason = _prefer_pcs_candidate(
        ref, prompt_box, pvs_mask, pcs_mask, region_mode
    )
    if use_pcs:
        source = "pcs"
        mask, metadata = pcs_mask, pcs_metadata
    else:
        source = "pvs"
        mask, metadata = pvs_mask, pvs_metadata
    if mask is None and pcs_mask is not None:
        source = "pcs"
        mask, metadata = pcs_mask, pcs_metadata
        selection_reason = "PVS_UNAVAILABLE"
    if mask is None:
        source = "box"
        mask = mask_from_box(prompt_box, shape)
        metadata = {"reason": "BOX_FALLBACK"}
        selection_reason = "NO_SAM_CANDIDATE"

    # The adaptive box expansion is only a SAM prompt aid.  Coverage is audited
    # against the original MLLM box; requiring the expanded margin itself caused
    # otherwise correct limb masks to become large rectangles.
    x_cover, y_cover = _directional_coverage(mask, raw_box)
    misses_coverage = (
        x_cover < CFG.min_directional_box_coverage
        or y_cover < CFG.min_directional_box_coverage
    )
    # MLLM boxes are conservative SAM search regions.  A valid semantic mask is
    # not expected to touch every side of that box, especially for a scattered
    # aggregate.  Keep directional coverage as an audit signal; never turn a
    # successful SAM mask into a filled rectangle merely to span the search box.
    sparse_semantic = source == "pcs" and (
        region_mode == "aggregate_region" or _semantic_detail_target(ref)
    )
    coverage_suppressed = bool(misses_coverage and source != "box")
    coverage_union = False
    semantic_source = source
    metadata.update(
        {
            "mask_source": source,
            "semantic_mask_source": semantic_source,
            "bbox_xyxy": [round(float(value), 2) for value in prompt_box],
            "directional_coverage": [round(x_cover, 4), round(y_cover, 4)],
            "coverage_box_union": coverage_union,
            "coverage_suppressed_for_sparse_semantics": coverage_suppressed,
            "semantic_detail_target": _semantic_detail_target(ref),
            "region_mode": region_mode,
            "mask_density": mask_density,
            "selection_reason": selection_reason,
            "sam_prompt": sam_prompt,
            "errors": errors,
        }
    )
    return mask.astype(np.uint8), metadata


def aspect_ratio_delta(source_size: Tuple[int, int], target_size: Tuple[int, int]) -> float:
    source_width, source_height = source_size
    target_width, target_height = target_size
    source_ar = source_width / max(source_height, 1)
    target_ar = target_width / max(target_height, 1)
    return float(abs(target_ar / max(source_ar, 1e-8) - 1.0))


def map_target_mask_to_source(
    mask: np.ndarray,
    source_shape: Tuple[int, int],
    ar_mismatch: bool,
    dilate_frac: Optional[float] = None,
) -> np.ndarray:
    mapped = resize_mask(mask, source_shape)
    fraction = (
        CFG.target_map_dilate_frac if dilate_frac is None else float(dilate_frac)
    ) + (CFG.ar_mismatch_extra_dilate_frac if ar_mismatch else 0.0)
    return dilate_mask(mapped, max(1, round(min(source_shape) * fraction)))


def encode_rle(mask: np.ndarray) -> Dict:
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(value) for value in encoded["size"]], "counts": str(counts)}


def _union(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        result |= (mask > 0).astype(np.uint8)
    return result


def _aggregate_semantic_connected_coverage(
    mask: np.ndarray,
    metadata: Dict,
    shape: Tuple[int, int],
    grounding_image: str,
    item_index: int,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """Turn a nearby group of tiny semantic islands into one filled region.

    The MLLM aggregate bbox is only a search region, so filling its rectangle
    would absorb unrelated content.  Conversely, keeping dozens of tiny PCS
    masks independent is fragile and violates the desired region-level label.
    For a compact bbox with strong many-instance PCS evidence, use the convex
    envelope of the *observed semantic masks*.  This guarantees one connected
    region without consuming unused rectangle corners.  Large/global groups
    remain semantic instance unions, avoiding whole-image hulls for petals or
    flowers distributed around a subject.
    """

    if metadata.get("region_mode") != "aggregate_region":
        return None
    if not _CONNECTABLE_SMALL_GROUP_RE.search(str(metadata.get("ref", ""))):
        return None
    if metadata.get("semantic_mask_source") != "pcs":
        return None
    selected_count = int(metadata.get("selected_count", 0) or 0)
    if selected_count < CFG.connected_region_min_selected_instances:
        return None
    bbox_2d = metadata.get("bbox_2d")
    if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox_2d]
    bbox_area_frac = max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1_000_000.0
    if bbox_area_frac > CFG.connected_region_max_bbox_area_frac:
        return None

    semantic = (mask > 0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        semantic, connectivity=8
    )
    foreground_count = component_count - 1
    if foreground_count < CFG.connected_region_min_components:
        return None
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    median_area_frac = float(np.median(component_areas) / max(semantic.size, 1))
    if median_area_frac > CFG.connected_region_max_median_component_area_frac:
        return None
    largest_component_ratio = float(
        component_areas.max() / max(component_areas.sum(), 1.0)
    )
    if largest_component_ratio > CFG.connected_region_max_largest_component_ratio:
        return None

    points = cv2.findNonZero(semantic)
    if points is None or len(points) < 3:
        return None
    hull = cv2.convexHull(points)
    coverage = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(coverage, hull, color=1, lineType=cv2.LINE_8)
    radius = max(1, round(min(shape) * CFG.connected_region_hull_dilate_frac))
    coverage = dilate_mask(coverage, radius)
    coverage_box = mask_to_box(coverage)
    if coverage_box is None:
        return None
    normalized_box = normalized_box_to_pixels(bbox_2d, shape)
    directional = _directional_coverage(coverage, normalized_box)
    return coverage, {
        "instance_id": f"{grounding_image}_{item_index}_connected_region",
        "role": "edit_region_coverage",
        "grounding_image": (
            "source" if grounding_image == "source_foreground" else grounding_image
        ),
        "ref": f"connected region for {metadata.get('ref', 'semantic group')}",
        "bbox_2d": [float(value) for value in bbox_2d],
        "bbox_xyxy": [round(float(value), 2) for value in coverage_box],
        "mask_source": "group",
        "semantic_mask_source": "group",
        "predicted_iou": math.nan,
        "box_iou": math.nan,
        "inside_ratio": 1.0,
        "candidate_count": foreground_count,
        "selected_count": selected_count,
        "pcs_query_mode": "",
        "selection_reason": "AGGREGATE_SEMANTIC_CONVEX_HULL",
        "sam_prompt": "",
        "region_mode": "aggregate_region",
        "mask_density": str(metadata.get("mask_density", "sparse")),
        "semantic_detail_target": True,
        "mapped_from_target": False,
        "coverage_box_union": False,
        "coverage_suppressed_for_sparse_semantics": False,
        "directional_coverage": [round(value, 4) for value in directional],
        "errors": [],
    }


def _segment_items(
    processor,
    state: Dict,
    items: Sequence[Dict],
    shape: Tuple[int, int],
    grounding_image: str,
    edit_type: str,
) -> Tuple[List[np.ndarray], List[Dict]]:
    masks, metadata_rows = [], []
    for index, item in enumerate(items):
        region_mode = _region_mode(item)
        mask_density = _mask_density(item, region_mode)
        mask, metadata = segment_grounded_box(
            processor,
            state,
            str(item["ref"]),
            item["bbox_2d"],
            shape,
            edit_type=edit_type,
            region_mode=region_mode,
            mask_density=mask_density,
        )
        metadata_row = {
            "instance_id": f"{grounding_image}_{index}",
            "role": "preserve_foreground" if grounding_image == "source_foreground" else "edit_region",
            "grounding_image": "source" if grounding_image == "source_foreground" else grounding_image,
            "ref": str(item["ref"]),
            "bbox_2d": [float(value) for value in item["bbox_2d"]],
            "region_mode": region_mode,
            "mask_density": mask_density,
            **metadata,
        }
        masks.append(mask)
        metadata_rows.append(metadata_row)
        if edit_type != "background":
            connected = _aggregate_semantic_connected_coverage(
                mask, metadata_row, shape, grounding_image, index
            )
            if connected is not None:
                connected_mask, connected_metadata = connected
                masks.append(connected_mask)
                metadata_rows.append(connected_metadata)
    return masks, metadata_rows


def annotate_grounded_sample(processor, sample: Dict, ground_row: Dict, sam_version: str) -> Dict:
    source = sample["input_img"].convert("RGB")
    target = sample["output_img"].convert("RGB")
    source_shape = (source.height, source.width)
    target_shape = (target.height, target.width)
    etype = canonicalize_type(sample["type"])
    payload = json.loads(ground_row["ground_json"])
    boxes = payload.get("boxes", {})
    ar_delta = aspect_ratio_delta(source.size, target.size)
    ar_mismatch = ar_delta > CFG.ar_mismatch_threshold

    if ground_row.get("qc_flag") == "GROUND_FAIL":
        return {
            "mask": np.zeros(source_shape, dtype=np.uint8),
            "instances": [],
            "mask_source": "box",
            "qc_flag": "GROUND_FAIL",
            "qc_flags": ["GROUND_FAIL"],
            "ar_delta": ar_delta,
            "sam_version": sam_version,
        }

    if etype == "style":
        mask = np.ones(source_shape, dtype=np.uint8)
        return {
            "mask": mask,
            "instances": [],
            "mask_source": "box",
            "qc_flag": "OK",
            "qc_flags": ["STYLE_FULL_IMAGE"],
            "ar_delta": ar_delta,
            "sam_version": sam_version,
        }

    image_states: Dict[str, Dict] = {}
    if boxes.get("source"):
        image_states["source"] = processor.set_image(source)
    if boxes.get("target"):
        image_states["target"] = processor.set_image(target)

    all_masks: List[np.ndarray] = []
    all_instances: List[Dict] = []
    if boxes.get("source"):
        role = "source_foreground" if etype == "background" else "source"
        source_masks, source_metadata = _segment_items(
            processor,
            image_states["source"],
            boxes["source"],
            source_shape,
            role,
            etype,
        )
        all_masks.extend(source_masks)
        all_instances.extend(source_metadata)

    if boxes.get("target"):
        target_masks, target_metadata = _segment_items(
            processor,
            image_states["target"],
            boxes["target"],
            target_shape,
            "target",
            etype,
        )
        for mask, metadata in zip(target_masks, target_metadata):
            connected_region = (
                metadata.get("selection_reason")
                == "AGGREGATE_SEMANTIC_CONVEX_HULL"
            )
            mapped = map_target_mask_to_source(
                mask,
                source_shape,
                ar_mismatch,
                dilate_frac=(
                    CFG.target_map_connected_region_dilate_frac
                    if connected_region
                    else None
                ),
            )
            metadata["mapped_from_target"] = True
            mapped_box = mask_to_box(mapped)
            if mapped_box is None:
                mapped_box = np.zeros(4, dtype=np.float32)
            metadata["bbox_xyxy"] = [round(float(value), 2) for value in mapped_box]
            all_masks.append(mapped)
            all_instances.append(metadata)

    if etype == "background":
        foreground = _union(all_masks, source_shape)
        radius = max(1, round(min(source_shape) * CFG.background_foreground_dilate_frac))
        mask = (1 - dilate_mask(foreground, radius)).astype(np.uint8)
    else:
        mask = _union(all_masks, source_shape)

    source_rank = {"pvs": 0, "pcs": 1, "group": 2, "box": 3}
    mask_source = max(
        (instance["mask_source"] for instance in all_instances),
        key=lambda value: source_rank[value],
        default="box",
    )
    flags = []
    if mask_source == "box":
        flags.append("BOX_FALLBACK")
    if any(
        instance.get("selection_reason") == "AGGREGATE_SEMANTIC_CONVEX_HULL"
        for instance in all_instances
    ):
        flags.append("CONNECTED_REGION_COVERAGE")
    if ar_mismatch:
        flags.append("AR_MISMATCH")
    qc_flag = "BOX_FALLBACK" if "BOX_FALLBACK" in flags else ("AR_MISMATCH" if ar_mismatch else "OK")

    for mask_item, instance in zip(all_masks, all_instances):
        rle = encode_rle(mask_item)
        instance["rle_size"] = rle["size"]
        instance["rle_counts"] = rle["counts"]
        instance["area"] = int(mask_item.sum())
        instance["predicted_iou"] = float(instance.get("predicted_iou", math.nan))
        instance["box_iou"] = float(instance.get("box_iou", math.nan))
        instance["mapped_from_target"] = bool(instance.get("mapped_from_target", False))

    return {
        "mask": mask,
        "instances": all_instances,
        "mask_source": mask_source,
        "qc_flag": qc_flag,
        "qc_flags": flags or ["OK"],
        "ar_delta": ar_delta,
        "sam_version": sam_version,
    }
