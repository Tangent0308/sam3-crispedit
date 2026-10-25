"""Opt-in, source-only background selector repair.

Two independently cropped SAM observations must agree. This module never
infers permission from an edited image, fills holes blindly, or changes a
hidden execution mask without updating the public selector.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from .geometry import load_mask, mask_geometry
from .production_state import atomic_json, file_hash

VERSION = "background-selector-consensus-1"
BACKGROUND_QUERIES = {"water", "sky", "grass", "sand", "snow", "road"}


def candidate_crop_pairs(source, selected, *, max_pairs=8):
    """Find bounded source-only ambiguities; no manual ROI or target image."""
    from PIL import ImageFilter
    from scipy.ndimage import find_objects
    selected = np.asarray(selected, dtype=bool)
    if not selected.any() or selected.mean() < .1:
        return []
    rgb = np.asarray(source.convert("RGB").filter(ImageFilter.GaussianBlur(1)), dtype=np.int16)
    if rgb.shape[:2] != selected.shape:
        raise ValueError("source dimensions differ")
    distance, nearest = distance_transform_edt(~selected, return_indices=True)
    possible = (~selected & (distance <= max(8, round(max(selected.shape)*.04)))
                & (np.max(np.abs(rgb-rgb[nearest[0], nearest[1]]), axis=2) <= 32))
    possible[:3] = False; possible[-3:] = False
    possible[:, :3] = False; possible[:, -3:] = False
    labels, _ = label(possible)
    candidates = {}
    for component, slices in enumerate(find_objects(labels), 1):
        if slices is None:
            continue
        y, x = slices
        if min(y.stop-y.start, x.stop-x.start) < 4:
            continue
        for top in range(y.start, y.stop, 320):
            for left in range(x.start, x.stop, 320):
                patch = labels[top:min(top+384,y.stop), left:min(left+384,x.stop)] == component
                if patch.sum() < 64:
                    continue
                yy, xx = np.where(patch)
                # Enough scene context to distinguish a surface from its host.
                cx, cy = int(left+np.median(xx)), int(top+np.median(yy))
                boxes = []
                for radius in (160,208):
                    boxes.append([max(0,cx-radius),max(0,cy-radius),
                                  min(source.width,cx+radius),min(source.height,cy+radius)])
                if boxes[0] == boxes[1]:
                    continue
                candidates[tuple(boxes[0])] = (int(patch.sum()), boxes)
    return [v[1] for v in sorted(candidates.values(), key=lambda v:(-v[0],v[1]))[:max_pairs]]


def propose(selected, observations, *, source, protected=None, max_growth=.05):
    """Observations: pairs of full-resolution masks with score and recall.

    Every pair must come from distinct crop boxes of the SAME source/query.
    Missing or conflicting evidence means abstention, not rejection of data.
    """
    selected = np.asarray(selected, dtype=bool)
    protected = np.zeros_like(selected) if protected is None else np.asarray(protected, dtype=bool)
    if protected.shape != selected.shape:
        raise ValueError("protected mask dimensions differ")
    added = np.zeros_like(selected)
    accepted_pairs = 0
    if selected.mean() < .1:
        return selected.copy(), {"status": "abstain_small_selector", "added_pixels": 0}
    for pair in observations:
        if len(pair) != 2 or pair[0]["box"] == pair[1]["box"]:
            raise ValueError("two distinct crop observations required")
        if any(x["score"] < .85 or x["recall"] < .9 for x in pair):
            continue
        masks = [np.asarray(x["mask"], dtype=bool) for x in pair]
        if any(x.shape != selected.shape for x in masks):
            raise ValueError("proposal dimensions differ")
        # Avoid inventing membership on a thin contour or antialiased railing.
        consensus = binary_erosion(masks[0] & masks[1], iterations=1)
        added |= consensus & ~selected & ~protected
        accepted_pairs += 1
    distance, nearest = distance_transform_edt(~selected, return_indices=True)
    added &= distance <= max(8, round(max(selected.shape)*.04))
    # Semantic agreement alone can bleed across a narrow hull/rail boundary.
    # Require local source appearance support as well; never use target colors.
    from PIL import ImageFilter
    rgb = np.asarray(source.convert("RGB").filter(ImageFilter.GaussianBlur(1)), dtype=np.int16)
    if rgb.shape[:2] != selected.shape:
        raise ValueError("source dimensions differ")
    adjacent = rgb[nearest[0], nearest[1]]
    added &= np.max(np.abs(rgb-adjacent), axis=2) <= 32
    labels, _ = label(added)
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero(counts >= 16)
    keep = keep[keep != 0]
    added &= np.isin(labels, keep)
    pixels = int(added.sum())
    growth = pixels/max(1, int(selected.sum()))
    metadata = dict(added_pixels=pixels, growth_fraction=growth, accepted_pairs=accepted_pairs)
    if growth > max_growth:
        return selected.copy(), dict(metadata, status="abstain_excess_growth")
    if not pixels:
        return selected.copy(), dict(metadata, status="abstain_no_consensus")
    return selected | added, dict(metadata, status="candidate")


def apply_candidate(group, uid, proposal, directory, *, query, evidence):
    """Return a new pre-generation record with a synchronized public selector.

    Caller must validate source-only semantic membership before adoption.
    Existing generated records and resolved add sites cannot be rewritten.
    """
    if query not in BACKGROUND_QUERIES:
        raise ValueError("unsupported background query")
    result = deepcopy(group)
    key = "units" if "units" in result else "members"
    member = next(m for m in result[key] if m["unit_id"] == uid)
    attributes = member["instruction_spec"].get("attributes", [])
    if member["operation"] != "attribute" or not attributes or any(
            a["property"] != "color" for a in attributes):
        raise ValueError("only background color edits are eligible")
    if member.get("edited_image") or member.get("step_output_image") or member.get("mask_v1"):
        raise ValueError("only unresolved pre-generation records can be refined")
    source = Image.open(result["source_image"])
    old = load_mask({"mask_path": member["mask"]}, source.size)
    proposal = np.asarray(proposal, dtype=bool)
    if proposal.shape != old.shape or np.any(old & ~proposal):
        raise ValueError("repair must preserve all original selected pixels")
    if int((proposal & ~old).sum()) > .05*int(old.sum()):
        raise ValueError("repair exceeds growth limit")
    protected = np.zeros_like(old)
    for other in result[key]:
        if other["unit_id"] != uid:
            protected |= load_mask({"mask_path": other["mask"]}, source.size)
    for path in result.get("protected_source_masks", []):
        protected |= load_mask({"mask_path": path}, source.size)
    if np.any(proposal & ~old & protected):
        raise ValueError("repair overlaps another protected selector")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/(uid+"_source_mask.png")
    if path.exists():
        if not np.array_equal(np.asarray(Image.open(path)) > 0, proposal):
            raise ValueError("existing refined mask differs")
    else:
        Image.fromarray(proposal.astype("uint8")*255).save(path)
    geometry = mask_geometry(proposal)
    record = dict(version=VERSION, query=query, original_mask=member["mask"],
                  original_sha256=file_hash(member["mask"]), mask=str(path),
                  mask_sha256=file_hash(path), source_sha256=file_hash(result["source_image"]),
                  added_pixels=int((proposal & ~old).sum()), evidence=evidence)
    member.update(mask=str(path), bbox=geometry["bbox"], point=geometry["point"],
                  area_fraction=float(proposal.mean()), support_mask=None, mask_refinement=record)
    unit = member["unit"]
    unit.update(mask_path=str(path), bbox=geometry["bbox"], point=geometry["point"],
                execution_mask_path=None, protected_mask_path=None,
                preflight_input={"execution_mask_path": None, "protected_mask_path": None})
    for name in ("source_grounding_evidence", "grounding_evidence"):
        member.pop(name, None)
    result.pop("dependency_schedule", None)
    atomic_json(directory/(uid+"_refinement.json"), record)
    return result
