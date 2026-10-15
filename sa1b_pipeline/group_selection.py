"""Deterministic, quota-aware selection of grouped source masks."""
from __future__ import annotations

import hashlib
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import re
from typing import Any, Iterable

from PIL import Image

from .geometry import load_mask, operation_quality_gate
from .group_geometry import (adjacent_pair_gate, category_allowed, object_features,
                             twin_pair_gate, normalized_category, support_mask, bbox_gap,
                             gap_pixels, candidate_granularity, candidate_category)
from .schema import OPERATIONS


_FEATURE_CACHE: dict[tuple[str, str, tuple[int, int]], dict[str, Any]] = {}
_PAIR_CACHE: dict[tuple[str, str, str, tuple[int, int]], dict[str, Any]] = {}
_IMAGE_SIZE_CACHE: dict[str, tuple[int, int]] = {}
_HOST_CACHE: dict[str, list[dict[str, Any]]] = {}

# Components that can naturally be edited as a scope in benchmark-style
# instructions.  Matching is token-aware so ``arm`` does not accidentally
# classify ``armchair`` as a body part.
_PART_HINTS = {
    "sleeve", "handle", "wheel", "tire", "arm", "leg", "foot", "hand",
    "finger", "strap", "belt", "frame", "window frame", "door handle",
    "cross", "beam", "shutter", "window", "door", "roof", "railing",
    "handrail", "latch", "hinge", "button", "pipe", "downspout", "spoke",
    "pedal", "handlebar", "armrest", "branch", "stem", "sign", "awning",
    "chimney", "gutter", "antenna", "mirror arm", "headlight", "taillight",
    "arch", "balcony", "bumper", "column", "pillar", "post", "pole", "fence",
    "stair", "step", "sill", "planter", "flower box", "panel", "screen",
    "grille", "grill", "lamp", "light", "flag", "gate", "gutter",
    "rail", "brace", "wire", "cable", "chain", "net", "board", "plank",
    "fender", "mirror", "lens", "socket", "switch", "drawer", "shelf",
    "faucet", "knob", "hook", "rope", "string", "cord",
}
_HOST_HINTS = {
    "person", "man", "woman", "child", "boy", "girl", "people", "human",
    "building", "house", "home", "door", "car", "truck", "bus", "vehicle",
    "bicycle", "bike", "motorcycle", "boat", "airplane", "train", "table",
    "chair", "sofa", "couch", "cabinet", "machine", "animal", "horse",
    "dog", "cat", "bird", "room", "vehicle",
}


def _is_part_category(value: Any) -> bool:
    text = str(value or "").lower().replace("_", " ")
    for hint in _PART_HINTS:
        if re.search(r"(?<![a-z])" + re.escape(hint) + r"(?![a-z])", text):
            return True
    return False


def _is_host_category(value: Any) -> bool:
    text = str(value or "").lower().replace("_", " ")
    return any(re.search(r"(?<![a-z])" + re.escape(hint) + r"(?![a-z])", text)
               for hint in _HOST_HINTS)


def _candidate_dict(candidate: Any) -> dict[str, Any]:
    return candidate.to_dict() if hasattr(candidate, "to_dict") else candidate


def _image_size(candidate: dict[str, Any]) -> tuple[int, int]:
    key = str(candidate.get("source_id") or candidate["image_path"])
    size = _IMAGE_SIZE_CACHE.get(key)
    if size is None:
        # Every dense SA-1B object stores the COCO RLE canvas as [height,
        # width].  Use it before opening the JPEG: the source cache may live
        # on a high-latency shared filesystem, and selection only needs the
        # canvas dimensions at this point.
        for obj in candidate.get("objects", []):
            rle_size = (obj.get("rle") or {}).get("size")
            if isinstance(rle_size, (list, tuple)) and len(rle_size) == 2:
                try:
                    height, width = int(rle_size[0]), int(rle_size[1])
                    if width > 0 and height > 0:
                        size = (width, height)
                        break
                except (TypeError, ValueError):
                    pass
    if size is None:
        with Image.open(candidate["image_path"]) as image:
            size = image.size
        _IMAGE_SIZE_CACHE[key] = size
    else:
        _IMAGE_SIZE_CACHE[key] = size
    return size


def _operation_allowed(obj: dict[str, Any], operation: str, size: tuple[int, int]) -> bool:
    try:
        key = (str(obj.get("mask_path") or obj.get("object_id")), str(obj.get("object_id")), size)
        feature = _FEATURE_CACHE.get(key)
        if feature is None:
            feature = object_features(obj, size)
            # Keep geometry only; the binary mask can be large and is not
            # needed for operation quality gates.
            feature = {k: v for k, v in feature.items() if k != "mask"}
            _FEATURE_CACHE[key] = feature
        return bool(operation_quality_gate(feature, operation)["eligible"])
    except (ValueError, TypeError, OSError, RuntimeError):
        return False


def _cheap_operation_allowed(obj: dict[str, Any], operation: str, size: tuple[int, int]) -> bool:
    """Conservative bbox-only prefilter; exact masks are checked on shortlisted pairs."""
    category = str(obj.get("category", "")).lower()
    if operation in {"replace", "remove"}:
        # Alias lists usually denote semantic parts rather than complete
        # editable objects. They need a trusted execution sub-mask, which
        # ordinary SA-1B candidates do not carry by default.
        if "," in category:
            return False
        if any(token in category for token in ("shadow", "reflection", "glare", "letter", "text", "logo", "word")):
            return False
    if operation == "remove":
        # Removing a visible limb/person part is not a natural local edit and
        # violates the benchmark's remove semantics even when the mask is
        # geometrically clean.
        if any(re.search(r"(?<![a-z])" + token + r"(?![a-z])", category)
               for token in ("arm", "hand", "leg", "foot", "finger", "head", "face", "person",
                             "cheek", "eye", "ear", "nose", "mouth", "lip", "neck", "torso")):
            return False
    if operation == "add":
        # These labels describe a visual effect, text/mark, or a tiny
        # wearable part rather than a stable host that can naturally carry a
        # new physical item. Keeping them out here avoids wasting a planner
        # call on an add instruction that cannot satisfy mask semantics.
        banned = ("shadow", "reflection", "glare", "letter", "text", "logo",
                  "word", "armband", "bracelet", "wristband", "finger")
        if any(token in category for token in banned):
            return False
    bbox = obj.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return True
    x0, y0, x1, y1 = [float(v) for v in bbox]
    # Dense labels use xyxy; tolerate xywh by falling back to positive extents.
    width, height = max(1.0, x1 - x0), max(1.0, y1 - y0)
    image_area = max(1.0, float(size[0] * size[1]))
    if operation in {"replace", "remove"} and (x0 <= 0 or y0 <= 0 or x1 >= size[0] or y1 >= size[1]):
        return False
    return width * height >= image_area * (0.00015 if operation == "add" else 0.00005)


def twin_operation_priority(objects: list[dict[str, Any]], operation: str) -> int:
    """Rank plausible edits within quota cells, without treating dense labels as ground truth."""
    preferred = {
        "add": {"table", "chair", "backpack", "bag", "hat", "pot", "planter", "cart", "shelf", "door", "screen"},
        "remove": {"bottle", "cup", "bag", "backpack", "bicycle", "chair", "umbrella", "vase", "sign", "lamp", "flowerpot"},
        "replace": {"chair", "stool", "bottle", "cup", "vase", "rim", "handle", "knob", "lamp", "pot", "hat", "shoe"},
        "attribute": {"shirt", "sleeve", "backpack", "bag", "chair", "door", "frame", "cap", "hat", "rim", "handle", "shutter"},
    }
    risky = {"strip", "band", "concrete", "wood", "beef", "wire", "wires", "strut", "struts"}
    scores = []
    for obj in objects:
        text = normalized_category(obj.get("category", ""))
        heads = {s.strip().split()[-1] for s in text.split(",") if s.strip()}
        score = 2 if heads & preferred.get(operation, set()) else 0
        if heads & risky:
            score -= 1
        if operation == "attribute" and heads & {"banner", "logo", "text", "sign"}:
            score -= 1
        scores.append(score)
    score = min(scores, default=0)
    if len(objects)>1 and len({normalized_category(o.get('category','')) for o in objects})>1:
        # Ranking only: shared physical function/support improves proposal
        # efficiency; the planner still verifies pixels and may reject.
        families = [
            {'chair','armchair','stool','bench','sofa','couch'},
            {'bottle','vase','pot','planter','urn','cup','bowl'},
            {'shirt','jacket','coat','dress','sweater','garment'},
            {'backpack','bag','sack','suitcase','package'},
            {'lamp','light','lantern'},
        ]
        if operation == 'add':
            families += [{'table','desk','shelf','cart','chair','armchair','bench'},
                         {'shirt','jacket','coat','backpack','bag','hat'}]
        words = [set(re.findall(r'[a-z]+', normalized_category(o.get('category','')))) for o in objects]
        if operation in {'add','replace'} and any(all(w & f for w in words) for f in families):
            score += 4
    return score


def _short_side(obj: dict[str, Any]) -> float:
    bbox = obj.get("bbox") or [0, 0, 1, 1]
    return max(1.0, min(abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1]))))


def _bbox_overlap_fraction(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Intersection over the right bbox area for cheap conflict screening."""
    try:
        lx0, ly0, lx1, ly1 = (float(v) for v in left.get("bbox", (0, 0, 0, 0)))
        rx0, ry0, rx1, ry1 = (float(v) for v in right.get("bbox", (0, 0, 0, 0)))
    except (TypeError, ValueError):
        return 1.0
    inter = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    area = max(1.0, (rx1 - rx0) * (ry1 - ry0))
    return inter / area


def _local_pair_view(a: dict[str, Any], b: dict[str, Any], ma, mb, short: float):
    """Crop aligned masks before the exact distance transform.

    SA-1B masks often live on a 4K canvas while adjacent objects occupy a
    small region. Running ``distance_transform_edt`` on the whole canvas for
    every pair causes avoidable memory growth; a margin around the union of
    the two bboxes preserves the nearest foreground pixels and the overlap
    test while keeping the temporary arrays bounded.
    """
    height, width = ma.shape
    boxes = [a.get("bbox") or [0, 0, width, height], b.get("bbox") or [0, 0, width, height]]
    try:
        x0 = min(float(box[0]) for box in boxes)
        y0 = min(float(box[1]) for box in boxes)
        x1 = max(float(box[2]) for box in boxes)
        y1 = max(float(box[3]) for box in boxes)
    except (TypeError, ValueError, IndexError):
        return ma, mb
    margin = max(8, min(256, int(round(0.2 * max(1.0, short))) + 8))
    left = max(0, int(x0) - margin)
    top = max(0, int(y0) - margin)
    right = min(width, int(x1) + margin + 1)
    bottom = min(height, int(y1) + margin + 1)
    return ma[top:bottom, left:right], mb[top:bottom, left:right]


def _near_bbox_pairs(objects: list[dict[str, Any]], *, limit: int = 8) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    """Return nearest bbox pairs without materialising all O(n²) combinations.

    Dense SA-1B scenes can contain hundreds of masks.  The old combinations()
    loop spent most of strict selection comparing obviously distant boxes,
    before the exact RLE gate had a chance to reject them.  A sweep sorted by
    x0 keeps the same conservative lower-bound test while only examining boxes
    whose horizontal gap could fit the largest admissible proximity window.
    """
    if len(objects) < 2:
        return []
    entries = []
    max_window = 32.0
    for obj in objects:
        bbox = obj.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        short = max(1.0, min(x1 - x0, y1 - y0))
        max_window = max(max_window, min(512.0, 0.2 * short))
        entries.append((x0, x1, y0, y1, obj))
    entries.sort(key=lambda item: (item[0], item[1], str(item[4].get("object_id", ""))))
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for i, (_, _, _, _, left) in enumerate(entries):
        left_box = left.get("bbox")
        if not isinstance(left_box, (list, tuple)) or len(left_box) != 4:
            continue
        try:
            lx1 = float(left_box[2])
        except (TypeError, ValueError):
            continue
        for x0, _, _, _, right in entries[i + 1:]:
            if x0 - lx1 > max_window:
                break
            gap = bbox_gap(left, right)
            threshold = 0.2 * max(1.0, min(_short_side(left), _short_side(right)))
            if gap <= threshold:
                candidates.append((gap, left, right))
    candidates.sort(key=lambda item: (item[0], str(item[1].get("object_id", "")),
                                      str(item[2].get("object_id", ""))))
    return candidates[:limit]


def _bbox_union(objects: Iterable[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    boxes = []
    for obj in objects:
        bbox = obj.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            boxes.append(tuple(float(v) for v in bbox))
        except (TypeError, ValueError):
            continue
    if not boxes:
        return None
    return (min(x[0] for x in boxes), min(x[1] for x in boxes),
            max(x[2] for x in boxes), max(x[3] for x in boxes))


def _bbox_area(obj: dict[str, Any]) -> float:
    bbox = obj.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError):
        return 0.0


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float], margin: float = 0.02) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    width, height = max(1.0, ox1 - ox0), max(1.0, oy1 - oy0)
    pad_x, pad_y = width * margin, height * margin
    return (ox0 - pad_x <= ix0 and oy0 - pad_y <= iy0
            and ox1 + pad_x >= ix1 and oy1 + pad_y >= iy1)


def _group_scope(a: dict[str, Any], b: dict[str, Any],
                 all_objects: Iterable[dict[str, Any]] | None = None) -> str:
    pa = a.get("parent_id") or (a.get("metadata") or {}).get("parent_id")
    pb = b.get("parent_id") or (b.get("metadata") or {}).get("parent_id")
    if pa and pb and str(pa) == str(pb):
        return "part_on_same_host"
    def is_part(obj: dict[str, Any]) -> bool:
        if (obj.get("metadata") or {}).get("is_part") is True:
            return True
        return candidate_granularity(obj) == "part"
    if is_part(a) or is_part(b):
        # Dense SA-1B labels do not carry parent IDs.  Recover a conservative
        # same-host signal when a third, non-part mask encloses both parts
        # (for example two sleeves inside one person or two handles inside a
        # door).  This is only a scope statistic/selection hint; the planner
        # still has to produce unambiguous refs and the group audit can reject
        # it later.
        union = _bbox_union((a, b))
        if union is not None:
            for host in all_objects or ():
                if host is a or host is b or is_part(host) or not _is_host_category(host.get("category", "")):
                    continue
                host_bbox = host.get("bbox")
                if not isinstance(host_bbox, (list, tuple)) or len(host_bbox) != 4:
                    continue
                try:
                    if _contains(tuple(float(v) for v in host_bbox), union):
                        return "part_on_same_host"
                except (TypeError, ValueError):
                    continue
        return "part_on_different_hosts"
    return "object"


def select_twin_candidates(candidates: Iterable[Any], *, target: int, operation: str,
                           seed: int = 0, max_groups_per_source: int = 2,
                           reserved_source_ids: set[str] | None = None,
                           strict_geometry: bool = False, member_count: int = 2,
                           category_relation: str = "any") -> list[dict[str, Any]]:
    """Select comparable-granularity pairs/triples, allowing different categories.

    This function is intentionally conservative: no operation fallback is
    performed after selection and no candidate is rejected for a textual
    ambiguity that only the planner can judge.
    """
    if category_relation not in {"same", "cross", "any"}:
        raise ValueError("unknown twin category relation")
    if operation not in OPERATIONS or target < 1 or member_count not in (2, 3):
        raise ValueError("invalid twin target or operation")
    reserved = {str(x) for x in (reserved_source_ids or set())}
    rng = random.Random(seed)
    rows = []
    for candidate in candidates:
        c = _candidate_dict(candidate)
        if str(c.get("source_id")) in reserved:
            continue
        try:
            size = _image_size(c)
        except (OSError, ValueError):
            continue
        groups = defaultdict(list)
        for obj in c.get("objects", []):
            ok, _ = category_allowed(obj.get("category", ""))
            if ok:
                key = (candidate_granularity(obj), candidate_category(obj) if category_relation == "same" else "all")
                groups[key].append(obj)
        # Same-host recovery only needs plausible enclosing masks. Keeping the
        # largest non-part boxes avoids scanning hundreds of tiny labels for
        # every part pair while retaining persons, vehicles, buildings, etc.
        source_key = str(c.get("source_id") or c.get("image_path"))
        host_objects = _HOST_CACHE.get(source_key)
        if host_objects is None:
            host_objects = sorted(
                (obj for obj in c.get("objects", []) if not _is_part_category(obj.get("category", ""))),
                key=_bbox_area, reverse=True,
            )[:32]
            _HOST_CACHE[source_key] = host_objects
        scope_cache: dict[tuple[str, str], str] = {}
        area_cache: dict[str, int] = {}

        def exact_area(obj):
            oid = str(obj["object_id"])
            if oid not in area_cache:
                rle = obj.get("rle")
                if rle and isinstance(rle.get("counts"), (str, bytes)):
                    # Compressed RLE area requires no full-resolution bitmap.
                    try:
                        from pycocotools import mask as coco_mask
                        area_cache[oid] = int(coco_mask.area(rle))
                    except ImportError:
                        area_cache[oid] = int(load_mask(obj, size).sum())
                else:
                    area_cache[oid] = int(load_mask(obj, size).sum())
            return area_cache[oid]

        attempted_total = 0
        for category in sorted(groups):
            values = [obj for obj in groups[category] if _cheap_operation_allowed(obj, operation, size)]
            if len(values) < 2:
                continue
            rng.shuffle(values)
            pair_attempts = 0
            for group_objects in combinations(values, member_count):
                categories = {candidate_category(obj) for obj in group_objects}
                relation = "same" if len(categories) == 1 else "cross"
                if category_relation != "any" and relation != category_relation:
                    continue
                pair_attempts += 1
                attempted_total += 1
                if pair_attempts > 2000 or attempted_total > 50000:
                    break
                # A twin group requires separated support regions. Dense
                # overlapping bbox candidates (for example rows of windows)
                # are almost never eligible and need not trigger RLE decode.
                if any(bbox_gap(a, b) < 8 for a, b in combinations(group_objects, 2)):
                    continue
                areas = [exact_area(obj) for obj in group_objects]
                if min(areas) <= 0 or max(areas) > 3 * min(areas):
                    continue
                gates = []
                for a, b in combinations(group_objects, 2):
                    key = (str(c["source_id"]), str(a.get("object_id")), str(b.get("object_id")), size)
                    if strict_geometry:
                        gate = _PAIR_CACHE.get(key)
                        if gate is None:
                            gate = twin_pair_gate(a, b, image_size=size)
                            _PAIR_CACHE[key] = gate
                    else:
                        gate = {"eligible": True, "geometry_exact": False,
                                "area_ratio_exact": max(exact_area(a), exact_area(b)) / min(exact_area(a), exact_area(b)),
                                "gap_pixels_lower_bound": bbox_gap(a, b),
                                "same_host": bool((a.get("parent_id") or (a.get("metadata") or {}).get("parent_id")) and
                                                   (a.get("parent_id") or (a.get("metadata") or {}).get("parent_id")) ==
                                                   (b.get("parent_id") or (b.get("metadata") or {}).get("parent_id")))}
                    gates.append(gate)
                if all(g["eligible"] for g in gates):
                    same_host = all(g.get("same_host") for g in gates)
                    scope_key = tuple(sorted(str(x.get("object_id")) for x in group_objects[:2]))
                    scope_level = scope_cache.get(scope_key)
                    if scope_level is None:
                        scope_level = "part_on_same_host" if same_host else _group_scope(
                            group_objects[0], group_objects[1], host_objects)
                        scope_cache[scope_key] = scope_level
                    rows.append({"candidate": c, "objects": list(group_objects),
                                 "gate": {"member_count": member_count, "pairs": gates,
                                          "eligible": True, "geometry_exact": strict_geometry},
                                 "scope_level": scope_level, "category_relation": relation,
                                 "candidate_granularity": candidate_granularity(group_objects[0]),
                                 "operation": operation})
                    if len(rows) >= target * 8:
                        break
            if len(rows) >= target * 8:
                break
            if attempted_total > 50000:
                break
    rng.shuffle(rows)
    rows.sort(key=lambda row: twin_operation_priority(row["objects"], operation), reverse=True)
    selected, per_source, used_objects = [], Counter(), set()
    for row in rows:
        sid = str(row["candidate"]["source_id"])
        key = (sid, tuple(sorted(str(o["object_id"]) for o in row["objects"])))
        if key in used_objects or per_source[sid] >= max_groups_per_source:
            continue
        used_objects.add(key)
        per_source[sid] += 1
        selected.append(row)
        if len(selected) >= target:
            break
    return selected


def _conflict(a: dict[str, Any], b: dict[str, Any], *, size: tuple[int, int]) -> bool:
    ma, mb = load_mask(a, size), load_mask(b, size)
    sa = support_mask(ma, "attribute", "target_only")
    sb = support_mask(mb, "attribute", "target_only")
    return bool((sa & mb).sum() > 0.10 * max(1, int(mb.sum())) or (sb & ma).sum() > 0.10 * max(1, int(ma.sum())))


def _target_sizes(total: int) -> dict[int, int]:
    # Largest remainder allocation for 20/30/25/25 percent.
    raw = {2: total * .20, 3: total * .30, 4: total * .25, 5: total * .25}
    result = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda x: -(raw[x] - result[x]))[:total - sum(result.values())]:
        result[k] += 1
    return result


def select_multi_candidates(candidates: Iterable[Any], *, target: int, seed: int = 0,
                            reserved_source_ids: set[str] | None = None,
                            max_groups_per_source: int = 2,
                            strict_geometry: bool = True) -> list[dict[str, Any]]:
    """Preselect 2-5 targets with at least one adjacent pair.

    The MLLM still decides semantic feasibility and exact refs; this stage only
    enforces geometry, quotas, object uniqueness and operation availability.
    """
    if target < 1:
        raise ValueError("target must be positive")
    reserved = {str(x) for x in (reserved_source_ids or set())}
    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = {n: [] for n in range(2, 6)}
    quotas = _target_sizes(target)
    # Keep a modest reserve for source-cap collisions, then stop scanning the
    # manifest once every target-count bucket has enough strict candidates.
    pool_caps = {n: max(quotas[n] * 2, quotas[n] + 8) for n in quotas}
    for candidate in candidates:
        c = _candidate_dict(candidate)
        if str(c.get("source_id")) in reserved:
            continue
        try:
            size = _image_size(c)
        except (OSError, ValueError):
            continue
        objects = [o for o in c.get("objects", []) if category_allowed(o.get("category", ""))[0]]
        if len(objects) < 2:
            continue
        # Shortlist near bbox pairs before decoding masks. Dense SA-1B scenes
        # can contain hundreds of masks; use a sweep instead of materialising
        # every O(n²) combination.
        near_pairs = _near_bbox_pairs(objects, limit=8)
        feature_cache: dict[str, dict[str, Any]] = {}
        mask_cache: dict[str, Any] = {}

        def object_key(obj: dict[str, Any]) -> str:
            return str(obj.get("object_id") or id(obj))

        def cached_mask(obj: dict[str, Any]):
            key = object_key(obj)
            mask = mask_cache.get(key)
            if mask is None:
                mask = load_mask(obj, size)
                mask_cache[key] = mask
            return mask
        support_cache: dict[str, Any] = {}

        def cached_feature(obj: dict[str, Any]) -> dict[str, Any]:
            key = object_key(obj)
            feature = feature_cache.get(key)
            if feature is None:
                feature = object_features(obj, size)
                feature_cache[key] = feature
            return feature

        def cached_support(obj: dict[str, Any]):
            key = object_key(obj)
            support = support_cache.get(key)
            if support is None:
                # target_only support is exactly the source mask; avoid a
                # second full-canvas copy for every adjacent pair.
                support = cached_mask(obj)
                support_cache[key] = support
            return support

        def cached_operation_allowed(obj: dict[str, Any], operation: str) -> bool:
            if not _cheap_operation_allowed(obj, operation, size):
                return False
            if not strict_geometry:
                # Fast capacity mode only estimates whether an operation is
                # plausible from category/bbox metadata. Exact mask quality is
                # enforced by the normal strict selection and group audit.
                return True
            # Operation quality is semantic rather than adjacency geometry.
            # The planner and 27B group audit are the authoritative gates; an
            # optional exact pre-gate is available for offline diagnostics but
            # is disabled in production selection because it decodes a dense
            # scene's mask once per operation and dominates wall time.
            if os.environ.get("SA1B_EXACT_MULTI_OPERATION_GATE", "0") != "1":
                return True
            feature = cached_feature(obj)
            quality = {k: v for k, v in feature.items() if k != "mask"}
            return bool(operation_quality_gate(quality, operation)["eligible"])

        pairs = []
        for _, a, b in near_pairs:
            try:
                short = min(_short_side(a), _short_side(b))
                if not strict_geometry:
                    # Fast capacity mode: the bbox lower bound is a safe
                    # candidate filter; the group editor/auditor performs the
                    # exact mask check before release.
                    gap = bbox_gap(a, b)
                    evidence = {
                        "eligible": gap <= 0.1 * max(1, short),
                        "gap_pixels": gap,
                        "threshold_pixels": 0.1 * max(1, short),
                        "support_a_over_source_b": None,
                        "geometry_exact": False,
                        "features": {"a": {"bbox": a.get("bbox")},
                                      "b": {"bbox": b.get("bbox")}},
                    }
                else:
                    ma, mb = cached_mask(a), cached_mask(b)
                    sa, sb = _local_pair_view(a, b, cached_support(a), cached_support(b), short)
                    gap = gap_pixels(sa, sb)
                    overlap = float((sa & sb).sum()) / max(1, int(sb.sum()))
                    evidence = {
                        "eligible": gap <= 0.1 * max(1, short) and overlap <= 0.10,
                        "gap_pixels": gap,
                        "threshold_pixels": 0.1 * max(1, short),
                        "support_a_over_source_b": overlap,
                        "geometry_exact": True,
                        "features": {
                            "a": {"bbox": a.get("bbox"), "area_px": int(ma.sum())},
                            "b": {"bbox": b.get("bbox"), "area_px": int(mb.sum())},
                        },
                    }
                if evidence["eligible"]:
                    pairs.append((a, b, evidence))
                    # A group needs one adjacent pair. Keep a few alternatives
                    # for operation feasibility without retaining every dense
                    # scene mask in memory.
                    if len(pairs) >= 2:
                        break
                else:
                    # Failed pairs are not reused. Release their decoded masks
                    # immediately; a few 4K SA-1B masks can otherwise retain
                    # hundreds of MB across a high-density candidate.
                    mask_cache.clear()
                    support_cache.clear()
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
        if not pairs:
            continue
        rng.shuffle(pairs)
        for n in range(2, 6):
            if len(objects) < n:
                continue
            a, b, adjacency = pairs[0]
            # Only nearby, operation-plausible objects are needed to fill the
            # remaining target slots. This avoids decoding hundreds of dense
            # scene masks merely to prove they conflict with the pair.
            plausible = [o for o in objects if o not in (a, b)
                         and any(_cheap_operation_allowed(o, op, size) for op in OPERATIONS)]
            plausible.sort(key=lambda o: (min(bbox_gap(o, a), bbox_gap(o, b)),
                                          str(o.get("object_id", ""))))
            others = plausible[:32]
            rng.shuffle(others)
            chosen = [a, b]
            for obj in others:
                try:
                    if strict_geometry:
                        # Exact RLE geometry is reserved for the required
                        # adjacent pair above.  Additional target conflicts
                        # are conservatively screened by bbox overlap here;
                        # generation/audit repeats the support-mask check on
                        # the chosen group before publication.
                        conflict = any(_bbox_overlap_fraction(obj, prev) > 0.10
                                       or _bbox_overlap_fraction(prev, obj) > 0.10
                                       for prev in chosen)
                    else:
                        conflict = any(_bbox_overlap_fraction(obj, prev) > 0.10
                                       for prev in chosen)
                except (OSError, RuntimeError, TypeError, ValueError):
                    conflict = True
                if not conflict:
                    chosen.append(obj)
                if len(chosen) == n:
                    break
            if len(chosen) != n:
                continue
            # Operation assignment is fixed before planning. For n<=4 all are
            # distinct; for n=5 exactly one operation repeats.
            if n <= 4:
                ops = list(OPERATIONS[:n])
            else:
                ops = ["remove", "replace", "attribute", "add", "attribute"]
            if any(not cached_operation_allowed(obj, op) for obj, op in zip(chosen, ops)):
                # Search a few permutations rather than fallback after plan.
                found = None
                for perm in rng.sample(list(__import__("itertools").permutations(OPERATIONS)), min(24, 24)):
                    trial = list(perm) + [perm[0]] if n == 5 else list(perm[:n])
                    if all(cached_operation_allowed(obj, op) for obj, op in zip(chosen, trial)):
                        found = trial
                        break
                if found is None:
                    continue
                ops = found
            buckets[n].append({"candidate": c, "objects": chosen, "operations": ops,
                               "adjacent_pairs": [{"object_ids": [a["object_id"], b["object_id"]], **adjacency}],
                               "image_size": size})
        if all(len(buckets[n]) >= pool_caps[n] for n in buckets):
            break
    selected, per_source = [], Counter()
    for n in range(2, 6):
        rng.shuffle(buckets[n])
        for row in buckets[n]:
            sid = str(row["candidate"]["source_id"])
            if per_source[sid] >= max_groups_per_source:
                continue
            per_source[sid] += 1
            selected.append(row)
            if sum(1 for x in selected if len(x["objects"]) == n) >= quotas[n]:
                break
    rng.shuffle(selected)
    # A fixed operation order is biased by the naturalness gate: masks that
    # cannot support the early slots silently push the aggregate toward the
    # operations that are easiest to accept.  Reassign only among feasible
    # schedules after the n/source quotas are fixed, so the requested four-way
    # balance is an explicit invariant before planning.
    from .multi_operation_balance import balance_multi_operations
    balance_multi_operations(
        selected,
        image_size_for=lambda row: tuple(row.get("image_size") or _image_size(row["candidate"])),
        allowed=lambda obj, op, size: _cheap_operation_allowed(obj, op, size),
        seed=seed,
    )
    return selected[:target]
