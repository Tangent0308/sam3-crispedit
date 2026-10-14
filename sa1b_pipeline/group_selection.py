"""Deterministic, quota-aware selection of grouped source masks."""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .geometry import load_mask, operation_quality_gate
from .group_geometry import (adjacent_pair_gate, category_allowed, object_features,
                             twin_pair_gate, normalized_category, support_mask, bbox_gap)
from .schema import OPERATIONS


_FEATURE_CACHE: dict[tuple[str, str, tuple[int, int]], dict[str, Any]] = {}
_PAIR_CACHE: dict[tuple[str, str, str, tuple[int, int]], dict[str, Any]] = {}


def _candidate_dict(candidate: Any) -> dict[str, Any]:
    return candidate.to_dict() if hasattr(candidate, "to_dict") else candidate


def _image_size(candidate: dict[str, Any]) -> tuple[int, int]:
    with Image.open(candidate["image_path"]) as image:
        return image.size


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
    if operation == "add":
        category = str(obj.get("category", "")).lower()
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


def _short_side(obj: dict[str, Any]) -> float:
    bbox = obj.get("bbox") or [0, 0, 1, 1]
    return max(1.0, min(abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1]))))


def _group_scope(a: dict[str, Any], b: dict[str, Any]) -> str:
    pa = a.get("parent_id") or (a.get("metadata") or {}).get("parent_id")
    pb = b.get("parent_id") or (b.get("metadata") or {}).get("parent_id")
    if pa and pb and str(pa) == str(pb):
        return "part_on_same_host"
    if a.get("category") != "object" or b.get("category") != "object":
        return "part_on_different_hosts"
    return "object"


def select_twin_candidates(candidates: Iterable[Any], *, target: int, operation: str,
                           seed: int = 0, max_groups_per_source: int = 2,
                           reserved_source_ids: set[str] | None = None,
                           strict_geometry: bool = False) -> list[dict[str, Any]]:
    """Select same-category pairs/triples before the MLLM sees them.

    This function is intentionally conservative: no operation fallback is
    performed after selection and no candidate is rejected for a textual
    ambiguity that only the planner can judge.
    """
    if operation not in OPERATIONS or target < 1:
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
                groups[normalized_category(obj.get("category", ""))].append(obj)
        attempted_total = 0
        for category in sorted(groups):
            values = [obj for obj in groups[category] if _cheap_operation_allowed(obj, operation, size)]
            if len(values) < 2:
                continue
            rng.shuffle(values)
            pair_attempts = 0
            for a, b in combinations(values, 2):
                pair_attempts += 1
                attempted_total += 1
                if pair_attempts > 2000 or attempted_total > 50000:
                    break
                # A twin group requires separated support regions. Dense
                # overlapping bbox candidates (for example rows of windows)
                # are almost never eligible and need not trigger RLE decode.
                if bbox_gap(a, b) < 8:
                    continue
                key = (str(c["source_id"]), str(a.get("object_id")), str(b.get("object_id")), size)
                if strict_geometry:
                    gate = _PAIR_CACHE.get(key)
                    if gate is None:
                        gate = twin_pair_gate(a, b, image_size=size)
                        _PAIR_CACHE[key] = gate
                else:
                    gate = {"eligible": True, "geometry_exact": False,
                            "gap_pixels_lower_bound": bbox_gap(a, b),
                            "same_host": bool((a.get("parent_id") or (a.get("metadata") or {}).get("parent_id")) and
                                               (a.get("parent_id") or (a.get("metadata") or {}).get("parent_id")) ==
                                               (b.get("parent_id") or (b.get("metadata") or {}).get("parent_id")))}
                if gate["eligible"]:
                    rows.append({"candidate": c, "objects": [a, b], "gate": gate,
                                 "scope_level": "part_on_same_host" if gate["same_host"] else _group_scope(a, b),
                                 "operation": operation})
                    if len(rows) >= target * 8:
                        break
            if len(rows) >= target * 8:
                break
            if attempted_total > 50000:
                break
    rng.shuffle(rows)
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
                            max_groups_per_source: int = 2) -> list[dict[str, Any]]:
    """Preselect 2-5 targets with at least one adjacent pair.

    The MLLM still decides semantic feasibility and exact refs; this stage only
    enforces geometry, quotas, object uniqueness and operation availability.
    """
    if target < 1:
        raise ValueError("target must be positive")
    reserved = {str(x) for x in (reserved_source_ids or set())}
    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = {n: [] for n in range(2, 6)}
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
        pairs = [(a, b, e) for a, b in combinations(objects, 2)
                 if bbox_gap(a, b) <= 0.2 * max(1, min(_short_side(a), _short_side(b)))
                 and (e := adjacent_pair_gate(a, b, image_size=size))["eligible"]]
        if not pairs:
            continue
        rng.shuffle(pairs)
        for n in range(2, 6):
            if len(objects) < n:
                continue
            a, b, adjacency = pairs[0]
            others = [o for o in objects if o not in (a, b)]
            rng.shuffle(others)
            chosen = [a, b]
            for obj in others:
                if all(not _conflict(obj, prev, size=size) for prev in chosen):
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
            if any(not _operation_allowed(obj, op, size) for obj, op in zip(chosen, ops)):
                # Search a few permutations rather than fallback after plan.
                found = None
                for perm in rng.sample(list(__import__("itertools").permutations(OPERATIONS)), min(24, 24)):
                    trial = list(perm) + [perm[0]] if n == 5 else list(perm[:n])
                    if all(_operation_allowed(obj, op, size) for obj, op in zip(chosen, trial)):
                        found = trial
                        break
                if found is None:
                    continue
                ops = found
            buckets[n].append({"candidate": c, "objects": chosen, "operations": ops,
                               "adjacent_pairs": [{"object_ids": [a["object_id"], b["object_id"]], **adjacency}],
                               "image_size": size})
    quotas = _target_sizes(target)
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
    return selected[:target]
