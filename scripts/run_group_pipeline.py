#!/usr/bin/env python3
"""Build resumable twin or multi-target data in an isolated root.

Examples (planner-only smoke):
  python scripts/run_group_pipeline.py --kind twins --manifest candidates.jsonl \
      --output-root /mnt/.../twins_first5000 --target 8 --plan-only \
      --planner-model /path/to/Qwen3.8-27B --planner-tp 1

The script never writes to the frozen 50k run. Final publication requires a
semantic audit model and the explicit ``--publish`` flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from sa1b_pipeline.geometry import load_mask, mask_geometry
from sa1b_pipeline.group_generation import (multi_support_preflight, generate_multi_group, generate_twin_group,
                                             twin_support_preflight, parallel_group_generation)
from sa1b_pipeline.group_geometry import normalized_category
from sa1b_pipeline.group_audit import audit_multi_groups, audit_twin_groups, audit_twin_groups_joint
from sa1b_pipeline.group_planning import (build_multi_group, build_twin_group, group_views,
                                           multi_prompt, twin_prompt, MULTI_PLAN_VERSION)
from sa1b_pipeline.group_schema import validate_multi_record, validate_twin_record
from sa1b_pipeline.group_selection import select_multi_candidates, select_twin_candidates, twin_operation_priority
from sa1b_pipeline.io import load_candidates, write_jsonl
from sa1b_pipeline.multi_operation_balance import _target_counts
from sa1b_pipeline.production_state import atomic_json, immutable_json, portable
from sa1b_pipeline.source_isolation import assert_no_eval_leakage, load_reservation


FROZEN_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014")
_AREA_CACHE: dict[tuple[str, str], float] = {}


def _quota(total: int) -> dict[str, int]:
    base, rem = divmod(total, 4)
    return {op: base + int(i < rem) for i, op in enumerate(("add", "replace", "remove", "attribute"))}


def _row_area_bucket(row: dict[str, Any]) -> str:
    """Bucket the smaller member area using the exact dense RLE mask."""
    fractions = []
    source_id = str(row["candidate"].get("source_id", ""))
    for obj in row.get("objects", []):
        key = (source_id, str(obj.get("object_id", "")))
        value = _AREA_CACHE.get(key)
        if value is None:
            rle = obj.get("rle") or {}
            size = rle.get("size")
            try:
                height, width = int(size[0]), int(size[1])
                if isinstance(rle.get("counts"), (str, bytes)):
                    try:
                        from pycocotools import mask as coco_mask
                    except ImportError:
                        area = float(load_mask(obj, (width, height)).sum())
                    else:
                        area = float(coco_mask.area(rle))
                else:
                    area = float(load_mask(obj, (width, height)).sum())
                value = area / max(1, width * height)
            except (TypeError, ValueError, KeyError, IndexError, OSError, RuntimeError):
                bbox = obj.get("bbox") or [0, 0, 0, 0]
                try:
                    value = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
                    value /= max(1, int(size[0]) * int(size[1]))
                except (TypeError, ValueError, KeyError, IndexError):
                    value = 0.0
            _AREA_CACHE[key] = value
        fractions.append(value)
    area = min(fractions) if fractions else 0.0
    if area < 0.005:
        return "lt0.5%"
    if area <= 0.02:
        return "0.5-2%"
    return "gt2%"


def _planner(args):
    if not args.planner_model:
        return None
    if args.planner_backend == "hf":
        from sa1b_pipeline.hf_mllm import HFSelector
        return HFSelector(model_id=args.planner_model, max_new_tokens=args.max_new_tokens,
                          device_map=args.planner_device_map)
    from sa1b_pipeline.mllm import VLLMSelector
    model = VLLMSelector(args.planner_model, tensor_parallel_size=args.planner_tp,
                        max_new_tokens=args.max_new_tokens, batch_size=args.planner_batch_size,
                        enable_thinking=(not getattr(args,"planner_disable_thinking",False)) and (getattr(args,"multi_plan_style","legacy") in ("grounded-thinking-v8", "grounded-thinking-v9", "lean-v1", "minimal-v1", "scene-v1", "content-v1")
                                         or (getattr(args,"twin_planning_style","legacy") in ("minimal-v1", "scene-v1", "content-v1") and args.planner_thinking_budget is not None)),
                        thinking_token_budget=None if getattr(args,"planner_disable_thinking",False) else getattr(args,"planner_thinking_budget",None))
    if args.structured_planning_json:
        if model.params is None:
            model.close()
            raise ValueError('structured planning requires SA1B_FORCE_DIRECT_VLLM=1')
        from vllm.sampling_params import StructuredOutputsParams
        if getattr(args, "kind", None) == "multi" and getattr(args, "multi_grounding_mode", "legacy") == "legacy":
            from sa1b_pipeline.multi_plan_schema import multi_plan_schema
            if getattr(args, "multi_plan_style", "legacy") in ("grounded-thinking-v8", "grounded-thinking-v9", "lean-v1", "minimal-v1", "scene-v1", "content-v1"):
                from sa1b_pipeline.multi_compact_target import group_schema
                schema = group_schema()
            else:
                schema = multi_plan_schema(execution_contract=getattr(args,"multi_execution_contract",False))
            model.params.structured_outputs = StructuredOutputsParams(json=schema)
        elif getattr(args, 'kind', None) == 'twins' and getattr(args, 'twin_planning_style', 'legacy') in ('concise-27b-v5', 'concise-27b-v8', 'lean-v1', 'minimal-v1', 'scene-v1', 'content-v1'):
            from sa1b_pipeline.twin_planning import output_schema
            model.params.structured_outputs = StructuredOutputsParams(json=output_schema())
        else:
            model.params.structured_outputs = StructuredOutputsParams(json_object=True)
    if getattr(args, 'lean_planning', False):
        from sa1b_pipeline.lean_planning import schema
        if args.minimal_planning:
            from sa1b_pipeline.minimal_planning import schema
        if getattr(args, "scene_planning", False):
            from sa1b_pipeline.scene_planning import schema
        if getattr(args, "content_planning", False):
            from sa1b_pipeline.content_planning import schema
        model.params.structured_outputs = StructuredOutputsParams(json=schema(args.kind))
    return model


def _editor(args):
    if not args.editor_model:
        return None
    from sa1b_pipeline.editor import DiffusersQwenImageEditor
    return DiffusersQwenImageEditor(args.editor_model, steps=40,
                                    dtype=args.editor_dtype, crop_instruction=False)


def _auditor(args):
    if not args.audit_model:
        return None
    from sa1b_pipeline.audit_backend import VLLMAuditor
    return VLLMAuditor(args.audit_model, batch_size=args.audit_batch_size,
                       max_tokens=1000 if args.twin_audit_mode == "joint" else 512,
                       check_operation=False, tensor_parallel_size=args.audit_tp)


def _manual_review_pass(row: dict[str, Any]) -> bool:
    """Return true only for an explicitly signed, all-fields-passing review."""
    decision = str(row.get("decision", "")).strip().lower()
    status = str(row.get("status", "")).strip().lower()
    return bool(str(row.get("reviewer", "")).strip()) and decision in {"pass", "manual_pass", "approved"} \
        and status in {"pass", "manual_pass", "approved", "reviewed"} \
        and all(row.get(field) is True for field in
                ("reference_correct", "target_done", "non_target_preserved", "natural"))


def _require_manual_review(groups: list[dict[str, Any]], path: Path, *, kind: str, split: str) -> dict[str, Any]:
    """Gate publication on the required human sample without trusting 27B as human."""
    if not groups:
        raise RuntimeError("human review gate cannot publish an empty group shard")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row.get("group_id")): row for row in rows if row.get("group_id")}
    expected = [str(group.get("twin_id") or group.get("multi_id")) for group in groups]
    reviewed_failures = [gid for gid in expected if gid in by_id
                         and str(by_id[gid].get("status", "")).lower() in
                         {"reviewed", "pass", "fail", "manual_pass", "approved"}
                         and not _manual_review_pass(by_id[gid])]
    if reviewed_failures:
        raise RuntimeError("human review gate contains explicitly rejected or incomplete reviews: "
                           + ", ".join(reviewed_failures[:10]))
    valid_ids = {gid for gid in expected if _manual_review_pass(by_id.get(gid, {}))}
    required = len(expected) if split == "eval" else (200 if kind == "twins" else 100)
    required = min(required, len(expected))
    if len(valid_ids) < required:
        raise RuntimeError(
            f"human review gate failed: {len(valid_ids)} passing groups, "
            f"need {required} for {split} {kind}")
    return {"path": str(path), "passing_groups": len(valid_ids), "required_groups": required,
            "all_eval_groups": split == "eval" and len(valid_ids) == len(expected)}


def _geometry(mask_paths: list[str]) -> list[dict[str, Any]]:
    result = []
    from sa1b_pipeline.group_geometry import precise_point
    for path in mask_paths:
        mask = load_mask({"mask_path": path}, (Path(path).stat().st_size, 1)) if False else None
        from PIL import Image
        with Image.open(path) as image:
            arr = load_mask({"mask_path": path}, image.size)
        geometry = mask_geometry(arr)
        geometry["point"] = precise_point(arr)
        result.append(geometry)
    return result


def _group_id(kind: str, row: dict[str, Any]) -> str:
    ids = ",".join(str(o["object_id"]) for o in row["objects"])
    op = row.get("operation") or "-".join(row.get("operations", []))
    digest = hashlib.sha256(f"{kind}|{row['candidate']['source_id']}|{ids}|{op}".encode()).hexdigest()[:12]
    return f"{kind}-{row['candidate']['source_id']}-{digest}"


def _validate_multi_selection_operations(rows: list[dict[str, Any]], *, aggregate_balance: bool = True) -> None:
    """Fail before planning if a shard violates the multi-target contract.

    Selection is the only stage allowed to assign operation types.  A stale
    fixed-order shard would otherwise consume planner/editor capacity and
    quietly reintroduce the historical add/attribute skew.
    """
    counts = Counter()
    for index, row in enumerate(rows):
        objects = row.get("objects") or []
        operations = row.get("operations") or []
        n = len(objects)
        if len(operations) != n:
            raise RuntimeError(f"multi selection row {index} has {len(operations)} operations for {n} objects")
        if n <= 4 and len(set(operations)) != n:
            raise RuntimeError(f"multi selection row {index} repeats an operation for n={n}")
        if n == 5 and set(operations) != set(("add", "replace", "remove", "attribute")):
            raise RuntimeError("five-target multi selection must contain all four operations")
        if n not in (2, 3, 4, 5):
            raise RuntimeError(f"multi selection row {index} has invalid target count {n}")
        counts.update(operations)
    targets = _target_counts(sum(counts.values()))
    deviations = {op: counts[op] - targets[op] for op in targets}
    # Naturalness eligibility can make exact equality impossible.  Keep the
    # release preflight tight enough to reject the old fixed-order skew while
    # allowing a small, recorded residual imbalance.
    limit = max(8, int(math.ceil(max(targets.values(), default=0) * 0.10)))
    if aggregate_balance and max((abs(value) for value in deviations.values()), default=0) > limit:
        raise RuntimeError(f"multi operation quota skew exceeds {limit}: {deviations}")


def _candidate_rows(args, reservation_ids: set[str]) -> list[dict[str, Any]]:
    candidates = [{**c.to_dict(), "objects": [{**o.__dict__} for o in c.objects]} for c in load_candidates(args.manifest)]
    ids = {str(c["source_id"]).removeprefix("sa1b-") for c in candidates}
    if args.split == "train" and reservation_ids:
        candidates = [c for c in candidates if str(c["source_id"]).removeprefix("sa1b-") not in reservation_ids]
    elif args.split == "eval" and reservation_ids:
        candidates = [c for c in candidates if str(c["source_id"]).removeprefix("sa1b-") in reservation_ids]
    if not candidates:
        raise RuntimeError("no candidates remain after source isolation")
    return candidates


def _select(args, candidates: list[dict[str, Any]], reservation_ids: set[str]) -> list[dict[str, Any]]:
    if args.kind == "twins":
        quotas = _quota(args.target)
        factor = max(1.0, float(args.candidate_pool_factor))
        # The small held-out pool is more sensitive to one scarce operation;
        # give it a little more deterministic oversampling while keeping the
        # training selection cost bounded.
        if args.split == "eval":
            factor = max(factor, 4.0)
        three_fraction = 0.0 if args.split == "eval" else min(1.0, max(0.0, float(args.twin_three_member_fraction)))
        pools: dict[str, list[dict[str, Any]]] = {}
        for index, (op, count) in enumerate(quotas.items()):
            pool_count = max(count, int(math.ceil(count * factor)))
            triple_count = int(math.ceil(pool_count * three_fraction))
            pair_count = pool_count - triple_count
            op_rows = []
            # Build both relation pools independently so abundant cross-class
            # combinations cannot crowd out the hard same-class examples.
            for relation in ('same', 'cross'):
                for members, number in ((3,triple_count),(2,pair_count)):
                    if number:
                        op_rows.extend(select_twin_candidates(
                            candidates, target=number, operation=op,
                            seed=args.seed+index*1009+(17 if members==3 else 0),
                            reserved_source_ids=reservation_ids if args.split=='train' else set(),
                            member_count=members, category_relation=relation))
            # Keep operation-specific pools separate until quotas are filled;
            # truncating one concatenated pool by source order can silently
            # turn a nominally balanced run into an attribute-heavy run.
            random.Random(args.seed + index * 7919).shuffle(op_rows)
            pools[op] = op_rows

        # Compute exact area buckets once per pooled row. The object-level
        # cache makes repeated groups sharing a source mask cheap.
        for op_rows in pools.values():
            for row in op_rows:
                row["_selection_area_bucket"] = _row_area_bucket(row)

        # Balance operation/member/area cells. Object/part scope is reporting
        # metadata only; mixed object/part pairs are eligible by size.
        scope_counts: Counter[str] = Counter()
        area_targets = {
            "lt0.5%": int(round(args.target * 0.30)),
            "0.5-2%": int(round(args.target * 0.40)),
        }
        area_targets["gt2%"] = args.target - sum(area_targets.values())
        area_counts: Counter[str] = Counter()
        category_cap = max(1, int(math.ceil(args.target * 0.05)))
        category_counts: Counter[str] = Counter()
        seen_sources: Counter[str] = Counter()
        seen_groups: set[tuple[str, tuple[str, ...], str]] = set()
        relation_targets = {'same': int(round(args.target*getattr(args,'twin_same_category_fraction',.30)))}
        relation_targets['cross'] = args.target-relation_targets['same']
        relation_counts = Counter()
        requirements = []
        for member_count in (3, 2):
            needs_by_op = {
                op: (int(round(count * three_fraction)) if member_count == 3
                     else count - int(round(count * three_fraction)))
                for op, count in quotas.items()
            }
            for index in range(max(needs_by_op.values(), default=0)):
                for op in quotas:
                    if index < needs_by_op[op]:
                        requirements.append((op, member_count))
        # Interleave operation cells. This prevents one operation from
        # consuming all available same-host/part candidates before the other
        # operations get a chance to meet their quotas.
        selected: list[dict[str, Any]] = []
        shortages: Counter[str] = Counter()
        relaxed_quota_count = 0
        for op, member_count in requirements:
            candidates_for_cell = []
            for row in pools[op]:
                relation = row.get('category_relation','same')
                if relation_counts[relation] >= relation_targets[relation]:
                    continue
                if len(row["objects"]) != member_count:
                    continue
                sid = str(row["candidate"]["source_id"])
                group_key = (sid, tuple(sorted(str(o["object_id"]) for o in row["objects"])), op)
                if seen_sources[sid] >= 2 or group_key in seen_groups:
                    continue
                scope = str(row.get("scope_level", "object"))
                area_bucket = str(row.get("_selection_area_bucket", "lt0.5%"))
                if area_counts[area_bucket] >= area_targets[area_bucket]:
                    continue
                category = normalized_category(row["objects"][0].get("category", "object")).split(",", 1)[0].strip()
                if category_counts[category] >= category_cap:
                    continue
                deficit = 0  # no object/part preference
                area_deficit = area_targets[area_bucket] - area_counts[area_bucket]
                candidates_for_cell.append((area_deficit, deficit, category_counts[category], sid, row))
            if not candidates_for_cell:
                # A tiny final relaxation handles a scarce operation/member
                # cell caused by source/category collisions.  It preserves
                # source uniqueness and chooses the least quota-violating
                # row, while larger shortages still fail closed instead of
                # silently destroying the requested distribution.
                fallback = []
                for row in pools[op]:
                    relation = row.get('category_relation','same')
                    if relation_counts[relation] >= relation_targets[relation]:
                        continue
                    if len(row["objects"]) != member_count:
                        continue
                    sid = str(row["candidate"]["source_id"])
                    group_key = (sid, tuple(sorted(str(o["object_id"]) for o in row["objects"])), op)
                    if seen_sources[sid] >= 2 or group_key in seen_groups:
                        continue
                    scope = str(row.get("scope_level", "object"))
                    area_bucket = str(row.get("_selection_area_bucket", "lt0.5%"))
                    category = normalized_category(row["objects"][0].get("category", "object")).split(",", 1)[0].strip()
                    violation = (int(area_counts[area_bucket] >= area_targets[area_bucket])
                                 + int(category_counts[category] >= category_cap))
                    fallback.append((violation, 0, area_counts[area_bucket],
                                     category_counts[category], sid, row))
                if not fallback:
                    shortages[f"{op}:{member_count}"] += 1
                    continue
                _, _, _, _, sid, row = min(fallback, key=lambda x: x[:-1])
                relaxed_quota_count += 1
            else:
                # Prefer area buckets furthest below their targets,
                # then underused categories; source ID makes the final choice
                # deterministic.
                _, _, _, sid, row = max(candidates_for_cell, key=lambda x: (
                    twin_operation_priority(x[4]["objects"], op), x[0], x[1], -x[2], x[3]))
            category = normalized_category(row["objects"][0].get("category", "object")).split(",", 1)[0].strip()
            scope = str(row.get("scope_level", "object"))
            group_key = (sid, tuple(sorted(str(o["object_id"]) for o in row["objects"])), op)
            seen_sources[sid] += 1
            seen_groups.add(group_key)
            scope_counts[scope] += 1
            area_counts[str(row["_selection_area_bucket"])] += 1
            category_counts[category] += 1
            selected.append(row)
            relation_counts[row.get("category_relation","same")] += 1
        if shortages or len(selected) != args.target or relaxed_quota_count > max(1, int(math.ceil(args.target * 0.005))):
            raise RuntimeError(
                f"twin selection cannot satisfy quotas: shortages={dict(shortages)} "
                f"selected={len(selected)}/{args.target} scopes={dict(scope_counts)} "
                f"areas={dict(area_counts)} relations={dict(relation_counts)} relaxed={relaxed_quota_count}")
        for row in selected:
            row.pop("_selection_area_bucket", None)
        # The loops above preserve both operation quotas and the requested
        # two/three-member mix. Shuffle only after those invariants hold.
        random.Random(args.seed + 424242).shuffle(selected)
        return selected
    return select_multi_candidates(
        candidates, target=args.target, seed=args.seed,
        reserved_source_ids=reservation_ids if args.split == "train" else set(),
        strict_geometry=not args.fast_selection,
    )


def main(argv=None, *, planner_override=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["twins", "multi"], required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--reservation", type=Path, help="source isolation reservation JSON")
    p.add_argument("--planner-model")
    p.add_argument('--twin-planning-style', choices=['legacy','concise-27b-v5','concise-27b-v8','lean-v1','minimal-v1','scene-v1','content-v1'], default='legacy')
    p.add_argument('--planning-view-style', choices=['aligned', 'clean-contours', 'twin-crops'], default='aligned',
                   help='content-v1: clean-contours uses two full frames; twin-crops uses a clean scene plus two labelled crops')
    p.add_argument('--planning-source-hints', choices=['none', 'category', 'annotation'], default='none',
                   help='content-v1: optional source category hints bound to the displayed masks')
    p.add_argument('--instruction-content-policy', choices=['baseline', 'ranked', 'scene'], default='baseline',
                   help='multi planner content policy; scene uses image-grounded rules without examples; ranked is a compatibility alias')
    p.add_argument("--structured-planning-json", action="store_true",
                   help="use vLLM JSON-constrained output in the same planning request")
    p.add_argument("--planner-backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--planner-device-map", default="auto",
                   help="Transformers device map for the HF planner; auto shards large MoE models across visible GPUs")
    p.add_argument("--planner-tp", type=int, default=1)
    p.add_argument("--planner-disable-thinking", action="store_true", help="use a non-thinking Instruct planner; included in resume configuration")
    p.add_argument("--max-new-tokens", type=int, default=1600)
    p.add_argument("--planner-batch-size", type=int, default=8,
                   help="number of multimodal planning requests submitted per vLLM batch")
    p.add_argument("--editor-model")
    p.add_argument("--site-model", default=os.environ.get("SA1B_SITE_MODEL"),
                   help="optional local SAM3 model for add site refinement before the existing audit")
    p.add_argument("--editor-dtype", default="bf16")
    p.add_argument("--editor-devices", help="explicit GPU IDs for persistent parallel editors, e.g. 0,1; generation-only")
    p.add_argument("--audit-model")
    p.add_argument("--audit-tp", type=int, default=1)
    p.add_argument("--audit-batch-size", type=int, default=4)
    p.add_argument("--twin-audit-mode", choices=["legacy", "joint"], default="legacy",
                   help="experimental joint audit: one structured request per group; legacy remains the default")
    p.add_argument("--human-review", type=Path,
                   help="JSONL ledger with explicit human decisions; required for --publish")
    p.add_argument("--outline-planning-crops", action="store_true",
                   help="append an outside-contour locator column beside clean crop and binary mask")
    p.add_argument("--skip-unused-planning-views", action="store_true",
                   help="omit diagnostic images that are never sent to the planner; model inputs stay identical")
    p.add_argument("--locate-planning-scene", action="store_true",
                   help="experimental full-scene outside contours and unit tags for crop-to-scene grounding")
    p.add_argument("--planner-thinking-budget", type=int, default=None, help="Optional internal thinking token limit for grounded-thinking-v9 (v8 is a compatibility alias).")
    p.add_argument("--multi-plan-style", choices=["legacy", "grounded-thinking-v8", "grounded-thinking-v9", "lean-v1", "minimal-v1", "scene-v1", "content-v1"], default="legacy")
    p.add_argument("--multi-execution-contract", action="store_true",
                   help="Experimental v4 observations/partial planning; not enabled by default after quality regression.")
    p.add_argument("--multi-grounding-mode", choices=["legacy", "selected-only", "context-cutout"], default="legacy",
                   help="ground all regions before edits; selected-only omits local context, context-cutout retains paired crops")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--defer-blind", action="store_true",
                   help="plan-only: persist raw twin plans for a separate frozen blind-model stage")
    p.add_argument("--generate-only", action="store_true",
                   help="persist generated groups without loading the audit model")
    p.add_argument("--audit-only", action="store_true",
                   help="audit existing generated.jsonl in the audit runtime without loading planner/editor")
    p.add_argument("--selection-only", action="store_true",
                   help="select and persist grouped inputs without loading a planner model")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--max-groups", type=int)
    p.add_argument("--candidate-pool-factor", type=float, default=3.0,
                   help="oversample planning candidates so rejected plans can be refilled")
    p.add_argument("--dynamic-operation-quotas", action="store_true",
                   help="external coordinator enforces final multi operation quotas; retain per-group rules")
    p.add_argument("--quota-from-selection", action="store_true",
                   help="use frozen shard operation/relation counts to avoid per-rank rounding drift")
    p.add_argument("--twin-same-category-fraction", type=float, default=.30,
                   help="target same-category share; remaining groups may mix categories")
    p.add_argument("--twin-three-member-fraction", type=float, default=0.15,
                   help="target fraction of three-member twin groups")
    p.add_argument("--fast-selection", action="store_true",
                   help="use bbox-only multi-target capacity selection; exact geometry remains a release gate")
    p.add_argument('--planning-operation-retry', type=int, choices=[0, 1], default=0,
                   help='lean-v1 only: at most one operation-change retry on explicit model infeasibility')
    p.add_argument('--planning-operation-counts', type=Path,
                   help='JSON counts of FINAL accepted operations; read-only quota priority for retries')
    p.add_argument('--planning-view-workers', type=int, default=1,
                   help='with --per-mask-planning: prepare planning views and mask geometry in this many processes '
                        'before planning (same files and results; 1 keeps the serial loop)')
    p.add_argument('--per-mask-planning', action='store_true',
                   help='content-v1: two images and one stateless request per mask; twins share operation only')
    p.add_argument('--group-mask-planning', action='store_true',
                   help='content-v1: original + all mask contours, one call per group; object-adapted edits')
    p.add_argument('--evidence-gates', action='store_true',
                   help='per-mask only: after program checks, drop targets failing the independent reader '
                        '(clean photo + Ref -> boxes), companion-category identity and replace payload checks; '
                        'twins keep two passing members when selection over-generated a third')
    p.add_argument('--evidence-gate-iou', type=float, default=.5)
    p.add_argument('--replace-retry', action='store_true',
                   help='Opt-in with --evidence-gates: one re-plan for replace targets dropped as appearance-only')
    p.add_argument('--evidence-gate-lighting', action='store_true',
                   help='Opt-in: colour edits of parts in shade or backlit (lit/shade/silhouette judgement '
                        'without reasoning) are dropped')
    p.add_argument('--evidence-gate-medium', action='store_true',
                   help='Opt-in: additions whose anchor is inside a printed, painted or displayed picture, or part of a '
                        'statue or relief, are dropped (real/depicted/sculpted judgement without reasoning)')
    p.add_argument('--reader-retry', action='store_true',
                   help='Opt-in with --evidence-gates: one re-plan, with the reader\'s finding, for targets whose only '
                        'gate failure was reader=wrong/none')
    p.add_argument('--evidence-gate-viewpoint', action='store_true',
                   help='Opt-in with --evidence-gates: drop additions in photos taken from high above (aerial)')
    p.add_argument('--evidence-gate-add-site', action='store_true',
                   help='Opt-in: an add Ref may name the attachment site on its anchor (reader containment, '
                        'identity "part"); add twins keep the most similar anchors')
    p.add_argument('--explicit-text', action='store_true',
                   help='per-mask only: text-bearing additions/replacements carry exact wording in a text slot '
                        '(rendered as reading "..."), text attribute values are short English wording')
    p.add_argument('--twin-single-add', action='store_true',
                   help='per-mask twins: an add member adds exactly one compact item (comparable twin regions)')
    p.add_argument('--garment-extent-check', action='store_true',
                   help='per-mask: drop replacements of shorts/short skirts by long trousers, leggings or long '
                        'dresses (content_guard.garment_extent)')
    args = p.parse_args(argv)
    args.minimal_planning = (args.twin_planning_style if args.kind == 'twins' else args.multi_plan_style) in ('minimal-v1', 'scene-v1', 'content-v1')
    args.scene_planning = (args.twin_planning_style if args.kind == 'twins' else args.multi_plan_style) in ('scene-v1', 'content-v1')
    args.content_planning = (args.twin_planning_style if args.kind == 'twins' else args.multi_plan_style) == 'content-v1'
    if args.per_mask_planning and args.group_mask_planning:
        p.error('choose per-mask-planning OR group-mask-planning')
    if (args.evidence_gate_add_site or args.evidence_gate_lighting or args.evidence_gate_medium or args.evidence_gate_viewpoint
            or args.replace_retry or args.reader_retry) and not args.evidence_gates:
        p.error('--evidence-gate-add-site / --evidence-gate-lighting / --evidence-gate-medium / --evidence-gate-viewpoint / '
                '--replace-retry require --evidence-gates')
    if args.evidence_gates and not args.per_mask_planning:
        p.error('--evidence-gates requires --per-mask-planning')
    if (args.explicit_text or args.twin_single_add or args.garment_extent_check) and not args.per_mask_planning:
        p.error('--explicit-text/--twin-single-add/--garment-extent-check require --per-mask-planning')
    if not 0 < args.evidence_gate_iou < 1:
        p.error('--evidence-gate-iou must be within (0, 1)')
    if (args.per_mask_planning or args.group_mask_planning) and (not args.content_planning or args.planning_operation_retry or args.planning_view_style != 'aligned'):
        p.error('mask planning requires content-v1, default view style, and retry=0')
    if args.planning_view_style != 'aligned' and not args.content_planning:
        p.error('clean-contours and twin-crops require content-v1 planning')
    if args.planning_view_style == 'twin-crops' and args.kind != 'twins':
        p.error('twin-crops requires twins planning with exactly two members')
    if args.planning_source_hints != 'none' and not args.content_planning:
        p.error('source category hints require content-v1 planning')
    args.lean_planning = args.minimal_planning or (args.twin_planning_style if args.kind == 'twins' else args.multi_plan_style) == 'lean-v1'
    if args.planning_operation_retry and not args.lean_planning:
        p.error('operation retry requires lean-v1 or minimal-v1')
    if args.lean_planning:
        if not args.structured_planning_json or args.planner_backend != 'vllm':
            p.error('lean-v1 requires structured vLLM planning')
        if args.multi_grounding_mode != 'legacy' or args.multi_execution_contract:
            p.error('lean-v1 cannot mix older grounding contracts')
        args.locate_planning_scene = True
    if args.multi_plan_style in ('grounded-thinking-v8', 'grounded-thinking-v9', 'lean-v1', 'minimal-v1', 'scene-v1', 'content-v1'):
        if args.kind != 'multi' or not args.structured_planning_json:
            p.error('grounded-thinking-v8/v9 requires --kind multi and --structured-planning-json')
        if args.planner_thinking_budget is None:
            args.planner_thinking_budget = 1024
        if args.planner_thinking_budget < 0:
            p.error('--planner-thinking-budget must be non-negative')
        args.locate_planning_scene = True
    elif args.kind == "twins" and args.minimal_planning and args.planner_thinking_budget is not None:
        if args.planner_thinking_budget < 0:
            p.error("--planner-thinking-budget must be non-negative")
    elif args.planner_thinking_budget is not None:
        p.error('--planner-thinking-budget requires --multi-plan-style grounded-thinking-v8/v9')
    if sum((args.plan_only, args.selection_only, args.generate_only, args.audit_only)) > 1:
        p.error("selection/plan/generate/audit-only modes are mutually exclusive")
    if args.defer_blind and not args.plan_only:
        p.error("--defer-blind requires --plan-only")
    if args.audit_only and (not args.audit_model or args.planner_model or args.editor_model):
        p.error("--audit-only requires --audit-model and no planner/editor model")
    if args.publish and (args.plan_only or args.selection_only or args.generate_only):
        p.error("publication requires the audit stage")
    if not 0 <= args.twin_same_category_fraction <= 1:
        p.error("same-category fraction must be within 0..1")
    if args.target < 1:
        p.error("--target must be positive")
    if args.editor_devices and not args.generate_only:
        p.error("--editor-devices requires --generate-only to isolate GPU runtime ownership")
    if args.site_model:
        os.environ["SA1B_SITE_MODEL"] = str(Path(args.site_model).resolve())
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if root == FROZEN_ROOT or FROZEN_ROOT in root.parents or root in FROZEN_ROOT.parents:
        raise RuntimeError("refusing to write inside the protected 50k run root")
    reservation_ids, reservation_hashes = load_reservation(args.reservation) if args.reservation else (set(), set())
    from sa1b_pipeline.instruction_diversity import VERSION as planning_content_version
    from sa1b_pipeline.lean_planning import VERSION as lean_planning_version
    if args.minimal_planning:
        from sa1b_pipeline.minimal_planning import VERSION as lean_planning_version
    if args.scene_planning:
        from sa1b_pipeline.scene_planning import VERSION as lean_planning_version
    if args.content_planning:
        from sa1b_pipeline.content_planning import VERSION as lean_planning_version
    immutable_json(root / "config.json", {"kind": args.kind, "target": args.target, "split": args.split,
                                           "seed": args.seed, "manifest": str(args.manifest.resolve()),
                                           "reservation": str(args.reservation.resolve()) if args.reservation else None,
                                           **({'per_mask_planning': __import__('sa1b_pipeline.per_mask_planning', fromlist=['VERSION']).VERSION} if args.per_mask_planning else {}),
                                           **({'evidence_gates': {'version': __import__('sa1b_pipeline.evidence_gates', fromlist=['VERSION']).VERSION,
                                                                  'iou_threshold': args.evidence_gate_iou,
                                                                  **({'add_site': True} if args.evidence_gate_add_site else {}),
                                                                  **({'lighting': True} if args.evidence_gate_lighting else {}),
                                                                  **({'medium': True} if args.evidence_gate_medium else {}),
                                                                  **({'viewpoint': True} if args.evidence_gate_viewpoint else {})}} if args.evidence_gates else {}),
                                           **({'planning_options': {k: True for k, v in (('explicit_text', args.explicit_text),
                                                                                        ('twin_single_add', args.twin_single_add),
                                                                                        ('replace_retry', args.replace_retry),
                                                                                        ('garment_extent', args.garment_extent_check),
                                                                                        ('reader_retry', args.reader_retry)) if v},
                                               'text_edits_version': __import__('sa1b_pipeline.text_edits', fromlist=['VERSION']).VERSION}
                                              if args.explicit_text or args.twin_single_add or args.replace_retry
                                              or args.garment_extent_check or args.reader_retry else {}),
                                           **({'group_mask_planning': __import__('sa1b_pipeline.group_mask_planning', fromlist=['VERSION']).VERSION} if args.group_mask_planning else {}),
                                           "protocol": "2.0", "mask_semantics": "2.0",
                                           **({'planning_source_hints': args.planning_source_hints,
                                               'planning_source_hints_version': __import__('sa1b_pipeline.content_planning', fromlist=['SOURCE_HINT_VERSION']).SOURCE_HINT_VERSION}
                                              if args.planning_source_hints != 'none' else {}),
                                           **({'planning_view_style': args.planning_view_style,
                                               'planning_view_version': __import__('sa1b_pipeline.' + ('twin_crop_views' if args.planning_view_style == 'twin-crops' else 'full_contour_views'), fromlist=['VERSION']).VERSION}
                                              if args.planning_view_style in ('clean-contours', 'twin-crops') else {}),
                                           **({"planner_disable_thinking": True} if args.planner_disable_thinking else {}),
                                           "planning_content_version": planning_content_version,
                                           **({"lean_planning": lean_planning_version, "planning_operation_retry": args.planning_operation_retry,
                                               "operation_count_basis": json.loads(args.planning_operation_counts.read_text()) if args.planning_operation_counts else {}} if args.lean_planning else {}),
                                           "instruction_style": "benchmark-concise-3-blind-1", "steps": 40,
                                           **({"twin_contract": "same-operation-adapted-3" if (args.per_mask_planning or args.group_mask_planning) else "blind-size-matched-2",
                                               "twin_same_category_fraction": args.twin_same_category_fraction,
                                               **({"quota_from_selection": True} if args.quota_from_selection else {})} if args.kind == "twins" else {}),
                                           **({"dynamic_operation_quotas": True} if args.dynamic_operation_quotas else {}),
                                           **({"structured_planning_json": True} if args.structured_planning_json else {}),
                                           **({'twin_planning_style':args.twin_planning_style} if args.twin_planning_style != 'legacy' else {}),
                                           **({'twin_thinking_budget': args.planner_thinking_budget} if args.kind == 'twins' and args.minimal_planning and args.planner_thinking_budget is not None else {}),
                                           **({"site_model": str(Path(args.site_model).resolve())} if args.site_model else {})})
    candidates = _candidate_rows(args, reservation_ids)
    selected_path = root / "inputs/selected.jsonl"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_path.exists():
        selected = [json.loads(line) for line in selected_path.read_text().splitlines() if line.strip()]
    else:
        selected = _select(args, candidates, reservation_ids)
        if args.max_groups:
            selected = selected[:args.max_groups]
        write_jsonl(selected_path, selected)
    if not selected:
        raise RuntimeError("candidate selection produced no groups")
    if args.kind == "multi":
        _validate_multi_selection_operations(selected, aggregate_balance=not args.dynamic_operation_quotas)
    if args.split == "train" and args.reservation:
        assert_no_eval_leakage((r["candidate"] for r in selected), args.reservation)
    selection_operations = Counter()
    selection_scopes = Counter()
    selection_members = Counter()
    selection_areas = Counter()
    selection_categories = Counter()
    selection_relations = Counter()
    for row in selected:
        if args.kind == "twins":
            selection_operations[str(row.get("operation"))] += 1
            selection_relations[row.get("category_relation", "same")] += 1
            selection_scopes[str(row.get("scope_level", "object"))] += 1
            selection_areas[_row_area_bucket(row)] += 1
            selection_categories[normalized_category(row["objects"][0].get("category", "object")).split(",", 1)[0].strip()] += 1
        else:
            selection_operations.update(row.get("operations", []))
            selection_members[len(row.get("objects", []))] += 1
    if args.kind == "twins":
        selection_members = Counter(len(row.get("objects", [])) for row in selected)
    atomic_json(root / "selection.json", {"requested": args.target, "selected": len(selected),
                                          "operations": selection_operations,
                                          "scope_levels": selection_scopes,
                                          "member_counts": selection_members,
                                          "area_buckets": selection_areas,
                                          "category_counts": selection_categories,
                                          "category_relations": selection_relations,
                                          "source_count": len({r["candidate"]["source_id"] for r in selected}),
                                          "source_ids": sorted({r["candidate"]["source_id"] for r in selected})})
    if args.selection_only:
        print(json.dumps({"status": "selected", "groups": len(selected),
                          "selection": str(selected_path),
                          "summary": str(root / "selection.json")}, ensure_ascii=False))
        return 0
    plans_dir = root / "planned"; plans_dir.mkdir(exist_ok=True)
    if args.kind == "multi" and args.planner_model:
        contract = root / "planning_contract.json"
        if (args.multi_grounding_mode != "legacy" and not contract.exists()
                and (any(plans_dir.glob("*.json")) or any((root / "rejected").glob("*.json")))):
            raise ValueError("new grounding mode requires a fresh experiment root; existing plans are not upgraded")
        immutable_json(contract, {"multi_plan_version": MULTI_PLAN_VERSION,
                                  "multi_grounding_mode": args.multi_grounding_mode,
                                  "multi_execution_contract": args.multi_execution_contract,
                                  **({"multi_plan_style": args.multi_plan_style, "thinking_token_budget": args.planner_thinking_budget} if args.multi_plan_style != "legacy" else {}),
                                  "locate_planning_scene": args.locate_planning_scene,
                                  "instruction_content_policy": args.instruction_content_policy})
    needs_planning = any(not (plans_dir / f"{_group_id(args.kind, row)}.json").exists()
                         and not (root / "rejected" / f"{_group_id(args.kind, row)}.json").exists()
                         for row in selected)
    import time
    planner_started = time.monotonic()
    planner = (planner_override if planner_override is not None else _planner(args)) if needs_planning else None
    if planner is not None and args.evidence_gates:
        from sa1b_pipeline.evidence_gates import EvidenceGates
        planner.evidence_gates = EvidenceGates(iou_threshold=args.evidence_gate_iou,
                                               add_site=args.evidence_gate_add_site,
                                               lighting=args.evidence_gate_lighting,
                                               medium=args.evidence_gate_medium,
                                               viewpoint=args.evidence_gate_viewpoint)
    elif planner is not None and getattr(planner, 'evidence_gates', None) is not None:
        planner.evidence_gates = None  # A shared planner must not leak gates into an ungated run.
    if planner is not None:
        # Same rule for planning options: set them per run, never inherit.
        planner.planning_options = {k: True for k, v in (('explicit_text', args.explicit_text),
                                                         ('twin_single_add', args.twin_single_add),
                                                         ('replace_retry', args.replace_retry),
                                                         ('garment_extent', args.garment_extent_check),
                                                         ('reader_retry', args.reader_retry)) if v}
    planner_initialized = time.monotonic()
    planned = []
    prepared = {}
    if planner is not None and args.per_mask_planning and args.planning_view_workers > 1:
        # Serial view preparation left the loaded planners idle for minutes on large waves
        # (smoke wave of 239 twin groups: 422 s); spawn, not fork, from this multi-threaded process.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from sa1b_pipeline.per_mask_planning import views_with_geometry
        todo = [(index, selection) for index, selection in enumerate(selected)
                if not (plans_dir / f"{_group_id(args.kind, selection)}.json").exists()
                and not (root / "rejected" / f"{_group_id(args.kind, selection)}.json").exists()]
        if todo:
            with ProcessPoolExecutor(min(args.planning_view_workers, len(todo)),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                tasks = [(selection['candidate'], selection['objects'], str(root / 'planning_views')) for _, selection in todo]
                for (index, _), result in zip(todo, pool.map(views_with_geometry, tasks, chunksize=2)):
                    prepared[index] = result
    try:
        pending = []
        for index, selection in enumerate(selected):
            gid = _group_id(args.kind, selection)
            path = plans_dir / f"{gid}.json"
            if path.exists():
                planned.append(json.loads(path.read_text())); continue
            rejected_path = root / "rejected" / f"{gid}.json"
            if rejected_path.exists():
                # Resume never spends another planning call on a completed
                # rejection. A new prompt experiment needs a separate root.
                continue
            objects = selection["objects"]
            grounding_first = args.kind == "multi" and args.multi_grounding_mode != "legacy"
            per_mask_views = None
            if args.group_mask_planning:
                from sa1b_pipeline.group_mask_planning import views as all_views
                views, mask_paths, view_digest = all_views(selection['candidate'], objects, root/'planning_views')
            elif args.per_mask_planning and index in prepared:
                per_mask_views, mask_paths, view_digest, _ = prepared[index]
                views = per_mask_views[0]  # legacy group metadata only; actual requests use per_mask_views.
            elif args.per_mask_planning:
                from sa1b_pipeline.per_mask_planning import views as single_views
                per_mask_views, mask_paths, view_digest = single_views(selection['candidate'], objects, root/'planning_views')
                views = per_mask_views[0]  # legacy group metadata only; actual requests use per_mask_views.
            else:
                views, mask_paths, view_digest = group_views(selection["candidate"], objects, root / "planning_views",
                                                              locator_outline=args.outline_planning_crops,
                                                              locate_in_scene=args.locate_planning_scene,
                                                              locator_cutout=args.kind == "multi" or args.twin_planning_style in ('concise-27b-v5','concise-27b-v8','lean-v1', 'minimal-v1', 'scene-v1', 'content-v1'),
                                                              selected_only=grounding_first,
                                                              context_cutout=args.kind == "multi" and args.multi_grounding_mode == "context-cutout",
                                                              save_debug_views=not args.skip_unused_planning_views,
                                                              grounded_zoom=args.kind == "multi" and args.multi_plan_style in ("grounded-thinking-v8", "grounded-thinking-v9", "lean-v1", "minimal-v1", "scene-v1", "content-v1"),
                                                              aligned_outline=args.scene_planning,
                                                              clean_contours=args.planning_view_style == 'clean-contours',
                                                              twin_crops=args.planning_view_style == 'twin-crops',
                                                              grounded_binary=args.multi_plan_style in ("grounded-thinking-v8", "grounded-thinking-v9", "lean-v1", "minimal-v1", "scene-v1", "content-v1"))
            geometries = prepared[index][3] if index in prepared else _geometry(mask_paths)
            if planner is None:
                # Generation-only resume is allowed: selections rejected by a
                # prior planning batch have no plan file and must simply be
                # skipped. A fresh --plan-only run still fails closed below.
                if args.plan_only:
                    raise RuntimeError("planner model is required to create new group plans")
                continue
            if args.kind == "twins":
                areas = [g["area_fraction"] for g in geometries]
                if min(areas) <= 0 or max(areas) > 3 * min(areas):
                    rejection = {"group_id": gid, "status": "preplan_rejected",
                                 "reason": "twin exact mask area ratio exceeds 3",
                                 "area_fractions": areas, "selection": selection}
                    atomic_json(root / "rejected" / f"{gid}.json", rejection)
                    print(json.dumps({"id": gid, "status": "preplan_rejected",
                                      "reason": rejection["reason"]}), flush=True)
                    continue
                op = selection["operation"]
                prompt = '' if args.lean_planning else twin_prompt(op, objects, style=args.twin_planning_style)
                if args.outline_planning_crops and args.twin_planning_style == 'legacy':
                    prompt += "\nImage 2 has THREE aligned columns per row: clean source, binary mask, and source with a yellow contour OUTSIDE the mask. Use the contour to identify the exact object, but read colors from column 1 only. Do not confuse an arch with a light inside it or a whole wheel with its rim. When front/rear or body-left/right is uncertain, prefer concise image-left/image-right references. A placement site must be a subregion (e.g. top edge), not repeat the host noun; use site=\"\" (the empty JSON string, never the literal word empty) for on/in the host itself. A green stripe changed to a red stripe is attribute, not replace; reject if no genuine new design fits."

            else:
                assignments = {str(obj["object_id"]): op for obj, op in zip(objects, selection["operations"])}
                order = [f"U{i+1}" for i in sorted(range(len(objects)), key=lambda i: ("remove", "replace", "attribute", "add").index(selection["operations"][i]))]
                prompt = multi_prompt(assignments, objects, order, grounding_first=grounding_first,
                                      context_cutout=args.multi_grounding_mode == "context-cutout",
                                      content_policy=args.instruction_content_policy,
                                      execution_contract=args.multi_execution_contract)
                if args.multi_plan_style in ("grounded-thinking-v8", "grounded-thinking-v9", "lean-v1", "minimal-v1", "scene-v1", "content-v1"):
                    if grounding_first or args.multi_execution_contract:
                        raise ValueError("grounded-thinking-v8 is a separate protocol; do not mix grounding/contract flags")
                    from sa1b_pipeline.multi_compact_target import group_prompt
                    prompt = group_prompt(assignments,objects,order)
            if args.locate_planning_scene or grounding_first:
                if args.locate_planning_scene:
                    appearance_rule = (
                        "Contours do not change selected pixels. Match the tags to their crops; never name the yellow contour or tags in a ref."
                        if grounding_first else
                        "Read appearance from the clean crops in image 2; never name the yellow contour or tags in a ref.")
                    prompt = prompt.replace("Image 1 is the clean ORIGINAL scene." if grounding_first else "Image 1 is the clean original scene.",
                        "Image 1 is the original scene with yellow outside contours and U1/U2/U3 tags for localization only. "
                        + appearance_rule)
                with Image.open(selection['candidate']['image_path']) as source:
                    width, height = source.size
                locations = [{"unit_id": f"U{i+1}", "object_id": str(obj['object_id']),
                              "bbox_xyxy_percent": [round(100 * v / (width if j % 2 == 0 else height), 1)
                                                    for j, v in enumerate(geom['bbox'])]}
                             for i, (obj, geom) in enumerate(zip(objects, geometries))]
                location_kind = "full-image" if grounding_first else "tagged"
                prompt += ("\nFull-image locations (0=left/top, 100=right/bottom): " + json.dumps(locations)
                           + f"\nMatch each crop to its {location_kind} location before writing refs. Left/right refer to the ORIGINAL image, "
                           "never to the crop or row order. Distinguish the target from ALL similar instances in the full scene, "
                           "not just the selected members. If multiple similar objects share one side, use a verified ordinal "
                           "or a short unique host relation. Never put coordinates, unit IDs or annotation terms in ref phrases or rendered instructions. "
                           "Internal contact_xy is required for add: normalized original-image coordinates in [0,1], not the percent scale of the boxes above.")
            source_hints = None
            if args.content_planning and args.planning_source_hints == 'category':
                from sa1b_pipeline.content_planning import category_hints
                source_hints = category_hints(objects)
            elif args.content_planning and args.planning_source_hints == 'annotation':
                from sa1b_pipeline.content_planning import annotation_hints
                source_hints = annotation_hints(objects)
            if args.lean_planning:
                from sa1b_pipeline.lean_planning import prompt as lean_prompt
                if args.minimal_planning:
                    from sa1b_pipeline.minimal_planning import prompt as lean_prompt
                if args.scene_planning:
                    from sa1b_pipeline.scene_planning import prompt as lean_prompt
                if args.content_planning:
                    from sa1b_pipeline.content_planning import prompt as lean_prompt
                lean_assignments = {f'U{i+1}': (selection['operation'] if args.kind == 'twins' else selection['operations'][i]) for i in range(len(objects))}
                prompt = lean_prompt(args.kind, lean_assignments, locations=None if args.scene_planning else locations,
                                     **({'view_style': args.planning_view_style, 'source_hints': source_hints} if args.content_planning else {}))
            pending.append({"index": index, "selection": selection, "gid": gid, "path": path,
                            "objects": objects, "views": views, "mask_paths": mask_paths,
                            "geometries": geometries, "view_digest": view_digest,
                            "prompt": prompt, "assignments": assignments if args.kind == "multi" else None})
            pending[-1]['per_mask_views'] = per_mask_views
            pending[-1]['planning_view_style'] = args.planning_view_style
            pending[-1]['planning_source_hints'] = source_hints
        if pending:
            print(json.dumps({'planning_preparation': {'requests': len(pending),
                              'planner_initialization_seconds': planner_initialized-planner_started,
                              'views_and_checks_seconds': time.monotonic()-planner_initialized}}), flush=True)
            # Submit bounded batches and persist every completed plan before
            # asking for the next batch. A rank may be preempted while a
            # 1,250-group shard is running; per-group checkpoints make resume
            # reuse successful calls instead of losing the whole shard.
            batch_size = max(1, int(getattr(planner, "batch_size", args.planner_batch_size)))
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start:batch_start + batch_size]
                if args.group_mask_planning:
                    from sa1b_pipeline.group_mask_planning import plan_batch
                    raws = plan_batch(planner, batch, kind=args.kind, root=root)
                elif args.per_mask_planning:
                    from sa1b_pipeline.per_mask_planning import plan_batch
                    raws = plan_batch(planner, batch, kind=args.kind, root=root)
                elif args.lean_planning:
                    from sa1b_pipeline.lean_retry import plan_batch
                    raws = plan_batch(planner, batch, kind=args.kind, root=root,
                                      retry=args.planning_operation_retry,
                                      accepted_counts=json.loads(args.planning_operation_counts.read_text()) if args.planning_operation_counts else None,
                                      existing_groups=planned, minimal=args.minimal_planning, scene=args.scene_planning, content=args.content_planning)
                else:
                    raws = planner.generate_multimage([(item["views"], item["prompt"]) for item in batch])
                if args.kind == 'multi' and args.multi_plan_style in ('grounded-thinking-v8', 'grounded-thinking-v9'):
                    from sa1b_pipeline.multi_compact_target import normalize_group
                    raws=[normalize_group(raw) for raw in raws]
                if len(raws) != len(batch):
                    raise RuntimeError(
                        f"planner returned an incomplete batch: {len(raws)}/{len(batch)}")
                for item, raw in zip(batch, raws):
                    selection, objects = item["selection"], item["objects"]
                    try:
                        if raw.get('_generation_metadata', {}).get('finish_reason') == 'length':
                            raise ValueError('planner output reached max_new_tokens; incomplete response, not a source-mask rejection')
                        if args.kind == "twins":
                            twin_objects, twin_masks, twin_geometry = objects, item["mask_paths"], item["geometries"]
                            if raw.get('_twin_kept_object_ids'):
                                # Evidence gates kept two of the over-generated candidates.
                                keep = [i for i, o in enumerate(objects) if str(o['object_id']) in raw['_twin_kept_object_ids']]
                                if len(keep) != len(raw['_twin_kept_object_ids']):
                                    raise ValueError('kept twin members do not match the selection')
                                twin_objects = [objects[i] for i in keep]
                                twin_masks = [item["mask_paths"][i] for i in keep]
                                twin_geometry = [item["geometries"][i] for i in keep]
                            group = build_twin_group(selection["candidate"], twin_objects, raw, twin_masks,
                                                     split=args.split, scope_level=selection["scope_level"], group_id=item["gid"],
                                                     geometry=twin_geometry, assigned_operation=selection["operation"],
                                                     require_grounding=True)
                            if raw.get('twin_overgeneration'):
                                group['twin_overgeneration'] = raw['twin_overgeneration']
                            group["selection_evidence"] = selection["gate"]
                            preflight = twin_support_preflight(group, root / "generated" / item["gid"])
                            if not preflight["pass"]:
                                raise ValueError("pre-generation writable support gap failed")
                            group["group_checks"]["support_preflight"] = preflight
                        else:
                            if args.multi_grounding_mode != "legacy" and raw.get("grounding_version") != 2:
                                raise ValueError("multi grounding-first response missing version 2")
                            group = build_multi_group(selection["candidate"], objects, raw, item["mask_paths"],
                                                      split=args.split, group_id=item["gid"], geometry=item["geometries"],
                                                      assignments=item["assignments"], require_grounding=True)
                            if raw.get('grounding_version') in (4, 5, 6, 7):
                                from sa1b_pipeline.multi_execution_contract import retain_feasible_plan, schedule_content_dependencies
                                group = retain_feasible_plan(group)
                                group = schedule_content_dependencies(group)
                            else:
                                group["adjacent_pairs"] = []
                            group["support_preflight"] = multi_support_preflight(group, root / "generated" / item["gid"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        # One malformed/infeasible completion must not discard
                        # valid siblings from the same continuous batch. Keep the
                        # raw answer and selection evidence for audit/retry.
                        rejection = {"group_id": item["gid"], "status": "plan_rejected",
                                     "reason": str(exc), "raw": raw, "selection": selection,
                                     "failure_kind": ('generation_limit' if raw.get('_generation_metadata',{}).get('finish_reason') == 'length'
                                         else 'output_format' if raw.get('_raw_output') and not raw.get('targets',raw.get('members')) else 'plan_constraint'),
                                     "planning_prompt": item["prompt"], "planning_views": item["views"],
                                     "planner_model": getattr(planner, "model_id", "unknown")}
                        atomic_json(root / "rejected" / f"{item['gid']}.json", rejection)
                        print(json.dumps({"planned": len(planned), "total": len(selected),
                                          "id": item["gid"], "status": "rejected", "reason": str(exc)},
                                         ensure_ascii=False), flush=True)
                        continue
                    if args.minimal_planning and args.kind == "twins":
                        group["category_relation"] = selection.get("category_relation", "unknown")
                        group["category_relation_source"] = "source_selection_not_inferred_from_ref"
                    group["planning"] = {"model": getattr(planner, "model_id", "unknown"), "calls": raw.get("_planning_calls", raw.get("_generation_metadata", {}).get("calls", 1)),
                                          "views": item["views"], "view_digest": item["view_digest"], "raw": raw,
                                          "prompt": item["prompt"]}
                    if args.group_mask_planning:
                        group['planning']['request_granularity'] = 'all_masks'
                    if args.per_mask_planning:
                        from sa1b_pipeline.per_mask_planning import request_prompt as mask_prompt
                        group['planning'].update(request_granularity='one_mask',
                            views=item['per_mask_views'],
                            prompt=[mask_prompt(op, item['per_mask_views'][i], (item.get('planning_source_hints') or {}).get(uid))
                                    for i, (uid, op) in enumerate(item['_mask_assignments'].items())])
                        view_metadata = json.loads((Path(item['per_mask_views'][0][0]).parent/'views.json').read_text())
                        # Key by object id: kept over-generated twins are re-indexed U1/U2.
                        cleanup = {str(t['object_id']): t.get('mask_cleanup', {}) for t in view_metadata['targets']}
                        for member in group.get('members', group.get('units', [])):
                            evidence = cleanup.get(str(member['object_id']), {})
                            if evidence.get('removed_remote_pixels'):
                                member['source_mask_cleanup'] = evidence
                    if item.get('planning_source_hints'):
                        for member in group.get('members', group.get('units', [])):
                            hint = item['planning_source_hints'].get(member['unit_id'])
                            if hint:
                                member['source_annotation_hint'] = hint
                    # This is the resume checkpoint. It is written before the
                    # next request is submitted, so an interrupted rank never
                    # needs to repeat a successfully parsed MLLM response.
                    atomic_json(item["path"], group)
                    planned.append(group)
                    print(json.dumps({"planned": len(planned), "total": len(selected),
                                      "id": item["gid"]}, ensure_ascii=False), flush=True)
                print(json.dumps({"planning_batch_completed": min(batch_start + len(batch), len(pending)),
                                  "planning_batch_total": len(pending), "checkpointed": len(planned)},
                                 ensure_ascii=False), flush=True)
        if args.kind == 'twins' and planner is not None and not args.defer_blind:
            from sa1b_pipeline.blind_identifiability import run_blind_probes
            run_blind_probes(planned, planner, root / 'blind_probes')
            for group in planned:
                atomic_json(plans_dir / (group['twin_id']+'.json'), group)
    finally:
        if planner is not None and planner_override is None and hasattr(planner, "close"):
            planner.close()
    if args.kind == 'twins' and not args.defer_blind:
        from sa1b_pipeline.twin_contract import blind_probe_satisfied
        for group in planned:
            check = group.get('group_checks', {}).get('blind_disambiguation', {})
            if not blind_probe_satisfied(group):
                atomic_json(root / 'rejected' / (group['twin_id']+'_blind.json'),
                            {'group_id':group['twin_id'],'stage':'blind_preflight','check':check})
        planned = [g for g in planned if blind_probe_satisfied(g)]
    write_jsonl(root / "planned.jsonl", planned)
    if args.plan_only:
        print(json.dumps({"status": "planned", "groups": len(planned), "root": str(root)}, ensure_ascii=False))
        return 0
    generated_path = root / "generated.jsonl"
    if args.audit_only:
        if not generated_path.is_file():
            raise RuntimeError("--audit-only requires generated.jsonl from the completed generation stage")
        generated = [json.loads(line) for line in generated_path.read_text().splitlines() if line.strip()]
    else:
        editor = None if args.editor_devices or not planned else _editor(args)
        if not args.editor_model:
            raise RuntimeError("--editor-model is required unless --plan-only is set")
        generated = []
        try:
            if args.editor_devices:
                groups_to_generate = parallel_group_generation(planned, root / "generated", kind=args.kind,
                    model=args.editor_model, dtype=args.editor_dtype,
                    devices=[d.strip() for d in args.editor_devices.split(",") if d.strip()])
            else:
                groups_to_generate = iter(planned)
            for index, group in enumerate(groups_to_generate):
                out = root / "generated" / (group.get("twin_id") or group.get("multi_id"))
                if args.kind == "twins":
                    if not args.editor_devices:
                        group = generate_twin_group(group, out, editor, seed=0)
                    checks = group.get("group_checks", {})
                    accepted = bool(checks.get("deterministic", {}).get("pass")
                                    and checks.get("cross_target_preservation", {}).get("pass")
                                    and checks.get("support_gap", {}).get("pass"))
                else:
                    if not args.editor_devices:
                        group = generate_multi_group(group, out, editor, seed=0)
                    joint = group.get("joint_audit", {})
                    accepted = (joint.get("blind_protocol", {}).get("pass") is True and all(x.get("decision") != "fail" for x in joint.get("steps", []))
                                and joint.get("outside_union_support_fraction", 1.0) <= 0.02)
                # Preserve rejected groups too, including the exact gates that failed.
                atomic_json(out / "group.json", group)
                if accepted:
                    generated.append(group)
                print(json.dumps({"generated": index + 1, "accepted": len(generated), "total": len(planned)}, ensure_ascii=False), flush=True)
        finally:
            if hasattr(editor, "pipe"):
                editor.pipe.close()
        generated.sort(key=lambda g: g.get("twin_id") or g.get("multi_id"))
        write_jsonl(generated_path, generated)
    if args.generate_only:
        print(json.dumps({"status": "generated_pending_audit", "groups": len(generated),
                          "manifest": str(generated_path)}, ensure_ascii=False))
        return 0
    auditor = _auditor(args) if generated else None
    try:
        if auditor is not None and generated:
            if args.kind == "twins":
                audit_fn = audit_twin_groups_joint if args.twin_audit_mode == "joint" else audit_twin_groups
                generated = audit_fn(generated, auditor, root / "audit_panels")
            else:
                generated = audit_multi_groups(generated, auditor, root / "audit_panels")
    finally:
        if auditor is not None:
            auditor.close()
    write_jsonl(root / "audited.jsonl", generated)
    final = []
    final_counts = Counter()
    final_relations = Counter()
    relation_target = {'same': int(round(args.target*args.twin_same_category_fraction))}
    relation_target['cross'] = args.target-relation_target['same']
    final_quota = _quota(args.target) if args.kind == "twins" else None
    if args.kind == 'twins' and args.quota_from_selection:
        if len(selected) != args.target:
            raise ValueError('shard quota mode requires target equal to frozen selection size')
        final_quota = dict(selection_operations)
        relation_target = dict(selection_relations)
    for group in generated:
        if args.kind == "twins":
            semantic = group.get("group_checks", {}).get("semantic_audit_pass") is True
            deterministic = group.get("group_checks", {}).get("group_pass", False)
            if not (deterministic and semantic):
                continue
            try:
                validate_twin_record(group)
            except (ValueError, TypeError, KeyError) as exc:
                atomic_json(root / "rejected" / f"{group['twin_id']}_release.json",
                            {"group_id": group["twin_id"], "stage": "release", "reason": str(exc)})
                continue
            if final_quota is not None and final_counts[group["operation"]] >= final_quota[group["operation"]]:
                continue
            relation = group.get('category_relation', 'same')
            if final_relations[relation] >= relation_target.get(relation, 0):
                continue
            final_relations[relation] += 1
            final_counts[group["operation"]] += 1
        else:
            if not group.get("joint_audit", {}).get("pass"):
                continue
            validate_multi_record(group)
            if len(final) >= args.target:
                continue
        final.append(group)
    human_review_summary = {"required": False}
    if args.publish:
        if not args.audit_model:
            raise RuntimeError("--publish requires --audit-model and completed 27B semantic audits")
        if len(final) != args.target:
            raise RuntimeError(f"--publish requires the complete shard: {len(final)}/{args.target} groups")
        if not args.human_review or not args.human_review.is_file():
            raise RuntimeError("--publish requires --human-review with explicit human decisions")
        human_review_summary = {"required": True, **_require_manual_review(
            final, args.human_review, kind=args.kind, split=args.split)}
    final_path = root / "final" / ("twins_train.jsonl" if args.kind == "twins" and args.split == "train" else
                                    "twins_eval.jsonl" if args.kind == "twins" else
                                    "multi_train.jsonl" if args.split == "train" else "multi_eval.jsonl")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    exported = [portable(g, root) for g in final]
    if args.split == "train" and args.reservation:
        assert_no_eval_leakage(exported, args.reservation)
    write_jsonl(final_path, exported)
    atomic_json(root / "release.json", {"groups": len(exported), "requested": args.target,
                                        "accepted_by_category_relation": dict(final_relations) if args.kind == "twins" else None,
                                        "requested_by_category_relation": relation_target if args.kind == "twins" else None,
                                        "accepted_by_operation": dict(final_counts) if args.kind == "twins" else None,
                                        "remaining_by_operation": {op: max(0, n - final_counts[op]) for op, n in final_quota.items()} if final_quota else None,
                                        "human_review": human_review_summary,
                                        "semantic_audit": bool(args.audit_model),
                                        "published": bool(args.publish), "final": str(final_path)})
    print(json.dumps({"status": "generated", "groups": len(exported), "final": str(final_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
