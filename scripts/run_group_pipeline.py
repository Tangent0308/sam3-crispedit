#!/usr/bin/env python3
"""Build resumable twin or adjacent multi-target data in an isolated root.

Examples (planner-only smoke):
  python scripts/run_group_pipeline.py --kind twins --manifest candidates.jsonl \
      --output-root /mnt/.../twins_first5000 --target 8 --plan-only \
      --planner-model /path/to/Qwen3.8-Flash-Next-FP8

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

from sa1b_pipeline.geometry import load_mask, mask_geometry
from sa1b_pipeline.group_generation import generate_multi_group, generate_twin_group
from sa1b_pipeline.group_geometry import normalized_category
from sa1b_pipeline.group_audit import audit_multi_groups, audit_twin_groups
from sa1b_pipeline.group_planning import (build_multi_group, build_twin_group, group_views,
                                           multi_prompt, twin_prompt)
from sa1b_pipeline.group_schema import validate_multi_record, validate_twin_record
from sa1b_pipeline.group_selection import select_multi_candidates, select_twin_candidates
from sa1b_pipeline.io import load_candidates, write_jsonl
from sa1b_pipeline.production_state import atomic_json, immutable_json, portable
from sa1b_pipeline.source_isolation import assert_no_eval_leakage, load_reservation


FROZEN_ROOT = Path("/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014")
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
                value = float(load_mask(obj, (width, height)).sum()) / max(1, width * height)
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
    return VLLMSelector(args.planner_model, tensor_parallel_size=args.planner_tp,
                        max_new_tokens=args.max_new_tokens, batch_size=args.planner_batch_size)


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
                       max_tokens=512, check_operation=False)


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
        three_fraction = min(1.0, max(0.0, float(args.twin_three_member_fraction)))
        pools: dict[str, list[dict[str, Any]]] = {}
        for index, (op, count) in enumerate(quotas.items()):
            pool_count = max(count, int(math.ceil(count * factor)))
            triple_count = int(math.ceil(pool_count * three_fraction))
            pair_count = pool_count - triple_count
            op_rows = []
            if triple_count:
                op_rows.extend(select_twin_candidates(
                    candidates, target=triple_count, operation=op,
                    seed=args.seed + index * 1009 + 17,
                    reserved_source_ids=reservation_ids if args.split == "train" else set(),
                    member_count=3))
            if pair_count:
                op_rows.extend(select_twin_candidates(
                    candidates, target=pair_count, operation=op,
                    seed=args.seed + index * 1009,
                    reserved_source_ids=reservation_ids if args.split == "train" else set()))
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

        # Select jointly across operation/member cells so the distribution is
        # an actual invariant, rather than a side effect of concatenating and
        # truncating operation-specific pools.  The scope targets mirror the
        # data specification: >=40% part-level, about 10% same-host parts.
        scope_targets = {
            "part_on_same_host": int(round(args.target * 0.10)),
            "part_on_different_hosts": int(math.ceil(args.target * 0.30)),
        }
        scope_targets["object"] = args.target - sum(scope_targets.values())
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
                if len(row["objects"]) != member_count:
                    continue
                sid = str(row["candidate"]["source_id"])
                group_key = (sid, tuple(sorted(str(o["object_id"]) for o in row["objects"])), op)
                if seen_sources[sid] >= 2 or group_key in seen_groups:
                    continue
                scope = str(row.get("scope_level", "object"))
                if scope_counts[scope] >= scope_targets.get(scope, 0):
                    continue
                area_bucket = str(row.get("_selection_area_bucket", "lt0.5%"))
                if area_counts[area_bucket] >= area_targets[area_bucket]:
                    continue
                category = normalized_category(row["objects"][0].get("category", "object")).split(",", 1)[0].strip()
                if category_counts[category] >= category_cap:
                    continue
                deficit = scope_targets.get(scope, 0) - scope_counts[scope]
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
                    if len(row["objects"]) != member_count:
                        continue
                    sid = str(row["candidate"]["source_id"])
                    group_key = (sid, tuple(sorted(str(o["object_id"]) for o in row["objects"])), op)
                    if seen_sources[sid] >= 2 or group_key in seen_groups:
                        continue
                    scope = str(row.get("scope_level", "object"))
                    area_bucket = str(row.get("_selection_area_bucket", "lt0.5%"))
                    category = normalized_category(row["objects"][0].get("category", "object")).split(",", 1)[0].strip()
                    violation = (int(scope_counts[scope] >= scope_targets.get(scope, 0))
                                 + int(area_counts[area_bucket] >= area_targets[area_bucket])
                                 + int(category_counts[category] >= category_cap))
                    fallback.append((violation, scope_counts[scope], area_counts[area_bucket],
                                     category_counts[category], sid, row))
                if not fallback:
                    shortages[f"{op}:{member_count}"] += 1
                    continue
                _, _, _, _, sid, row = min(fallback, key=lambda x: x[:-1])
                relaxed_quota_count += 1
            else:
                # Prefer scopes and area buckets furthest below their targets,
                # then underused categories; source ID makes the final choice
                # deterministic.
                _, _, _, sid, row = max(candidates_for_cell, key=lambda x: (x[0], x[1], -x[2], x[3]))
            category = normalized_category(row["objects"][0].get("category", "object")).split(",", 1)[0].strip()
            scope = str(row.get("scope_level", "object"))
            group_key = (sid, tuple(sorted(str(o["object_id"]) for o in row["objects"])), op)
            seen_sources[sid] += 1
            seen_groups.add(group_key)
            scope_counts[scope] += 1
            area_counts[str(row["_selection_area_bucket"])] += 1
            category_counts[category] += 1
            selected.append(row)
        if shortages or len(selected) != args.target or relaxed_quota_count > max(1, int(math.ceil(args.target * 0.005))):
            raise RuntimeError(
                f"twin selection cannot satisfy quotas: shortages={dict(shortages)} "
                f"selected={len(selected)}/{args.target} scopes={dict(scope_counts)} "
                f"areas={dict(area_counts)} relaxed={relaxed_quota_count}")
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["twins", "multi"], required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--reservation", type=Path, help="source isolation reservation JSON")
    p.add_argument("--planner-model")
    p.add_argument("--planner-backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--planner-device-map", default="auto",
                   help="Transformers device map for the HF planner; auto shards large MoE models across visible GPUs")
    p.add_argument("--planner-tp", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--planner-batch-size", type=int, default=8,
                   help="number of multimodal planning requests submitted per vLLM batch")
    p.add_argument("--editor-model")
    p.add_argument("--editor-dtype", default="bf16")
    p.add_argument("--audit-model")
    p.add_argument("--audit-batch-size", type=int, default=4)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--selection-only", action="store_true",
                   help="select and persist grouped inputs without loading a planner model")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--max-groups", type=int)
    p.add_argument("--candidate-pool-factor", type=float, default=3.0,
                   help="oversample planning candidates so rejected plans can be refilled")
    p.add_argument("--twin-three-member-fraction", type=float, default=0.15,
                   help="target fraction of three-member twin groups")
    p.add_argument("--fast-selection", action="store_true",
                   help="use bbox-only multi-target capacity selection; exact geometry remains a release gate")
    args = p.parse_args(argv)
    if args.target < 1:
        p.error("--target must be positive")
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if root == FROZEN_ROOT or FROZEN_ROOT in root.parents or root in FROZEN_ROOT.parents:
        raise RuntimeError("refusing to write inside the protected 50k run root")
    reservation_ids, reservation_hashes = load_reservation(args.reservation) if args.reservation else (set(), set())
    immutable_json(root / "config.json", {"kind": args.kind, "target": args.target, "split": args.split,
                                           "seed": args.seed, "manifest": str(args.manifest.resolve()),
                                           "reservation": str(args.reservation.resolve()) if args.reservation else None,
                                           "protocol": "1.0", "mask_semantics": "1.1", "steps": 40})
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
    if args.split == "train" and args.reservation:
        assert_no_eval_leakage((r["candidate"] for r in selected), args.reservation)
    selection_operations = Counter()
    selection_scopes = Counter()
    selection_members = Counter()
    selection_areas = Counter()
    selection_categories = Counter()
    for row in selected:
        if args.kind == "twins":
            selection_operations[str(row.get("operation"))] += 1
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
                                          "source_count": len({r["candidate"]["source_id"] for r in selected}),
                                          "source_ids": sorted({r["candidate"]["source_id"] for r in selected})})
    if args.selection_only:
        print(json.dumps({"status": "selected", "groups": len(selected),
                          "selection": str(selected_path),
                          "summary": str(root / "selection.json")}, ensure_ascii=False))
        return 0
    planner = _planner(args)
    plans_dir = root / "planned"; plans_dir.mkdir(exist_ok=True)
    planned = []
    try:
        pending = []
        for index, selection in enumerate(selected):
            gid = _group_id(args.kind, selection)
            path = plans_dir / f"{gid}.json"
            if path.exists():
                planned.append(json.loads(path.read_text())); continue
            objects = selection["objects"]
            views, mask_paths, view_digest = group_views(selection["candidate"], objects, root / "planning_views")
            geometries = _geometry(mask_paths)
            if planner is None:
                # Generation-only resume is allowed: selections rejected by a
                # prior planning batch have no plan file and must simply be
                # skipped. A fresh --plan-only run still fails closed below.
                if args.plan_only:
                    raise RuntimeError("planner model is required to create new group plans")
                continue
            if args.kind == "twins":
                op = selection["operation"]
                prompt = twin_prompt(op, objects)
            else:
                assignments = {str(obj["object_id"]): op for obj, op in zip(objects, selection["operations"])}
                order = [f"U{i+1}" for i in range(len(objects))]
                prompt = multi_prompt(assignments, objects, order)
            pending.append({"index": index, "selection": selection, "gid": gid, "path": path,
                            "objects": objects, "views": views, "mask_paths": mask_paths,
                            "geometries": geometries, "view_digest": view_digest,
                            "prompt": prompt, "assignments": assignments if args.kind == "multi" else None})
        if pending:
            raws = planner.generate_multimage([(item["views"], item["prompt"]) for item in pending])
            if len(raws) != len(pending):
                raise RuntimeError("planner returned an incomplete batch")
            for item, raw in zip(pending, raws):
                selection, objects = item["selection"], item["objects"]
                try:
                    if args.kind == "twins":
                        group = build_twin_group(selection["candidate"], objects, raw, item["mask_paths"],
                                                 split=args.split, scope_level=selection["scope_level"], group_id=item["gid"],
                                                 geometry=item["geometries"])
                        group["selection_evidence"] = selection["gate"]
                    else:
                        group = build_multi_group(selection["candidate"], objects, raw, item["mask_paths"],
                                                  split=args.split, group_id=item["gid"], geometry=item["geometries"],
                                                  assignments=item["assignments"])
                        group["adjacent_pairs"] = selection["adjacent_pairs"]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    # One malformed/infeasible completion must not discard
                    # valid siblings from the same continuous batch. Keep the
                    # raw answer and selection evidence for audit/retry.
                    rejection = {"group_id": item["gid"], "status": "plan_rejected",
                                 "reason": str(exc), "raw": raw, "selection": selection}
                    atomic_json(root / "rejected" / f"{item['gid']}.json", rejection)
                    print(json.dumps({"planned": len(planned), "total": len(selected),
                                      "id": item["gid"], "status": "rejected", "reason": str(exc)},
                                     ensure_ascii=False), flush=True)
                    continue
                group["planning"] = {"model": getattr(planner, "model_id", "unknown"), "calls": 1,
                                      "views": item["views"], "view_digest": item["view_digest"], "raw": raw}
                atomic_json(item["path"], group)
                planned.append(group)
                print(json.dumps({"planned": len(planned), "total": len(selected), "id": item["gid"]}, ensure_ascii=False), flush=True)
    finally:
        if planner is not None and hasattr(planner, "close"):
            planner.close()
    write_jsonl(root / "planned.jsonl", planned)
    if args.plan_only:
        print(json.dumps({"status": "planned", "groups": len(planned), "root": str(root)}, ensure_ascii=False))
        return 0
    editor = _editor(args)
    if editor is None:
        raise RuntimeError("--editor-model is required unless --plan-only is set")
    generated = []
    try:
        for index, group in enumerate(planned):
            out = root / "generated" / (group.get("twin_id") or group.get("multi_id"))
            if args.kind == "twins":
                group = generate_twin_group(group, out, editor, seed=0)
                checks = group.get("group_checks", {})
                if (not checks.get("deterministic", {}).get("pass")
                        or not checks.get("cross_target_preservation", {}).get("pass")
                        or not checks.get("support_gap", {}).get("pass")):
                    continue
                generated.append(group)
            else:
                group = generate_multi_group(group, out, editor, seed=0)
                joint = group.get("joint_audit", {})
                if not all(x.get("decision") != "fail" for x in joint.get("steps", [])) or joint.get("outside_union_support_fraction", 1.0) > 0.02:
                    continue
                generated.append(group)
            print(json.dumps({"generated": index + 1, "accepted": len(generated), "total": len(planned)}, ensure_ascii=False), flush=True)
    finally:
        if hasattr(editor, "pipe"):
            editor.pipe.close()
    auditor = _auditor(args)
    try:
        if auditor is not None and generated:
            if args.kind == "twins":
                generated = audit_twin_groups(generated, auditor, root / "audit_panels")
            else:
                generated = audit_multi_groups(generated, auditor, root / "audit_panels")
    finally:
        if auditor is not None:
            auditor.close()
    final = []
    final_counts = Counter()
    final_quota = _quota(args.target) if args.kind == "twins" else None
    for group in generated:
        if args.kind == "twins":
            semantic = group.get("group_checks", {}).get("semantic_audit_pass") is True
            deterministic = group.get("group_checks", {}).get("group_pass", False)
            if not (deterministic and semantic):
                continue
            validate_twin_record(group)
            if final_quota is not None and final_counts[group["operation"]] >= final_quota[group["operation"]]:
                continue
            final_counts[group["operation"]] += 1
        else:
            if not group.get("joint_audit", {}).get("pass"):
                continue
            validate_multi_record(group)
            if len(final) >= args.target:
                continue
        final.append(group)
    if args.publish and not args.audit_model:
        raise RuntimeError("--publish requires --audit-model and completed 27B semantic audits")
    final_path = root / "final" / ("twins_train.jsonl" if args.kind == "twins" and args.split == "train" else
                                    "twins_eval.jsonl" if args.kind == "twins" else
                                    "multi_train.jsonl" if args.split == "train" else "multi_eval.jsonl")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    exported = [portable(g, root) for g in final]
    if args.split == "train" and args.reservation:
        assert_no_eval_leakage(exported, args.reservation)
    write_jsonl(final_path, exported)
    atomic_json(root / "release.json", {"groups": len(exported), "requested": args.target,
                                        "human_review": False, "semantic_audit": bool(args.audit_model),
                                        "published": bool(args.publish), "final": str(final_path)})
    print(json.dumps({"status": "generated", "groups": len(exported), "final": str(final_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
