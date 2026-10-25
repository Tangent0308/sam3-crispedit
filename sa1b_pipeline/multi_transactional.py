"""Commit only individually audited edits; rejected steps never advance the image."""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image

from .geometry import load_mask
from .group_audit import _single_member_audits, audit_committed_multi_groups
from .group_generation import (_render_member, _unit_dict, _support_for, _row,
                               _write_mask, _union_support, _outside_fraction,
                               multi_support_preflight)
from .group_geometry import adjacent_pair_gate
from .group_schema import validate_multi_record
from .production_state import atomic_json, immutable_json, file_hash, digest
from .source_regions import resolve_source_region, blind_group_check
from .validate import deterministic_check

POLICY = "keep-passed-v1"


def member_passes(audit):
    return audit.get("decision") == "pass" and all(audit.get(k) is True for k in
        ("operation_correct", "target_done", "non_target_preserved", "natural", "host_integrity"))


def _evidence_hashes(member):
    hashes = {p: file_hash(p) for key in ("step_input_image", "step_output_image", "mask",
            "support_mask", "chain_execution_mask", "chain_protected_mask")
            if (p := member.get(key))}
    for key in ("dependency_overlap_mask_path", "dependency_interface_mask_path"):
        if path := member.get("unit", {}).get(key): hashes[path] = file_hash(path)
    for evidence in (member.get("generation_context", {}).get("crop_evidence") or {}).values():
        if file_hash(evidence["path"]) != evidence["sha256"]:
            raise ValueError("raw crop evidence changed")
        hashes[evidence["path"]] = evidence["sha256"]
    return hashes


def _verify_hashes(hashes):
    if any(file_hash(p) != expected for p, expected in hashes.items()):
        raise ValueError("transactional checkpoint image changed")


def _retained_adjacency(group):
    size = Image.open(group["source_image"]).size
    objects = [{"object_id": u["object_id"], "mask_path": u.get("mask_v1") or u["mask"],
                "category": u.get("scope", "object")} for u in group["units"]]
    found = []
    for a, b in combinations(objects, 2):
        gate = adjacent_pair_gate(a, b, image_size=size)
        if gate["eligible"]:
            found.append({"object_ids": [a["object_id"], b["object_id"]],
                          **gate, "geometry_exact": True})
    return found


def run_committed_step(group, member, current, original, step_dir, editor, auditor, *,
                       index, seed=0, save_crop_evidence=False, defer_semantic=False):
    """Generate/audit one resumable attempt; the caller alone advances its chain."""
    step_dir = Path(step_dir)
    uid = member["unit_id"]
    size = (original.shape[1], original.shape[0])
    ledger = step_dir / "decision.json"
    input_hash = file_hash(current)
    if ledger.exists():
        decision = json.loads(ledger.read_text())
        if decision["input_sha256"] != input_hash or decision["unit_id"] != uid:
            raise ValueError("transactional step input changed")
        if decision.get('defer_semantic', False) != defer_semantic:
            raise ValueError('transactional step audit policy changed')
        payload = {k: v for k, v in decision.items() if k != 'ledger_sha256'}
        if decision.get('ledger_sha256') != digest(payload):
            raise ValueError("transactional decision changed")
        _verify_hashes(decision["image_hashes"])
        member = decision["member"]
    else:
        started = time.monotonic()
        member = deepcopy(member); unit = _unit_dict(member)
        own = load_mask({"mask_path": member.get("mask_v1") or member["mask"]}, size)
        # Other selectors are protected outside this target's own source extent;
        # shared source pixels belong to the serial component, not two writers.
        # Only committed edits contribute prior-change protection.
        before = np.asarray(Image.open(current).convert("RGB"), dtype=np.int16)
        prior = np.max(np.abs(original - before), axis=2) > 12
        prior_protected = prior & ~own
        if unit.get("dependency_overlap_mask_path"):
            permission = load_mask({"mask_path": unit["dependency_overlap_mask_path"]}, size)
            prior_protected &= ~permission
        protected = load_mask({"mask_path": unit["protected_mask_path"]}, size) | prior_protected
        from .specialist_editors import writable_mask
        writable = writable_mask(unit, size, require_add_placement=False)
        removal_effect = getattr(editor, "removal_effect", None)
        if removal_effect is not None and unit["operation"] == "remove":
            # Opt-in: the removed object's own shadow/reflection joins this step's writable support.
            unit["protected_mask_path"] = _write_mask(step_dir / "chain_protected.png", protected)
            effect_path, effect_info = removal_effect(current, unit, step_dir / "removal_effect.png")
            unit["removal_effects"] = effect_info
            if effect_path:
                writable |= load_mask({"mask_path": effect_path}, size)
                unit["removal_effect_mask_path"] = effect_path
        unit["execution_mask_path"] = _write_mask(step_dir / "chain_execution.png", writable & ~protected)
        unit["protected_mask_path"] = _write_mask(step_dir / "chain_protected.png", protected)
        if save_crop_evidence:
            unit["diagnostic_dir"] = str(step_dir / "crop_evidence")
        output = str(step_dir / "edited.png")
        context = _render_member(current, unit, Path(output), editor, seed)
        generation_seconds = time.monotonic() - started
        support = _support_for(editor, current, unit, step_dir / "support_mask.png")
        member.setdefault("unit", {}).update({k: unit[k] for k in ("execution_mask_path", "protected_mask_path")})
        resolve_source_region(member, current, output, step_dir)
        row = _row(group, member, current, output, support, input_image=current, output_image=output, step=index,
                   generation_context=context)
        check = deterministic_check(row, semantic_low_change=True)
        member.update(step_input_image=current, step_output_image=output, support_mask=support,
                      step_audit=check, generation_context=context, seed=seed,
                      generation_seconds=generation_seconds,
                      chain_execution_mask=unit["execution_mask_path"],
                      chain_protected_mask=unit["protected_mask_path"])
        reasons = list(check.get("failure_reasons") or [])
        eligible = check.get("decision") == "pending_semantic_audit" and member.get("blind_eligible") is True
        if not member.get("blind_eligible"):
            reasons.append("unresolved_source_region")
        if eligible and defer_semantic:
            # Provisional generation checkpoint, never a semantic acceptance.
            accepted = True
            member['semantic_audit_pending'] = True
        elif eligible:
            t = time.monotonic()
            _single_member_audits([{**group, "units": [member]}], auditor, step_dir / "audit_panels", kind="multi")
            member["audit_seconds"] = time.monotonic() - t
            accepted = member_passes(member["semantic_audit"])
            reasons.extend(member["semantic_audit"].get("failure_reasons") or [])
            if not accepted and not reasons:
                reasons.append("step_semantic_check_failed")
        else:
            accepted = False
        decision = {"unit_id": uid, "input_sha256": input_hash, "accepted": accepted,
                    "reasons": reasons, "member": member, "image_hashes": _evidence_hashes(member)}
        decision['defer_semantic'] = defer_semantic
        decision['ledger_sha256'] = digest(decision)
        atomic_json(ledger, decision)
    return decision


def generate_committed_multi_group(plan, directory, editor, auditor, *, seed=0, save_crop_evidence=False):
    """One attempt per original target, one joint audit, immutable commit ledgers.

    A failed semantic decision is final for this run. Infrastructure exceptions
    propagate and can resume; they must never be mistaken for rejected edits.
    """
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    if plan.get("commit_policy") or plan.get("joint_target"):
        raise ValueError("transactional generation requires an ungenerated plan")
    ids = [u["unit_id"] for u in plan["units"]]
    order = plan["generation_order"]
    if len(ids) != len(set(ids)) or len(order) != len(ids) or set(order) != set(ids):
        raise ValueError("invalid transactional generation order")
    code_hashes = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    contract = {"policy": POLICY, "plan": plan, "seed": seed,
                "source_sha256": file_hash(plan["source_image"]),
                "mask_sha256": {u["unit_id"]: file_hash(u.get("mask_v1") or u["mask"]) for u in plan["units"]},
                "editor": editor.config, "auditor": auditor.config, "code": code_hashes,
                "save_crop_evidence": save_crop_evidence}
    immutable_json(root / "transaction_contract.json", contract)
    final_path = root / "committed.json"
    if final_path.exists():
        saved = json.loads(final_path.read_text()); _verify_hashes(saved["image_hashes"])
        if saved["group_sha256"] != digest(saved["group"]):
            raise ValueError("transactional final record changed")
        return saved["group"]
    group = deepcopy(plan)
    group["commit_policy"] = POLICY
    preflight = multi_support_preflight(group, root)
    group["support_preflight"] = preflight
    by_id = {u["unit_id"]: u for u in group["units"]}
    original = np.asarray(Image.open(group["source_image"]).convert("RGB"), dtype=np.int16)
    current = group["source_image"]
    kept, decisions = [], []
    for index, uid in enumerate(order, 1):
        step_dir = root / f"step_{index:02d}_{uid}"
        decision = run_committed_step(group, by_id[uid], current, original, step_dir,
                                      editor, auditor, index=index, seed=seed,
                                      save_crop_evidence=save_crop_evidence)
        member = decision["member"]
        decisions.append({"unit_id": uid, "operation": member["operation"],
                          "accepted": decision["accepted"], "reasons": decision["reasons"],
                          "checkpoint": str(step_dir / "decision.json")})
        if decision["accepted"]:
            kept.append(member); current = member["step_output_image"]
        atomic_json(root / "progress.json", {"attempted": index, "planned": len(order),
                    "kept": len(kept), "current_image": current, "decisions": decisions})
        print(json.dumps({"group": group["multi_id"], "step": uid, "commit": decision["accepted"],
                          "kept": len(kept), "attempted": index, "planned": len(order)}), flush=True)
    return finalize_committed_group(plan, group, kept, decisions, current, root, auditor)


def finalize_committed_group(plan, group, kept, decisions, current, root, auditor, *, component_only=False):
    """Rebuild retained annotations and run exactly one final audit when eligible."""
    ids = [u["unit_id"] for u in plan["units"]]
    order = plan["generation_order"]
    keep_ids = {u["unit_id"] for u in kept}
    # Preserve canonical unit ordering and stable IDs, not the generation order.
    group["units"] = [next(u for u in kept if u["unit_id"] == uid) for uid in ids if uid in keep_ids]
    group.update(n_targets=len(kept), operations=[u["operation"] for u in group["units"]],
                 generation_order=[uid for uid in order if uid in keep_ids], joint_target=current)
    group["retention"] = {"parent_id": plan["multi_id"], "planned_n_targets": len(ids),
                          "planned_operations": plan["operations"], "decisions": decisions,
                          "kept_unit_ids": group["generation_order"],
                          "dropped_unit_ids": [uid for uid in plan["generation_order"] if uid not in keep_ids]}
    group["adjacent_pairs"] = []  # Legacy field; no distance requirement or mask-distance IO.
    with Image.open(group["source_image"]) as source:
        shape = (source.height, source.width)
    allowed = _union_support(group["source_image"], kept) if kept else np.zeros(shape, dtype=bool)
    group["joint_audit"] = {"pass": False, "status": "pending_joint_audit",
                           "blind_protocol": blind_group_check(group),
                           "steps": [u["step_audit"] for u in kept],
                           "outside_union_support_fraction": _outside_fraction(group["source_image"], current, allowed)}
    try:
        validate_multi_record(group)
    except (ValueError, KeyError, TypeError) as exc:
        group["joint_audit"].update(status="retained_contract_rejected", failure_reasons=[str(exc)])
    else:
        if component_only:
            from .multi_component_audit import verify_component_fusion
            proof = verify_component_fusion(group)
            group['joint_audit'].update(
                status='complete', audit_mode='component-final-v2',
                group_mllm_calls=0, fusion_verification=proof,
                member_semantic_audit_pass=all(member_passes(u.get('semantic_audit', {})) for u in kept),
                **{'pass': bool(proof['pass'] and group['joint_audit']['blind_protocol']['pass']
                               and all(member_passes(u.get('semantic_audit', {})) for u in kept))})
        elif group["joint_audit"]["outside_union_support_fraction"] <= .02:
            group = audit_committed_multi_groups([group], auditor, root / "audit_panels")[0]
            group["joint_audit"]["status"] = "complete"
        else:
            group["joint_audit"].update(status="outside_retained_support", failure_reasons=["outside_retained_support"])
    hashes = {p: h for u in kept for p, h in _evidence_hashes(u).items()}
    hashes[group["source_image"]] = file_hash(group["source_image"])
    hashes[current] = file_hash(current)
    atomic_json(root / "group.json", group)
    atomic_json(root / "committed.json", {"group": group, "group_sha256": digest(group), "image_hashes": hashes})
    return group
