"""27B semantic audits for grouped records.

The deterministic checks live in :mod:`group_generation`; this module keeps
the visual decisions auditable and all-or-nothing at group level.
"""
from __future__ import annotations

from .instruction_protocol import BLIND_RULE

import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .audit_backend import VLLMAuditor
from .validate import _target_panel, audit_prompt, deterministic_check


TWIN_GROUP_PROMPT = '''You are auditing one counterfactual twin group. Image 1 is the clean original full scene. Image 2 contains one row per member: SOURCE original crop (left), RESULT edited crop (middle), and MASK ONLY black-and-white selector (right). Judge generated appearance ONLY from the middle RESULT column. The right column is never a result. White selects the source object and black excludes it. The source scene and the member result images are provided. Different categories are allowed; comparable means all whole objects or all parts, with the shared edit natural for every member. The same no-ref edit was intended for each marked member, while each ref identifies a different original instance. Decide whether every reference is unique, every assigned operation is correct, every requested edit is complete and natural, and the member results are comparable. The group fails if any member is wrong, if a result edits another member, if the operation is misclassified, or if the shared request is not natural for all. Before deciding, match each white selector to the actual source part and exclude black holes and neighboring objects. Check every ref against all similar instances in the full scene; image-left/image-right refer to the original scene, never crop row order. For add, verify direct physical contact with the selected host/site, not a nearby surface. For remove, check for target remnants and unsupported remnants; deleting a wheel, door, support or other functional part is not natural unless the image clearly supports it. For replace, require a visibly different physical form/design, not only a color or material change. For attribute, verify the requested surface changed while geometry, markings and unrelated materials remain. Judge only visible evidence and fail a member with a concrete mismatch. Return JSON only: {"decision":"pass|fail","unique_refs":true,"operations_correct":true,"all_targets_done":true,"members_preserved":true,"comparable":true,"natural":true,"failure_reasons":[],"evidence":"brief"}.'''

MULTI_GROUP_PROMPT = '''You are auditing a joint multi-target edit group. The original scene, each chain step, and the final result are provided. Check that each original ref identifies one target, every assigned operation is correct and completed in its step and remains correct later, adjacent targets do not bleed into each other, and the final image is natural. The group fails if any operation type is wrong. Return JSON only: {"decision":"pass|fail","unique_refs":true,"operations_correct":true,"all_targets_done":true,"no_crosstalk":true,"outside_preserved":true,"natural":true,"failure_reasons":[],"evidence":"brief"}.'''

TWIN_REQUIRED_AUDIT_FIELDS = ("unique_refs", "operations_correct", "all_targets_done", "members_preserved", "comparable", "natural")
MULTI_REQUIRED_AUDIT_FIELDS = ("unique_refs", "operations_correct", "all_targets_done", "no_crosstalk", "outside_preserved", "natural")


def _semantic_group_pass(audit: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return isinstance(audit, dict) and audit.get("decision") == "pass" and all(
        audit.get(field) is True for field in fields)


def _call_batched(model: VLLMAuditor, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the auditor's configured batch size while preserving job order."""
    size = max(1, int(getattr(model, "batch_size", 1)))
    results = []
    for start in range(0, len(jobs), size):
        batch = jobs[start:start + size]
        outputs = model.call_batch(batch)
        if len(outputs) != len(batch):
            raise RuntimeError("group audit backend returned an incomplete batch")
        if [r.get("job_id") for r in outputs] != [j["job_id"] for j in batch]:
            raise RuntimeError("group audit backend returned mismatched job IDs")
        results.extend(outputs)
    return results


def _marked_scene(group: dict[str, Any]) -> str | None:
    views = (group.get("planning") or {}).get("views") or []
    if views and Path(views[0]).is_file():
        marked = Path(views[0]).with_name("marked_scene.png")
        return str(marked if marked.is_file() else views[0])
    return None


def _target_contract(group: dict[str, Any], *, kind: str) -> str:
    rows = group.get("members", []) if kind == "twins" else group.get("units", [])
    lines = []
    for index, member in enumerate(rows, 1):
        lines.append(
            f"{member.get('unit_id', f'U{index}')}: object_id={member.get('object_id')}; "
            f"ref={member.get('ref') or member.get('ref_phrase')}; "
            f"noref={member.get('instruction_noref')}; "
            f"site={(member.get('instruction_spec') or {}).get('placement', {}).get('site', '')}; "
            f"operation={member.get('operation')}; instruction={member.get('instruction') or member.get('instruction_ref')}"
        )
    return "\n".join(lines)


def _member_comparison(paths: list[str], out: Path, *, crop_panels: bool = False) -> str:
    """Pack every twin member panel into one audit image.

    The 27B adapter accepts at most three image inputs. Passing the source and
    one tiled comparison therefore keeps all two/three members visible instead
    of silently dropping the third member from a triple group.
    """
    panels = [Image.open(path).convert("RGB") for path in paths]
    tile_w, tile_h = (1536, 480) if crop_panels else (768, 520)
    tiles = []
    for index, panel in enumerate(panels, 1):
        panel.thumbnail((tile_w, tile_h - 64), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        ImageDraw.Draw(tile).text((12, 6), f"U{index} | MEMBER {index}", fill="black",
                                  font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36))
        tile.paste(panel, ((tile_w - panel.width) // 2, 64 + (tile_h - 64 - panel.height) // 2))
        tiles.append(tile)
    columns = 1 if crop_panels else min(2, max(1, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    panel = Image.new("RGB", (columns * tile_w, rows * tile_h), (235, 235, 235))
    for index, tile in enumerate(tiles):
        panel.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out, quality=94, subsampling=0)
    return str(out)


def _member_audit_row(group: dict[str, Any], member: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """Adapt a generated member/step to the ordinary 50k audit contract."""
    if kind == "twins":
        source = group["source_image"]
        edited = member.get("edited_image")
    else:
        source = member.get("step_input_image")
        edited = member.get("step_output_image")
    if not source or not edited:
        raise ValueError("generated member is missing single-step audit images")
    return {"source_image": source, "edited_image": edited,
            "mask": member["mask"], "operation": member["operation"],
            "instruction": member["instruction"],
            "instruction_spec": member.get("instruction_spec", {}),
            "support_mask": member.get("support_mask")}


def _single_member_audits(groups: list[dict[str, Any]], model: VLLMAuditor,
                          panel_dir: Path, *, kind: str) -> list[list[dict[str, Any]]]:
    """Run the standard operation-aware 27B audit for every member/step."""
    jobs = []
    locations = []
    for group in groups:
        members = group.get("members", []) if kind == "twins" else group.get("units", [])
        for member in members:
            row = _member_audit_row(group, member, kind=kind)
            panel = _target_panel(row, panel_dir, binary_locator=True)
            jobs.append({"job_id": f"{group.get('twin_id') or group.get('multi_id')}::{member['unit_id']}",
                         "images": [row["source_image"], row["edited_image"], panel],
                         "prompt": audit_prompt(row["instruction"], operation=row["operation"], binary_locator=True)})
            locations.append((group, member))
    results = _call_batched(model, jobs) if jobs else []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for (group, member), result in zip(locations, results):
        audit = result.get("semantic_audit", {})
        member["semantic_audit"] = audit
        gid = str(group.get("twin_id") or group.get("multi_id"))
        by_group.setdefault(gid, []).append(audit)
    ordered = []
    for group in groups:
        gid = str(group.get("twin_id") or group.get("multi_id"))
        ordered.append(by_group.get(gid, []))
    return ordered


def audit_twin_groups(groups: list[dict[str, Any]], model: VLLMAuditor, panel_dir: str | Path) -> list[dict[str, Any]]:
    panel_dir = Path(panel_dir)
    member_audits = _single_member_audits(groups, model, panel_dir / "members", kind="twins")
    jobs, refs = [], []
    for group in groups:
        rows = []
        for member in group["members"]:
            row = {"source_image": group["source_image"], "edited_image": member["edited_image"],
                   "mask": member["mask"], "operation": group["operation"],
                   "instruction": member["instruction"], "instruction_spec": member["instruction_spec"],
                   "support_mask": member["support_mask"]}
            rows.append(_target_panel(row, panel_dir, binary_locator=True))
        comparison = _member_comparison(rows,
                                        panel_dir / f"{group['twin_id']}_comparison.jpg", crop_panels=True)
        refs.append(group)
        # Colored source overlays were repeatedly mistaken for edited pixels.
        # Clean context plus labeled clean/binary crops localize without tint.
        images = [group["source_image"], comparison]
        jobs.append({"job_id": group["twin_id"], "images": images[:3],
                     "prompt": TWIN_GROUP_PROMPT + "\n" + BLIND_RULE + "\nTarget contract:\n" +
                     _target_contract(group, kind="twins")})
    # The source plus one tiled comparison stays within the three-image limit
    # while showing every member, including optional three-member groups.
    results = []
    for group, audits, result in zip(refs, member_audits, _call_batched(model, jobs)):
        audit = result["semantic_audit"]
        checks = group.setdefault("group_checks", {})
        member_pass = bool(audits) and all(
            item.get("decision") == "pass"
            and item.get("operation_correct") is True
            and all(item.get(key) is True for key in ("target_done", "non_target_preserved", "natural"))
            for item in audits)
        checks["member_semantic_audit_pass"] = member_pass
        checks["semantic_audit"] = audit
        semantic_pass = _semantic_group_pass(audit, TWIN_REQUIRED_AUDIT_FIELDS)
        checks["semantic_audit_pass"] = semantic_pass
        checks["group_pass"] = bool(
            member_pass
            and checks.get("blind_disambiguation", {}).get("pass") is True
            and
            checks.get("deterministic", {}).get("pass") is True
            and checks.get("cross_target_preservation", {}).get("pass") is True
            and checks.get("support_gap", {}).get("pass") is True
            and semantic_pass)
        results.append(group)
    return results


JOINT_TWIN_PROMPT = '''Audit one counterfactual twin group independently of the planner, using the photographs.
Image 1 is the clean full source. Image 2 has one row per member: SOURCE / RESULT / BINARY MASK.
Image 3 shows full member results. Each result starts from the SAME source and edits ONE member.
For each row identify the actual white-mask entity, compare SOURCE against RESULT, then judge.
Judge the edited object AND its immediate physical context; never repeat the requested change as an observation.

Reject concrete serious defects: wrong/ambiguous referent, wrong source region or extent, unchanged target,
wrong operation, missing substantial requested content, editing another member, major collateral damage,
implausible physical support/contact, or a shared request inappropriate for any member.
For remove check visible remnants, occupied seats and load-bearing/support relations. For replace check
actual shape change and whether the replacement fits each location's function and mounting; a generic
handle is not automatically appropriate to every host. Attribute must be a plausible appearance change,
not painted-looking food/skin; preserve glass, labels and unrelated materials. For add check the actual
contact footprint on the selected source site, size and support, not just proximity to the host.
Check refs against ALL similar objects in the full source, not just these members. Small/clipped/occluded
objects are valid if identity, scope and the edit are clear. Different categories are valid when the SAME
request is natural for both. Do not demand identical object names or identical attachment-site names.
shared_edit_valid requires comparable practical suitability of the EXACT payload at every intended site;
mere physical possibility is insufficient if ordinary use clearly favors one member (e.g. a water pitcher
on a dining table versus a chair seat). The independent blind probe's example categories are incomplete;
verify these actual members yourself, rather than treating blind ambiguity as proof of shared suitability.

Do NOT fail for mild texture/edge variation, minor local lighting changes, subjective style preferences,
or a slightly longer but clear instruction. Put those in warnings. Floating/unsupported objects, missing edits and wrong targets are ALWAYS failures, never minor warnings. Do not require physical perfection.
Uncertain means the CORE edit/reference cannot be judged, not inability to prove every pixel unchanged.
For each member write short observed_entity, ref_entity, observed_change (max 12 words each), followed by checks.
If a check fails give the visible contradiction; pass when the core requirement is visibly met.
Return JSON only, EVERY member in order:
{"members":[{"unit_id":"U1","observed_entity":"...","ref_entity":"...","observed_change":"...",
"checks":{"reference_correct":"pass|fail|uncertain","single_instance":"pass|fail|uncertain",
"scope_correct":"pass|fail|uncertain","operation_correct":"pass|fail|uncertain",
"target_done":"pass|fail|uncertain","non_target_preserved":"pass|fail|uncertain",
"natural":"pass|fail|uncertain"},"failure_reasons":[],"warnings":[]}],
"comparable_granularity":"pass|fail|uncertain","shared_edit_valid":"pass|fail|uncertain","evidence":"brief"}.
'''

JOINT_MEMBER_FIELDS = ("reference_correct", "single_instance", "scope_correct", "operation_correct",
                       "target_done", "non_target_preserved", "natural")


def _acknowledged_hard_defects(warnings):
    """Do not downgrade an explicitly reported serious defect to a soft note.

    This does not invent visual evidence: it only resolves contradictory model
    output. Negated statements are ignored. Style/texture comments stay soft.
    """
    defects=[]
    for item in warnings if isinstance(warnings,list) else []:
        text=str(item).strip();low=text.lower()
        if re.search(r"\b(no|without|not) (?:visible |obvious |significant |any )?(?:floating|unsupported|wrong|missing)\b",low):continue
        if (re.search(r"\b(?:person|object|item|part|plant|cup|glass|body)\b.{0,55}\b(?:floating|unsupported)\b",low)
                or re.search(r"\bwrong (?:target|object|region|referent)\b",low)
                or re.search(r"\b(?:target|object|item)\b.{0,40}\b(?:unchanged|not removed|not replaced|not edited)\b",low)):
            defects.append(text)
    return defects


def _apply_joint_twin_audit(group: dict[str, Any], result: dict[str, Any]) -> None:
    audit = result.get("semantic_audit") or {}
    rows = audit.get("members") or []
    if isinstance(rows, list) and rows and all(isinstance(r, dict) and "checks" in r for r in rows):
        rows = [{**r, **{k: (r.get("checks") or {}).get(k) == "pass" for k in JOINT_MEMBER_FIELDS}}
                for r in rows]
        for row in rows:
            contradictions = _acknowledged_hard_defects(row.get("warnings", []))
            if contradictions:
                row["failure_reasons"] = list(row.get("failure_reasons") or []) + contradictions
            if row.get("failure_reasons"):
                row["natural"] = False
            if not all(str(row.get(k, "")).strip() for k in ("observed_entity", "ref_entity", "observed_change")):
                row["scope_correct"] = False
        audit = {**audit, "members": rows, "comparable_granularity": audit.get("comparable_granularity") == "pass",
                 "shared_edit_valid": audit.get("shared_edit_valid") == "pass"}
    expected = [m["unit_id"] for m in group["members"]]
    valid_rows = isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
    ids_match = valid_rows and [r.get("unit_id") for r in rows] == expected
    members_pass = bool(ids_match) and all(all(r.get(k) is True for k in JOINT_MEMBER_FIELDS) for r in rows)
    if (result.get("finish_reason") == "length" or audit.get("decision") == "fail"
            or audit.get("_parse_error") or audit.get("failure_reasons")):
        members_pass = False
    semantic_pass = members_pass and audit.get("comparable_granularity") is True and audit.get("shared_edit_valid") is True
    for index, member in enumerate(group["members"]):
        row = rows[index] if ids_match else {"failure_reasons": ["missing_or_mismatched_member_ids"]}
        member["semantic_audit"] = {**row, "decision": "pass" if ids_match and
                                    all(row.get(k) is True for k in JOINT_MEMBER_FIELDS) else "fail"}
    checks = group.setdefault("group_checks", {})
    checks.update({"member_semantic_audit_pass": members_pass,
                   "semantic_audit_pass": semantic_pass,
                   "semantic_audit": {**audit, "decision": "pass" if semantic_pass else "fail"},
                   "audit_mode": "joint-core-defects-v4", "audit_calls": 1,
                   "audit_request": {k: result.get(k) for k in
                                     ("semantic_audit_raw", "input_tokens", "output_tokens", "finish_reason")}})
    checks["group_pass"] = (checks.get("blind_disambiguation", {}).get("pass") is True) and semantic_pass and all(checks.get(k, {}).get("pass") is True for k in
        ("deterministic", "cross_target_preservation", "support_gap"))


def refresh_twin_add_checks(group: dict[str, Any]) -> bool:
    """Apply the add presence gate to cached groups without another MLLM call."""
    checks = group.setdefault("group_checks", {})
    if group.get("operation") != "add" or checks.get("add_presence_version") == 1:
        return checks.get("deterministic", {}).get("pass") is not False
    results = []
    for member in group["members"]:
        result = deterministic_check(_member_audit_row(group, member, kind="twins"))
        member["deterministic"] = result
        results.append(result)
    # A new check must never erase an existing rejection from another gate.
    passed = checks.get("deterministic", {}).get("pass") is not False and all(
        row["decision"] != "fail" for row in results)
    checks["deterministic"] = {"pass": passed, "members": results}
    checks["add_presence_version"] = 1
    if not passed:
        checks["group_pass"] = False
    return passed


def audit_twin_groups_joint(groups: list[dict[str, Any]], model: VLLMAuditor,
                            panel_dir: str | Path) -> list[dict[str, Any]]:
    """One visual request per group; deterministic gates remain independent."""
    panel_dir = Path(panel_dir)
    jobs, audited_groups = [], []
    for group in groups:
        if not refresh_twin_add_checks(group):
            continue
        panels = [_target_panel(_member_audit_row(group, member, kind="twins"), panel_dir,
                                binary_locator=True) for member in group["members"]]
        comparison = _member_comparison(panels, panel_dir / f"{group['twin_id']}_local.jpg", crop_panels=True)
        full = _member_comparison([m["edited_image"] for m in group["members"]],
                                 panel_dir / f"{group['twin_id']}_full.jpg")
        jobs.append({"job_id": group["twin_id"], "images": [group["source_image"], comparison, full],
                     "prompt": JOINT_TWIN_PROMPT + "\n" + BLIND_RULE + "\nTarget contract:\n" + _target_contract(group, kind="twins")})
        audited_groups.append(group)
    for group, result in zip(audited_groups, _call_batched(model, jobs)):
        _apply_joint_twin_audit(group, result)
    return groups


def audit_multi_groups(groups: list[dict[str, Any]], model: VLLMAuditor, panel_dir: str | Path) -> list[dict[str, Any]]:
    panel_dir = Path(panel_dir)
    member_audits = _single_member_audits(groups, model, panel_dir / "steps", kind="multi")
    jobs = []
    for group in groups:
        images = [group["source_image"]]
        marked = _marked_scene(group)
        if marked:
            images.append(marked)
        comparison_paths = []
        for member in group["units"]:
            if member.get("step_output_image"):
                comparison_paths.append(member["step_output_image"])
        if group.get("joint_target") and group["joint_target"] not in images:
            comparison_paths.append(group["joint_target"])
        if comparison_paths:
            images.append(_member_comparison(comparison_paths, panel_dir / f"{group['multi_id']}_comparison.jpg"))
        jobs.append({"job_id": group["multi_id"], "images": images[:3],
                     "prompt": MULTI_GROUP_PROMPT + "\n" + BLIND_RULE + "\nTarget contract:\n" +
                     _target_contract(group, kind="multi")})
    results = []
    for group, audits, result in zip(groups, member_audits, _call_batched(model, jobs)):
        audit = result["semantic_audit"]
        joint = group["joint_audit"]
        member_pass = bool(audits) and all(
            item.get("decision") == "pass"
            and item.get("operation_correct") is True
            and all(item.get(key) is True for key in ("target_done", "non_target_preserved", "natural"))
            for item in audits)
        joint["member_semantic_audit_pass"] = member_pass
        joint["semantic_audit"] = audit
        semantic_pass = _semantic_group_pass(audit, MULTI_REQUIRED_AUDIT_FIELDS)
        joint["semantic_audit_pass"] = semantic_pass
        step_pass = all(x.get("decision") != "fail" for x in joint.get("steps", []))
        joint["pass"] = bool(joint.get("blind_protocol", {}).get("pass") is True and member_pass and step_pass and joint.get("outside_union_support_fraction", 1.0) <= 0.02
                              and semantic_pass)
        results.append(group)
    return results
