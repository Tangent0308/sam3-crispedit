"""27B semantic audits for grouped records.

The deterministic checks live in :mod:`group_generation`; this module keeps
the visual decisions auditable and all-or-nothing at group level.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .audit_backend import VLLMAuditor
from .validate import _target_panel, audit_prompt


TWIN_GROUP_PROMPT = '''You are auditing one counterfactual twin group. The source scene and the member result images are provided. The same no-ref edit was intended for each marked member, while each ref identifies a different original instance. Decide whether every reference is unique, every requested edit is complete and natural, and the member results are comparable. The group fails if any member is wrong, if a result edits another member, or if the shared request is not natural for all. Return JSON only: {"decision":"pass|fail","unique_refs":true,"all_targets_done":true,"members_preserved":true,"comparable":true,"natural":true,"failure_reasons":[],"evidence":"brief"}.'''

MULTI_GROUP_PROMPT = '''You are auditing a joint multi-target edit group. The original scene, each chain step, and the final result are provided. Check that each original ref identifies one target, every assigned operation is completed in its step and remains correct later, adjacent targets do not bleed into each other, and the final image is natural. Return JSON only: {"decision":"pass|fail","unique_refs":true,"all_targets_done":true,"no_crosstalk":true,"outside_preserved":true,"natural":true,"failure_reasons":[],"evidence":"brief"}.'''

TWIN_REQUIRED_AUDIT_FIELDS = ("unique_refs", "all_targets_done", "members_preserved", "comparable", "natural")
MULTI_REQUIRED_AUDIT_FIELDS = ("unique_refs", "all_targets_done", "no_crosstalk", "outside_preserved", "natural")


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
        results.extend(outputs)
    return results


def _marked_scene(group: dict[str, Any]) -> str | None:
    views = (group.get("planning") or {}).get("views") or []
    if views and Path(views[0]).is_file():
        return str(views[0])
    return None


def _target_contract(group: dict[str, Any], *, kind: str) -> str:
    rows = group.get("members", []) if kind == "twins" else group.get("units", [])
    lines = []
    for index, member in enumerate(rows, 1):
        lines.append(
            f"{member.get('unit_id', f'U{index}')}: object_id={member.get('object_id')}; "
            f"ref={member.get('ref') or member.get('ref_phrase')}; "
            f"operation={member.get('operation')}; instruction={member.get('instruction') or member.get('instruction_ref')}"
        )
    return "\n".join(lines)


def _member_comparison(paths: list[str], out: Path) -> str:
    """Pack every twin member panel into one audit image.

    The 27B adapter accepts at most three image inputs. Passing the source and
    one tiled comparison therefore keeps all two/three members visible instead
    of silently dropping the third member from a triple group.
    """
    panels = [Image.open(path).convert("RGB") for path in paths]
    tile_w, tile_h = 768, 520
    tiles = []
    for index, panel in enumerate(panels, 1):
        panel.thumbnail((tile_w, tile_h - 28), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        ImageDraw.Draw(tile).text((8, 6), f"MEMBER {index}", fill="black")
        tile.paste(panel, ((tile_w - panel.width) // 2, 28 + (tile_h - 28 - panel.height) // 2))
        tiles.append(tile)
    columns = min(2, max(1, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    panel = Image.new("RGB", (columns * tile_w, rows * tile_h), (235, 235, 235))
    for index, tile in enumerate(tiles):
        panel.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out, quality=94, subsampling=0)
    return str(out)


def audit_twin_groups(groups: list[dict[str, Any]], model: VLLMAuditor, panel_dir: str | Path) -> list[dict[str, Any]]:
    panel_dir = Path(panel_dir)
    jobs, refs = [], []
    for group in groups:
        rows = []
        for member in group["members"]:
            row = {"source_image": group["source_image"], "edited_image": member["edited_image"],
                   "mask": member["mask"], "operation": group["operation"],
                   "instruction": member["instruction"], "instruction_spec": member["instruction_spec"],
                   "support_mask": member["support_mask"]}
            rows.append(_target_panel(row, panel_dir))
        comparison = _member_comparison([m["edited_image"] for m in group["members"]],
                                        panel_dir / f"{group['twin_id']}_comparison.jpg")
        refs.append(group)
        marked = _marked_scene(group)
        images = [group["source_image"]]
        if marked:
            images.append(marked)
        images.append(comparison)
        jobs.append({"job_id": group["twin_id"], "images": images[:3],
                     "prompt": TWIN_GROUP_PROMPT + "\nTarget contract:\n" +
                     _target_contract(group, kind="twins")})
    # The source plus one tiled comparison stays within the three-image limit
    # while showing every member, including optional three-member groups.
    results = []
    for group, result in zip(refs, _call_batched(model, jobs)):
        audit = result["semantic_audit"]
        checks = group.setdefault("group_checks", {})
        checks["semantic_audit"] = audit
        semantic_pass = _semantic_group_pass(audit, TWIN_REQUIRED_AUDIT_FIELDS)
        checks["semantic_audit_pass"] = semantic_pass
        checks["group_pass"] = bool(
            checks.get("deterministic", {}).get("pass") is True
            and checks.get("cross_target_preservation", {}).get("pass") is True
            and checks.get("support_gap", {}).get("pass") is True
            and semantic_pass)
        for member in group.get("members", []):
            member["semantic_audit"] = audit
        results.append(group)
    return results


def audit_multi_groups(groups: list[dict[str, Any]], model: VLLMAuditor, panel_dir: str | Path) -> list[dict[str, Any]]:
    panel_dir = Path(panel_dir)
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
                     "prompt": MULTI_GROUP_PROMPT + "\nTarget contract:\n" +
                     _target_contract(group, kind="multi")})
    results = []
    for group, result in zip(groups, _call_batched(model, jobs)):
        audit = result["semantic_audit"]
        joint = group["joint_audit"]
        joint["semantic_audit"] = audit
        semantic_pass = _semantic_group_pass(audit, MULTI_REQUIRED_AUDIT_FIELDS)
        joint["semantic_audit_pass"] = semantic_pass
        step_pass = all(x.get("decision") != "fail" for x in joint.get("steps", []))
        joint["pass"] = bool(step_pass and joint.get("outside_union_support_fraction", 1.0) <= 0.02
                              and semantic_pass)
        for member in group.get("units", []):
            member["semantic_audit"] = audit
        results.append(group)
    return results
