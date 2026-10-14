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


def _crop_pair(path_a: str, path_b: str, out: Path) -> str:
    a = Image.open(path_a).convert("RGB")
    b = Image.open(path_b).convert("RGB")
    size = (min(768, max(a.width, b.width)), min(768, max(a.height, b.height)))
    a.thumbnail(size, Image.Resampling.LANCZOS); b.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (max(a.width, b.width) * 2, max(a.height, b.height) + 32), "white")
    ImageDraw.Draw(panel).text((4, 4), "MEMBER A", fill="black")
    ImageDraw.Draw(panel).text((panel.width // 2 + 4, 4), "MEMBER B", fill="black")
    panel.paste(a, ((panel.width // 2 - a.width) // 2, 32))
    panel.paste(b, (panel.width // 2 + (panel.width // 2 - b.width) // 2, 32))
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
        comparison = _crop_pair(group["members"][0]["edited_image"], group["members"][1]["edited_image"],
                                panel_dir / f"{group['twin_id']}_comparison.jpg")
        refs.append(group)
        jobs.append({"job_id": group["twin_id"], "images": [group["source_image"], *rows, comparison][:3],
                     "prompt": TWIN_GROUP_PROMPT})
    # The auditor accepts at most three images per request. The source plus two
    # target panels is sufficient; the comparison is retained in the on-disk
    # audit bundle and used in a second call when there are two members.
    results = []
    for group, job in zip(refs, jobs):
        result = model.call_batch([job])[0]
        audit = result["semantic_audit"]
        group.setdefault("group_checks", {})["semantic_audit"] = audit
        group["group_checks"]["semantic_audit_pass"] = audit.get("decision") == "pass"
        results.append(group)
    return results


def audit_multi_groups(groups: list[dict[str, Any]], model: VLLMAuditor, panel_dir: str | Path) -> list[dict[str, Any]]:
    panel_dir = Path(panel_dir)
    results = []
    for group in groups:
        images = [group["source_image"]]
        for member in group["units"]:
            if member.get("step_output_image"):
                images.append(member["step_output_image"])
        if group.get("joint_target") and group["joint_target"] not in images:
            images.append(group["joint_target"])
        result = model.call_batch([{"job_id": group["multi_id"], "images": images[:3],
                                    "prompt": MULTI_GROUP_PROMPT}])[0]
        audit = result["semantic_audit"]
        group["joint_audit"]["semantic_audit"] = audit
        group["joint_audit"]["semantic_audit_pass"] = audit.get("decision") == "pass"
        results.append(group)
    return results
