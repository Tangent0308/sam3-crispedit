"""Chain/twin generation and deterministic group checks."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .editor import region_support_mask
from .geometry import load_mask
from .group_geometry import gap_pixels, precise_point
from .production_state import atomic_json
from .validate import deterministic_check


def _write_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255), mode="L").save(path, compress_level=1)
    return str(path)


def _unit_dict(member: dict[str, Any]) -> dict[str, Any]:
    unit = dict(member.get("unit") or {})
    if not unit:
        unit = {k: v for k, v in member.items() if k in {
            "unit_id", "object_id", "operation", "instruction_ref", "instruction_noref",
            "mask_path", "execution_mask_path", "protected_mask_path", "support_policy",
            "instruction_spec", "mask_role", "bbox", "point", "ref_phrase", "scope", "payload",
        }}
    unit["mask_path"] = member.get("mask") or unit.get("mask_path")
    unit["instruction_ref"] = member.get("instruction") or unit.get("instruction_ref")
    unit["instruction_noref"] = member.get("instruction_noref") or unit.get("instruction_noref")
    unit["instruction_spec"] = member.get("instruction_spec") or unit.get("instruction_spec", {})
    unit["support_policy"] = member.get("support_policy") or unit.get("support_policy")
    unit["operation"] = member.get("operation") or unit.get("operation") or unit["instruction_spec"].get("operation")
    return unit


def _source_digest(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _render_member(source: str, unit: dict[str, Any], output: Path,
                   editor: Any, seed: int) -> dict[str, Any]:
    """Resume only a matching input contract, retaining its original context."""
    checkpoint = output.with_name("render.json")
    contract = {"version": 1, "source_sha256": _source_digest(source), "unit": unit,
                "seed": seed, "editor": getattr(editor, "config", {}),
                "mask_hashes": {key: _source_digest(unit[key]) for key in
                                ("mask_path", "execution_mask_path", "protected_mask_path")
                                if unit.get(key)}}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("fingerprint") != fingerprint:
            raise ValueError(f"generation input changed; use a new output directory: {output.parent}")
        if saved.get("status") == "complete" and output.exists():
            if saved.get("output_sha256") != _source_digest(str(output)):
                raise ValueError(f"cached generated image checksum mismatch: {output}")
            return saved["generation_context"]
    elif output.exists():
        raise ValueError(f"existing image has no verifiable render checkpoint: {output}")
    atomic_json(checkpoint, {"fingerprint": fingerprint, "status": "rendering"})
    result = editor.generate(source, unit, seed=seed)
    temporary = output.with_name("edited.partial.png")
    result.save(temporary)
    temporary.replace(output)
    context = {"editor": getattr(editor, "config", {}),
               **getattr(editor, "last_generation_context", {})}
    atomic_json(checkpoint, {"fingerprint": fingerprint, "status": "complete",
                            "output_sha256": _source_digest(str(output)),
                            "generation_context": context})
    return context


def _row(group: dict[str, Any], member: dict[str, Any], source: str, edited: str,
         support: str, *, input_image: str | None = None, output_image: str | None = None,
         step: int | None = None, generation_context: dict[str, Any] | None = None) -> dict[str, Any]:
    unit = _unit_dict(member)
    return {"group_id": group.get("twin_id") or group.get("multi_id"),
            "source_id": group["source_id"], "source_image": source,
            "edited_image": edited, "mask": member["mask"], "support_mask": support,
            "execution_mask_path": unit.get("execution_mask_path"),
            "protected_mask_path": unit.get("protected_mask_path"),
            "operation": unit["operation"], "instruction": member["instruction"],
            "instruction_noref": member["instruction_noref"], "instruction_spec": unit.get("instruction_spec", {}),
            "ref_phrase": member["ref"], "object_id": member["object_id"],
            "unit_id": member["unit_id"], "bbox": member.get("bbox"), "point": member.get("point"),
            "seed": 0, "step": step, "step_input_image": input_image,
            "step_output_image": output_image or edited, "generation_context": generation_context or {}}


def _support_for(editor: Any, source: str, unit: dict[str, Any], path: Path) -> str:
    with Image.open(source) as image:
        size = image.size
    mask = load_mask({"mask_path": unit["mask_path"]}, size)
    if hasattr(editor, "support_mask"):
        support = editor.support_mask(Image.open(source).convert("RGB"), unit)
    else:
        support = region_support_mask(mask, unit["operation"], unit.get("support_policy"))
    return _write_mask(path, support)


def _check_cross_preservation(source: str, edited: str, other_mask: str) -> float:
    src = np.asarray(Image.open(source).convert("RGB"), dtype=np.int16)
    out = np.asarray(Image.open(edited).convert("RGB"), dtype=np.int16)
    mask = load_mask({"mask_path": other_mask}, (src.shape[1], src.shape[0]))
    delta = np.max(np.abs(src - out), axis=2) > 12
    return float(delta[mask].mean()) if mask.any() else 0.0


def generate_twin_group(group: dict[str, Any], output_dir: str | Path, editor: Any, *, seed: int = 0,
                        semantic_audits: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate each twin from the same source/seed and gate the whole group."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    source = group["source_image"]
    if not Path(source).is_file():
        raise FileNotFoundError(source)
    outputs = []
    supports = []
    for index, member in enumerate(group["members"], 1):
        unit = _unit_dict(member)
        unit_dir = output_dir / f"member_{member['unit_id']}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        output = unit_dir / "edited.png"
        state = unit_dir / "generation.json"
        started = time.perf_counter()
        context = _render_member(source, unit, output, editor, seed)
        support = _support_for(editor, source, unit, unit_dir / "support_mask.png")
        row = _row(group, member, source, str(output), support,
                   generation_context=context)
        row["seed"] = seed
        check = deterministic_check(row)
        row["deterministic"] = check
        member.update({"edited_image": str(output), "support_mask": support, "seed": seed,
                       "generation_context": row["generation_context"], "deterministic": check,
                       "generation_seconds": time.perf_counter() - started})
        outputs.append(row); supports.append(support)
        atomic_json(state, {"row": row, "fingerprint": hashlib.sha256(json.dumps(unit, sort_keys=True).encode()).hexdigest()})
    cross = []
    support_gaps = []
    for i, left in enumerate(group["members"]):
        for right in group["members"][i + 1:]:
            with Image.open(source) as image:
                size = image.size
            left_support = load_mask({"mask_path": left["support_mask"]}, size)
            right_support = load_mask({"mask_path": right["support_mask"]}, size)
            gap = gap_pixels(left_support, right_support)
            left_short = min(left["bbox"][2] - left["bbox"][0], left["bbox"][3] - left["bbox"][1])
            right_short = min(right["bbox"][2] - right["bbox"][0], right["bbox"][3] - right["bbox"][1])
            required = max(16.0, 0.1 * min(left_short, right_short))
            support_gaps.append({"members": [left["unit_id"], right["unit_id"]],
                                 "gap_pixels": gap, "required_gap_pixels": required,
                                 "pass": gap >= required})
    for i, left in enumerate(group["members"]):
        for j, right in enumerate(group["members"]):
            if i == j:
                continue
            cross.append({"member": left["unit_id"], "other": right["unit_id"],
                          "changed_fraction_in_other_source_mask": _check_cross_preservation(
                              source, left["edited_image"], right["mask"])})
    deterministic_pass = all(r["deterministic"].get("decision") != "fail" for r in outputs)
    cross_pass = all(x["changed_fraction_in_other_source_mask"] <= 0.02 for x in cross)
    gap_pass = all(x["pass"] for x in support_gaps)
    semantic_audits = semantic_audits or {}
    semantic_pass = bool(semantic_audits) and all(
        semantic_audits.get(m["unit_id"], {}).get("decision") == "pass" for m in group["members"])
    group["group_checks"] = {**group.get("group_checks", {}),
                              "deterministic": {"pass": deterministic_pass, "members": [r["deterministic"] for r in outputs]},
                              "cross_target_preservation": {"pass": cross_pass, "pairs": cross},
                              "support_gap": {"pass": gap_pass, "pairs": support_gaps},
                              "semantic_audit": {"pass": semantic_pass, "audits": semantic_audits},
                              "group_pass": bool(deterministic_pass and cross_pass and gap_pass and semantic_pass)}
    group["members"] = group["members"]
    return group


def _union_support(source: str, members: list[dict[str, Any]]) -> np.ndarray:
    with Image.open(source) as image:
        shape = (image.height, image.width)
    union = np.zeros(shape, dtype=bool)
    for member in members:
        union |= load_mask({"mask_path": member["support_mask"]}, (shape[1], shape[0]))
    return union


def _outside_fraction(source: str, edited: str, allowed: np.ndarray) -> float:
    a = np.asarray(Image.open(source).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(edited).convert("RGB"), dtype=np.int16)
    delta = np.max(np.abs(a - b), axis=2) > 12
    return float(delta[~allowed].mean()) if (~allowed).any() else 0.0


def generate_multi_group(group: dict[str, Any], output_dir: str | Path, editor: Any, *, seed: int = 0,
                         semantic_audits: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a group as a source -> step chain; any failure drops all steps."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    source = group["source_image"]
    by_id = {m["unit_id"]: m for m in group["units"]}
    current = source
    rows = []
    for step, unit_id in enumerate(group["generation_order"], 1):
        member = by_id[unit_id]
        unit = _unit_dict(member)
        unit_dir = output_dir / f"step_{step:02d}_{unit_id}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        output = unit_dir / "edited.png"
        context = _render_member(current, unit, output, editor, seed)
        support = _support_for(editor, current, unit, unit_dir / "support_mask.png")
        row = _row(group, member, current, str(output), support, input_image=current,
                   output_image=str(output), step=step, generation_context=context)
        row["seed"] = seed
        check = deterministic_check(row)
        row["step_audit"] = check
        member.update({"step_input_image": current, "step_output_image": str(output),
                       "support_mask": support, "generation_context": context, "step_audit": check,
                       "seed": seed})
        rows.append(row)
        current = str(output)
    allowed = _union_support(source, group["units"])
    final_outside = _outside_fraction(source, current, allowed)
    semantic_audits = semantic_audits or {}
    step_pass = all(r["step_audit"].get("decision") != "fail" for r in rows)
    semantic_pass = bool(semantic_audits) and all(
        semantic_audits.get(m["unit_id"], {}).get("decision") == "pass" for m in group["units"])
    group["joint_target"] = current
    group["joint_audit"] = {"steps": [r["step_audit"] for r in rows],
                            "outside_union_support_fraction": final_outside,
                            "semantic_audit": semantic_audits,
                            "pass": bool(step_pass and final_outside <= 0.02 and semantic_pass)}
    return group
