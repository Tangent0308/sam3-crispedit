"""Pair-level deterministic checks plus optional Qwen3.8 semantic audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import load_mask
from .io import read_jsonl, write_jsonl
from .mllm import VLLMSelector


AUDIT_PROMPT = """Judge this fine-grained edit pair. Images are ordered as full source, full edited result, and a high-resolution target panel. The target panel contains source crop, edited crop, and cyan-mask crop left-to-right for precise localization. Compare the first two using the target panel and instruction.
Return JSON only: {\"decision\":\"pass\"|\"fail\",\"target_done\":bool,\"non_target_preserved\":bool,\"natural\":bool,\"failure_reasons\":[],\"evidence\":\"...\"}.
The cyan pixels in the third panel are an artificial location annotation, NEVER output pixels or an artifact. Judge rendering only on the clean source and edited views. Locate the intended instance from the mask and then verify that the instruction's source description actually matches it; reject a wrong source color or a whole-person color request that silently means clothing. Do not confuse the mask overlay with an edited result.
Failure-first rules: wrong instance, missing/partial edit, extra duplicate, changed neighboring instance, broken occlusion/contact, or visible artifact is fail. For add the host remains and the item is attached. For remove the entire visible target disappears. For replace the old identity disappears and a distinct new identity appears. For attribute geometry and identity remain.
The natural verdict must assess both photographic rendering and contextual plausibility. Even a seamless edit fails natural if the requested content is implausible for the visible scene. For add, check a conventional host-item relationship, usable attachment site and realistic scale without blocking the host's function. For replace, check compatible scene role, available space and existing support. For attribute, check credible requested properties, preserved unrequested markings and component boundaries, and natural shading rather than flat paint. Do not reject unusual taste alone; use visible physical or contextual contradictions. Report contextual mismatch as implausible_edit_content with concrete visual evidence, distinguishing it from rendering artifacts. Do not excuse implausible content merely because the instruction requested it.
Protocol rules: the source mask is a semantic selector, not the entire required change area. For named part removal/trim, remove only that part and retain the host. Obey explicit item counts and placement relations, including relative reference placement. For attribute, change requested text/pattern/finish/opacity when stated and preserve only unrequested properties; tint preserves transparency. Never penalize a required small part edit just because most host pixels remain unchanged.
Instruction: {instruction}
Target mask file is supplied for localization only; it is not a pixel-difference ground truth."""


OPERATION_AUDIT_RULES = '''
Also return operation_correct: bool. Check the expected dataset operation against
the actual requested change AND the visible result, not merely the imperative verb.
add introduces a new object/detail while keeping its host; remove deletes the named
existing object/detail; replace exchanges an object/component for a different
physical form or category; attribute changes properties while keeping that object.
Changing only color/material/finish/texture/pattern/text/transparency is attribute,
even if the instruction says Replace. Turning rippled water into calm water is a
texture/state change, not object replacement. Repainting or resurfacing a continuous
surface without a new physical structure is not replace. A new tattoo/logo is add;
changing an existing printed design is attribute; erasing it is remove.
If the expected type is wrong, set operation_correct=false and decision=fail with
failure_reasons containing operation_type_mismatch. Do not reclassify the sample
to save its quota. The usual identity, naturalness and preservation checks still apply.
Expected operation: {operation}.
'''

OPERATION_AUDIT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'decision': {'enum': ['pass', 'fail']},
        **{k: {'type': 'boolean'} for k in ('operation_correct', 'target_done', 'non_target_preserved', 'natural')},
        'failure_reasons': {'type': 'array', 'items': {'type': 'string'}},
        'evidence': {'type': 'string'},
    },
    'required': ['decision', 'operation_correct', 'target_done', 'non_target_preserved', 'natural', 'failure_reasons', 'evidence'],
}


def audit_prompt(instruction: str, *, binary_locator: bool = False, operation: str | None = None) -> str:
    """Pair the experimental locator panel with its correct image-role prompt."""
    prompt = AUDIT_PROMPT
    if binary_locator:
        prompt = prompt.replace(
            'source crop, edited crop, and cyan-mask crop left-to-right for precise localization.',
            'SOURCE clean crop, RESULT clean edited crop, and MASK ONLY binary locator left-to-right. '
            'All three use the same crop, scale and aspect ratio. White in MASK ONLY identifies '
            'the source target; black excludes it. Gray margins and text labels are panel layout only.')
        prompt = prompt.replace(
            'The cyan pixels in the third panel are an artificial location annotation, NEVER output pixels or an artifact.',
            'The black-and-white MASK ONLY panel is a binary location map, NEVER output pixels or an artifact.')
    if operation is not None:
        if operation not in {'add', 'replace', 'remove', 'attribute'}:
            raise ValueError('unknown expected operation')
        prompt = prompt.replace('"target_done":bool', '"operation_correct":bool,"target_done":bool', 1)
        prompt += OPERATION_AUDIT_RULES.replace('{operation}', operation)
    return prompt.replace('{instruction}', instruction)


def _fraction(a: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(a[mask])) if mask.any() else 0.0


def deterministic_check(row: dict[str, Any]) -> dict[str, Any]:
    source = np.asarray(Image.open(row["source_image"]).convert("RGB"), dtype=np.int16)
    edited = np.asarray(Image.open(row["edited_image"]).convert("RGB"), dtype=np.int16)
    if source.shape != edited.shape:
        return {"decision": "fail", "failure_reasons": ["size_mismatch"]}
    mask = load_mask({"mask_path": row["mask"]}, (source.shape[1], source.shape[0]))
    delta = np.max(np.abs(source - edited), axis=2) > 12
    inside, outside = _fraction(delta, mask), _fraction(delta, ~mask)
    # ``mask`` is the benchmark target.  ``support_mask`` is the explicit
    # operation-specific region that the editor is allowed to touch (for
    # example, host plus attachment band for add).  Keep both diagnostics:
    # target-local change is useful for attribute/replace/remove, while the
    # support contract is the correct leakage check for add.
    support_path = row.get("support_mask")
    support_inside = support_outside = None
    if support_path and Path(str(support_path)).exists():
        support = load_mask({"mask_path": str(support_path)}, (source.shape[1], source.shape[0]))
        support_inside = _fraction(delta, support)
        support_outside = _fraction(delta, ~support)
    from .instruction_protocol import needs_local_execution
    partial = needs_local_execution(row.get('instruction_spec') or {})
    # Host-area ratios cannot veto a small named subpart. The aligned execution
    # region is the measurable target for those tasks.
    change_fraction = inside
    if partial:
        if not row.get('execution_mask_path'):
            return {'decision': 'fail', 'failure_reasons': ['missing_subpart_execution_region']}
        execution = load_mask({'mask_path': row['execution_mask_path']}, (source.shape[1], source.shape[0]))
        change_fraction = _fraction(delta, execution)
    reasons = []
    if partial and change_fraction == 0:
        reasons.append('no_visible_target_change')
    if row["operation"] == "remove" and not partial and change_fraction < 0.50:
        reasons.append("remove_mask_change_too_weak")
    if row["operation"] in {"attribute", "replace"} and change_fraction < 0.05:
        reasons.append("no_visible_target_change")
    # A few anti-aliased pixels can cross the support boundary, but a broad
    # edit outside the explicit support is a hard localization failure.
    if support_outside is not None and support_outside > 0.02:
        reasons.append("support_leakage")
    deterministic = {"inside_changed_fraction": inside, "outside_changed_fraction": outside}
    if support_inside is not None:
        deterministic.update({
            "support_changed_fraction": support_inside,
            "outside_support_changed_fraction": support_outside,
        })
    return {"deterministic": deterministic,
            "decision": "fail" if reasons else "pending_semantic_audit", "failure_reasons": reasons}


def validate(manifest: Path, output: Path, selector: VLLMSelector | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest)
    results = []
    jobs = []
    for row in rows:
        check = deterministic_check(row)
        results.append({**row, **check})
        if selector and check["decision"] != "fail":
            panel = _target_panel(row, output.parent / "audit_overlays")
            jobs.append(([row["source_image"], row["edited_image"], panel], AUDIT_PROMPT.replace('{instruction}', row['instruction'])))
    if selector:
        audits = selector.generate_multimage(jobs)
        index = 0
        for result in results:
            if result["decision"] == "fail":
                continue
            audit = audits[index]; index += 1
            result["semantic_audit"] = audit
            if audit.get("decision") != "pass" or not all(audit.get(k) is True for k in ("target_done", "non_target_preserved", "natural")):
                result["decision"] = "fail"
                result["failure_reasons"] = list(result.get("failure_reasons", [])) + ["semantic_audit_failed"]
            else:
                result["decision"] = "pass"
    write_jsonl(output, results)
    return results


def _mask_overlay(source_path: str, mask_path: str, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (Path(source_path).stem + "__" + Path(mask_path).stem + ".jpg")
    if path.exists():
        return str(path)
    source = Image.open(source_path).convert("RGB")
    mask = load_mask({"mask_path": mask_path}, source.size)
    overlay = Image.new("RGB", source.size, (0, 210, 220))
    alpha = Image.fromarray((mask.astype("uint8") * 90), mode="L")
    Image.composite(overlay, source, alpha).save(path, quality=94)
    return str(path)


def _target_panel(row: dict[str, Any], directory: Path, *, binary_locator: bool = False) -> str:
    """Build a high-resolution source/edited/mask crop for the semantic auditor."""
    directory.mkdir(parents=True, exist_ok=True)
    # Include the crop policy in the cache key so a pre-existing wide panel
    # cannot silently defeat the tighter add audit crop on a rerun.
    operation = str(row.get("operation") or "generic")
    # Multiple backends edit the same source mask. Include actual pair contents
    # so one backend cannot accidentally be audited using another's crop.
    digest = hashlib.sha256()
    for key in ('source_image', 'edited_image', 'mask'):
        digest.update(Path(row[key]).read_bytes())
    if row.get('execution_mask_path'):
        digest.update(Path(row['execution_mask_path']).read_bytes())
    digest.update(json.dumps(row.get('instruction_spec') or {}, sort_keys=True).encode())
    path = directory / (
        Path(row["source_image"]).stem + "__" + Path(row["mask"]).stem
        + f"__panel_{operation}_{'v5_binary' if binary_locator else 'v4'}_{digest.hexdigest()[:20]}.jpg"
    )
    if path.exists():
        return str(path)
    source = Image.open(row["source_image"]).convert("RGB")
    edited = Image.open(row["edited_image"]).convert("RGB")
    mask = load_mask({"mask_path": row["mask"]}, source.size)
    crop_mask = mask
    if row.get('execution_mask_path'):
        from .instruction_protocol import needs_local_execution
        execution = load_mask({'mask_path': row['execution_mask_path']}, source.size)
        crop_mask = execution if needs_local_execution(row.get('instruction_spec') or {}) else (mask | execution)
    ys, xs = np.where(crop_mask)
    if len(xs):
        # Add edits often introduce a small attached item on a larger host.
        # A tighter crop gives the semantic auditor enough pixels to see the
        # new item while the full source/edited images still check global
        # preservation. Other operations keep the wider context so contact,
        # removal boundaries, and neighboring instances remain judgeable.
        margin_factor = 0.55 if row.get("operation") == "add" else 0.9
        margin = max(32, int(round(margin_factor * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))))
        left = max(0, int(xs.min()) - margin)
        top = max(0, int(ys.min()) - margin)
        right = min(source.width, int(xs.max()) + 1 + margin)
        bottom = min(source.height, int(ys.max()) + 1 + margin)
        box = (left, top, right, bottom)
    else:
        box = (0, 0, source.width, source.height)
    source_crop = source.crop(box)
    edited_crop = edited.crop(box)
    if binary_locator:
        from PIL import ImageDraw
        # Experimental audit view: keep clean photographic evidence separate
        # from localization, and preserve a shared scale/aspect ratio.
        local = mask[box[1]:box[3], box[0]:box[2]]
        locator = Image.fromarray(local.astype('uint8') * 255).convert('RGB')
        panel = Image.new('RGB', (1536, 512), (48, 48, 48))
        scale = min(512 / source_crop.width, 464 / source_crop.height)
        size = (max(1, round(source_crop.width * scale)),
                max(1, round(source_crop.height * scale)))
        for i, (tile, label) in enumerate(zip(
                (source_crop, edited_crop, locator),
                ('SOURCE - ORIGINAL', 'RESULT - JUDGE THIS', 'MASK ONLY - WHITE = TARGET'))):
            header = Image.new('RGB', (256, 24), 'white')
            ImageDraw.Draw(header).text((5, 6), label, fill='black')
            panel.paste(header.resize((512, 48)), (i * 512, 0))
            tile = tile.resize(size, Image.Resampling.NEAREST if i == 2 else Image.Resampling.LANCZOS)
            panel.paste(tile, (i * 512 + (512-size[0])//2, 48 + (464-size[1])//2))
        panel.save(path, quality=94, subsampling=0)
        return str(path)
    mask_crop = source.crop(box)
    local = mask[box[1]:box[3], box[0]:box[2]]
    cyan = Image.new("RGB", mask_crop.size, (0, 210, 220))
    alpha = Image.fromarray((local.astype("uint8") * 120), mode="L")
    mask_crop = Image.composite(cyan, mask_crop, alpha)
    tiles = [im.resize((512, 512), Image.Resampling.LANCZOS) for im in (source_crop, edited_crop, mask_crop)]
    panel = Image.new("RGB", (1536, 512))
    for i, tile in enumerate(tiles):
        panel.paste(tile, (i * 512, 0))
    panel.save(path, quality=94)
    return str(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="generated.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--disable-custom-all-reduce", action="store_true",
                   help="use standard NCCL all-reduce when shared custom buffers are unavailable")
    a = p.parse_args(argv)
    selector = VLLMSelector(a.model_id, tensor_parallel_size=a.tensor_parallel_size,
                             disable_custom_all_reduce=a.disable_custom_all_reduce) if a.model_id else None
    try:
        results = validate(a.manifest, a.output, selector)
        print(json.dumps({"rows": len(results), "pass": sum(x["decision"] == "pass" for x in results), "output": str(a.output)}))
    finally:
        if selector:
            selector.close()


if __name__ == "__main__":
    main()
