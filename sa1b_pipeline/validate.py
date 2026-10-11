"""Pair-level deterministic checks plus optional Qwen3.8 semantic audit."""
from __future__ import annotations

import argparse
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
Failure-first rules: wrong instance, missing/partial edit, extra duplicate, changed neighboring instance, broken occlusion/contact, or visible artifact is fail. For add the host remains and the item is attached. For remove the entire visible target disappears. For replace the old identity disappears and a distinct new identity appears. For attribute geometry and identity remain.
Instruction: {instruction}
Target mask file is supplied for localization only; it is not a pixel-difference ground truth."""


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
    reasons = []
    if row["operation"] == "remove" and inside < 0.50:
        reasons.append("remove_mask_change_too_weak")
    if row["operation"] in {"attribute", "replace"} and inside < 0.05:
        reasons.append("no_visible_target_change")
    return {"deterministic": {"inside_changed_fraction": inside, "outside_changed_fraction": outside},
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


def _target_panel(row: dict[str, Any], directory: Path) -> str:
    """Build a high-resolution source/edited/mask crop for the semantic auditor."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (Path(row["source_image"]).stem + "__" + Path(row["mask"]).stem + "__panel.jpg")
    if path.exists():
        return str(path)
    source = Image.open(row["source_image"]).convert("RGB")
    edited = Image.open(row["edited_image"]).convert("RGB")
    mask = load_mask({"mask_path": row["mask"]}, source.size)
    ys, xs = np.where(mask)
    if len(xs):
        margin = max(32, int(round(0.9 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))))
        left = max(0, int(xs.min()) - margin)
        top = max(0, int(ys.min()) - margin)
        right = min(source.width, int(xs.max()) + 1 + margin)
        bottom = min(source.height, int(ys.max()) + 1 + margin)
        box = (left, top, right, bottom)
    else:
        box = (0, 0, source.width, source.height)
    source_crop = source.crop(box)
    edited_crop = edited.crop(box)
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
    a = p.parse_args(argv)
    selector = VLLMSelector(a.model_id, tensor_parallel_size=a.tensor_parallel_size) if a.model_id else None
    try:
        results = validate(a.manifest, a.output, selector)
        print(json.dumps({"rows": len(results), "pass": sum(x["decision"] == "pass" for x in results), "output": str(a.output)}))
    finally:
        if selector:
            selector.close()


if __name__ == "__main__":
    main()
