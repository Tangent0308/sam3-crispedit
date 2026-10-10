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


AUDIT_PROMPT = """Judge this fine-grained edit pair. Images are ordered clean source, edited result, then a cyan target-mask overlay for localization. Compare the first two using the third mask and instruction.
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
    if inside < 0.01 and row["operation"] != "add":
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
            overlay = _mask_overlay(row["source_image"], row["mask"], output.parent / "audit_overlays")
            jobs.append(([row["source_image"], row["edited_image"], overlay], AUDIT_PROMPT.replace('{instruction}', row['instruction'])))
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="generated.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    a = p.parse_args(argv)
    selector = VLLMSelector(a.model_id, tensor_parallel_size=a.tensor_parallel_size) if a.model_id else None
    results = validate(a.manifest, a.output, selector)
    print(json.dumps({"rows": len(results), "pass": sum(x["decision"] == "pass" for x in results), "output": str(a.output)}))


if __name__ == "__main__":
    main()
