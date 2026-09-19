"""Paired audit ablations and an independent instruction-reconstruction stage."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
import utils.vlm_utils as vlm
from synthesis_pipeline.audit_edit_pairs import (
    mask_array,
    locality_metrics,
    write_jsonl,
)
from synthesis_pipeline.audit_edit_pairs_v2 import (
    build_prompt,
    parse_audit_json,
    normalize_result,
    apply_low_change_veto,
)
from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs
from synthesis_pipeline.audit_edit_pairs import parse_json_object

COMPACT_PROMPT = """Review the two aligned image views. IMAGE 1 is the original; the black/white outline marks the target without changing its colors. IMAGE 2 is the edited result. The outline is an annotation, not an object or an edit.
Instruction: {instruction}
Edit type: {task_type}
Compare what actually changed on that specific target and inspect the surrounding scene. Set visual_quality to pass only for a clearly visible, localized, broadly natural edit. Fail recognizable target remnants, ghost contours, obvious blur/fill seams, distorted objects, implausible support, or substantial unrelated changes. Small plausible texture differences are acceptable.
Set instruction_match independently: pass only if the requested instance, operation, identity/property and count visibly match. For add, the new item must be new and the anchor retained; for remove, the target must disappear; for replace, a new identity must replace the old one; for attribute, the property must change while identity remains. Verify the before/after pixels; do not assume the request happened. A natural actual edit can pass visual_quality while failing instruction_match.
Return only JSON with three fields: {{"visual_quality":"pass|fail","instruction_match":"pass|fail","reason":"Brief concrete before/after evidence for the decision."}}"""

QUALITY_PROMPT = """Inspect this pair for use as localized image-edit training data. IMAGE 1 is the original, with the target mask marked by a black/white outline; IMAGE 2 is the edited image at the same coordinates. The outline is only an annotation. No requested instruction is supplied: judge the actual visible operation.
Identify the real change at the outlined object or its immediate placement area. The result must contain a clear, describable local edit, preserve unrelated objects, and look broadly natural at the source image's level of detail. Reject no visible change, a different edited instance, partial target remnants, transparent ghosts, conspicuous blur/hole/seam, malformed objects, impossible scale or support, and unrelated insertions/deletions. For apparent removal inspect the whole old footprint and dependent parts; for an addition inspect its contact/support; for a color/material change inspect retained geometry. Original source blur or occlusion alone is not a new defect. Minor plausible texture differences are acceptable.
Return only JSON with three fields: {"quality":"pass|fail","observed_change":"One short factual description, or no visible edit.","reason":"Specific visual evidence, including any decisive defect."}. Judge the pictures, not an imagined edit request."""

GROUNDED_PROMPT = """Compare the actual pixels in two aligned photographs. IMAGE 1 is BEFORE; its black/white outline marks the target or placement anchor. IMAGE 2 is AFTER. The outline is annotation, not appearance.
Before deciding, inspect the original target's actual color, shape and parts, then the same coordinates AFTER. Establish a meaningful visible change rather than guessing it from the request. An existing hand, existing silver finish or pre-existing occlusion is not a new edit. Inspect the entire old footprint and immediately adjacent objects, not just the most attractive new object.
Visual quality passes for a clear, useful localized change with broadly natural geometry and boundaries at the source photo's level of detail. Reject recognizable old-object fragments or ghost edges, conspicuous fill/blur/seams, unexplained attached parts or cast shadows left after removal, and separate unrelated additions/deletions. Check physical contact: an inserted object must plausibly rest on a surface, be held, hang, or otherwise fit the depicted physics. A clean-looking object floating unsupported is not enough. Background objects already partly visible before must not be mistaken for remnants; retain those objects. Minor plausible texture differences and blur already present BEFORE are acceptable.
Judge instruction match separately. Add must introduce a new item at the selected anchor; remove must eliminate the complete named target; replace must substitute the target's identity; attribute must visibly change the requested property of the same instance. Quality may pass even when the actual natural edit differs from the request; that mismatch must fail instruction_match. A target indistinguishable before and after fails both.
The following is intended action, NOT evidence that it occurred:
Type: {task_type}
Request: {instruction}
Return only three JSON fields, writing evidence FIRST: {{"reason":"Actual BEFORE appearance -> actual AFTER appearance; decisive boundary/support/collateral evidence.","visual_quality":"pass|fail","instruction_match":"pass|fail"}}. Do not repeat the request as if observed; cite what the images visibly establish."""

REWRITE_PROMPT = """Write a concise training instruction describing the ACTUAL localized change between these images. IMAGE 1 is the original with a black/white outline marking the target or placement anchor. IMAGE 2 is the clean edited result. The outline is annotation only. You are not given an old instruction.
First compare the SAME outlined object in both images: establish what was already there before deciding what is new. A small rendering/hand-detail difference alone is not a useful edit. Changes on a different instance or only outside the target/placement area are not eligible. Do not invent an instruction for an unchanged target.
Use the original image to identify the target unambiguously, especially among similar instances: include a visible position or relation. Describe only the dominant clear change. Use attribute and the verb Change/Make for color, material, surface finish or orientation changes of the same object; do NOT call those replace. Use replace only for a different object identity/category. Add requires a genuinely new object; remove requires the complete selected object to disappear. If the mask selects several components, the instruction must cover their actual changed count.
Keep one short imperative sentence, preferably under 25 words and never over 35. Do not mention masks, outlines, image numbers, unchanged regions, reconstruction procedures, or speculative details. The instruction must be actionable from the original image alone. Do not shrink the target's definition to excuse remnants, describe an unsupported floating object as intentional, or silently omit a substantial collateral change. Partial removal, ghosts, wrong-instance changes and broadly damaged results must return null for both fields, not a clever caption. If a clean useful edit cannot be established from the pixels, return null for both fields.
Return only JSON: {"task_type":"add|remove|replace|attribute or null","instruction":"concise English imperative or null"}."""


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def messages(before, after, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before},
                {"type": "image", "image": after},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def paired_views(source, edited, mask, scope):
    if scope != "overview":
        return audit_two_image_inputs(
            source,
            edited,
            mask,
            longest_side=max(source.size) if scope == "full" else 1280,
            scope="full" if scope == "full" else "context_crop",
        )
    full = audit_two_image_inputs(source, edited, mask, longest_side=1024, scope="full")
    detail = audit_two_image_inputs(
        source, edited, mask, longest_side=1024, scope="context_crop"
    )
    panels = []
    for whole, crop in zip(full, detail):
        panel = Image.new("RGB", (1024, 1344), "white")
        draw = ImageDraw.Draw(panel)
        for y, picture, label in [
            (0, whole, "FULL SCENE: USE THIS FOR TARGET LOCATION"),
            (672, crop, "MAGNIFIED TARGET CONTEXT: SAME PHOTO"),
        ]:
            draw.text((8, y + 8), label, fill="black")
            tile = ImageOps.contain(picture, (1024, 640))
            panel.paste(
                tile, ((1024 - tile.width) // 2, y + 32 + (640 - tile.height) // 2)
            )
        panels.append(panel)
    return tuple(panels)


def parse_compact(raw, quality_only=False):
    if "<think>" in raw and "</think>" not in raw:
        return None
    raw = raw.rsplit("</think>", 1)[-1]
    p = parse_json_object(raw)
    if (
        not isinstance(p, dict)
        or not isinstance(p.get("reason"), str)
        or not p["reason"].strip()
    ):
        return None
    if quality_only:
        if p.get("quality") not in ("pass", "fail") or not isinstance(
            p.get("observed_change"), str
        ):
            return None
        return {
            "visual_quality": p["quality"],
            "instruction_match": None,
            "quality": p["quality"],
            "observed_edit": p["observed_change"],
            "reason": p["reason"],
        }
    if any(
        p.get(k) not in ("pass", "fail")
        for k in ("visual_quality", "instruction_match")
    ):
        return None
    p["quality"] = (
        "pass" if p["visual_quality"] == p["instruction_match"] == "pass" else "fail"
    )
    return p


def parse_rewrite(raw):
    """Reject malformed/annotation-dependent labels; null means abstention."""
    if "<think>" in raw and "</think>" not in raw:
        return None
    val = parse_json_object(raw.rsplit("</think>", 1)[-1])
    if not isinstance(val, dict):
        return None
    instruction = val.get("instruction")
    if not isinstance(instruction, str) or not 0 < len(instruction.split()) <= 35:
        return None
    if val.get("task_type") not in {"add", "remove", "replace", "attribute"}:
        return None
    # Do not ban legitimate scene objects such as a surgical mask. Reject
    # explicit annotation references rather than arbitrary word substrings.
    if re.search(
        r"\b(masked|highlighted|selected|outlined)\b|\b(?:target mask|mask region|marked region|black/white outline|image\s*[12])\b",
        instruction.lower(),
    ):
        return None
    return {"task_type": val["task_type"], "editing_instruction": instruction.strip()}


def corrected_annotation(row, rewrite, quality_audit):
    """Quality gate cannot be overridden by an attractive rewritten instruction."""
    if (quality_audit.get("audit") or {}).get("visual_quality") != "pass":
        return None
    if rewrite.get("status") != "candidate":
        return None
    result = dict(row)
    result["editing_instruction"] = rewrite["editing_instruction"]
    result["task_type"] = rewrite["task_type"]
    result["instruction_revision"] = {
        "original_instruction": row["editing_instruction"],
        "original_task_type": row["task_type"],
        "method": "independent_two_image_reconstruction",
        "quality_reason": quality_audit["audit"].get("reason"),
        "verification": "model_accepted_not_manually_verified",
    }
    return result


def apply_target_change_gate(audit, task_type, metrics, threshold):
    """Conservative geometry gate, preserving the raw VLM judgment separately.

    An add may legitimately leave its anchor unchanged. Other tasks must have
    a measurable change inside the supplied target, not only on a neighbor.
    This rejects some tiny real edits; it is not a semantic quality oracle.
    """
    if audit is None or task_type == "add" or threshold <= 0:
        return audit
    if metrics["inside_changed_fraction"] >= threshold:
        return audit
    result = dict(audit)
    result.update(
        visual_quality="fail",
        instruction_match="fail",
        quality="fail",
        target_change_gate="rejected",
    )
    result["reason"] = (
        audit.get("reason", "")
        + f" [Geometry gate: only {metrics['inside_changed_fraction']:.2%} "
        f"of target pixels visibly changed; minimum {threshold:.2%}. "
        "Changes outside the target cannot substitute for editing it.]"
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--annotations-jsonl", type=Path)
    p.add_argument("--edited-dir", type=Path)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--variants",
        default="compact_crop",
        help="legacy_full, compact_full, compact_crop, compact_overview, quality_full, quality_crop",
    )
    p.add_argument("--rewrite-from", type=Path)
    p.add_argument(
        "--rewrite-scope", choices=["full", "crop", "overview"], default="overview"
    )
    p.add_argument(
        "--rewrite-after-audit",
        action="store_true",
        help="Reuse the resident model to reconstruct instructions for visual passes only",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Opt-in Qwen3.8 thinking experiment; more tokens, not more calls",
    )
    p.add_argument("--thinking-max-tokens", type=int, default=2048)
    p.add_argument(
        "--reasoning-effort", choices=["low", "medium", "xhigh"], default="low"
    )
    p.add_argument(
        "--target-change-threshold",
        type=float,
        default=0.0,
        help="Optional conservative non-add target gate; 0 disables; pilot recommendation .02",
    )
    p.add_argument(
        "--model-id",
        default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B",
    )
    args = p.parse_args()
    if not 0 <= args.target_change_threshold <= 1:
        p.error("--target-change-threshold must be between 0 and 1")
    if args.batch_size < 1 or args.thinking_max_tokens < 1:
        p.error("Batch size and token limit must be positive")
    rows = read(args.annotations_jsonl or args.data_root / "annotations.jsonl")
    if len({row["image"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case image names in the input manifest")
    edited_dir = args.edited_dir or args.data_root / "edited"
    if args.rewrite_from:
        quality_audits = {r["image"]: r for r in read(args.rewrite_from)}
        for result in quality_audits.values():
            result["audit"] = apply_target_change_gate(
                result.get("audit"),
                result["task_type"],
                result["locality_metrics"],
                args.target_change_threshold,
            )
        approved = {
            image
            for image, r in quality_audits.items()
            if (r.get("audit") or {}).get("visual_quality") == "pass"
        }
        rows = [r for r in rows if r["image"] in approved]
        variants = [f"rewrite_{args.rewrite_scope}"]
    else:
        variants = args.variants.split(",")
    for variant in variants:
        parts = variant.rsplit("_", 1)
        if (
            len(parts) != 2
            or parts[0] not in {"legacy", "compact", "grounded", "quality", "rewrite"}
            or parts[1] not in {"full", "crop", "overview"}
        ):
            p.error(f"Unsupported variant: {variant}")
        if parts[0] == "rewrite" and not args.rewrite_from:
            p.error(
                "Standalone rewrite variants require --rewrite-from quality results"
            )
    if args.rewrite_after_audit and (args.rewrite_from or len(variants) != 1):
        raise ValueError(
            "--rewrite-after-audit requires exactly one audit variant, without --rewrite-from"
        )
    if not rows:
        if not args.rewrite_from:
            raise ValueError("No cases to process")
        empty_out = args.out_root / variants[0]
        empty_out.mkdir(parents=True, exist_ok=True)
        if (empty_out / "rewrites.jsonl").exists():
            raise FileExistsError(empty_out / "rewrites.jsonl")
        write_jsonl(empty_out / "rewrites.jsonl", [])
        write_jsonl(empty_out / "model_accepted_annotations.jsonl", [])
        (empty_out / "summary.json").write_text(
            json.dumps(
                {
                    "cases": 0,
                    "vlm_calls": 0,
                    "reason": "No visual-quality passes; model not loaded",
                }
            )
        )
        return
    vlm.configure_backend(
        "qwen38-vllm", model_id=args.model_id, device="cuda:0", dtype="bf16"
    )
    start = time.perf_counter()
    backend = vlm.get_backend()
    load_seconds = time.perf_counter() - start
    if args.thinking:
        # Keep the tested 27B memory settings; only switch the chat template and
        # sampling to the settings documented in the local official model card.
        backend.enable_thinking = True
        backend.chat_template_overrides = {"reasoning_effort": args.reasoning_effort}
        backend.sampling_overrides = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
    try:
        for variant in variants:
            variant_start = time.perf_counter()
            mode, scope = variant.rsplit("_", 1)
            if mode not in {
                "legacy",
                "compact",
                "grounded",
                "quality",
                "rewrite",
            } or scope not in {"full", "crop", "overview"}:
                raise ValueError(variant)
            out = args.out_root / variant
            out.mkdir(parents=True, exist_ok=True)
            output = out / (
                "rewrites.jsonl" if mode == "rewrite" else "edit_audit.jsonl"
            )
            if output.exists():
                raise FileExistsError(f"Use a new experiment directory: {output}")
            results = []
            elapsed = 0
            for offset in tqdm(range(0, len(rows), args.batch_size), desc=variant):
                batch = rows[offset : offset + args.batch_size]
                jobs = []
                for row in batch:
                    source = Image.open(
                        args.data_root / "sources" / row["source_image"]
                    ).convert("RGB")
                    edited = Image.open(edited_dir / row["image"]).convert("RGB")
                    mask = mask_array(source.size, row["mask"])
                    before, after = paired_views(source, edited, mask, scope)
                    inputs = args.out_root / f"inputs_{scope}"
                    inputs.mkdir(exist_ok=True)
                    stem = Path(row["image"]).stem
                    before.save(inputs / f"{stem}_source.png")
                    after.save(inputs / f"{stem}_edited.png")
                    prompt = (
                        build_prompt(row, "legacy")
                        if mode == "legacy"
                        else (
                            COMPACT_PROMPT.format(
                                instruction=row["editing_instruction"],
                                task_type=row["task_type"],
                            )
                            if mode == "compact"
                            else QUALITY_PROMPT if mode == "quality" else REWRITE_PROMPT
                        )
                    )
                    if mode == "grounded":
                        prompt = GROUNDED_PROMPT.format(
                            instruction=row["editing_instruction"],
                            task_type=row["task_type"],
                        )
                    if scope == "overview":
                        prompt = (
                            "Each input contains the FULL SCENE above and its aligned magnified target context below. They are two views of the SAME photograph. Use the FULL SCENE for unambiguous position/instance references, and the detail for appearance. Never say highlighted or selected; use visible distinguishing attributes or stable relations to neighboring objects.\n"
                            + prompt
                        )
                    jobs.append(
                        (
                            messages(before, after, prompt),
                            (
                                None
                                if mode == "rewrite"
                                else locality_metrics(source, edited, mask, 24)
                            ),
                            prompt,
                        )
                    )
                max_tokens = (
                    args.thinking_max_tokens
                    if args.thinking
                    else (512 if mode == "legacy" else 256)
                )
                t = time.perf_counter()
                raw = backend.chat_batch(
                    [j[0] for j in jobs], max_new_tokens=max_tokens
                )
                elapsed += time.perf_counter() - t
                for row, text, j in zip(batch, raw, jobs):
                    parse_text = text if not args.thinking or "</think>" in text else ""
                    if mode == "rewrite":
                        val = parse_rewrite(parse_text)
                        result = {
                            "image": row["image"],
                            "original_instruction": row["editing_instruction"],
                            "original_task_type": row["task_type"],
                            "task_type": val["task_type"] if val else None,
                            "editing_instruction": (
                                val["editing_instruction"] if val else None
                            ),
                            "status": "candidate" if val else "rejected_or_parse_error",
                            "raw_response": text,
                            "prompt": j[2],
                        }
                    else:
                        val = (
                            normalize_result(
                                parse_audit_json(parse_text), row["task_type"]
                            )
                            if mode == "legacy"
                            else parse_compact(parse_text, mode == "quality")
                        )
                        if mode != "quality":
                            val = apply_low_change_veto(
                                val, row["task_type"], j[1], 0.3
                            )
                        model_val = val
                        val = apply_target_change_gate(
                            val, row["task_type"], j[1], args.target_change_threshold
                        )
                        result = {
                            "image": row["image"],
                            "task_type": row["task_type"],
                            "editing_instruction": row["editing_instruction"],
                            "quality": val["quality"] if val else "parse_error",
                            "audit": val,
                            "locality_metrics": j[1],
                            "raw_response": text,
                            "prompt": j[2],
                            "rewrite_candidate": None,
                            "salvage_status": "none",
                        }
                        result["model_audit_before_target_gate"] = model_val
                    result["reasoning_complete"] = (
                        "</think>" in text if args.thinking else None
                    )
                    results.append(result)
                    with output.open("a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if mode == "rewrite":
                by_image = {r["image"]: r for r in results}
                corrected = [
                    corrected_annotation(
                        r, by_image[r["image"]], quality_audits[r["image"]]
                    )
                    for r in rows
                ]
                write_jsonl(
                    out / "model_accepted_annotations.jsonl",
                    [r for r in corrected if r is not None],
                )
            wall_seconds = time.perf_counter() - variant_start
            summary = {
                "variant": variant,
                "cases": len(results),
                "counts": dict(
                    Counter(r.get("quality", r.get("status")) for r in results)
                ),
                "model": args.model_id,
                "load_seconds": load_seconds,
                "inference_seconds": elapsed,
                "cases_per_minute": len(results) * 60 / elapsed,
                "wall_seconds_excluding_load": wall_seconds,
                "wall_cases_per_minute": len(results) * 60 / wall_seconds,
                "images_per_case": 2,
                "vlm_calls": len(results),
            }
            summary.update(
                thinking=args.thinking,
                max_new_tokens=max_tokens,
                reasoning_effort=args.reasoning_effort if args.thinking else None,
                incomplete_reasoning=sum(
                    r.get("reasoning_complete") is False for r in results
                ),
                target_change_threshold=args.target_change_threshold,
                sampling=getattr(backend, "sampling_overrides", {"temperature": 0.0}),
            )
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
            if args.rewrite_after_audit and mode != "rewrite":
                quality_audits = {r["image"]: r for r in results}
                rows = [
                    r
                    for r in rows
                    if (quality_audits[r["image"]].get("audit") or {}).get(
                        "visual_quality"
                    )
                    == "pass"
                ]
                if rows:
                    variants.append(f"rewrite_{args.rewrite_scope}")
                else:
                    write_jsonl(args.out_root / "model_accepted_annotations.jsonl", [])
    finally:
        vlm.shutdown_backend()


if __name__ == "__main__":
    main()
