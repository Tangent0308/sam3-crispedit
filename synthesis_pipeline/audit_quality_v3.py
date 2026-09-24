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

CORRESPONDENCE_PROMPT = """Compare BEFORE (image 1, target outlined) with AFTER (image 2, clean), at matching coordinates. The outline is not a real color, shape or object.
Do this in order:
1. Find the whole target in BEFORE, then track that SAME instance in AFTER. Compare nearby same-category instances too.
2. Establish the actual operation. Search BEFORE for anything you think is new: an already-visible person/object revealed behind a removed target is background, NOT a replacement. If the original target remains and another object appears beside it, that is addition, NOT replacement. A surface/shape rendering change to the same recognizable object is attribute, NOT replacement.
3. Inspect the old footprint, its boundary, adjacent instances, and contact with the scene. Fail a remaining target fragment, ghost/double edge, newly smeared fill, clipped feet/support, an unsupported floating insertion, or an unrelated instance changed/deleted. Compare with BEFORE: do not call pre-existing blur, partial visibility, or perspective a new defect. Small plausible texture variations are fine. Do not require a perfect photograph.
4. Only now compare with the requested instruction. Visual quality may pass when the request differs, but not when the target is unchanged, the wrong instance changed, or unrelated objects were damaged. The edit must be a visible local change associated with the outlined target/anchor.
Requested type: {task_type}
Request: {instruction}
For this requested type: {type_check}
Return ONLY three fields. Write reason first, with concrete BEFORE -> AFTER evidence and the decisive boundary/contact/neighbor finding, not a restatement of the request:
{{"reason":"brief visual comparison", "visual_quality":"pass|fail", "instruction_match":"pass|fail"}}"""

TYPE_CHECK = {
    'add': 'A genuinely new item appears at the named anchor, which remains. Its contact/placement must be plausible.',
    'remove': 'The entire named target disappears. Other previously visible objects must remain; revealed background is not a new object.',
    'replace': 'The named old target disappears and a genuinely different identity occupies its place; retaining it beside an addition does not qualify.',
    'attribute': 'The same named instance has the requested visible property change, without changing another instance or its basic identity.',
}

FORENSIC_PROMPT = """Decide whether this pair is usable as a LOCALIZED image-edit training example. Image 1 is BEFORE; its black/white contour is an annotation identifying the intended target/anchor. Image 2 is AFTER. No instruction is supplied: judge the actual change, not what might have been requested.
Compare the two pictures at identical positions. First identify what genuinely changed. Track neighboring instances individually: a second removed or recolored object is a defect even if the intended target looks good. Look along the entire old silhouette for surviving fragments, double contours, pasted edges, and texture that abruptly stops. Inspect the edited object's bottom/contact point and any narrow attachments; do not assume a support exists when it is not visible. Compare apparent defects with BEFORE so source blur/occlusion is not unfairly penalized. Background already visible BEFORE can be revealed by removal, not counted as a new replacement.
PASS requires an obvious useful change at the target, broadly natural edges, coherent physical placement and preserved unrelated instances. FAIL a barely visible/no edit, wrong instance, recognizable leftover piece, ghosted boundary, smeared reconstruction, clipped or floating object, or substantial collateral edit. Do not excuse a visible defect merely because most of the picture is unchanged. Minor plausible texture variation is acceptable. If the change cannot be established clearly, fail.
Return exactly three fields, describing observations before the verdict:
{"observed_change":"short factual BEFORE -> AFTER difference", "reason":"specific boundary, physical contact and neighbor evidence; decisive defect if present", "quality":"pass|fail"}."""

CORRESPONDENCE_REWRITE_PROMPT = """Write a short edit instruction for this BEFORE/AFTER pair. Image 1 is the original with an annotation outline around the target/anchor. Image 2 is the clean result. Ignore the outline when identifying objects or their color.
First track the same target across BOTH images, and search the original for any object you think is newly introduced. A background person/object already partly visible BEFORE and exposed by a removal is NOT a replacement. A retained object plus a new nearby item is ADD, not replace. The same recognizable object with a new finish, shape detail, or style is ATTRIBUTE, not replace. Do not infer materials, relationships or activities that the pixels cannot establish.
Use remove only when the complete target disappears; replace only when a different identity actually substitutes for it; add only for a genuinely new item; attribute for a changed property of the same instance. If multiple unrelated changes occurred, the wrong instance changed, fragments remain, a new object floats unsupported, or the change is indistinct, return null for both fields. Do not rescue a bad image by changing the target definition.
One imperative sentence, at most 25 words: operation + target uniquely locatable in the ORIGINAL full scene + visible result if applicable. Use the shortest sufficient position or stable landmark; shared category/color alone cannot identify one of several similar objects. No preservation boilerplate, reconstruction recipes, annotations, or speculative detail. You are not given the old instruction.
Return only JSON: {"task_type":"add|remove|replace|attribute or null","instruction":"English command or null"}."""

BALANCED_EVIDENCE_PROMPT = """Judge the photographic quality of a localized edit from BEFORE (image 1) and AFTER (image 2). No desired instruction is supplied. The thin black/white outline is a label drawn only on BEFORE; its absence AFTER is NOT an edit. Ignore that annotation and compare photographic content at identical coordinates.
First establish a visible change to the outlined instance or its immediate placement area. If the object looks essentially unchanged apart from small rendering differences, fail rather than inventing an operation. Track other instances in BEFORE before deciding they were added or removed: a partly visible background object becoming exposed is not newly inserted.
Then inspect the changed footprint and its boundary. Fail clearly visible leftover target pieces, double edges, pasted flat slivers, broken continuation of a tabletop/wall/rail, conspicuous blurred fill, or unrelated objects being damaged. A smooth boundary cannot excuse a recognizable leftover part inside the footprint. Compare with BEFORE: source blur, compression, perspective and pre-existing occlusion are not new defects.
Pass a clearly visible useful edit that looks broadly natural at the source image's level of detail. Judge visible defects, not speculative scene rules: do not demand a visible stand, shadow, attachment or all wheels if occlusion/resolution can reasonably explain their absence. A distant vehicle is not floating simply because its wheels are higher in image coordinates. Do not reject an ordinary object solely for being unusual at its location. Conversely, a clear detached sliver or a plainly broken contact is a defect. Do not invent hidden supports to excuse an obvious one.
Return only three fields, observations first: {"observed_change":"Actual photographic BEFORE -> AFTER change, or no clear change","reason":"Specific visible defect and its location, or why boundaries and neighboring objects look acceptable; no guesses about intention","quality":"pass|fail"}."""


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
        r"\b(masked|highlighted|selected|outlined|annotation|contour)\b|\b(?:target mask|mask region|marked region|(?:black|white|black/white|black and white) outline|image\s*[12])\b",
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


def rewrite_policy_error(row, rewrite, allow_task_type_change=False):
    """Conservative admission rules, not a claim of semantic verification.

    A mistaken replacement label for revealed background must not silently
    enter training. Cross-operation salvage remains available for explicit
    experiments/manual review rather than automatic acceptance.
    """
    if not rewrite:
        return 'null_or_invalid_response'
    if rewrite['task_type'] != row['task_type'] and not allow_task_type_change:
        return 'task_type_change_requires_manual_verification'
    if row.get('edit_unit_status') in {'complete_object', 'complete_part'} and re.search(
        r'^(?:remove|replace)\s+(?:the\s+)?(?:two|three|four|both|all|[2-9])\b',
        rewrite['editing_instruction'], re.I,
    ):
        return 'rewrite_changes_single_target_count'
    if rewrite['task_type'] == 'replace':
        replacement = re.split(r'\bwith\b', rewrite['editing_instruction'], flags=re.I)[-1]
        if re.search(r'\b(?:and|plus|along with|together with)\s+(?:a|an|another|a different|the)\b',replacement,re.I):
            return 'replacement_introduces_multiple_entities'
        if re.search(r'\b(?:walking|pushing|riding|carrying|holding)\b\s+(?:a|an|the)\b', replacement, re.I):
            return 'replacement_introduces_multiple_interacting_entities'
    return None


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


def apply_add_scope_gate(audit, task_type, metrics, maximum):
    """Reject outsized additions under an explicit fine-grained data policy.

    This measures change extent, not perceptual quality or exact object area.
    It is opt-in because some valid add datasets intentionally use tiny anchors.
    """
    if audit is None or task_type != 'add' or maximum <= 0:
        return audit
    ratio = metrics.get('changed_to_target_area_ratio')
    if ratio is None or ratio <= maximum:
        return audit
    return {**audit, 'visual_quality': 'fail', 'quality': 'fail',
            'instruction_match': 'fail', 'scope_gate': 'oversized_addition',
            'reason': audit.get('reason', '') +
            f' [Fine-grained scope gate: changed area is {ratio:.2f}x anchor area; maximum {maximum:.2f}x.]'}


def apply_flat_fill_gate(audit, task_type, metrics, enabled=False):
    if audit is None or task_type != 'attribute' or not enabled:
        return audit
    old = metrics.get('source_dominant_color_fraction', 1.)
    new = metrics.get('edited_dominant_color_fraction', 0.)
    if old >= .35 or new < .85:
        return audit
    return {**audit, 'visual_quality': 'fail', 'quality': 'fail', 'instruction_match': 'fail',
            'flat_fill_gate': 'rejected', 'reason': audit.get('reason', '') +
            f' [Texture gate: dominant RGB bin rises from {old:.1%} to {new:.1%}; likely flat fill replacing photographic detail.]'}


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
    p.add_argument('--rewrite-policy', choices=['legacy', 'correspondence'], default='correspondence')
    p.add_argument('--allow-task-type-change', action='store_true',
                   help='Experimental cross-operation rewrites; otherwise require manual verification')
    p.add_argument('--max-add-change-ratio', type=float, default=0,
                   help='Optional fine-grained size gate: changed pixels / anchor area; 0 disables')
    p.add_argument('--reject-flat-attribute', action='store_true', help='Conservative photographic texture-collapse gate')
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
    if args.max_add_change_ratio < 0:
        p.error('--max-add-change-ratio must be nonnegative')
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
            metrics = result['locality_metrics']
            if args.max_add_change_ratio and 'changed_to_target_area_ratio' not in metrics:
                raise ValueError('Scope metrics missing from saved audit; regenerate metrics/audit before enabling the size gate')
            if args.reject_flat_attribute and 'edited_dominant_color_fraction' not in metrics:
                raise ValueError('Texture metrics missing from saved audit; regenerate metrics/audit before enabling the texture gate')
            result['audit'] = apply_add_scope_gate(result['audit'], result['task_type'], metrics, args.max_add_change_ratio)
            result['audit'] = apply_flat_fill_gate(result['audit'], result['task_type'], metrics, args.reject_flat_attribute)
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
            or parts[0] not in {"legacy", "compact", "grounded", "quality", "rewrite", "correspondence", "forensic", "balanced"}
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
                "correspondence",
                "forensic",
                "balanced",
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
                    if mode == 'correspondence':
                        prompt = CORRESPONDENCE_PROMPT.format(
                            instruction=row['editing_instruction'], task_type=row['task_type'],
                            type_check=TYPE_CHECK[row['task_type']],
                        )
                    if mode == 'rewrite' and args.rewrite_policy == 'correspondence':
                        prompt = CORRESPONDENCE_REWRITE_PROMPT
                    if mode == 'forensic':
                        prompt = FORENSIC_PROMPT
                    if mode == 'balanced':
                        prompt = BALANCED_EVIDENCE_PROMPT
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
                        policy_error = rewrite_policy_error(row, val, args.allow_task_type_change)
                        result = {
                            "image": row["image"],
                            "original_instruction": row["editing_instruction"],
                            "original_task_type": row["task_type"],
                            "task_type": val["task_type"] if val else None,
                            "editing_instruction": (
                                val["editing_instruction"] if val else None
                            ),
                            "status": "candidate" if policy_error is None else "rejected_policy_or_parse",
                            "policy_error": policy_error,
                            "raw_response": text,
                            "prompt": j[2],
                        }
                    else:
                        val = (
                            normalize_result(
                                parse_audit_json(parse_text), row["task_type"]
                            )
                            if mode == "legacy"
                            else parse_compact(parse_text, mode in {"quality", "forensic", "balanced"})
                        )
                        if mode not in {"quality", "forensic", "balanced"}:
                            val = apply_low_change_veto(
                                val, row["task_type"], j[1], 0.3
                            )
                        model_val = val
                        val = apply_target_change_gate(
                            val, row["task_type"], j[1], args.target_change_threshold
                        )
                        val = apply_add_scope_gate(val, row['task_type'], j[1], args.max_add_change_ratio)
                        val = apply_flat_fill_gate(val, row['task_type'], j[1], args.reject_flat_attribute)
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
                    result['edited_path'] = str((edited_dir / row['image']).resolve())
                    result['source_path'] = str((args.data_root / 'sources' / row['source_image']).resolve())
                    result['input_scope'] = scope
                    result['input_images'] = [str(args.out_root / f'inputs_{scope}' / f'{Path(row["image"]).stem}_{suffix}.png')
                                              for suffix in ('source', 'edited')]
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
                max_add_change_ratio=args.max_add_change_ratio,
                rewrite_policy=args.rewrite_policy,
                allow_task_type_change=args.allow_task_type_change,
                reject_flat_attribute=args.reject_flat_attribute,
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
