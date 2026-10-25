"""Experimental, bounded postprocessing of an existing Qwen edit.

These functions do not generate, audit, or promote samples. Callers must audit
the returned pixels and record whether an experimental proposal was accepted.
"""
from __future__ import annotations

import re
import hashlib
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .specialist_editors import writable_mask, read_binary_mask


def noop_remove_evidence(source, edited, unit):
    """Flag a nearly unchanged remove result, never certify an edit as good.

    This is a proposal trigger for a cheap alternate editor, not a rejection
    rule. White objects on white backgrounds can also have small pixel changes.
    The normal semantic audit must examine whichever final proposal is used.
    """
    if source.size != edited.size:
        raise ValueError("source and edit must have identical dimensions")
    spec = unit.get("instruction_spec", {})
    if unit["operation"] != "remove" or spec.get("remove_mode", "whole") != "whole":
        return {"trigger": False, "reason": "not whole-target removal"}
    if re.search(r"\b(folds?|creases?|wrinkles?)\b", spec.get("ref", ""), re.I):
        return {"trigger": False, "reason": "surface retouching"}
    support = writable_mask(unit, source.size)
    mask = read_binary_mask(unit["mask_path"], source.size)
    if mask.sum() < 64 or support.mean() > .035:
        return {"trigger": False, "reason": "small selector or large writable region"}
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
    local = mask[box[1]:box[3], box[0]:box[2]]
    change = np.max(np.abs(np.asarray(source.crop(box).convert("RGB"), dtype=np.int16)
                           - np.asarray(edited.crop(box).convert("RGB"), dtype=np.int16)), axis=2)[local]
    unchanged = float((change <= 12).mean())
    mean_change = float(change.mean())
    trigger = unchanged >= .75 and mean_change <= 12
    return {"trigger": trigger, "reason": "near-noop proposal trigger" if trigger else "visible change",
            "unchanged_fraction": unchanged, "mean_max_rgb_change": mean_change,
            "does_not_prove_semantic_failure": True}


class QwenWithTools:
    """Opt-in wrapper: one Qwen generation, at most one cheap removal proposal.

    Keep the original Qwen proposal in diagnostic artifacts. Run the existing
    semantic audit after this wrapper; cached audits of Qwen's old pixels are
    not valid for a replacement proposal.
    """
    def __init__(self, qwen, lama=None, *, repair_noop_remove=False, restore_texture=False,
                 diagnostic_directory=None):
        if repair_noop_remove and lama is None:
            raise ValueError("remove repair requires an explicit LaMa editor")
        self.qwen, self.lama = qwen, lama
        self.repair_noop_remove, self.restore_texture = repair_noop_remove, restore_texture
        self.diagnostic_directory = Path(diagnostic_directory) if diagnostic_directory else None
        self.config = {**getattr(qwen, "config", {}), "tools_version": "experimental-2",
                       "repair_noop_remove": repair_noop_remove, "restore_texture": restore_texture}
        if lama is not None:
            self.config['lama'] = getattr(lama, 'config', {})

    @property
    def pipe(self):
        return self.qwen.pipe

    def generate(self, image_path, unit, seed=0):
        source = Image.open(image_path).convert("RGB")
        start = time.monotonic()
        original = self.qwen.generate(image_path, unit, seed=seed)
        qwen_seconds = time.monotonic() - start
        tool_start = time.monotonic()
        result = original
        context = dict(getattr(self.qwen, "last_generation_context", {}))
        evidence = noop_remove_evidence(source, original, unit)
        used = []
        if self.repair_noop_remove and evidence["trigger"]:
            result = self.lama.generate(image_path, unit, seed=seed)
            used.append("lama_remove_proposal")
        if self.restore_texture:
            result, texture = restore_color_texture(source, result, unit)
            context["texture_proposal"] = texture
            if texture["applied"]:
                used.append("source_texture_proposal")
        # This is a proposal, not a substitute for source/target semantic audit.
        self.last_qwen_proposal = original
        self.last_generation_context = {
            **context, "tools_config": self.config, "noop_evidence": evidence, "tools_applied": used,
            "requires_semantic_audit": True, "audit_cached_qwen_pixels_invalid": bool(used),
            "qwen_seconds": qwen_seconds, "tool_seconds": time.monotonic() - tool_start,
            "qwen_pixels_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
            "final_pixels_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        }
        if self.diagnostic_directory is not None:
            self.diagnostic_directory.mkdir(parents=True, exist_ok=True)
            path = self.diagnostic_directory / (uuid.uuid4().hex + ".png")
            original.save(path)
            self.last_generation_context["qwen_proposal_path"] = str(path)
        return result


def restore_color_texture(source, edited, unit, *, strength=.5, max_delta_l=5.):
    """Restore lost luminance detail only for geometrically aligned plain color edits.

    Qwen supplies the edited color/low-frequency shading. Do not infer a material
    mask from dominant source color. Refuse geometry mismatch and compound
    finish/pattern requirements. Return the original proposal when ineligible.
    """
    import cv2
    if source.size != edited.size:
        raise ValueError("source and edit must have identical dimensions")
    if not 0 <= strength <= 1 or not 0 <= max_delta_l <= 10:
        raise ValueError("invalid bounded texture restoration settings")
    spec = unit.get("instruction_spec", {})
    attrs = spec.get("attributes", [])
    if (unit["operation"] != "attribute" or len(attrs) != 1
            or attrs[0].get("property") != "color"):
        return edited.copy(), {"applied": False, "reason": "not a single color edit"}
    if re.search(r"\b(glossy|shiny|metallic|matte|pattern|striped|solid)\b",
                 attrs[0].get("value", ""), re.I):
        return edited.copy(), {"applied": False, "reason": "finish/pattern request"}
    support = writable_mask(unit, source.size)
    selector = read_binary_mask(unit["mask_path"], source.size)
    inside = cv2.erode((support & selector).astype("uint8"), np.ones((5, 5), np.uint8)) > 0
    if inside.sum() < 128:
        return edited.copy(), {"applied": False, "reason": "small interior"}
    src = np.asarray(source.convert("RGB"))
    tar = np.asarray(edited.convert("RGB"))
    sg = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    tg = cv2.cvtColor(tar, cv2.COLOR_RGB2GRAY)
    se, te = cv2.Canny(sg, 60, 140) > 0, cv2.Canny(tg, 60, 140) > 0
    k = np.ones((5, 5), np.uint8)
    sm = cv2.dilate(se.astype("uint8"), k) > 0
    tm = cv2.dilate(te.astype("uint8"), k) > 0
    denom = (se & inside).sum() + (te & inside).sum()
    mismatch = float(((se & ~tm & inside).sum() + (te & ~sm & inside).sum()) / max(1, denom))
    if mismatch > .18:
        return edited.copy(), {"applied": False, "reason": "geometry/edge mismatch",
                               "edge_mismatch": mismatch}
    sl = cv2.cvtColor(src.astype("float32")/255., cv2.COLOR_RGB2LAB)
    tl = cv2.cvtColor(tar.astype("float32")/255., cv2.COLOR_RGB2LAB)
    sh = sl[..., 0] - cv2.GaussianBlur(sl[..., 0], (0, 0), 1.2)
    th = tl[..., 0] - cv2.GaussianBlur(tl[..., 0], (0, 0), 1.2)
    # Strong edges are excluded: a misregistered boundary cannot be "sharpened".
    active = inside & ~sm & ~tm & (np.linalg.norm(tl-sl, axis=2) > 8)
    if active.sum() < 64:
        return edited.copy(), {"applied": False, "reason": "insufficient changed texture area"}
    srms = float(np.sqrt(np.mean(sh[active]**2)))
    trms = float(np.sqrt(np.mean(th[active]**2)))
    if srms <= max(.3, trms*1.10):
        return edited.copy(), {"applied": False, "reason": "no measurable detail loss",
                               "source_detail_rms": srms, "edited_detail_rms": trms}
    delta = np.clip(strength*(sh-th), -max_delta_l, max_delta_l)
    refined = tl.copy()
    refined[..., 0] = np.clip(refined[..., 0] + delta*active, 0, 100)
    rgb = np.rint(np.clip(cv2.cvtColor(refined, cv2.COLOR_LAB2RGB), 0, 1)*255).astype("uint8")
    out = tar.copy()
    out[active] = rgb[active]
    return Image.fromarray(out), {
        "applied": True, "reason": "bounded source detail with Qwen color",
        "edge_mismatch": mismatch, "active_pixels": int(active.sum()),
        "source_detail_rms": srms, "edited_detail_rms": trms,
        "strength": strength, "max_delta_l": max_delta_l,
        "requires_semantic_audit": True,
    }


def poisson_proposal(source, edited, unit):
    """Diagnostic only: preserve edit gradients while matching the source boundary.

    Poisson can change requested colors or restore unwanted source structure;
    it is intentionally not an automatically enabled production operation.
    """
    import cv2
    if source.size != edited.size:
        raise ValueError("source and edit must have identical dimensions")
    support = writable_mask(unit, source.size)
    # OpenCV's solver needs an exterior boundary. Never wrap image-edge masks.
    if support[:2].any() or support[-2:].any() or support[:, :2].any() or support[:, -2:].any():
        return edited.copy(), {"applied": False, "reason": "support touches image boundary"}
    ys, xs = np.nonzero(support)
    if len(xs) < 64:
        return edited.copy(), {"applied": False, "reason": "small support"}
    src = np.asarray(source.convert("RGB"))
    tar = np.asarray(edited.convert("RGB"))
    center = (int((xs.min()+xs.max()+1)//2), int((ys.min()+ys.max()+1)//2))
    result = cv2.seamlessClone(tar[..., ::-1], src[..., ::-1],
                               support.astype("uint8")*255, center, cv2.NORMAL_CLONE)[..., ::-1]
    out = tar.copy()
    out[support] = result[support]
    return Image.fromarray(out), {"applied": True, "reason": "diagnostic Poisson proposal",
                                 "requires_semantic_audit": True}
