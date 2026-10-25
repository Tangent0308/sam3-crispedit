"""Opt-in, single-attempt specialist routing; semantic audit remains mandatory.

Only image/mask measurements are used. Color clustering is a conservative
proposal, not a material segmentation guarantee. Qwen remains the fallback.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import re

from .specialist_editors import crop_box, target_color, writable_mask, composite


def remove_route(image, unit):
    import cv2
    support = writable_mask(unit, image.size)
    spec = unit.get("instruction_spec", {})
    if spec.get("remove_mode", "whole") != "whole":
        return {"backend": "qwen", "reason": "partial/contents removal needs semantic editing"}
    # Removing a fold/crease is surface retouching; erasing its pixels can
    # cut through the host. This guard does not reject the task itself.
    if re.search(r"\b(folds?|creases?|wrinkles?)\b", spec.get("ref", ""), re.I):
        return {"backend": "qwen", "reason": "surface retouching requires host continuity"}
    box = crop_box(support, 1.0)
    local = support[box[1]:box[3], box[0]:box[2]]
    gray = cv2.cvtColor(np.asarray(image.crop(box)), cv2.COLOR_RGB2GRAY)
    # Measure a band around the hole, not the removed object's own texture.
    scale = min(1., 512 / max(local.shape))
    size = (max(8, round(local.shape[1]*scale)), max(8, round(local.shape[0]*scale)))
    gray = cv2.resize(gray, size)
    m = cv2.resize(local.astype("uint8"), size, interpolation=cv2.INTER_NEAREST)
    ring = (cv2.dilate(m, np.ones((17, 17), np.uint8)) > 0) & (m == 0)
    edges = cv2.Canny(gray, 60, 140) > 0
    edge_density = float(edges[ring].mean()) if ring.any() else 1.
    fraction = float(support.mean())
    eligible = fraction <= .035 and edge_density <= .12 and local.sum() >= 16
    return {"backend": "lama" if eligible else "qwen",
            "reason": "small hole with low surrounding edge density" if eligible else
                      "large hole or structured surrounding background",
            "support_fraction": fraction, "ring_edge_density": edge_density}


def material_proposal(image, unit):
    """Select the dominant Lab color population inside the actual source mask.

    No object-specific source color, model call, or hand-painted mask is used.
    Color and material are not equivalent: borderline populations fall back.
    """
    import cv2
    target = target_color(unit).astype("float32")
    support = writable_mask(unit, image.size)
    box = crop_box(support, .1)
    rgb = np.asarray(image.crop(box), dtype=np.float32) / 255.
    local = support[box[1]:box[3], box[0]:box[2]]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    pixels = lab[local]
    if len(pixels) < 64:
        raise ValueError("too few pixels for material selection")
    sampled = pixels[::max(1, len(pixels)//4096)].copy()
    sampled[:, 0] *= .35  # do not split a shaded material solely by brightness
    # Deterministic farthest-point initialization and Lloyd updates.
    centers = [sampled[np.argmin(np.linalg.norm(sampled-np.median(sampled, axis=0), axis=1))]]
    for _ in range(3):
        distance = np.min([np.sum((sampled-c)**2, axis=1) for c in centers], axis=0)
        centers.append(sampled[np.argmax(distance)])
    centers = np.asarray(centers)
    for _ in range(12):
        labels = np.argmin(((sampled[:, None]-centers)**2).sum(2), axis=1)
        for i in range(4):
            if (labels == i).any():
                centers[i] = sampled[labels == i].mean(0)
    counts = np.bincount(labels, minlength=4)
    center = centers[counts.argmax()]
    weighted = lab.copy()
    weighted[..., 0] *= .35
    distance = np.linalg.norm(weighted-center, axis=2)
    active = local & (distance < 13)
    coverage = float(active.sum()/local.sum())
    median_l = float(np.median(lab[..., 0][active])) if active.any() else 0.
    if coverage < .70 or median_l < 18:
        raise ValueError(f"ambiguous/mixed/dark surface: coverage={coverage:.3f}, lightness={median_l:.2f}")
    desired = cv2.cvtColor(target.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]
    center_lab = np.median(lab[active], axis=0)
    if np.linalg.norm(desired-center_lab) < 8:
        raise ValueError("color change too small")
    return {"box": box, "active": active, "lab": lab, "desired": desired,
            "center_lab": center_lab, "support": support,
            "coverage": coverage, "median_lightness": median_l}


def choose_route(image, unit):
    if unit["operation"] == "remove":
        return remove_route(image, unit)
    if unit["operation"] == "attribute":
        try:
            p = material_proposal(image, unit)
        except ValueError as exc:
            return {"backend": "qwen", "reason": str(exc)}
        return {"backend": "material-color", "reason": "dominant color population",
                "coverage": p["coverage"], "median_lightness": p["median_lightness"]}
    return {"backend": "qwen", "reason": "semantic generation required"}


class MaterialColorEditor:
    backend_name = "material-color-lab-experimental"

    def __init__(self):
        self.config = {"backend": self.backend_name, "version": 2, "experimental": True}

    def generate(self, image_path, unit, seed=0):
        import cv2
        source = Image.open(image_path).convert("RGB")
        p = material_proposal(source, unit)
        lab = p["lab"].copy()
        delta = p["desired"]-p["center_lab"]
        # Preserve source local shading and chromatic variations.
        delta[0] = np.clip(delta[0], -32, 32)
        lab[..., 0] += delta[0]
        # Reduce gamut clipping rather than flattening highlights into RGB red.
        lab[..., 1:] = .75*p["desired"][1:] + .35*(lab[..., 1:]-p["center_lab"][1:])
        lab[..., 0] = np.clip(lab[..., 0], 0, 100)
        rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
        edited = source.copy()
        edited.paste(Image.fromarray(np.rint(rgb*255).astype("uint8")), p["box"][:2])
        material = np.zeros_like(p["support"])
        x0, y0, x1, y1 = p["box"]
        material[y0:y1, x0:x1] = p["active"]
        result = composite(source, edited, material, feather=2)
        self.last_material_mask = material
        self.last_generation_context = {"editor": self.config, "material_coverage": p["coverage"]}
        return result


class MiganEditor:
    """Official ONNX pipeline: uint8 NCHW, white means known/retained pixels."""
    backend_name = "migan-onnx"

    def __init__(self, model_id, providers=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model_id), sess_options=options,
                                           providers=providers or ["CPUExecutionProvider"])
        self.config = {"backend": self.backend_name, "model_id": str(model_id),
                       "providers": self.session.get_providers()}

    def generate(self, image_path, unit, seed=0):
        if unit["operation"] != "remove":
            raise ValueError("MI-GAN is only eligible for remove")
        source = Image.open(image_path).convert("RGB")
        support = writable_mask(unit, source.size)
        rgb = np.asarray(source).transpose(2, 0, 1)[None].copy()
        known = ((~support).astype("uint8")*255)[None, None]
        result = self.session.run(None, {"image": rgb, "mask": known})[0]
        edited = Image.fromarray(result[0].transpose(1, 2, 0).astype("uint8"))
        return composite(source, edited, support)


class SpecialistRouter:
    """Optional wrapper. One selected editor call, no hidden regeneration."""

    def __init__(self, qwen, lama, *, enable_material=False):
        self.qwen, self.lama = qwen, lama
        self.material = MaterialColorEditor()
        self.enable_material = enable_material
        self.config = {"backend": "specialist-router-experimental", "version": 1,
                       "default": "qwen", "material_enabled": enable_material}

    def generate(self, image_path, unit, seed=0):
        with Image.open(image_path) as src:
            route = choose_route(src.convert("RGB"), unit)
        if route["backend"] == "material-color" and not self.enable_material:
            route = {"backend": "qwen", "reason": "material routing not validated for production"}
        editor = {"lama": self.lama, "material-color": self.material}.get(route["backend"], self.qwen)
        result = editor.generate(image_path, unit, seed=seed)
        self.last_generation_context = {**getattr(editor, "last_generation_context", {}),
                                        "specialist_route": route, "requires_semantic_audit": True}
        return result
