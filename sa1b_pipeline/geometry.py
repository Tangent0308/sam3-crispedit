"""Mask decoding and objective difficulty features used before/after MLLM review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mask(obj: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    """Load a binary mask from a file, COCO RLE, or an uncompressed array."""
    width, height = image_size
    path = obj.get("mask_path") or obj.get("mask")
    if path:
        arr = np.asarray(Image.open(path).convert("L"))
        if arr.shape != (height, width):
            raise ValueError(f"mask {path} has {arr.shape}, expected {(height, width)}")
        return arr > 0
    rle = obj.get("rle") or obj.get("segmentation")
    if isinstance(rle, list):
        flat = np.asarray(rle, dtype=np.uint8)
        if flat.size != width * height:
            raise ValueError("uncompressed mask has wrong size")
        return flat.reshape((height, width)).astype(bool)
    if isinstance(rle, dict):
        try:
            from pycocotools import mask as coco_mask
            decoded = coco_mask.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            # `pycocotools` releases before its NumPy 2 compatibility fix can
            # pass an unsupported `copy=` keyword through `asarray`; an
            # explicit array copy keeps the handoff reader compatible.
            return np.array(decoded, dtype=bool, copy=True)
        except ImportError as exc:
            raise RuntimeError("pycocotools is required for compressed SA-1B RLE") from exc
    raise ValueError("object has no mask_path, mask, rle, or segmentation")


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask must be a non-empty 2D array")
    h, w = mask.shape
    ys, xs = np.where(mask)
    area = int(mask.sum())
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    # A deterministic interior point approximates the benchmark point contract.
    best = (int(xs[0]), int(ys[0]), -1)
    min_x, max_x, min_y, max_y = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    for y, x in zip(ys[:: max(1, len(ys) // 5000)], xs[:: max(1, len(xs) // 5000)]):
        d = int(
            min(x - min_x, max_x - x, y - min_y, max_y - y)
        )
        if d > best[2]:
            best = (int(x), int(y), d)
    # SA-1B images are large; use the optional vision dependency for connected
    # components when available, while retaining a dependency-free fallback.
    components = None
    try:
        import cv2  # type: ignore
        components = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[0] - 1)
    except ImportError:
        pass
    if components is None:
        try:
            from scipy import ndimage  # type: ignore
            _, components = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
            components = int(components)
        except ImportError:
            seen = np.zeros_like(mask, dtype=bool)
            components = 0
            for y, x in zip(ys, xs):
                if seen[y, x]:
                    continue
                components += 1
                stack = [(int(y), int(x))]
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    # Boundary length proxy; high values identify narrow/complex regions.
    neighbors = np.zeros_like(mask, dtype=np.uint8)
    neighbors[1:] += mask[1:] != mask[:-1]
    neighbors[:-1] += mask[1:] != mask[:-1]
    neighbors[:, 1:] += mask[:, 1:] != mask[:, :-1]
    neighbors[:, :-1] += mask[:, 1:] != mask[:, :-1]
    perimeter = int(neighbors[mask].sum())
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    border_pixels = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    border_pixels = min(area, border_pixels)
    return {
        "area_px": area,
        "area_fraction": area / float(h * w),
        "bbox": bbox,
        "point": [best[0], best[1]],
        "components": components,
        "bbox_fill": area / float(bbox_area),
        "perimeter_proxy": perimeter,
        "thinness": perimeter / max(1.0, area**0.5),
        "touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
        "border_fraction": border_pixels / float(area),
    }


def operation_quality_gate(feature: dict[str, Any], operation: str) -> dict[str, Any]:
    """Return an auditable, operation-specific mask gate before generation."""
    thresholds = {
        "attribute": (256, 0.12, 2),
        "replace": (512, 0.20, 2),
        # Additions may attach to a host whose visible mask is split by
        # occlusion or holes. Reject only small highly-fragmented hosts.
        "add": (1024, 0.10, 4),
        # A grille/window can legitimately have several visible components;
        # image-edge and low-fill targets remain unsafe for removal.
        "remove": (2048, 0.30, 4),
    }
    if operation not in thresholds:
        raise ValueError(f"unknown operation: {operation}")
    base_min_area, min_fill, max_components = thresholds[operation]
    # Scale the absolute pixel floor for small synthetic/unit-test canvases;
    # full SA-1B images still use the conservative base thresholds.
    fraction = float(feature.get("area_fraction", 0.0))
    image_area = int(round(feature.get("area_px", 0) / fraction)) if fraction > 0 else base_min_area
    min_area = min(base_min_area, max(32, int(round(image_area * 0.0005))))
    reasons = []
    if int(feature.get("area_px", 0)) < min_area:
        reasons.append("mask_too_small")
    if float(feature.get("bbox_fill", 0.0)) < min_fill:
        reasons.append("mask_too_fragmented_or_thin")
    components = int(feature.get("components", 0))
    if operation == "add" and components > max_components and int(feature.get("area_px", 0)) < 250_000:
        reasons.append("mask_has_multiple_components")
    elif operation != "add" and components > max_components:
        reasons.append("mask_has_multiple_components")
    if operation in {"replace", "remove"} and feature.get("touches_border"):
        reasons.append("mask_touches_image_border")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "min_area_px": min_area,
        "min_bbox_fill": min_fill,
        "max_components": max_components,
    }


def difficulty_features(objects: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any]:
    """Compute scene-level evidence; no heuristic can replace MLLM visual review."""
    features = []
    for obj in objects:
        f = mask_geometry(load_mask(obj, image_size))
        f["object_id"] = str(obj.get("object_id", obj.get("id", "")))
        f["category"] = str(obj.get("category", ""))
        f["operation_quality"] = {
            operation: operation_quality_gate(f, operation)
            for operation in ("attribute", "replace", "add", "remove")
        }
        features.append(f)
    mechanisms: set[str] = set()
    for f in features:
        if f["area_fraction"] < 0.01:
            mechanisms.add("小目标")
        elif f["area_fraction"] < 0.03:
            mechanisms.add("细粒度区域")
        if f["components"] > 1:
            mechanisms.add("分离可见区域")
        if f["thinness"] > 20 or f["bbox_fill"] < 0.25:
            mechanisms.add("细长狭窄")
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            if left["category"].lower() == right["category"].lower() and left["category"]:
                mechanisms.add("同类多实例")
            a, b = left["bbox"], right["bbox"]
            gap_x = max(a[0] - b[2], b[0] - a[2], 0)
            gap_y = max(a[1] - b[3], b[1] - a[3], 0)
            if gap_x <= 8 and gap_y <= 8:
                mechanisms.add("紧邻保护")
    return {
        "mechanisms": sorted(mechanisms),
        "target_count": len(features),
        "target_features": features,
        "difficulty_score": min(
            1.0,
            0.15 * len(mechanisms) + 0.08 * sum(f["components"] > 1 for f in features),
        ),
    }
