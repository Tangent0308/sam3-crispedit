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
            return np.asarray(decoded, dtype=bool)
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
    for y, x in zip(ys[:: max(1, len(ys) // 5000)], xs[:: max(1, len(xs) // 5000)]):
        d = int(
            min(x - xs.min(), xs.max() - x, y - ys.min(), ys.max() - y)
        )
        if d > best[2]:
            best = (int(x), int(y), d)
    # Connected components without scipy/cv2.
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
    return {
        "area_px": area,
        "area_fraction": area / float(h * w),
        "bbox": bbox,
        "point": [best[0], best[1]],
        "components": components,
        "bbox_fill": area / float(bbox_area),
        "perimeter_proxy": perimeter,
        "thinness": perimeter / max(1.0, area**0.5),
    }


def difficulty_features(objects: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any]:
    """Compute scene-level evidence; no heuristic can replace MLLM visual review."""
    features = []
    for obj in objects:
        f = mask_geometry(load_mask(obj, image_size))
        f["object_id"] = str(obj.get("object_id", obj.get("id", "")))
        f["category"] = str(obj.get("category", ""))
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
