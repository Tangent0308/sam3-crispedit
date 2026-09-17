"""Audit edited pairs visually and measure whether pixel changes stay localized."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.vlm_utils as vlm


AUDIT_PROMPT = """Compare three images in order: (1) SOURCE, (2) SOURCE WITH RED TARGET MASK, and (3) EDITED.
Requested edit: <<<INSTRUCTION>>>
Target region(s): <<<TARGETS>>>

Judge the edited result for training-pair quality:
- The requested change should be visibly present on the intended instance/part.
- Other same-category instances must remain unedited.
- Background, composition, identity, and unrelated objects should remain substantially preserved.
- Small diffusion/rendering differences are acceptable; reject a wrong-instance edit, absent requested change, severe artifacts, or broad scene replacement.

Output one JSON object only:
{"edit_visible":true,"correct_instance":true,"non_target_preserved":true,"scene_preserved":true,"quality":"pass","reason":"brief visible evidence"}
`quality` must be `pass`, `review`, or `fail`. No markdown or extra text."""

AUDIT_VERSION = "edit_pair_visual_locality_v2_cached"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--guard-pixels", type=int, default=24)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--vlm", choices=["gemma4", "qwen8b", "qwen4b", "qwen35"], default="qwen8b"
    )
    parser.add_argument("--vlm-model-id", default=None)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--vlm-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse audit rows whose source, edited image, annotation, prompt, "
            "and model settings have the same content fingerprint."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(
    row: dict[str, Any], source_path: Path, edited_path: Path, args: argparse.Namespace
) -> str:
    payload = {
        "audit_version": AUDIT_VERSION,
        "audit_prompt": AUDIT_PROMPT,
        "image": row.get("image"),
        "editing_instruction": row.get("editing_instruction"),
        "refer_object": row.get("refer_object"),
        "mask": row.get("mask"),
        "source_sha256": file_sha256(source_path),
        "edited_sha256": file_sha256(edited_path),
        "guard_pixels": args.guard_pixels,
        "vlm": args.vlm,
        "vlm_model_id": args.vlm_model_id,
        "vlm_dtype": args.vlm_dtype,
        "max_new_tokens": args.max_new_tokens,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = str(text).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_polygons(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if all(isinstance(item, (int, float)) for item in value):
        return [[float(item) for item in value]] if len(value) >= 6 else []
    return [
        [float(number) for number in polygon]
        for polygon in value
        if isinstance(polygon, list)
        and len(polygon) >= 6
        and all(isinstance(number, (int, float)) for number in polygon)
    ]


def decode_coco_rle(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, dict) or "counts" not in value or "size" not in value:
        return None
    rle = {"counts": value["counts"], "size": value["size"]}
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def mask_array(size: tuple[int, int], masks: Any) -> np.ndarray:
    values = masks if isinstance(masks, list) else [masks]
    rle_canvas = np.zeros((size[1], size[0]), dtype=bool)
    found_rle = False
    for value in values:
        decoded = decode_coco_rle(value)
        if decoded is None:
            continue
        found_rle = True
        if decoded.shape != rle_canvas.shape:
            decoded = cv2.resize(
                decoded.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        rle_canvas |= decoded
    if found_rle:
        return rle_canvas

    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    if isinstance(values, list):
        for mask in values:
            for polygon in normalize_polygons(mask):
                draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=255)
    return np.asarray(canvas, dtype=np.uint8) > 0


def mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[mask] = np.round(base[mask] * 0.45 + red[mask] * 0.55).astype(np.uint8)
    return Image.fromarray(base)


def locality_metrics(
    source: Image.Image, edited: Image.Image, mask: np.ndarray, guard_pixels: int
) -> dict[str, float]:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    if edited.size != source.size:
        edited = edited.resize(source.size, Image.Resampling.LANCZOS)
    edited_array = np.asarray(edited.convert("RGB"), dtype=np.float32) / 255.0
    delta = np.abs(source_array - edited_array).mean(axis=2)
    kernel_size = max(1, guard_pixels * 2 + 1)
    dilated = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    ).astype(bool)
    outside = ~dilated

    def mean_where(selection: np.ndarray) -> float:
        return float(delta[selection].mean()) if selection.any() else 0.0

    def changed_where(selection: np.ndarray) -> float:
        return float((delta[selection] >= 0.05).mean()) if selection.any() else 0.0

    return {
        "mask_area_fraction": round(float(mask.mean()), 8),
        "inside_mean_abs_diff": round(mean_where(mask), 8),
        "inside_changed_fraction": round(changed_where(mask), 8),
        "outside_guard_mean_abs_diff": round(mean_where(outside), 8),
        "outside_guard_changed_fraction": round(changed_where(outside), 8),
        "global_mean_abs_diff": round(float(delta.mean()), 8),
    }


def audit_messages(
    source: Image.Image, overlay: Image.Image, edited: Image.Image, row: dict[str, Any]
) -> list[dict[str, Any]]:
    prompt = AUDIT_PROMPT.replace("INSTRUCTION", str(row.get("editing_instruction", "")))
    targets = "; ".join(str(value) for value in row.get("refer_object", []))
    prompt = prompt.replace("TARGETS", targets)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source},
                {"type": "image", "image": overlay},
                {"type": "image", "image": edited},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(array.min()), 8),
        "p25": round(float(np.quantile(array, 0.25)), 8),
        "median": round(float(np.median(array)), 8),
        "p75": round(float(np.quantile(array, 0.75)), 8),
        "max": round(float(array.max()), 8),
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.annotations_jsonl)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    existing_path = args.out_dir / "edit_audit.jsonl"
    existing_rows = load_jsonl(existing_path) if args.resume and existing_path.exists() else []
    existing_by_image = {
        str(row.get("image")): row
        for row in existing_rows
        if isinstance(row.get("image"), str) and row.get("input_fingerprint")
    }
    tasks = []
    output_by_image: dict[str, dict[str, Any]] = {}
    reused_cases = 0
    for row in rows:
        image_name = str(row.get("image", ""))
        source_path = args.source_dir / image_name
        edited_path = args.edited_dir / image_name
        fingerprint = input_fingerprint(row, source_path, edited_path, args)
        existing = existing_by_image.get(image_name)
        if existing and existing.get("input_fingerprint") == fingerprint:
            output_by_image[image_name] = existing
            reused_cases += 1
            continue
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")
        with Image.open(edited_path) as handle:
            edited = handle.convert("RGB")
        mask = mask_array(source.size, row.get("mask", []))
        overlay = mask_overlay(source, mask)
        tasks.append(
            {
                "row": row,
                "source": source,
                "edited": edited,
                "overlay": overlay,
                "metrics": locality_metrics(source, edited, mask, args.guard_pixels),
                "input_fingerprint": fingerprint,
            }
        )

    if tasks:
        vlm.configure_backend(
            name=args.vlm,
            model_id=args.vlm_model_id,
            device=args.vlm_device,
            dtype=args.vlm_dtype,
        )
        batch_size = max(1, args.batch_size)
        for start in tqdm(range(0, len(tasks), batch_size), desc="edit-pair audit"):
            batch = tasks[start : start + batch_size]
            outputs = vlm.get_backend().chat_batch(
                [
                    audit_messages(
                        task["source"], task["overlay"], task["edited"], task["row"]
                    )
                    for task in batch
                ],
                max_new_tokens=args.max_new_tokens,
            )
            for task, raw in zip(batch, outputs):
                parsed = parse_json_object(raw)
                quality = (
                    str(parsed.get("quality", "parse_error"))
                    if parsed
                    else "parse_error"
                )
                output_by_image[str(task["row"].get("image", ""))] = {
                    "image": task["row"].get("image"),
                    "editing_instruction": task["row"].get("editing_instruction"),
                    "refer_object": task["row"].get("refer_object"),
                    "quality": quality,
                    "audit": parsed,
                    "locality_metrics": task["metrics"],
                    "raw_response": raw,
                    "input_fingerprint": task["input_fingerprint"],
                }

    output_rows = [output_by_image[str(row.get("image", ""))] for row in rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "edit_audit.jsonl", output_rows)
    metric_names = list(output_rows[0]["locality_metrics"]) if output_rows else []
    summary = {
        "cases": len(output_rows),
        "quality_counts": dict(sorted(Counter(row["quality"] for row in output_rows).items())),
        "metric_quantiles": {
            name: quantiles([row["locality_metrics"][name] for row in output_rows])
            for name in metric_names
        },
        "guard_pixels": args.guard_pixels,
        "audit_version": AUDIT_VERSION,
        "audited_cases": len(tasks),
        "reused_cases": reused_cases,
        "resume": args.resume,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
