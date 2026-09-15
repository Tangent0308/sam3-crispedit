#!/usr/bin/env python3
"""Create four-panel review pages for native RefEdit grounding and masks."""

from __future__ import annotations

import argparse
import io
import json
import math
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from refedit.io import discover_shards, sample_id
from scaleedit.io import decode_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-page", type=int, default=5)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=250)
    parser.add_argument("--img-id", action="append", default=[])
    parser.add_argument("--selection-file", type=Path)
    return parser.parse_args()


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size=size)
        if path.exists()
        else ImageFont.load_default()
    )


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), "#eeeeee")
    thumbnail = image.convert("RGB").copy()
    thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel.paste(
        thumbnail,
        ((width - thumbnail.width) // 2, (height - thumbnail.height) // 2),
    )
    return panel


def _pixel_box(box: Sequence[float], image: Image.Image) -> Tuple[float, ...]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return (
        x1 * image.width / 1000.0,
        y1 * image.height / 1000.0,
        x2 * image.width / 1000.0,
        y2 * image.height / 1000.0,
    )


def _boxes(image: Image.Image, items: Sequence[Dict], color: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width = max(6, min(canvas.size) // 90)
    font = _font(max(18, min(canvas.size) // 42))
    for index, item in enumerate(items):
        box = item.get("bbox_2d")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        pixels = _pixel_box(box, canvas)
        draw.rectangle(pixels, outline="#101010", width=width + 3)
        draw.rectangle(pixels, outline=color, width=width)
        label = f"{index}: {item.get('ref', '')}"
        origin = (pixels[0] + width + 2, pixels[1] + width + 2)
        text_box = draw.textbbox(origin, label, font=font, stroke_width=1)
        draw.rectangle(text_box, fill="#101010")
        draw.text(
            origin,
            label,
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="#101010",
        )
    return canvas


def _decode_mask(payload: bytes, shape: Tuple[int, int]) -> np.ndarray:
    if not payload:
        return np.zeros(shape, dtype=np.uint8)
    return (np.asarray(Image.open(io.BytesIO(payload)).convert("L")) > 0).astype(
        np.uint8
    )


def _overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.shape != (image.height, image.width):
        mask = np.asarray(
            Image.fromarray(mask * 255, mode="L").resize(
                image.size, Image.Resampling.NEAREST
            )
        ) > 0
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 40
    rgba[..., 3] = mask.astype(np.uint8) * 112
    return Image.alpha_composite(
        image.convert("RGBA"), Image.fromarray(rgba, mode="RGBA")
    ).convert("RGB")


def _load_rows(args: argparse.Namespace) -> List[Dict]:
    source_by_name = {path.name: path for path in discover_shards(args.input_dir)}
    requested = {sample_id(value) for value in args.img_id}
    if args.selection_file:
        raw_selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
        values = (
            raw_selection.get("img_ids", [])
            if isinstance(raw_selection, dict)
            else raw_selection
        )
        if not isinstance(values, list):
            raise ValueError("selection file must contain a list of img_id values")
        requested.update(sample_id(value) for value in values)
    result = []
    for ground_path in sorted(args.grounding_dir.glob("train-*.parquet")):
        if requested:
            shard_ids = set(
                str(value)
                for value in pq.read_table(
                    ground_path, columns=["sample_id"]
                )[0].to_pylist()
            )
            if not requested.intersection(shard_ids):
                continue
        source_path = source_by_name.get(ground_path.name)
        mask_path = args.mask_dir / ground_path.name
        if source_path is None or not mask_path.is_file():
            raise FileNotFoundError(f"incomplete review inputs for {ground_path.name}")
        ground_rows = pq.read_table(ground_path).to_pylist()
        mask_rows = pq.read_table(mask_path).to_pylist()
        if len(ground_rows) != len(mask_rows):
            raise ValueError(f"ground/mask row mismatch for {ground_path.name}")
        source_rows = pq.read_table(source_path).to_pylist()
        for ground, mask in zip(ground_rows, mask_rows):
            if str(ground["sample_id"]) != str(mask["sample_id"]):
                raise ValueError(f"ground/mask identity mismatch in {ground_path.name}")
            if requested and str(mask["sample_id"]) not in requested:
                continue
            row_idx = int(ground["row_idx"])
            raw = source_rows[row_idx]
            if sample_id(raw["img_id"]) != str(mask["sample_id"]):
                raise ValueError(f"source identity mismatch in {ground_path.name}:{row_idx}")
            result.append(
                {"shard": ground_path.name, "raw": raw, "ground": ground, "mask": mask}
            )
    if requested:
        found = {str(item["mask"]["sample_id"]) for item in result}
        if found != requested:
            raise KeyError(f"missing requested img_id(s): {sorted(requested - found)}")
    if not result:
        raise ValueError("no RefEdit mask rows found")
    return result


def _render(item: Dict, panel_width: int, panel_height: int) -> Image.Image:
    raw, mask_row = item["raw"], item["mask"]
    source = decode_image(raw["source_img"])
    target = decode_image(raw["target_img"])
    payload = json.loads(mask_row["ground_json"])
    mode = str(payload.get("mask_mode", "unresolved"))
    source_items = payload.get("source", [])
    target_items = payload.get("target", [])
    if mode == "protect_foreground":
        source_items = payload.get("protected_foreground", [])
    mask = _decode_mask(
        mask_row["mask_png"],
        (
            int(mask_row.get("mask_height") or source.height),
            int(mask_row.get("mask_width") or source.width),
        ),
    )
    panels = [
        ("source + MLLM boxes (green)", _boxes(source, source_items, "#39ff72")),
        ("target + MLLM boxes (cyan)", _boxes(target, target_items, "#00e5ff")),
        ("source + final mask", _overlay(source, mask)),
        ("binary final mask", Image.fromarray(mask * 255, mode="L").convert("RGB")),
    ]
    header_height = 90
    label_height = 24
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), header_height + label_height + panel_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    area = float(mask_row.get("area_frac", math.nan))
    title = (
        f"{mask_row['sample_id']} | {mask_row['final_task']} | mode={mode} | "
        f"source={mask_row['mask_source']} | qc={mask_row['qc_flag']} | area={area:.3f}"
    )
    draw.text((8, 5), title, fill="black", font=_font(18))
    for line_no, line in enumerate(
        textwrap.wrap(str(mask_row["final_instruction"]), width=155)[:2]
    ):
        draw.text((8, 33 + 19 * line_no), line, fill="#222", font=_font(14))
    for index, (label, panel) in enumerate(panels):
        x = index * panel_width
        draw.text((x + 7, header_height + 3), label, fill="black", font=_font(14))
        canvas.paste(_fit(panel, panel_width, panel_height), (x, header_height + label_height))
    return canvas


def main() -> None:
    args = parse_args()
    for name in ("input_dir", "grounding_dir", "mask_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.selection_file:
        args.selection_file = args.selection_file.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args)
    rows.sort(
        key=lambda item: (
            str(item["mask"]["final_task"]),
            str(item["mask"]["sample_id"]),
        )
    )
    page_names = []
    for page_no, start in enumerate(range(0, len(rows), args.rows_per_page), 1):
        rendered = [
            _render(item, args.panel_width, args.panel_height)
            for item in rows[start : start + args.rows_per_page]
        ]
        page = Image.new(
            "RGB",
            (
                max(item.width for item in rendered),
                sum(item.height for item in rendered),
            ),
            "white",
        )
        y = 0
        for image in rendered:
            page.paste(image, (0, y))
            y += image.height
        name = f"refedit_mask_preview_page_{page_no:02d}.jpg"
        page.save(args.output_dir / name, quality=92, subsampling=1)
        page_names.append(name)
    summary = {
        "rows": len(rows),
        "tasks": dict(Counter(str(item["mask"]["final_task"]) for item in rows)),
        "qc_flags": dict(Counter(str(item["mask"]["qc_flag"]) for item in rows)),
        "mask_sources": dict(Counter(str(item["mask"]["mask_source"]) for item in rows)),
        "pages": page_names,
        "samples": [str(item["mask"]["sample_id"]) for item in rows],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
