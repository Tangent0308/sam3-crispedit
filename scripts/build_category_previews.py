#!/usr/bin/env python3
"""Build high-resolution, one-contact-sheet-per-category mask previews.

Unlike the runtime preview (which is intentionally compact), this script reads
the original source/target image bytes and renders large panels before composing
the category sheets. It is intended for human quality review, not training.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


CATEGORY_ORDER = ["add", "background change", "color", "motion change", "remove", "replace", "style"]
CARD_HEADER = 172


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-resolution category mask contact sheets")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=400)
    parser.add_argument("--columns", type=int, default=1, help="Cards per category sheet row; 1 maximizes clarity")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Reserved for future JPEG export")
    return parser.parse_args()


def _slug(value: str) -> str:
    return "_".join(value.split())


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_selection(path: Optional[Path]) -> Optional[Dict[str, set[int]]]:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("selection JSON must be a list or an object with cases")
    selected: Dict[str, set[int]] = defaultdict(set)
    for item in cases:
        selected[str(item["shard"])].add(int(item["row_idx"]))
    return selected


def _read_rows(
    path: Path,
    indices: Optional[Sequence[int]] = None,
    row_key_column: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Read selected rows, keyed by physical index or an explicit row_idx field.

    ``iter_batches`` is deliberate here: CrispEdit shards are hundreds of MB to
    GB, while a review slice generally needs one row per shard.  Reading the
    entire row group would unnecessarily materialize every image in the shard.
    """

    wanted = set(indices) if indices is not None else None
    result: Dict[int, Dict[str, Any]] = {}
    offset = 0
    parquet = pq.ParquetFile(path)
    for group_index in range(parquet.num_row_groups):
        group_meta = parquet.metadata.row_group(group_index)
        group_start = offset
        group_end = group_start + group_meta.num_rows
        if wanted is not None and row_key_column is None and not any(group_start <= item < group_end for item in wanted):
            offset = group_end
            continue
        local = 0
        for batch in parquet.iter_batches(batch_size=64, columns=columns, row_groups=[group_index]):
            for row in batch.to_pylist():
                row_idx = int(row[row_key_column]) if row_key_column else offset + local
                if wanted is None or row_idx in wanted:
                    result[row_idx] = row
                local += 1
            if wanted is not None and wanted.issubset(result):
                break
        offset = group_end
        if wanted is not None and wanted.issubset(result):
            break
    return result


def _decode_image(cell: Dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def _decode_mask(data: Any, size: Tuple[int, int]) -> np.ndarray:
    if not data:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    image = Image.open(io.BytesIO(data)).convert("L")
    image = image.resize(size, Image.Resampling.NEAREST) if image.size != size else image
    return (np.asarray(image) > 0).astype(np.uint8)


def _normalized_box(box: Sequence[float], size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    return (
        max(0.0, min(width, x1 * width / 1000.0)),
        max(0.0, min(height, y1 * height / 1000.0)),
        max(0.0, min(width, x2 * width / 1000.0)),
        max(0.0, min(height, y2 * height / 1000.0)),
    )


def _draw_boxes(image: Image.Image, items: Sequence[Dict[str, Any]], color: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, min(canvas.size) // 220)
    label_font = _font(max(16, min(canvas.size) // 55), bold=True)
    for index, item in enumerate(items):
        box = _normalized_box(item.get("bbox_2d", [0, 0, 1, 1]), canvas.size)
        draw.rectangle(box, outline=color, width=line_width)
        # Full semantic refs overlap badly for dense sets (flowers, petals,
        # piercings, etc.).  The readable Markdown/JSON export contains the
        # full refs; the image only needs a stable compact id per box.
        label = f"#{index + 1}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=label_font)
        text_width, text_height = right - left, bottom - top
        tx, ty = int(box[0]) + line_width + 3, int(box[1]) + line_width + 3
        if ty + text_height + 8 > canvas.height:
            ty = max(0, int(box[1]) - text_height - 10)
        draw.rectangle((tx - 3, ty - 3, tx + text_width + 5, ty + text_height + 4), fill="#111111")
        draw.text((tx, ty), label, fill=color, font=label_font)
    return canvas


def _overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 70
    rgba[..., 3] = mask.astype(np.uint8) * 128
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def _fit(image: Image.Image, size: Tuple[int, int], background: str = "#eeeeee") -> Image.Image:
    canvas = Image.new("RGB", size, background)
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _payload(mask_row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = json.loads(str(mask_row.get("ground_json", "{}")))
    except (TypeError, ValueError):
        result = {}
    return result if isinstance(result, dict) else {}


def _card(
    raw_row: Dict[str, Any],
    mask_row: Dict[str, Any],
    panel_width: int,
    panel_height: int,
) -> Image.Image:
    source = _decode_image(raw_row["input_img"])
    target = _decode_image(raw_row["output_img"])
    mask = _decode_mask(mask_row.get("mask_png"), source.size)
    payload = _payload(mask_row)
    boxes = payload.get("boxes", {}) if isinstance(payload, dict) else {}
    source_boxes = boxes.get("source", []) if isinstance(boxes, dict) else []
    target_boxes = boxes.get("target", []) if isinstance(boxes, dict) else []
    source_panel = _draw_boxes(source, source_boxes, "#38ff75")
    target_panel = _draw_boxes(target, target_boxes, "#20d9ff")
    overlay_panel = _overlay(source, mask)
    binary_panel = Image.fromarray(mask * 255, mode="L").convert("RGB")

    gap = 12
    header = CARD_HEADER
    labels = ["source + MLLM boxes", "target + MLLM boxes", "final mask overlay (source)", "final mask (binary)"]
    images = [source_panel, target_panel, overlay_panel, binary_panel]
    card = Image.new("RGB", (panel_width * 4 + gap * 3, header + panel_height), "white")
    draw = ImageDraw.Draw(card)
    title_font = _font(24, bold=True)
    body_font = _font(18)
    bold_font = _font(18, bold=True)
    category = str(mask_row.get("canonical_type", mask_row.get("raw_type", "")))
    shard = str(mask_row.get("_shard", ""))
    row_idx = int(mask_row.get("row_idx", -1))
    area = float(mask_row.get("area_frac", 0.0) or 0.0)
    source_name = str(mask_row.get("mask_source", ""))
    qc = str(mask_row.get("qc_flag", ""))
    instances = mask_row.get("instance_masks") or []
    draw.text((12, 8), f"{category} | {shard} row={row_idx}", fill="black", font=title_font)
    instruction_lines = textwrap.wrap(str(raw_row.get("instruction", "")), width=170)[:2]
    for line_index, line in enumerate(instruction_lines):
        draw.text((12, 42 + line_index * 24), line, fill="#222222", font=body_font)
    status = f"mask_source={source_name} | qc={qc} | area={area:.4f} | instances={len(instances)}"
    draw.text((12, 94), status, fill="#222222", font=bold_font)
    draw.text(
        (12, 122),
        f"MLLM boxes: source={len(source_boxes)} target={len(target_boxes)}  (labels #1, #2, ...)",
        fill="#444444",
        font=body_font,
    )
    for index, (label, image) in enumerate(zip(labels, images)):
        x = index * (panel_width + gap)
        draw.text((x + 8, header - 28), label, fill="black", font=bold_font)
        card.paste(_fit(image, (panel_width, panel_height)), (x, header))
    return card


def main() -> None:
    args = parse_args()
    if args.columns < 1 or args.panel_width < 200 or args.panel_height < 160:
        raise ValueError("columns and panel dimensions are too small")
    selection = _load_selection(args.selection_file.resolve() if args.selection_file else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Image.Image]] = defaultdict(list)
    entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for mask_path in sorted(args.mask_dir.glob("*.parquet")):
        selected = selection.get(mask_path.name) if selection is not None else None
        mask_rows = _read_rows(
            mask_path,
            sorted(selected) if selected is not None else None,
            row_key_column="row_idx",
            columns=[
                "row_idx", "canonical_type", "raw_type", "instruction", "ground_json",
                "mask_png", "instance_masks", "mask_source", "area_frac", "qc_flag",
            ],
        )
        if not mask_rows:
            continue
        raw_path = args.input_dir / mask_path.name
        if not raw_path.exists():
            raise FileNotFoundError(f"raw shard is missing: {raw_path}")
        raw_rows = _read_rows(
            raw_path,
            sorted(mask_rows),
            columns=["input_img", "instruction", "output_img", "type"],
        )
        for row_idx, mask_row in sorted(mask_rows.items()):
            if row_idx not in raw_rows:
                raise IndexError(f"row {row_idx} missing from {raw_path.name}")
            mask_row = dict(mask_row)
            mask_row["_shard"] = mask_path.name
            category = str(mask_row.get("canonical_type", mask_row.get("raw_type", "unknown")))
            grouped[category].append(_card(raw_rows[row_idx], mask_row, args.panel_width, args.panel_height))
            entries[category].append(
                {
                    "shard": mask_path.name,
                    "row_idx": row_idx,
                    "instruction": str(raw_rows[row_idx].get("instruction", "")),
                    "output": f"{_slug(category)}.png",
                }
            )

    if not grouped:
        raise SystemExit("no mask rows matched")
    order = [category for category in CATEGORY_ORDER if category in grouped]
    order.extend(sorted(set(grouped) - set(order)))
    category_paths = {}
    for category in order:
        cards = grouped[category]
        columns = min(args.columns, len(cards))
        gap = 24
        title_height = 74
        card_width = args.panel_width * 4 + 12 * 3
        card_height = CARD_HEADER + args.panel_height
        rows = math.ceil(len(cards) / columns)
        sheet = Image.new(
            "RGB",
            (columns * card_width + (columns + 1) * gap, title_height + rows * card_height + (rows + 1) * gap),
            "#d9dde3",
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((gap, 17), f"CrispEdit mask review — {category} ({len(cards)} samples)", fill="#111111", font=_font(34, bold=True))
        for index, card in enumerate(cards):
            row, column = divmod(index, columns)
            x = gap + column * (card_width + gap)
            y = title_height + gap + row * (card_height + gap)
            sheet.paste(card, (x, y))
        path = args.output_dir / f"{_slug(category)}.png"
        sheet.save(path, format="PNG", optimize=True)
        category_paths[category] = str(path)

    index = {
        "categories": category_paths,
        "entries": entries,
        "panel_width": args.panel_width,
        "panel_height": args.panel_height,
        "columns": args.columns,
        "note": "Rendered from original input image bytes and final mask_png; masks are source-coordinate masks.",
    }
    (args.output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"categories": {key: len(value) for key, value in grouped.items()}, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
