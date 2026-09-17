"""Create per-case and contact-sheet visualizations for a SAMTok edit pilot."""

from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, default=None)
    parser.add_argument("--manual-review-jsonl", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument(
        "--page-size",
        type=int,
        default=10,
        help="Number of cases per contact sheet; avoids one oversized image.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def decode_union(rles: list[dict[str, Any]]) -> np.ndarray:
    union = None
    for raw in rles:
        value = dict(raw)
        if isinstance(value["counts"], str):
            value["counts"] = value["counts"].encode("ascii")
        decoded = mask_utils.decode(value).astype(bool)
        union = decoded if union is None else union | decoded
    if union is None:
        raise ValueError("No masks in annotation")
    return union


def difference_panel(source: Image.Image, edited: Image.Image, rles: list[dict]) -> Image.Image:
    edited = edited.resize(source.size, Image.Resampling.LANCZOS)
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32)
    edited_array = np.asarray(edited.convert("RGB"), dtype=np.float32)
    delta = np.abs(source_array - edited_array).mean(axis=2)
    heat = np.clip(delta * 4.0, 0, 255).astype(np.uint8)
    rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=2)
    mask = decode_union(rles)
    edge = np.zeros_like(mask)
    edge[1:] |= mask[1:] != mask[:-1]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    rgb[edge] = [255, 255, 255]
    return Image.fromarray(rgb)


def case_sheet(
    row: dict[str, Any],
    source: Image.Image,
    overlay: Image.Image,
    edited: Image.Image,
    audit: dict[str, Any] | None,
    manual_review: dict[str, Any] | None,
    panel_width: int,
) -> Image.Image:
    panel_height = round(panel_width * 0.68)
    labels = ["SOURCE", "TARGET MASK", "EDITED", "ABS DIFF + MASK EDGE"]
    panels = [
        source,
        overlay,
        edited,
        difference_panel(source, edited, row["mask"]),
    ]
    header_height = 176
    sheet = Image.new("RGB", (panel_width * 4, header_height + panel_height + 36), "white")
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{row['image']}  |  type={row.get('task_type')}  |  "
        f"regions={len(row.get('mask', []))}"
    )
    draw.text((12, 10), title, fill="black", font=font(20, True))
    instruction = "Instruction: " + str(row.get("editing_instruction", ""))
    y = 40
    for line in textwrap.wrap(instruction, width=150)[:3]:
        draw.text((12, y), line, fill=(25, 25, 25), font=font(17))
        y += 23
    if audit:
        reason = (audit.get("audit") or {}).get("reason", "")
        status = f"VLM audit: {audit.get('quality')} — {reason}"
        draw.text((12, 116), status[:180], fill=(120, 30, 20), font=font(16, True))
    if manual_review:
        status = (
            f"Manual review: {manual_review.get('quality')} — "
            f"{manual_review.get('reason', '')}"
        )
        draw.text((12, 140), status[:180], fill=(20, 75, 145), font=font(16, True))
    for index, (label, panel) in enumerate(zip(labels, panels)):
        x = index * panel_width
        sheet.paste(fit(panel, panel_width, panel_height), (x, header_height))
        draw.text((x + 10, header_height + panel_height + 7), label, fill="black", font=font(16, True))
    return sheet


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.annotations_jsonl)
    audit_by_image: dict[str, dict[str, Any]] = {}
    if args.audit_jsonl and args.audit_jsonl.exists():
        audit_by_image = {
            str(row["image"]): row for row in load_jsonl(args.audit_jsonl)
        }
    manual_by_image: dict[str, dict[str, Any]] = {}
    if args.manual_review_jsonl and args.manual_review_jsonl.exists():
        manual_by_image = {
            str(row["image"]): row for row in load_jsonl(args.manual_review_jsonl)
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = args.out_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    html_rows = []
    for row in rows:
        name = str(row["image"])
        with Image.open(args.source_dir / str(row.get("source_image") or name)) as handle:
            source = handle.convert("RGB")
        with Image.open(args.overlay_dir / name) as handle:
            overlay = handle.convert("RGB")
        with Image.open(args.edited_dir / name) as handle:
            edited = handle.convert("RGB")
        sheet = case_sheet(
            row,
            source,
            overlay,
            edited,
            audit_by_image.get(name),
            manual_by_image.get(name),
            args.panel_width,
        )
        path = case_dir / name.replace(".png", ".jpg")
        sheet.save(path, quality=92)
        sheets.append(sheet)
        audit = audit_by_image.get(name, {})
        manual = manual_by_image.get(name, {})
        html_rows.append(
            "<section><h2>" + html.escape(name) + "</h2>"
            "<p><b>Instruction:</b> "
            + html.escape(str(row.get("editing_instruction", "")))
            + "</p><p><b>Audit:</b> "
            + html.escape(str(audit.get("quality", "not run")))
            + "</p><p><b>Manual review:</b> "
            + html.escape(str(manual.get("quality", "not run")))
            + " — "
            + html.escape(str(manual.get("reason", "")))
            + "</p><img src='cases/"
            + html.escape(path.name)
            + "'></section>"
        )

    if sheets:
        if args.page_size <= 0:
            raise ValueError("--page-size must be positive")
        gap = 12
        for start in range(0, len(sheets), args.page_size):
            page = sheets[start : start + args.page_size]
            contact = Image.new(
                "RGB",
                (
                    max(sheet.width for sheet in page),
                    sum(sheet.height for sheet in page) + gap * (len(page) - 1),
                ),
                (225, 225, 225),
            )
            y = 0
            for sheet in page:
                contact.paste(sheet, (0, y))
                y += sheet.height + gap
            end = start + len(page) - 1
            contact.save(
                args.out_dir / f"contact_sheet_{start:03d}_{end:03d}.jpg",
                quality=90,
            )
    (args.out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>SAMTok edit pilot</title>"
        "<style>body{font-family:sans-serif;max-width:1500px;margin:auto;background:#eee}"
        "section{background:white;margin:20px;padding:18px}img{width:100%;height:auto}</style>"
        + "".join(html_rows),
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(rows), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
