#!/usr/bin/env python3
"""Render representative PASS/DROP boards for the CrispEdit scene filter."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, ImageOps


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def image_from(value: object) -> Image.Image:
    if isinstance(value, dict):
        value = value["bytes"]
    if isinstance(value, memoryview):
        value = value.tobytes()
    return Image.open(io.BytesIO(value)).convert("RGB")


def wrap(draw: ImageDraw.ImageDraw, text: object, width: int, limit: int) -> list[str]:
    words = str(text or "").replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font(18)) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1].rstrip(".,;:") + "..."
    return lines


def paste_image(canvas: Image.Image, value: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    canvas.paste("#111827", box)
    fitted = ImageOps.contain(value, (x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas.paste(fitted, (x0 + (x1 - x0 - fitted.width) // 2, y0 + (y1 - y0 - fitted.height) // 2))


def load_featured(args: argparse.Namespace) -> dict[str, list[dict]]:
    cases = json.loads(args.case_file.read_text(encoding="utf-8"))["cases"]
    grouped = {"PASS": [], "DROP": []}
    audit_cache: dict[str, dict[int, dict]] = {}
    raw_cache: dict[str, object] = {}
    for case in cases:
        if not case.get("featured"):
            continue
        shard = str(case["shard"])
        row_idx = int(case["row_idx"])
        if shard not in raw_cache:
            raw_cache[shard] = pq.read_table(
                args.input_dir / shard,
                columns=["input_img", "output_img", "instruction"],
            )
            audit_cache[shard] = {
                int(row["row_idx"]): row
                for row in pq.read_table(args.audit_dir / shard).to_pylist()
            }
        raw = raw_cache[shard].slice(row_idx, 1).to_pylist()[0]
        audit = audit_cache[shard][row_idx]
        expected = str(case["expected"])
        if audit["scene_decision"] != expected:
            raise ValueError(f"decision mismatch for {shard}:{row_idx}")
        grouped[expected].append(
            {
                "case": f"{shard}:{row_idx}",
                "instruction": raw["instruction"],
                "reason": audit["scene_reason"],
                "source": image_from(raw["input_img"]),
                "target": image_from(raw["output_img"]),
            }
        )
    return grouped


def draw_board(decision: str, rows: list[dict], output: Path) -> None:
    width, margin, gap = 2200, 34, 24
    columns, card_height, header = 2, 620, 130
    card_width = (width - 2 * margin - gap) // columns
    height = header + math.ceil(len(rows) / columns) * (card_height + gap) + margin
    background = "#eef2f7"
    accent = "#16845b" if decision == "PASS" else "#c54747"
    soft = "#e5f6ef" if decision == "PASS" else "#fbeaea"
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 100), fill="#18243a")
    draw.text((margin, 22), f"CrispEdit difficult local edit · {decision}", font=font(34, bold=True), fill="white")
    draw.text((margin, 68), "Model input: SOURCE only · TARGET is shown only for human review", font=font(18), fill="#cbd5e1")

    for index, row in enumerate(rows):
        x = margin + (index % columns) * (card_width + gap)
        y = header + (index // columns) * (card_height + gap)
        draw.rounded_rectangle((x, y, x + card_width, y + card_height), radius=18, fill="white", outline=accent, width=4)
        draw.text((x + 20, y + 18), row["case"], font=font(20, bold=True), fill="#172033")
        badge_width = 92
        draw.rounded_rectangle((x + card_width - badge_width - 20, y + 13, x + card_width - 20, y + 51), radius=19, fill=soft)
        draw.text((x + card_width - badge_width - 5, y + 20), decision, font=font(18, bold=True), fill=accent)

        image_y, image_height = y + 66, 325
        image_width = (card_width - 50) // 2
        for offset, key, label in ((0, "source", "SOURCE"), (image_width + 10, "target", "TARGET")):
            box = (x + 20 + offset, image_y, x + 20 + offset + image_width, image_y + image_height)
            paste_image(canvas, row[key], box)
            draw.rectangle(box, outline="#cbd5e1", width=2)
            draw.rectangle((box[0], box[1], box[0] + 88, box[1] + 29), fill="#111827")
            draw.text((box[0] + 8, box[1] + 5), label, font=font(15, bold=True), fill="white")

        text_y = image_y + image_height + 18
        draw.text((x + 20, text_y), "Instruction", font=font(17, bold=True), fill="#356fc0")
        for line in wrap(draw, row["instruction"], card_width - 40, 2):
            text_y += 25
            draw.text((x + 20, text_y), line, font=font(18), fill="#172033")
        text_y += 32
        draw.text((x + 20, text_y), "Reason", font=font(17, bold=True), fill=accent)
        for line in wrap(draw, row["reason"], card_width - 40, 3):
            text_y += 24
            draw.text((x + 20, text_y), line, font=font(17), fill="#5d687b")
    canvas.save(output, quality=92)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_featured(args)
    for decision, rows in grouped.items():
        draw_board(decision, rows, args.output_dir / f"final_{decision.lower()}_examples.jpg")
    print(json.dumps({key: len(value) for key, value in grouped.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
