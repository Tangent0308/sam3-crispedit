#!/usr/bin/env python3
"""Render source/target review pages for RefEdit quality-prefilter audits."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from refedit.io import discover_shards, sample_id
from scaleedit.io import decode_image


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "#ededed")
    value = image.convert("RGB").copy()
    value.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(value, ((width - value.width) // 2, (height - value.height) // 2))
    return canvas


def _source_rows(input_dir: Path, audit_rows: List[Dict]) -> Dict[str, Dict]:
    wanted = {str(row["sample_id"]) for row in audit_rows}
    result: Dict[str, Dict] = {}
    for path in discover_shards(input_dir):
        shard_wanted = {
            str(row["sample_id"])
            for row in audit_rows
            if Path(
                str(row.get("source_shard") or row.get("source_relative_path", ""))
            ).name
            == path.name
        }
        if not shard_wanted:
            continue
        for record in pq.read_table(path).to_pylist():
            identity = sample_id(record["img_id"])
            if identity in shard_wanted:
                result[identity] = record
    if set(result) != wanted:
        raise KeyError(f"missing source rows: {sorted(wanted - set(result))}")
    return result


def _load_audit_rows(
    path: Path,
    img_ids: List[str],
    selection_file: Optional[Path],
    verdicts: List[str],
    limit: Optional[int],
) -> List[Dict]:
    paths = sorted(path.glob("train-*.parquet")) if path.is_dir() else [path]
    if not paths or not all(item.is_file() for item in paths):
        raise FileNotFoundError(f"no quality audit parquet found at {path}")
    requested = {sample_id(value) for value in img_ids}
    if selection_file:
        raw = json.loads(selection_file.read_text(encoding="utf-8"))
        values = raw.get("img_ids", []) if isinstance(raw, dict) else raw
        if not isinstance(values, list):
            raise ValueError("selection file must contain img_ids")
        requested.update(sample_id(value) for value in values)
    allowed_verdicts = {value.upper() for value in verdicts}
    result = []
    for audit_path in paths:
        for row in pq.read_table(audit_path).to_pylist():
            if requested and str(row["sample_id"]) not in requested:
                continue
            if allowed_verdicts and str(row["verdict"]).upper() not in allowed_verdicts:
                continue
            result.append(row)
            if limit is not None and len(result) >= limit:
                break
        if limit is not None and len(result) >= limit:
            break
    if requested:
        found = {str(row["sample_id"]) for row in result}
        if found != requested:
            raise KeyError(f"missing selected audit rows: {sorted(requested - found)}")
    if not result:
        raise ValueError("quality audit selection is empty")
    return result


def _render(row: Dict, record: Dict, panel_width: int, panel_height: int) -> Image.Image:
    source = _fit(decode_image(record["source_img"]), panel_width, panel_height)
    target = _fit(decode_image(record["target_img"]), panel_width, panel_height)
    assessment = json.loads(row["assessment_json"])
    reasons = ", ".join(assessment.get("reason_codes", [])) or "none"
    failed = ", ".join(json.loads(row["failed_dimensions_json"])) or "none"
    lines = [
        f"{row['sample_id']} | {row['task']} | verdict={row['verdict']}",
        f"instruction: {row['instruction']}",
        f"failed: {failed}",
        f"reasons: {reasons}",
        f"summary: {row['summary']}",
    ]
    wrapped = []
    for index, line in enumerate(lines):
        width = 105 if index else 90
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    header_height = max(145, 12 + len(wrapped) * 20)
    label_height = 28
    canvas = Image.new(
        "RGB", (panel_width * 2, header_height + label_height + panel_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    y = 7
    for index, line in enumerate(wrapped):
        draw.text(
            (8, y),
            line,
            fill="#9d1111" if index == 0 and row["verdict"] == "FAIL" else "#222222",
            font=_font(17 if index == 0 else 14),
        )
        y += 21 if index == 0 else 19
    draw.text((8, header_height + 4), "Image 1: SOURCE", fill="black", font=_font(15))
    draw.text(
        (panel_width + 8, header_height + 4),
        "Image 2: TARGET",
        fill="black",
        font=_font(15),
    )
    canvas.paste(source, (0, header_height + label_height))
    canvas.paste(target, (panel_width, header_height + label_height))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-page", type=int, default=3)
    parser.add_argument("--panel-width", type=int, default=520)
    parser.add_argument("--panel-height", type=int, default=390)
    parser.add_argument("--img-id", action="append", default=[])
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--verdict", action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    audit_rows = _load_audit_rows(
        args.audit_parquet,
        args.img_id,
        args.selection_file,
        args.verdict,
        args.limit,
    )
    records = _source_rows(args.input_dir, audit_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for page_no, start in enumerate(range(0, len(audit_rows), args.rows_per_page), 1):
        rendered = [
            _render(row, records[str(row["sample_id"])], args.panel_width, args.panel_height)
            for row in audit_rows[start : start + args.rows_per_page]
        ]
        page = Image.new(
            "RGB",
            (max(image.width for image in rendered), sum(image.height for image in rendered)),
            "white",
        )
        y = 0
        for image in rendered:
            page.paste(image, (0, y))
            y += image.height
        page.save(
            args.output_dir / f"refedit_quality_review_page_{page_no:02d}.jpg",
            quality=92,
            subsampling=1,
        )


if __name__ == "__main__":
    main()
