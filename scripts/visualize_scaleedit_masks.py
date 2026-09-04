#!/usr/bin/env python3
"""Build review pages and aggregate metrics for ScaleEdit mask outputs."""

from __future__ import annotations

import argparse
import io
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from scaleedit.io import decode_image, discover_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ScaleEdit mask labels")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--rows-per-page", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=340)
    parser.add_argument("--panel-height", type=int, default=230)
    parser.add_argument("--task", action="append", default=[], help="Keep this final_task")
    parser.add_argument("--sample-id", action="append", default=[], help="Keep this exact sample_id")
    return parser.parse_args()


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), "#eeeeee")
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel.paste(thumb, ((width - thumb.width) // 2, (height - thumb.height) // 2))
    return panel


def _pixel_box(box: Sequence[float], image: Image.Image) -> Tuple[float, float, float, float]:
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
    width = max(2, min(canvas.size) // 250)
    font = _font(max(12, min(canvas.size) // 60))
    for item in items:
        box = _pixel_box(item["bbox_2d"], canvas)
        draw.rectangle(box, outline=color, width=width)
        label = f"{item.get('ref', '')} [{item.get('mask_method', 'sam')}]"
        draw.text((box[0] + width, box[1] + width), label, fill=color, font=font)
    return canvas


def _overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.shape != (image.height, image.width):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                image.size, Image.Resampling.NEAREST
            )
        ) > 0
    base = image.convert("RGBA")
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 45
    rgba[..., 3] = mask.astype(np.uint8) * 112
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def _decode_mask(payload: bytes) -> np.ndarray:
    return (np.asarray(Image.open(io.BytesIO(payload)).convert("L")) > 0).astype(np.uint8)


def _load_rows(args: argparse.Namespace) -> List[Dict]:
    result = []
    for raw_path in discover_shards(args.input_dir):
        ground_path = args.grounding_dir / raw_path.name
        mask_path = args.mask_dir / raw_path.name
        if not ground_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing aligned output for {raw_path.name}")
        raw_rows = pq.read_table(raw_path).to_pylist()
        ground_rows = pq.read_table(ground_path).to_pylist()
        mask_rows = pq.read_table(mask_path).to_pylist()
        if not (len(raw_rows) == len(ground_rows) == len(mask_rows)):
            raise ValueError(
                f"row mismatch for {raw_path.name}: "
                f"raw={len(raw_rows)} ground={len(ground_rows)} mask={len(mask_rows)}"
            )
        for raw, ground, mask in zip(raw_rows, ground_rows, mask_rows):
            identities = {str(raw["sample_id"]), str(ground["sample_id"]), str(mask["sample_id"])}
            if len(identities) != 1:
                raise ValueError(f"sample identity mismatch in {raw_path.name}")
            result.append({"shard": raw_path.name, "raw": raw, "ground": ground, "mask": mask})
    return result


def _selection_score(item: Dict) -> Tuple[int, float]:
    mask_row = item["mask"]
    payload = json.loads(mask_row["ground_json"])
    flag = str(mask_row["qc_flag"])
    flag_rank = {"OK": 0, "AR_MISMATCH": 1, "BOX_FALLBACK": 2}.get(flag, 3)
    area = float(mask_row.get("area_frac", math.nan))
    if payload.get("mask_mode") == "regions":
        area_penalty = abs(area - 0.18) if math.isfinite(area) else 10.0
    else:
        area_penalty = 0.0 if math.isfinite(area) else 10.0
    return flag_rank, area_penalty


def _select(rows: Sequence[Dict], count: int) -> List[Dict]:
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for item in rows:
        by_task[str(item["mask"]["final_task"])].append(item)
    selected = []
    for task in sorted(by_task):
        selected.extend(sorted(by_task[task], key=_selection_score)[:count])
    return selected


def _render_sample(item: Dict, panel_width: int, panel_height: int) -> Image.Image:
    raw, mask_row = item["raw"], item["mask"]
    source = decode_image(raw["source_image"])
    target = decode_image(raw["edited_image"])
    payload = json.loads(mask_row["ground_json"])
    mode = str(payload.get("mask_mode", "unresolved"))
    source_items = payload.get("source", [])
    if mode == "protect_foreground":
        source_items = payload.get("protected_foreground", [])
    mask = _decode_mask(mask_row["mask_png"])
    source_boxes = _boxes(source, source_items, "#00b050" if mode != "protect_foreground" else "#ff8c00")
    target_boxes = _boxes(target, payload.get("target", []), "#008cff")
    panels = [
        ("source + boxes", source_boxes),
        ("edited + boxes", target_boxes),
        ("source mask overlay", _overlay(source, mask)),
        ("binary edit mask", Image.fromarray(mask * 255, mode="L").convert("RGB")),
    ]
    header_height = 82
    width = panel_width * len(panels)
    canvas = Image.new("RGB", (width, header_height + panel_height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font = _font(18), _font(14)
    title = (
        f"{mask_row['final_task']} | mode={mode} | source={mask_row['mask_source']} | "
        f"qc={mask_row['qc_flag']} | area={float(mask_row['area_frac']):.3f}"
    )
    draw.text((8, 5), title, fill="black", font=title_font)
    instruction = str(mask_row["final_instruction"])
    for line_index, line in enumerate(textwrap.wrap(instruction, width=150)[:2]):
        draw.text((8, 31 + line_index * 19), line, fill="#222222", font=body_font)
    for index, (label, image) in enumerate(panels):
        x = index * panel_width
        canvas.paste(_fit(image, panel_width, panel_height), (x, header_height + 24))
        draw.text((x + 7, header_height + 3), label, fill="black", font=body_font)
    return canvas


def _write_pages(selected: Sequence[Dict], args: argparse.Namespace) -> List[str]:
    names = []
    for page_index, start in enumerate(range(0, len(selected), args.rows_per_page), start=1):
        items = selected[start : start + args.rows_per_page]
        rendered = [
            _render_sample(item, args.panel_width, args.panel_height) for item in items
        ]
        width = max(image.width for image in rendered)
        height = sum(image.height for image in rendered)
        page = Image.new("RGB", (width, height), "white")
        y = 0
        for image in rendered:
            page.paste(image, (0, y))
            y += image.height
        name = f"scaleedit_mask_preview_page_{page_index}.jpg"
        page.save(args.output_dir / name, quality=90, subsampling=1)
        names.append(name)
    return names


def _summary(rows: Sequence[Dict], selected: Sequence[Dict], pages: Sequence[str]) -> Dict:
    tasks = Counter(str(item["mask"]["final_task"]) for item in rows)
    flags = Counter(str(item["mask"]["qc_flag"]) for item in rows)
    sources = Counter(str(item["mask"]["mask_source"]) for item in rows)
    modes = Counter(
        str(json.loads(item["mask"]["ground_json"]).get("mask_mode", "unresolved"))
        for item in rows
    )
    areas: Dict[str, List[float]] = defaultdict(list)
    for item in rows:
        value = float(item["mask"].get("area_frac", math.nan))
        if math.isfinite(value):
            areas[str(item["mask"]["final_task"])].append(value)
    per_task = {}
    for task in sorted(tasks):
        values = areas.get(task, [])
        per_task[task] = {
            "rows": tasks[task],
            "mean_area_frac": round(sum(values) / len(values), 6) if values else None,
            "min_area_frac": round(min(values), 6) if values else None,
            "max_area_frac": round(max(values), 6) if values else None,
        }
    return {
        "rows": len(rows),
        "task_count": len(tasks),
        "tasks": dict(sorted(tasks.items())),
        "qc_flags": dict(sorted(flags.items())),
        "mask_sources": dict(sorted(sources.items())),
        "mask_modes": dict(sorted(modes.items())),
        "per_task": per_task,
        "preview_pages": list(pages),
        "selected_samples": [
            {
                "sample_id": item["mask"]["sample_id"],
                "final_task": item["mask"]["final_task"],
                "shard": item["shard"],
                "row_idx": item["mask"]["row_idx"],
            }
            for item in selected
        ],
    }


def main() -> None:
    args = parse_args()
    for field in ("input_dir", "grounding_dir", "mask_dir", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args)
    requested_tasks = {str(value) for value in args.task}
    requested_ids = {str(value) for value in args.sample_id}
    if requested_tasks:
        rows = [item for item in rows if str(item["mask"]["final_task"]) in requested_tasks]
    if requested_ids:
        rows = [item for item in rows if str(item["mask"]["sample_id"]) in requested_ids]
        found_ids = {str(item["mask"]["sample_id"]) for item in rows}
        if found_ids != requested_ids:
            raise KeyError(f"sample_id(s) not found after filtering: {sorted(requested_ids - found_ids)}")
    if not rows:
        raise ValueError("no rows remain after visualization filters")
    selected = _select(rows, args.samples_per_task)
    pages = _write_pages(selected, args)
    summary = _summary(rows, selected, pages)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
