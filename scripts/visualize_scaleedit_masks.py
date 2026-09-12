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


COARSE_CATEGORY_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "01_object_composition",
        (
            "object_addition",
            "object_removal",
            "object_replacement",
            "count_change",
            "compositional_editing",
        ),
    ),
    (
        "02_attribute_action",
        ("action_editing", "color_change", "material_change", "size_change"),
    ),
    (
        "03_text_symbol",
        (
            "building_surface_text_editing",
            "gui_interface_text_editing",
            "movie_poster_text_editing",
            "object_surface_text_editing",
            "symbolic_reasoning",
        ),
    ),
    (
        "04_reasoning_repair",
        (
            "perceptual_reasoning",
            "scientific_reasoning",
            "social_reasoning",
            "visual_beautification",
        ),
    ),
    (
        "05_scene_global_extraction",
        (
            "background_replacement",
            "part_extraction",
            "style_transfer",
            "tone_adjustment",
            "viewpoint_transformation",
        ),
    ),
)


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
    parser.add_argument(
        "--coarse-category-groups",
        action="store_true",
        help="Write the five documented ScaleEdit category groups below output-dir",
    )
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
    # Boxes are drawn before the image is reduced to the contact-sheet panel.
    # A dark halo and thick colored stroke keep target boxes visible after
    # reduction, including over bright edited-image backgrounds.
    width = max(6, min(canvas.size) // 90)
    halo_width = width + max(3, width // 2)
    font = _font(max(18, min(canvas.size) // 38))
    for item in items:
        box = _pixel_box(item["bbox_2d"], canvas)
        draw.rectangle(box, outline="#101010", width=halo_width)
        draw.rectangle(box, outline=color, width=width)
        label = f"{item.get('ref', '')} [{item.get('mask_method', 'sam')}]"
        text_x, text_y = box[0] + halo_width, box[1] + halo_width
        text_box = draw.textbbox((text_x, text_y), label, font=font, stroke_width=1)
        draw.rectangle(text_box, fill="#101010")
        draw.text(
            (text_x, text_y),
            label,
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="#101010",
        )
    return canvas


def _semantic_box_items(
    grounded_items: Sequence[Dict], instances: Sequence[Dict], image_side: str
) -> List[Dict]:
    """Use the realized SAM contour box for review, with grounding fallback."""

    by_id = {str(item.get("instance_id", "")): item for item in instances}
    result = []
    for index, grounded in enumerate(grounded_items):
        item = dict(grounded)
        candidate_id = int(grounded.get("candidate_id", index))
        member_index = int(grounded.get("member_index", 0))
        instance = by_id.get(
            f"{image_side}_c{candidate_id}_m{member_index}",
            by_id.get(f"{image_side}_{index}", {}),
        )
        semantic = instance.get("semantic_bbox_2d")
        if (
            isinstance(semantic, (list, tuple))
            and len(semantic) == 4
            and float(semantic[2]) > float(semantic[0])
            and float(semantic[3]) > float(semantic[1])
        ):
            item["bbox_2d"] = semantic
        result.append(item)
    return result


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


def _decode_mask(payload: bytes, shape: Tuple[int, int] | None = None) -> np.ndarray:
    if not payload:
        if shape is None or min(shape) <= 0:
            raise ValueError("empty mask payload has no valid fallback shape")
        return np.zeros(shape, dtype=np.uint8)
    return (np.asarray(Image.open(io.BytesIO(payload)).convert("L")) > 0).astype(np.uint8)


def _load_rows(args: argparse.Namespace) -> List[Dict]:
    result = []
    raw_by_name = {path.name: path for path in discover_shards(args.input_dir)}
    for ground_path in sorted(args.grounding_dir.glob("part-*.parquet")):
        raw_path = raw_by_name.get(ground_path.name)
        if raw_path is None:
            raise FileNotFoundError(f"missing source shard for {ground_path.name}")
        mask_path = args.mask_dir / ground_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing mask output for {ground_path.name}")
        ground_rows = pq.read_table(ground_path).to_pylist()
        mask_rows = pq.read_table(mask_path).to_pylist()
        if len(ground_rows) != len(mask_rows):
            raise ValueError(
                f"row mismatch for {raw_path.name}: "
                f"ground={len(ground_rows)} mask={len(mask_rows)}"
            )
        all_raw_rows = pq.read_table(raw_path).to_pylist()
        raw_rows = []
        for ground in ground_rows:
            row_idx = int(ground["row_idx"])
            if row_idx < 0 or row_idx >= len(all_raw_rows):
                raise IndexError(
                    f"grounding row_idx out of range in {raw_path.name}: {row_idx}"
                )
            raw_rows.append(all_raw_rows[row_idx])
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
    instances = mask_row.get("instance_masks", []) or []
    source_items = _semantic_box_items(source_items, instances, "source")
    target_items = _semantic_box_items(payload.get("target", []), instances, "target")
    mask = _decode_mask(
        mask_row["mask_png"],
        (
            int(mask_row["mask_height"]) or source.height,
            int(mask_row["mask_width"]) or source.width,
        ),
    )
    source_boxes = _boxes(
        source,
        source_items,
        "#39ff72" if mode != "protect_foreground" else "#ffb000",
    )
    target_boxes = _boxes(target, target_items, "#00e5ff")
    source_label = "source + SOURCE boxes (green)"
    if mode == "protect_foreground":
        source_label = "source + PROTECTED boxes (orange)"
    target_label = (
        "edited + TARGET boxes (cyan)"
        if target_items
        else (
            "edited | no target boxes "
            f"({'full-image route' if mode == 'full_image' else 'source-only route'})"
        )
    )
    panels = [
        (source_label, source_boxes),
        (target_label, target_boxes),
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


def _write_review(rows: Sequence[Dict], args: argparse.Namespace, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_args = argparse.Namespace(**vars(args))
    render_args.output_dir = output_dir
    selected = _select(rows, args.samples_per_task)
    pages = _write_pages(selected, render_args)
    summary = _summary(rows, selected, pages)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    for field in ("input_dir", "grounding_dir", "mask_dir", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    rows = _load_rows(args)
    requested_tasks = {str(value) for value in args.task}
    requested_ids = {str(value) for value in args.sample_id}
    if args.coarse_category_groups:
        if requested_tasks or requested_ids:
            raise ValueError(
                "--coarse-category-groups cannot be combined with --task or --sample-id"
            )
        groups = []
        for name, tasks in COARSE_CATEGORY_GROUPS:
            task_set = set(tasks)
            group_rows = [
                item
                for item in rows
                if str(item["mask"]["final_task"]) in task_set
            ]
            if not group_rows:
                raise ValueError(f"no rows found for coarse category group {name}")
            summary = _write_review(group_rows, args, args.output_dir / name)
            groups.append(
                {
                    "name": name,
                    "tasks": list(tasks),
                    "rows": len(group_rows),
                    "summary": f"{name}/summary.json",
                    "preview_pages": [f"{name}/{page}" for page in summary["preview_pages"]],
                }
            )
        index = {"groups": groups, "samples_per_task": args.samples_per_task}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(index, indent=2, ensure_ascii=False))
        return
    if requested_tasks:
        rows = [item for item in rows if str(item["mask"]["final_task"]) in requested_tasks]
    if requested_ids:
        rows = [item for item in rows if str(item["mask"]["sample_id"]) in requested_ids]
        found_ids = {str(item["mask"]["sample_id"]) for item in rows}
        if found_ids != requested_ids:
            raise KeyError(f"sample_id(s) not found after filtering: {sorted(requested_ids - found_ids)}")
    if not rows:
        raise ValueError("no rows remain after visualization filters")
    summary = _write_review(rows, args, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
