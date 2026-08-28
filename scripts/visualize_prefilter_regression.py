#!/usr/bin/env python3
"""Render visual boards for a CrispEdit prefilter regression run."""

from __future__ import annotations

import argparse
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, ImageOps


BG = "#eef2f7"
INK = "#172033"
MUTED = "#5d687b"
NAVY = "#18243a"
WHITE = "#ffffff"
GREEN = "#16845b"
GREEN_SOFT = "#e5f6ef"
RED = "#c54747"
RED_SOFT = "#fbeaea"
AMBER = "#ba7215"
AMBER_SOFT = "#fff3da"
BLUE = "#356fc0"
BLUE_SOFT = "#e8f0fc"
BORDER = "#d6deea"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

GROUP_ORDER = [
    "A_FALSE_KEEP_NOOP",
    "B_KEEP_REASON_MISMATCH",
    "HIGH_CONFIDENCE_KEEP_CONTROL",
]

GROUP_LABELS = {
    "A_FALSE_KEEP_NOOP": "Class A · old false-keeps / no-op edits",
    "B_KEEP_REASON_MISMATCH": "Class B · valid edits with old reason mismatch",
    "HIGH_CONFIDENCE_KEEP_CONTROL": "Legacy high-confidence keep probes",
}

GROUP_NOTES = {
    "A_FALSE_KEEP_NOOP": "Human-reviewed regression cases. Desired result: DROP.",
    "B_KEEP_REASON_MISMATCH": "Human-reviewed regression cases. Desired result: KEEP with a better reason.",
    "HIGH_CONFIDENCE_KEEP_CONTROL": (
        "Pseudo-gold controls sampled from the old audit, not manually verified ground truth."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--wall-seconds", type=float, default=None)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str | None = None,
    width: int = 1,
    radius: int = 16,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=text_font)
    return box[2] - box[0]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = str(text or "").replace("\n", " ").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, text_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) <= max_lines:
        return lines
    lines = lines[:max_lines]
    while lines[-1] and text_width(draw, lines[-1] + "…", text_font) > max_width:
        lines[-1] = lines[-1][:-1]
    lines[-1] += "…"
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    max_lines: int,
    line_gap: int = 4,
) -> int:
    x, y = xy
    lines = wrap_text(draw, text, text_font, max_width, max_lines)
    line_height = text_font.size + line_gap
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * line_height), line, font=text_font, fill=fill)
    return y + len(lines) * line_height


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str,
    text_fill: str,
    text_font: ImageFont.FreeTypeFont,
    pad_x: int = 13,
    height: int = 34,
) -> int:
    x, y = xy
    width = text_width(draw, text, text_font) + 2 * pad_x
    rounded_box(draw, (x, y, x + width, y + height), fill=fill, radius=height // 2)
    bbox = draw.textbbox((0, 0), text, font=text_font)
    text_y = y + (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x + pad_x, text_y), text, font=text_font, fill=text_fill)
    return width


def bytes_image(value: Any) -> Image.Image:
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, memoryview):
        value = value.tobytes()
    return Image.open(io.BytesIO(value)).convert("RGB")


def load_rows(input_dir: Path, report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(report_path.read_text())
    rows = [dict(item) for item in report["rows"]]
    anchor_indices_by_shard: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if str(row["issue_class"]) in GROUP_ORDER:
            anchor_indices_by_shard[str(row["mini_shard"])].append(idx)
    for shard_name, indices in anchor_indices_by_shard.items():
        table = pq.read_table(
            input_dir / shard_name,
            columns=["input_img", "output_img"],
        )
        for idx in indices:
            record = table.slice(int(rows[idx]["row_idx"]), 1).to_pylist()[0]
            rows[idx]["source_image"] = bytes_image(record["input_img"])
            rows[idx]["target_image"] = bytes_image(record["output_img"])
    return report, rows


def paste_contained(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    background: str = "#111827",
) -> None:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    canvas.paste(background, box)
    contained = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    x = x0 + (width - contained.width) // 2
    y = y0 + (height - contained.height) // 2
    canvas.paste(contained, (x, y))


def decision_colors(decision: str) -> tuple[str, str]:
    if str(decision).lower() == "keep":
        return GREEN_SOFT, GREEN
    return RED_SOFT, RED


def predicate_summary(row: dict[str, Any]) -> str:
    predicates = row.get("predicates") or {}
    failures = [name for name, value in predicates.items() if str(value).upper() == "FALSE"]
    unknown = [name for name, value in predicates.items() if str(value).upper() == "UNKNOWN"]
    parts = []
    if failures:
        parts.append("FALSE: " + ", ".join(failures))
    if unknown:
        parts.append("UNKNOWN: " + ", ".join(unknown))
    return " · ".join(parts) if parts else "All required predicates passed"


def step5_summary(row: dict[str, Any]) -> str:
    if not row.get("review_triggered"):
        return "Step 5: not triggered"
    resolution = row.get("review_resolution") or row.get("review_decision") or "completed"
    return f"Step 5: triggered → {str(resolution).upper()}"


def draw_pair(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: dict[str, Any],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    gap = 10
    each = (width - gap) // 2
    boxes = [
        (x, y, x + each, y + height),
        (x + each + gap, y, x + width, y + height),
    ]
    for image_value, box, label in zip(
        (row["source_image"], row["target_image"]), boxes, ("SOURCE", "TARGET")
    ):
        paste_contained(canvas, image_value, box)
        draw.rectangle(box, outline="#bac5d5", width=2)
        label_font = font(17, bold=True)
        label_w = text_width(draw, label, label_font) + 18
        draw.rectangle((box[0], box[1], box[0] + label_w, box[1] + 31), fill="#111827")
        draw.text((box[0] + 9, box[1] + 5), label, font=label_font, fill=WHITE)


def draw_card(canvas: Image.Image, row: dict[str, Any], x: int, y: int, width: int, height: int) -> None:
    draw = ImageDraw.Draw(canvas)
    correct = bool(row["correct"])
    outline = GREEN if correct else AMBER
    rounded_box(draw, (x, y, x + width, y + height), fill=WHITE, outline=outline, width=4, radius=18)

    pad = 22
    shard_label = f'{row["source_shard"]}:{row["source_row_idx"]}'
    draw.text((x + pad, y + 18), shard_label, font=font(21, bold=True), fill=INK)

    decision = str(row["actual_decision"]).upper()
    expected = str(row["expected_decision"]).upper()
    badge_text = f"{decision}  {'✓' if correct else '≠ expected ' + expected}"
    badge_fill, badge_ink = decision_colors(decision)
    badge_width = text_width(draw, badge_text, font(18, bold=True)) + 26
    pill(
        draw,
        (x + width - pad - badge_width, y + 13),
        badge_text,
        fill=badge_fill if correct else AMBER_SOFT,
        text_fill=badge_ink if correct else AMBER,
        text_font=font(18, bold=True),
        height=38,
    )

    pair_y = y + 67
    draw_pair(canvas, draw, row, x + pad, pair_y, width - 2 * pad, 286)

    text_y = pair_y + 304
    draw.text((x + pad, text_y), "Instruction", font=font(17, bold=True), fill=BLUE)
    text_y = draw_wrapped(
        draw,
        (x + pad, text_y + 25),
        row.get("instruction", ""),
        font(18),
        INK,
        width - 2 * pad,
        2,
        5,
    )
    text_y += 5
    text_y = draw_wrapped(
        draw,
        (x + pad, text_y),
        predicate_summary(row),
        font(16),
        MUTED,
        width - 2 * pad,
        2,
        3,
    )
    text_y += 2
    draw.text((x + pad, text_y), step5_summary(row), font=font(16, bold=True), fill=INK)
    text_y += 28
    reason = row.get("reason") or row.get("actual_reason") or ""
    draw_wrapped(
        draw,
        (x + pad, text_y),
        f"Reason: {reason}",
        font(15),
        MUTED,
        width - 2 * pad,
        2,
        3,
    )


def render_group_board(group: str, rows: list[dict[str, Any]], output_path: Path) -> None:
    board_width = 1880
    margin = 30
    gap = 22
    header_height = 145
    card_width = (board_width - 2 * margin - gap) // 2
    card_height = 590
    grid_rows = math.ceil(len(rows) / 2)
    board_height = header_height + grid_rows * card_height + max(0, grid_rows - 1) * gap + margin
    canvas = Image.new("RGB", (board_width, board_height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, board_width, 105), fill=NAVY)
    draw.text((margin, 23), GROUP_LABELS[group], font=font(34, bold=True), fill=WHITE)
    draw.text((margin, 70), GROUP_NOTES[group], font=font(18), fill="#cbd5e5")
    correct_count = sum(bool(row["correct"]) for row in rows)
    score = f"{correct_count}/{len(rows)} matched expectation"
    score_w = text_width(draw, score, font(19, bold=True)) + 30
    pill(
        draw,
        (board_width - margin - score_w, 32),
        score,
        fill=GREEN_SOFT if correct_count == len(rows) else AMBER_SOFT,
        text_fill=GREEN if correct_count == len(rows) else AMBER,
        text_font=font(19, bold=True),
        height=40,
    )

    for idx, row in enumerate(rows):
        col = idx % 2
        grid_row = idx // 2
        x = margin + col * (card_width + gap)
        y = header_height + grid_row * (card_height + gap)
        draw_card(canvas, row, x, y, card_width, card_height)

    canvas.save(output_path, quality=95)


def draw_metric(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    note: str,
    color: str,
) -> None:
    rounded_box(draw, (x, y, x + width, y + 144), fill=WHITE, outline=BORDER, width=2, radius=18)
    draw.rectangle((x, y, x + 9, y + 144), fill=color)
    draw.text((x + 27, y + 19), label, font=font(18, bold=True), fill=MUTED)
    draw.text((x + 27, y + 48), value, font=font(42, bold=True), fill=INK)
    draw.text((x + 27, y + 108), note, font=font(15), fill=MUTED)


def draw_compact_tile(
    canvas: Image.Image,
    row: dict[str, Any],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    correct = bool(row["correct"])
    rounded_box(
        draw,
        (x, y, x + width, y + height),
        fill=WHITE,
        outline=GREEN if correct else AMBER,
        width=3,
        radius=14,
    )
    pair_height = 119
    draw_pair(canvas, draw, row, x + 12, y + 12, width - 24, pair_height)
    decision = str(row["actual_decision"]).upper()
    status = f"{decision} {'✓' if correct else '≠ ' + str(row['expected_decision']).upper()}"
    status_fill, status_ink = decision_colors(decision)
    draw.text((x + 14, y + 146), status, font=font(18, bold=True), fill=status_ink if correct else AMBER)
    draw.text(
        (x + 14, y + 175),
        f'{row["source_shard"]}:{row["source_row_idx"]}',
        font=font(14),
        fill=MUTED,
    )
    draw_wrapped(
        draw,
        (x + 14, y + 199),
        row.get("instruction", ""),
        font(14),
        INK,
        width - 28,
        2,
        3,
    )


def render_overview(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "CrispEdit fact prefilter · visual regression",
    subtitle: str | None = None,
) -> None:
    width, height = 1900, 1350
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 126), fill=NAVY)
    draw.text((40, 25), title, font=font(38, bold=True), fill=WHITE)
    draw.text(
        (40, 78),
        subtitle or f"{len(rows)}-row Qwen3-VL-8B test · exact source/target pairs shown below",
        font=font(19),
        fill="#cbd5e5",
    )

    summary = report["summary"]
    accuracy = 100.0 * float(summary.get("accuracy", 0.0))
    class_a_correct = int(summary.get("class_A_FALSE_KEEP_NOOP_correct", 0))
    class_a_rows = int(summary.get("class_A_FALSE_KEEP_NOOP_rows", 0))
    class_b_correct = int(summary.get("class_B_KEEP_REASON_MISMATCH_correct", 0))
    class_b_rows = int(summary.get("class_B_KEEP_REASON_MISMATCH_rows", 0))
    probe_correct = int(summary.get("class_HIGH_CONFIDENCE_KEEP_CONTROL_correct", 0))
    probe_rows = int(summary.get("class_HIGH_CONFIDENCE_KEEP_CONTROL_rows", 0))
    metric_gap = 18
    metric_margin = 40
    metric_width = (width - 2 * metric_margin - 3 * metric_gap) // 4
    metrics = [
        ("OVERALL", f'{summary["correct"]}/{summary["rows"]}', f"{accuracy:.1f}% matched expected labels", BLUE),
        ("CLASS A", f"{class_a_correct}/{class_a_rows}", "false-keeps corrected", GREEN),
        ("CLASS B", f"{class_b_correct}/{class_b_rows}", "valid edits preserved", GREEN),
        (
            "KEEP PROBES",
            f"{probe_correct}/{probe_rows}",
            "legacy pseudo-gold controls",
            GREEN if probe_correct == probe_rows else AMBER,
        ),
    ]
    for idx, metric in enumerate(metrics):
        draw_metric(draw, metric_margin + idx * (metric_width + metric_gap), 154, metric_width, *metric)

    legend_y = 324
    draw.text((40, legend_y), "Decision map", font=font(25, bold=True), fill=INK)
    draw.text(
        (250, legend_y + 3),
        "Green border = matched expectation · amber border = disagreement · KEEP/DROP is the new policy result",
        font=font(17),
        fill=MUTED,
    )

    cols = 5
    tile_gap = 15
    tile_margin = 40
    tile_width = (width - 2 * tile_margin - (cols - 1) * tile_gap) // cols
    tile_height = 278
    start_y = 372
    for idx, row in enumerate(rows):
        x = tile_margin + (idx % cols) * (tile_width + tile_gap)
        y = start_y + (idx // cols) * (tile_height + tile_gap)
        draw_compact_tile(canvas, row, x, y, tile_width, tile_height)

    note_y = 1269
    perfect = int(summary.get("incorrect", 0)) == 0
    rounded_box(
        draw,
        (40, note_y, width - 40, note_y + 56),
        fill=GREEN_SOFT if perfect else AMBER_SOFT,
        radius=12,
    )
    footer = (
        "All 15 expectations matched; Class A/B are human-reviewed regressions and keep probes are old-pipeline pseudo-gold."
        if perfect
        else "Class A/B are human-reviewed regressions; keep probes are old-pipeline pseudo-gold and amber cases need review."
    )
    draw.text(
        (60, note_y + 17),
        f"Interpretation: {footer}",
        font=font(17, bold=True),
        fill=GREEN if perfect else AMBER,
    )
    canvas.save(output_path, quality=95)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        issue_class = str(row["issue_class"])
        correct = bool(row["correct"])
        counts["rows"] += 1
        counts["correct"] += int(correct)
        counts["incorrect"] += int(not correct)
        counts[f"class_{issue_class}_rows"] += 1
        counts[f"class_{issue_class}_correct"] += int(correct)
        counts[f"actual_{row['actual_decision']}"] += 1
    result = dict(counts)
    result["accuracy"] = counts["correct"] / max(counts["rows"], 1)
    return result


def draw_horizontal_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    value: int,
    total: int,
    color: str,
    label: str,
    value_label: str | None = None,
) -> None:
    label_font = font(17, bold=True)
    value_font = font(16, bold=True)
    draw.text((x, y), label, font=label_font, fill=INK)
    shown = value_label or f"{value}/{total}"
    shown_w = text_width(draw, shown, value_font)
    draw.text((x + width - shown_w, y + 1), shown, font=value_font, fill=MUTED)
    bar_y = y + 30
    draw.rounded_rectangle((x, bar_y, x + width, bar_y + 18), radius=9, fill="#e2e8f0")
    filled = int(width * value / max(total, 1))
    if filled:
        draw.rounded_rectangle((x, bar_y, x + filled, bar_y + 18), radius=9, fill=color)


def render_batch_dashboard(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    run_summary: dict[str, Any],
    output_path: Path,
    *,
    batch_size: int | None,
    wall_seconds: float | None,
) -> None:
    width, height = 1900, 1320
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 126), fill=NAVY)
    batch_label = f"batch {batch_size}" if batch_size else "batch run"
    draw.text((40, 25), f"CrispEdit fact prefilter · {batch_label}", font=font(38, bold=True), fill=WHITE)
    draw.text(
        (40, 78),
        "8 × H100 · 295-row stratified evaluation · legacy labels are diagnostic references, not human gold",
        font=font(19),
        fill="#cbd5e5",
    )

    summary = report["summary"]
    rows_count = int(summary["rows"])
    metric_gap = 18
    metric_margin = 40
    metric_width = (width - 2 * metric_margin - 3 * metric_gap) // 4
    wall_value = f"{wall_seconds:.1f} s" if wall_seconds is not None else "n/a"
    throughput = rows_count / wall_seconds if wall_seconds else 0.0
    metrics = [
        ("WALL TIME", wall_value, f"{throughput:.3f} rows/s" if wall_seconds else "not recorded", BLUE),
        (
            "MODEL.GENERATE",
            str(run_summary.get("generation_calls", "n/a")),
            f'{summary.get("mllm_conversations", 0)} logical conversations',
            BLUE,
        ),
        (
            "DECISIONS",
            f'{summary.get("actual_keep", 0)} / {summary.get("actual_drop", 0)}',
            "keep / drop",
            GREEN,
        ),
        (
            "ERRORS",
            str(run_summary.get("errors", 0)),
            f'{summary.get("review_triggered", 0)} boundary reviews',
            GREEN if int(run_summary.get("errors", 0)) == 0 else RED,
        ),
    ]
    for idx, metric in enumerate(metrics):
        draw_metric(draw, metric_margin + idx * (metric_width + metric_gap), 154, metric_width, *metric)

    panel_y = 326
    left_x, left_w = 40, 570
    mid_x, mid_w = 632, 570
    right_x, right_w = 1224, 636
    panel_h = 330
    for x, w in ((left_x, left_w), (mid_x, mid_w), (right_x, right_w)):
        rounded_box(draw, (x, panel_y, x + w, panel_y + panel_h), fill=WHITE, outline=BORDER, width=2, radius=18)

    draw.text((left_x + 24, panel_y + 20), "Output distribution", font=font(24, bold=True), fill=INK)
    draw_horizontal_bar(
        draw,
        x=left_x + 24,
        y=panel_y + 68,
        width=left_w - 48,
        value=int(summary.get("actual_drop", 0)),
        total=rows_count,
        color=RED,
        label="DROP",
        value_label=f'{summary.get("actual_drop", 0)} · {100 * summary.get("actual_drop", 0) / rows_count:.1f}%',
    )
    draw_horizontal_bar(
        draw,
        x=left_x + 24,
        y=panel_y + 129,
        width=left_w - 48,
        value=int(summary.get("actual_keep", 0)),
        total=rows_count,
        color=GREEN,
        label="KEEP",
        value_label=f'{summary.get("actual_keep", 0)} · {100 * summary.get("actual_keep", 0) / rows_count:.1f}%',
    )
    verdicts = run_summary.get("verdict_counts", {})
    draw.text((left_x + 24, panel_y + 205), "Verdicts", font=font(18, bold=True), fill=MUTED)
    draw.text(
        (left_x + 24, panel_y + 240),
        f'PASS {verdicts.get("PASS", 0)}     FAIL {verdicts.get("FAIL", 0)}     UNSURE {verdicts.get("UNSURE", 0)}',
        font=font(21, bold=True),
        fill=INK,
    )

    draw.text((mid_x + 24, panel_y + 20), "Routing and cost", font=font(24, bold=True), fill=INK)
    routing = [
        ("Deterministic slots", int(summary.get("deterministic_slots", 0)), BLUE),
        ("Code state matches", int(summary.get("code_matches", 0)), GREEN),
        ("Early exits", int(summary.get("early_exits", 0)), GREEN),
        ("Boundary reviews", int(summary.get("review_triggered", 0)), AMBER),
    ]
    for idx, (label, value, color) in enumerate(routing):
        draw_horizontal_bar(
            draw,
            x=mid_x + 24,
            y=panel_y + 64 + idx * 59,
            width=mid_w - 48,
            value=value,
            total=rows_count,
            color=color,
            label=label,
        )

    draw.text((right_x + 24, panel_y + 20), "15 anchor cases", font=font(24, bold=True), fill=INK)
    anchor_specs = [
        ("Class A · false keeps", "A_FALSE_KEEP_NOOP", GREEN),
        ("Class B · reason mismatch", "B_KEEP_REASON_MISMATCH", GREEN),
        ("Expected-keep controls", "HIGH_CONFIDENCE_KEEP_CONTROL", AMBER),
    ]
    for idx, (label, group, color) in enumerate(anchor_specs):
        total = int(summary.get(f"class_{group}_rows", 0))
        correct = int(summary.get(f"class_{group}_correct", 0))
        draw_horizontal_bar(
            draw,
            x=right_x + 24,
            y=panel_y + 69 + idx * 67,
            width=right_w - 48,
            value=correct,
            total=total,
            color=color if correct == total else AMBER,
            label=label,
        )
    anchor_misses = [
        row for row in rows
        if not str(row["issue_class"]).startswith("LEGACY_") and not bool(row["correct"])
    ]
    miss_text = "No anchor misses" if not anchor_misses else "Miss: " + ", ".join(
        f'{row["source_shard"]}:{row["source_row_idx"]}' for row in anchor_misses
    )
    draw_wrapped(
        draw,
        (right_x + 24, panel_y + 270),
        miss_text,
        font(16, bold=True),
        AMBER if anchor_misses else GREEN,
        right_w - 48,
        2,
    )

    chart_x, chart_y, chart_w, chart_h = 40, 682, 1820, 556
    rounded_box(draw, (chart_x, chart_y, chart_x + chart_w, chart_y + chart_h), fill=WHITE, outline=BORDER, width=2, radius=18)
    draw.text(
        (chart_x + 24, chart_y + 20),
        "Legacy high-confidence reference agreement by edit type",
        font=font(25, bold=True),
        fill=INK,
    )
    draw.text(
        (chart_x + 24, chart_y + 56),
        "Each bar is agreement with 20 old-audit reference decisions; this is drift diagnostics, not accuracy.",
        font=font(17),
        fill=MUTED,
    )
    by_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        if not str(row["issue_class"]).startswith("LEGACY_"):
            continue
        expected = str(row["expected_decision"])
        values = by_type[str(row["raw_type"])][expected]
        values[0] += int(bool(row["correct"]))
        values[1] += 1
    type_order = ["add", "background change", "color", "motion change", "remove", "replace", "style"]
    row_start = chart_y + 104
    label_w = 220
    bar_w = 600
    gap = 90
    draw.text((chart_x + label_w + 24, chart_y + 85), "Old DROP reference", font=font(15, bold=True), fill=RED)
    draw.text((chart_x + label_w + 24 + bar_w + gap, chart_y + 85), "Old KEEP reference", font=font(15, bold=True), fill=GREEN)
    for idx, edit_type in enumerate(type_order):
        y = row_start + idx * 60
        label = edit_type.replace(" change", "")
        draw.text((chart_x + 24, y + 7), label.upper(), font=font(17, bold=True), fill=INK)
        for col, expected in enumerate(("drop", "keep")):
            correct, total = by_type[edit_type][expected]
            bar_x = chart_x + label_w + 24 + col * (bar_w + gap)
            draw_horizontal_bar(
                draw,
                x=bar_x,
                y=y,
                width=bar_w,
                value=correct,
                total=total,
                color=RED if expected == "drop" else GREEN,
                label="",
                value_label=f"{correct}/{total}",
            )

    draw.text(
        (40, 1264),
        "Interpretation: batch 16 is conservative on remove / motion / replace keeps; inspect disagreements before treating old labels as ground truth.",
        font=font(17, bold=True),
        fill=MUTED,
    )
    canvas.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, rows = load_rows(args.input_dir, args.report_json)
    prefix = f"batch{args.batch_size}" if args.batch_size else "fact_prefilter"
    anchor_rows = [
        row for row in rows if not str(row["issue_class"]).startswith("LEGACY_")
    ]
    expanded_report = len(anchor_rows) != len(rows)
    if expanded_report:
        run_summary_path = args.report_json.parent / "audit" / "run_summary.json"
        run_summary = (
            json.loads(run_summary_path.read_text()) if run_summary_path.exists() else {}
        )
        render_batch_dashboard(
            report,
            rows,
            run_summary,
            args.output_dir / f"{prefix}_summary.png",
            batch_size=args.batch_size,
            wall_seconds=args.wall_seconds,
        )
        anchor_report = {"summary": summarize_rows(anchor_rows), "rows": anchor_rows}
        render_overview(
            anchor_report,
            anchor_rows,
            args.output_dir / f"{prefix}_anchors.png",
            title=f"CrispEdit fact prefilter · {prefix} anchors",
            subtitle="8 reviewed regression cases + 7 expected-keep controls · exact source/target pairs",
        )
    else:
        render_overview(report, rows, args.output_dir / f"{prefix}_overview.png")
    filenames = {
        "A_FALSE_KEEP_NOOP": f"{prefix}_class_a.png",
        "B_KEEP_REASON_MISMATCH": f"{prefix}_class_b.png",
        "HIGH_CONFIDENCE_KEEP_CONTROL": f"{prefix}_keep_controls.png",
    }
    for group in GROUP_ORDER:
        group_rows = [row for row in rows if row["issue_class"] == group]
        render_group_board(group, group_rows, args.output_dir / filenames[group])

    print(f"Wrote visual boards to {args.output_dir}")


if __name__ == "__main__":
    main()
