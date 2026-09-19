"""Create review sheets from the exact two image inputs sent to an audit VLM."""

from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def rows_by_image(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        row["image"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("/usr/share/fonts/truetype/dejavu") / name), size
    )


def fit(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(
        image, ((size - image.width) // 2, (size - image.height) // 2)
    )
    return canvas


def display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def wrapped_lines(label: str, value: object, width: int = 120) -> list[str]:
    return textwrap.wrap(
        f"{label}: {display(value)}", width=width,
        break_long_words=True, break_on_hyphens=False,
    ) or [f"{label}: —"]


def model_detail_html(label: str, row: dict) -> str:
    audit = row.get("audit") or {}
    fields = [
        ("原图目标", "source_target"),
        ("编辑后同位置", "edited_target_area"),
        ("实际观察到的编辑", "observed_edit"),
        ("目标匹配", "target_match"),
        ("额外变化", "unexpected_change"),
        ("画面瑕疵", "artifact"),
        ("实际编辑指令", "observed_instruction"),
    ]
    evidence = "".join(
        f"<dt>{html.escape(title)}</dt><dd>{html.escape(display(audit.get(key)))}</dd>"
        for title, key in fields
    )
    raw_response = row.get("raw_response")
    raw_html = (
        "<details><summary>查看模型原始完整响应</summary>"
        f"<pre>{html.escape(display(raw_response))}</pre></details>"
        if raw_response is not None else ""
    )
    rewrite = row.get("rewrite_candidate")
    return (
        f"<div class='review'><h3>{html.escape(label)}：{html.escape(display(row.get('quality')))}</h3>"
        f"<p class='subverdict'>画面质量 {html.escape(display(audit.get('visual_quality')))}"
        f" · 指令匹配 {html.escape(display(audit.get('instruction_match')))}"
        f" · 改写状态 {html.escape(display(row.get('salvage_status')))}</p>"
        f"<p><strong>完整审核理由：</strong>{html.escape(display(audit.get('reason')))}</p>"
        f"<dl>{evidence}</dl>"
        f"<p><strong>建议改写：</strong>{html.escape(display(rewrite))}</p>"
        f"{raw_html}</div>"
    )


def manual_detail_html(row: dict, include_rewrite_validity: bool = True) -> str:
    rewrite_html = (
        f"<p><strong>8B 改写是否可用：</strong>{html.escape(display(row.get('rewrite_8b_valid')))}"
        f" · <strong>27B 改写是否可用：</strong>"
        f"{html.escape(display(row.get('rewrite_27b_valid')))}</p>"
        if include_rewrite_validity else ""
    )
    return (
        "<div class='review manual'><h3>人工复核："
        f"{html.escape(display(row.get('quality')))}</h3>"
        f"<p class='subverdict'>画面质量 {html.escape(display(row.get('visual_quality')))}"
        f" · 指令匹配 {html.escape(display(row.get('instruction_match')))}</p>"
        f"<p><strong>完整人工理由：</strong>{html.escape(display(row.get('reason')))}</p>"
        f"{rewrite_html}</div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit8-jsonl", type=Path, required=True)
    parser.add_argument("--audit27-jsonl", type=Path)
    parser.add_argument("--audit8-label", default="8B")
    parser.add_argument("--audit27-label", default="27B")
    parser.add_argument("--hide-rewrite-validity", action="store_true")
    parser.add_argument("--manual-jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--hide-audits", action="store_true",
        help="Show only image, edit type, and instruction for blind manual review.",
    )
    parser.add_argument("--tile-size", type=int, default=580)
    parser.add_argument("--page-size", type=int, default=4)
    args = parser.parse_args()
    if args.tile_size < 256 or args.page_size < 1:
        raise ValueError("tile-size must be >=256 and page-size must be >=1")

    annotations = rows_by_image(args.annotations_jsonl)
    audit8 = rows_by_image(args.audit8_jsonl)
    audit27 = rows_by_image(args.audit27_jsonl) if args.audit27_jsonl else {}
    manual = rows_by_image(args.manual_jsonl) if args.manual_jsonl else {}
    if set(annotations) != set(audit8):
        raise ValueError("The annotation and audit case sets must be identical")
    if not args.hide_audits and set(annotations) != set(audit27):
        raise ValueError("The 27B audit case set must match unless audits are hidden")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.out_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    cards: list[Image.Image] = []
    links: list[str] = []
    width = args.tile_size * 2 + 16
    for index, (name, row) in enumerate(annotations.items()):
        stem = Path(name).stem
        before_path = args.input_dir / f"{stem}_source.png"
        after_path = args.input_dir / f"{stem}_edited.png"
        with Image.open(before_path) as handle:
            before = fit(handle.copy(), args.tile_size)
        with Image.open(after_path) as handle:
            after = fit(handle.copy(), args.tile_size)
        instruction = str(row.get("editing_instruction", ""))
        verdict = "Blind manual review" if args.hide_audits else (
            f"{args.audit8_label}={audit8[name]['quality']}  "
            f"{args.audit27_label}={audit27[name]['quality']}  "
            f"human={manual.get(name, {}).get('quality', '?')}  "
            f"rewriteA={bool(audit8[name].get('rewrite_candidate'))}  "
            f"rewriteB={bool(audit27[name].get('rewrite_candidate'))}"
        )
        header_lines = wrapped_lines("Instruction", instruction, width=110)
        header_height = 64 + 23 * len(header_lines)
        reason_blocks = [] if args.hide_audits else [
            (f"{args.audit8_label} reason", (audit8[name].get("audit") or {}).get("reason")),
            (f"{args.audit27_label} reason", (audit27[name].get("audit") or {}).get("reason")),
            ("Human reason", manual.get(name, {}).get("reason")),
        ]
        reason_lines = [
            wrapped_lines(label, reason) for label, reason in reason_blocks
        ]
        reason_height = sum(12 + 21 * len(lines) for lines in reason_lines)
        row_height = header_height + args.tile_size + 18 + reason_height
        card = Image.new("RGB", (width, row_height), "white")
        draw = ImageDraw.Draw(card)
        draw.text(
            (8, 7), f"{index:03d} {name} [{row.get('task_type')}]",
            font=font(19, True), fill="black"
        )
        for line_index, line in enumerate(header_lines):
            draw.text((8, 34 + 23 * line_index), line, font=font(16), fill="black")
        draw.text(
            (8, header_height - 27), verdict,
            font=font(17, True), fill=(70, 40, 20)
        )
        card.paste(before, (0, header_height))
        card.paste(after, (args.tile_size + 16, header_height))
        y = header_height + args.tile_size + 11
        for lines in reason_lines:
            for line in lines:
                draw.text((8, y), line, font=font(16), fill="black")
                y += 21
            y += 12
        cards.append(card)
        card_path = cases_dir / f"{index:03d}_{stem}.jpg"
        card.save(card_path, quality=94)
        review_html = "" if args.hide_audits else (
            model_detail_html(args.audit8_label, audit8[name])
            + model_detail_html(args.audit27_label, audit27[name])
            + manual_detail_html(manual.get(name, {}), not args.hide_rewrite_validity)
        )
        links.append(
            f"<section id='case-{index:03d}'><h2>{index:03d} {html.escape(name)}</h2>"
            f"<p><strong>编辑类型：</strong>{html.escape(display(row.get('task_type')))}</p>"
            f"<p><strong>原始指令：</strong>{html.escape(instruction)}</p>"
            f"<p>{html.escape(verdict)}</p>"
            f"<img loading='lazy' src='cases/{html.escape(card_path.name)}' "
            "alt='左侧为标记 mask 的原图，右侧为编辑后图'>"
            f"{review_html}</section>"
        )

    for start in range(0, len(cards), args.page_size):
        page = cards[start : start + args.page_size]
        sheet = Image.new("RGB", (width, sum(card.height for card in page)
                           + 8 * (len(page) - 1)), (230, 230, 230))
        y = 0
        for offset, card in enumerate(page):
            sheet.paste(card, (0, y))
            y += card.height + 8
        sheet.save(
            args.out_dir / f"contact_{start:03d}_{start + len(page) - 1:03d}.jpg",
            quality=92,
        )
    (args.out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font-family:sans-serif;max-width:1250px;margin:auto;background:#eee;"
        "line-height:1.5}section{background:white;padding:18px;margin:18px 0}"
        "img{width:100%;height:auto}.review{border-left:4px solid #5879a5;"
        "padding:4px 16px;margin:16px 0;background:#f5f8fc}.manual{border-color:#a8783b;"
        "background:#fff9ee}.subverdict{color:#555}dl{display:grid;"
        "grid-template-columns:160px 1fr;gap:5px 12px}dt{font-weight:bold}"
        "dd{margin:0;white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#e7edf4;padding:12px}p{overflow-wrap:anywhere}</style>"
        + "".join(links),
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(cards), "pages": (len(cards) + args.page_size - 1) // args.page_size,
                      "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
