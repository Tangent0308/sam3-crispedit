"""Build a standalone HTML gallery and documentation thumbnails for the final dataset."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import html
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from pycocotools import mask as coco_mask


PANE = (440, 300)
GAP = 12
LABEL_HEIGHT = 28


def read_selection(path: Path) -> list[dict]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("selection must be a nonempty JSON list")
    ids = [value["case_id"] for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate selected case_id")
    return values


def read_selected(manifest: Path, selected: list[dict]) -> dict[str, dict]:
    wanted = {item["case_id"] for item in selected}
    found = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["case_id"] in wanted:
                found[row["case_id"]] = row
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"selected cases not present in final manifest: {sorted(missing)}")
    return found


def annotated_source(source: Image.Image, mask_rle: dict) -> tuple[Image.Image, Image.Image]:
    mask_array = coco_mask.decode(mask_rle)
    if mask_array.shape != (source.height, source.width):
        raise ValueError(f"mask/image size mismatch: {mask_array.shape} vs {source.size}")
    mask = Image.fromarray((mask_array * 255).astype("uint8"), "L")
    if not mask.getbbox():
        raise ValueError("empty mask in final gallery")
    edge = ImageChops.difference(mask.filter(ImageFilter.MaxFilter(7)),
                                 mask.filter(ImageFilter.MinFilter(7)))
    marked = source.copy()
    marked.paste((0, 255, 255), (0, 0), edge)
    return marked, mask


def context_box(mask: Image.Image) -> tuple[int, int, int, int]:
    left, top, right, bottom = mask.getbbox()
    width, height = right - left, bottom - top
    pad = max(80, int(max(width, height) * 0.55))
    return (max(0, left - pad), max(0, top - pad),
            min(mask.width, right + pad), min(mask.height, bottom + pad))


def pane(canvas: Image.Image, image: Image.Image, x: int, y: int, title: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((x + 4, y + 5), title, fill=(25, 31, 44))
    resized = ImageOps.contain(image, PANE, Image.Resampling.LANCZOS)
    canvas.paste(resized, (x + (PANE[0] - resized.width) // 2,
                           y + LABEL_HEIGHT + (PANE[1] - resized.height) // 2))


def composite(row: dict, dataset_root: Path) -> tuple[bytes, float]:
    source_path = dataset_root / row["source_image"]
    edited_path = dataset_root / row["edited_image"]
    with Image.open(source_path) as image:
        source = image.convert("RGB")
    with Image.open(edited_path) as image:
        edited = image.convert("RGB")
    if edited.size != source.size:
        raise ValueError(f"source/edit size mismatch: {row['case_id']}")
    marked, mask = annotated_source(source, row["mask_rle"])
    box = context_box(mask)
    width = PANE[0] * 2 + GAP * 3
    height = (PANE[1] + LABEL_HEIGHT) * 2 + GAP * 3
    canvas = Image.new("RGB", (width, height), (247, 249, 251))
    pane(canvas, marked, GAP, GAP, "SOURCE / cyan outline = target mask")
    pane(canvas, edited, PANE[0] + GAP * 2, GAP, "EDITED")
    y = PANE[1] + LABEL_HEIGHT + GAP * 2
    pane(canvas, marked.crop(box), GAP, y, "SOURCE / target context")
    pane(canvas, edited.crop(box), PANE[0] + GAP * 2, y, "EDITED / same context")
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=82, optimize=True)
    area_fraction = float(coco_mask.area(row["mask_rle"])) / (source.width * source.height)
    return buffer.getvalue(), area_fraction


def build(dataset_root: Path, selection_path: Path, doc_assets: Path | None) -> Path:
    dataset_root = dataset_root.resolve()
    selected = read_selection(selection_path)
    records = read_selected(dataset_root / "manifest.jsonl", selected)
    gallery = dataset_root / "gallery"
    assets = gallery / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    if doc_assets:
        doc_assets.mkdir(parents=True, exist_ok=True)
    cards = []
    totals = {task: 0 for task in ("remove", "add", "replace", "attribute")}
    for item in selected:
        row = records[item["case_id"]]
        task = row["task_type"]
        totals[task] += 1
        image_data, area_fraction = composite(row, dataset_root)
        filename = f"{row['case_id']}.jpg"
        (assets / filename).write_bytes(image_data)
        if doc_assets:
            (doc_assets / filename).write_bytes(image_data)
        embedded = base64.b64encode(image_data).decode("ascii")
        cards.append(f"""
<article class="case" id="{html.escape(row['case_id'])}">
  <div class="meta"><span class="badge {task}">{task}</span>
    <code>{html.escape(row['case_id'])}</code>
    <span>{html.escape(row['source_subset'].upper())} · mask {area_fraction:.2%} of image · {html.escape(item.get('note', ''))}</span></div>
  <p class="instruction">{html.escape(row['editing_instruction'])}</p>
  <img loading="lazy" src="data:image/jpeg;base64,{embedded}" alt="Source with target mask outline and edited result for {html.escape(row['case_id'])}">
  <details><summary>Model audit and provenance</summary>
    <p><strong>Audit:</strong> {html.escape(row['audit_reason'])}</p>
    <p><strong>Source:</strong> {html.escape(row['source_image'])}<br>
    <strong>Edited:</strong> {html.escape(row['edited_image'])}</p>
  </details>
</article>""")
    if any(count < 3 for count in totals.values()):
        raise ValueError(f"selection is not representative across four tasks: {totals}")
    summary = json.loads((dataset_root / "summary.json").read_text())
    headings = " ".join(f"<a href='#{task}'>{task} ({totals[task]})</a>" for task in totals)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAMTok derived edit: final model-pass gallery</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;margin:0;background:#eef2f6;color:#17212f}}
header{{padding:26px max(20px,calc((100vw - 1100px)/2));background:#12304b;color:white}}
header h1{{margin:0 0 8px}}header p{{margin:6px 0}}nav a{{color:#d6efff;margin-right:20px}}
main{{max-width:1100px;margin:auto;padding:16px}}h2{{margin:36px 0 12px;border-bottom:2px solid #cbd7e3}}
.case{{background:white;border:1px solid #d4dfe9;border-radius:12px;margin:16px 0;padding:16px;box-shadow:0 2px 8px #15273712}}
.meta{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:14px;color:#566477}}
.badge{{font-weight:700;text-transform:uppercase;border-radius:6px;padding:3px 8px;color:white}}
.remove{{background:#8a4151}}.add{{background:#23765b}}.replace{{background:#3868a1}}.attribute{{background:#7958a1}}
.instruction{{font-size:18px;font-weight:600;margin:12px 0}}.case img{{display:block;width:100%;height:auto;border:1px solid #d9e0e8;border-radius:5px}}
details{{margin-top:10px;color:#36475a}}summary{{cursor:pointer}}code{{font-size:13px}}
</style></head><body><header><h1>SAMTok 派生细粒度编辑：最终通过样例</h1>
<p>四类合并数据共 {summary['total_cases']:,} 条模型审核通过：remove {summary['task_type_counts']['remove']:,}，add {summary['task_type_counts']['add']:,}，replace {summary['task_type_counts']['replace']:,}，attribute {summary['task_type_counts']['attribute']:,}。</p>
<p>每条上排是全图，下排是同一位置的局部放大；青色轮廓仅标在原图上，指明训练 mask。图片已内嵌，单独打开此 HTML 也能显示。通过指模型审核，不代表全部经过人工复核。</p>
<nav>{headings}</nav></header><main>
"""
    for task in totals:
        document += f'<h2 id="{task}">{task.capitalize()}</h2>\n'
        for item, card in zip(selected, cards):
            if records[item["case_id"]]["task_type"] == task:
                document += card + "\n"
    document += "</main></body></html>\n"
    output = gallery / "index.html"
    output.write_text(document, encoding="utf-8")
    (gallery / "selection.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gallery": str(output), "counts": totals, "html_bytes": output.stat().st_size}, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--doc-assets", type=Path)
    args = parser.parse_args()
    build(args.dataset_root, args.selection_json, args.doc_assets)


if __name__ == "__main__":
    main()
