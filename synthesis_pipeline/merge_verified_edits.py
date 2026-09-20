"""Make a portable-in-workspace gallery/manifest from explicitly verified runs."""

import argparse
import base64
import html
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from synthesis_pipeline.assemble_edit_validation import link
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.evaluate_audit27 import read_jsonl
from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs


def comparison_sheet(source_path: Path, edited_path: Path, mask_data) -> Image.Image:
    """A local, aligned full-scene and target-crop comparison for one case."""
    source = Image.open(source_path).convert("RGB")
    edited = Image.open(edited_path).convert("RGB")
    if source.size != edited.size:
        raise ValueError(f"Source/edit sizes differ: {source_path}, {edited_path}")
    mask = mask_array(source.size, mask_data)
    full = audit_two_image_inputs(source, edited, mask, 1100, "full")
    crop = audit_two_image_inputs(source, edited, mask, 1100, "context_crop")
    tile_width, tile_height, gap = 620, 580, 8
    sheet = Image.new("RGB", (2 * tile_width + gap, 2 * tile_height + gap), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, pair in enumerate((full, crop)):
        for column_index, picture in enumerate(pair):
            tile = ImageOps.contain(picture, (tile_width - 8, tile_height - 8))
            x = column_index * (tile_width + gap) + (tile_width - tile.width) // 2
            y = row_index * (tile_height + gap) + (tile_height - tile.height) // 2
            sheet.paste(tile, (x, y))
    draw.line(
        (0, tile_height + 3, sheet.width, tile_height + 3), fill="#999999", width=2
    )
    return sheet


def jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue()


def preview_data_uri(image: Image.Image) -> str:
    """Keep the standalone HTML small; the linked JPG retains full detail."""
    thumbnail = ImageOps.contain(image, (880, 880))
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, action="append", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    args = p.parse_args()
    rows, cards, seen = [], [], set()
    for directory in ["sources", "edited"]:
        (args.out_root / directory).mkdir(parents=True, exist_ok=True)
    case_dir = args.out_root / "cases"
    case_dir.mkdir(exist_ok=True)
    for root in args.input_root:
        for image, row in read_jsonl(
            root / "assistant_verified_annotations.jsonl"
        ).items():
            if image in seen:
                raise ValueError(f"Duplicate region in merged release: {image}")
            seen.add(image)
            revision = row["instruction_revision"]
            if revision["verification"] != "assistant_visually_verified":
                raise ValueError(f"Unverified input: {image}")
            link(
                args.out_root / "sources" / row["source_image"],
                root / "sources" / row["source_image"],
            )
            link(args.out_root / "edited" / image, root / "edited" / image)
            result = dict(row, reviewed_source_root=str(root.resolve()))
            rows.append(result)
            sheet = comparison_sheet(
                args.out_root / "sources" / row["source_image"],
                args.out_root / "edited" / image,
                row["mask"],
            )
            jpeg = jpeg_bytes(sheet)
            case_name = Path(image).stem + ".jpg"
            (case_dir / case_name).write_bytes(jpeg)
            embedded = preview_data_uri(sheet)
            e = lambda value: html.escape(str(value), quote=True)
            cards.append(
                f'<article data-type="{e(row["task_type"])}" id="case-{e(Path(image).stem)}">'
                f'<h2>{e(image)} <span>{e(row["task_type"])}</span></h2>'
                f'<p class="instruction">{e(row["editing_instruction"])}</p>'
                f'<a href="cases/{e(case_name)}" download><img loading="lazy" '
                f'alt="Full scene and target crop, original on left and edited on right" '
                f'src="{embedded}"></a>'
                f'<p><a href="cases/{e(case_name)}" download>Download this comparison as JPG</a></p>'
                f"<details><summary>Original instruction and visual review</summary>"
                f'<p>Original: {e(revision["original_instruction"])}</p>'
                f'<p>Visual review: {e(revision["reason"])}</p>'
                f"<p>Source experiment: {e(root.name)}</p></details></article>"
            )
    manifest = args.out_root / "annotations.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError("Use a fresh release directory; manifest differs")
    manifest.write_text(content)
    counts = Counter(row["task_type"] for row in rows)
    filters = " ".join(
        f'<button type="button" onclick="showType(\'{e(kind)}\')">{e(kind)} ({counts[kind]})</button>'
        for kind in ("add", "remove", "replace", "attribute")
    )
    (args.out_root / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Verified fine-grained edits</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1160px;margin:24px auto;padding:0 16px;background:#f6f7f8;color:#222}"
        "nav{position:sticky;top:0;background:#f6f7f8;padding:12px 0;z-index:2}button{margin:2px;padding:8px 12px;cursor:pointer}"
        "article{background:white;border:1px solid #ddd;border-radius:10px;padding:16px;margin:24px 0}"
        "article img{display:block;width:100%;height:auto}h2{font-size:1.1em;overflow-wrap:anywhere}h2 span{font-size:.8em;color:#555}"
        ".instruction{font-size:1.1em}summary{cursor:pointer}</style>"
        f"<h1>{len(rows)} visually reviewed edit pairs</h1>"
        "<p>Top: full scene. Bottom: magnified target area. Original with a black/white mask outline is on the left; edited result is on the right. Each comparison is embedded in this HTML and also saved as a standalone JPG.</p>"
        "<p>These cases were individually selected across experiments. The original target masks are preserved as COCO RLE in annotations.jsonl; they are not exact change masks.</p>"
        f'<nav><button type="button" onclick="showType(\'all\')">All ({len(rows)})</button>{filters}</nav>'
        + "".join(cards)
        + '<script>function showType(kind){for(const card of document.querySelectorAll("article[data-type]")){card.hidden=kind!=="all"&&card.dataset.type!==kind;}}</script></html>',
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"verified_cases": len(rows), "gallery": str(args.out_root / "index.html")}
        )
    )


if __name__ == "__main__":
    main()
