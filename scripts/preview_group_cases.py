#!/usr/bin/env python3
"""Render complete grouped cases as readable source/mask/target PDF pages."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


PALETTE = ["#e34a65", "#159ac8", "#e6a323", "#9170dd", "#19a882"]


def font(size, bold=False):
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size)


def resolve(path, base):
    p = Path(path)
    return p if p.is_absolute() else base / p


def fit(image, size, mask=False):
    tile = Image.new("RGB", size, "#edf0f5")
    im = ImageOps.contain(image.convert("RGB"), size,
                         Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    tile.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return tile


def target(member):
    return member.get("edited_image") or member.get("step_output_image")


def status(group, kind):
    checks = group.get("group_checks", {}) if kind == "twins" else group.get("joint_audit", {})
    if checks.get("group_pass") is True or checks.get("pass") is True:
        return "AUTOMATIC GROUP AUDIT: PASS | EXPERIMENTAL, UNPUBLISHED", "#16754a"
    if kind == "multi" and (checks.get("blind_protocol", {}).get("pass") is False
                            or any(x.get("decision") == "fail" for x in checks.get("steps", []))):
        return "DETERMINISTIC GROUP CHECK: FAIL | DIAGNOSTIC, UNPUBLISHED", "#a63838"
    if any(checks.get(k, {}).get("pass") is False for k in
           ("deterministic", "support_gap", "cross_target_preservation")):
        return "DETERMINISTIC GROUP CHECK: FAIL | DIAGNOSTIC, UNPUBLISHED", "#a63838"
    semantic = checks.get("semantic_audit") or {}
    if semantic.get("decision") == "pending":
        return "AUTOMATIC GROUP AUDIT: PENDING | REVISED MASK, UNPUBLISHED", "#866319"
    if (checks.get("semantic_audit_pass") is False or semantic.get("decision") == "fail"
            or checks.get("member_semantic_audit_pass") is False):
        return "AUTOMATIC GROUP AUDIT: FAIL | DIAGNOSTIC, UNPUBLISHED", "#a63838"
    return "GENERATED PREVIEW | NOT RELEASED / AUDIT NOT PASSED", "#866319"


def overlay(source, members, base):
    im = source.convert("RGB")
    draw = ImageDraw.Draw(im)
    scale = max(source.size) / 1800
    for i, member in enumerate(members):
        with Image.open(resolve(member["mask"], base)) as raw:
            mask = raw.convert("L")
        if mask.size != source.size:
            raise ValueError("source/mask sizes differ")
        bbox = mask.getbbox()
        if not bbox:
            raise ValueError("empty selector mask")
        color = PALETTE[i % len(PALETTE)]
        im = Image.composite(Image.new("RGB", im.size, color), im,
                             mask.point(lambda v: round(v * .27)))
        edge = mask.filter(ImageFilter.MaxFilter(5))
        from PIL import ImageChops
        edge = ImageChops.subtract(edge, mask)
        im.paste(Image.new("RGB", im.size, color), (0, 0), edge)
        draw = ImageDraw.Draw(im)
        x, y = bbox[0], max(0, bbox[1] - int(34 * scale))
        label = member["unit_id"]
        f = font(max(18, round(27 * scale)), True)
        box = draw.textbbox((x, y), label, font=f)
        draw.rectangle((x, y, box[2] + 10, box[3] + 5), fill=color)
        draw.text((x + 4, y), label, font=f, fill="white")
    return im


def page(group, kind, number, total, base):
    members = group["members"] if kind == "twins" else group["units"]
    if kind == "multi":
        by_id = {m["unit_id"]: m for m in members}
        members = [by_id[uid] for uid in group["generation_order"]]
    empty_committed = kind == "multi" and group.get("commit_policy") in {"keep-passed-v1", "parallel-components-v1"} and not members
    if (not members and not empty_committed) or any(not target(m) for m in members):
        raise ValueError("preview requires every generated target in the group")
    width, gap, margin = 3300, 24, 52
    height = 1020 + 650 * len(members)
    canvas = Image.new("RGB", (width, height), "#f3f5f9")
    draw = ImageDraw.Draw(canvas)
    gid = group.get("twin_id") or group.get("multi_id")
    heading = "COUNTERFACTUAL TWINS" if kind == "twins" else ("MULTI-TARGET PARALLEL FUSION" if group.get("parallel") else "MULTI-TARGET CHAIN")
    draw.text((margin, 32), f"{heading}"
              f"  |  CASE {number:02d}/{total:02d}", font=font(42, True), fill="#162335")
    draw.text((margin, 94), f"{gid}   |   {len(members)} targets", font=font(26), fill="#5a6779")
    label, color = status(group, kind)
    draw.text((margin, 142), label, font=font(25, True), fill=color)
    model = group.get("planning", {}).get("model", "unknown")
    draw.text((margin, 187), "Planner: " + Path(model).name + "  |  Real generation; no manual retouching",
              font=font(23), fill="#5a6779")
    if kind == "twins":
        text = "Shared no-ref: " + group.get("instruction_noref", "")
    elif group.get("parallel"):
        text = "Parallel from original: " + " | ".join(" -> ".join(c) for c in group["parallel"]["components"]) + " ; fuse accepted regions"
    else:
        text = "Chain: original -> " + " -> ".join(group["generation_order"]) + " (each step uses previous output)"
    draw.multiline_text((margin, 227), "\n".join(textwrap.wrap(text, 163)),
                        font=font(25), fill="#26394e", spacing=6)
    review = group.get("preview_review", {})
    note = review.get("note_en") or "No complete manual acceptance recorded for this preview."
    decision = str(review.get("decision", "pending")).upper()
    draw.text((margin, 282), f"Visual review: {decision} | {note[:185]}", font=font(21),
              fill="#a63838" if decision == "FAIL" else "#5a6779")
    with Image.open(resolve(group["source_image"], base)) as raw:
        source = raw.convert("RGB")
    # Full-scene views retain the context needed to distinguish similar targets.
    panels = [("SRC / ORIGINAL", source), ("MASK OVERLAY / SOURCE SELECTORS", overlay(source, members, base))]
    if kind == "twins":
        for m in members:
            panels.append((f"TAR {m['unit_id']} / INDEPENDENT EDIT", Image.open(resolve(target(m), base)).convert("RGB")))
    else:
        panels.append(("TAR / FINAL JOINT RESULT", Image.open(resolve(group.get("joint_target") or target(members[-1]), base)).convert("RGB")))
    tile_width = (width - 2 * margin - gap * (len(panels) - 1)) // len(panels)
    for i, (title, im) in enumerate(panels):
        x = margin + i * (tile_width + gap)
        draw.text((x, 319), title, font=font(21, True), fill="#26394e")
        canvas.paste(fit(im, (tile_width, 536)), (x, 359))
    col_width = (width - 2 * margin - 2 * gap) // 3
    for i, m in enumerate(members):
        y = 924 + i * 650
        color = PALETTE[i % len(PALETTE)]
        draw.rounded_rectangle((margin, y, width - margin, y + 618), radius=18, fill="white")
        draw.rectangle((margin, y, margin + 10, y + 618), fill=color)
        instruction = m.get("instruction", "")
        draw.text((margin + 22, y + 16), f"{m['unit_id']}  |  {m.get('operation', group.get('operation', ''))}",
                  font=font(27, True), fill=color)
        draw.multiline_text((margin + 365, y + 17), "\n".join(textwrap.wrap(instruction, 155)),
                            font=font(24), fill="#26394e", spacing=5)
        if kind == "multi" and m.get("instruction_noref"):
            ref_lines = len(textwrap.wrap(instruction, 155))
            draw.text((margin + 365, y + 17 + 29 * ref_lines + 8),
                      "No-ref: " + m["instruction_noref"], font=font(20), fill="#5a6779")
        input_path = m.get("step_input_image") if kind == "multi" else group["source_image"]
        with Image.open(resolve(m["mask"], base)) as raw:
            mask = raw.convert("L")
        bbox = mask.getbbox()
        if not bbox:
            raise ValueError("empty mask")
        pad = max(32, round(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * .35))
        crop = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(mask.width, bbox[2] + pad), min(mask.height, bbox[3] + pad))
        tiles = [("SRC CROP" if kind == "twins" else "SRC / THIS STEP INPUT CROP", input_path),
                 ("MASK / WHITE = SOURCE SELECTOR", m["mask"]),
                 ("TAR CROP" if kind == "twins" else "TAR / THIS STEP OUTPUT CROP", target(m))]
        for j, (title, path) in enumerate(tiles):
            x = margin + j * (col_width + gap)
            draw.text((x + 20, y + 106), title, font=font(21, True), fill="#5a6779")
            with Image.open(resolve(path, base)) as raw:
                if raw.size != mask.size:
                    raise ValueError(f"image/mask size mismatch: {path}")
                im = raw.convert("RGB").crop(crop)
            canvas.paste(fit(im, (col_width - 28, 451), mask=j == 1), (x + 14, y + 148))
    draw.text((margin, height - 51), "Same crop coordinates within each row. Mask = input selector; it is not the output-difference mask.",
              font=font(21), fill="#5a6779")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=["twins", "multi"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if len(rows) != args.count:
        raise ValueError(f"expected {args.count} complete groups, received {len(rows)}")
    ids = [g.get("twin_id") or g.get("multi_id") for g in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate groups in preview")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i, group in enumerate(rows, 1):
        canvas = page(group, args.kind, i, len(rows), args.base_root or args.manifest.parent)
        jpg = args.output_dir / f"case_{i:02d}.jpg"
        canvas.save(jpg, quality=96, subsampling=0)
        index.append({"case": i, "group_id": ids[i - 1], "page": str(jpg), "size": list(canvas.size),
                      "status": status(group, args.kind)[0]})
    pages = [Image.open(item["page"]).convert("RGB") for item in index]
    pdf = args.output_dir / f"{args.kind}_{args.count}cases_src_mask_tar.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=160,
                  quality=95, subsampling=0, title=f"{args.kind}: {args.count} real source/mask/target cases")
    for im in pages:
        im.close()
    (args.output_dir / "page_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(json.dumps({"cases": len(rows), "pdf": str(pdf), "bytes": pdf.stat().st_size}))


if __name__ == "__main__":
    main()
