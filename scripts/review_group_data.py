#!/usr/bin/env python3
"""Create a visual human-review PDF and an explicit decision ledger.

The ledger is deliberately initialized as ``pending``. A group is publishable
only after every requested field has been filled and the reviewer signature is
recorded; automatic 27B decisions never masquerade as human confirmation.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _image(path: str, size=(620, 420)):
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _font(size=22):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _member_page(group: dict, member: dict, kind: str, index: int):
    """One readable page per member, including all five multi-target steps."""
    canvas = Image.new("RGB", (1600, 1320), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    gid = group.get("twin_id") or group.get("multi_id")
    draw.text((30, 20), f"{kind.upper()}  {gid} / {member.get('unit_id')} / member {index + 1}",
              font=_font(23), fill=(20, 20, 20))
    instruction = member.get("instruction", "")
    draw.multiline_text((30, 60), "\n".join(textwrap.wrap(instruction, 115)),
                        font=_font(), fill=(20, 20, 20), spacing=7)
    source = member.get("step_input_image") or group["source_image"]
    target = member.get("edited_image") or member.get("step_output_image")
    for x, path, label in [(30, source, "SOURCE / STEP INPUT"), (815, target, "RESULT")]:
        draw.text((x, 150), label, font=_font(), fill=(40, 40, 40))
        if path:
            canvas.paste(_image(path, (755, 500)), (x, 185))
    # Exact aligned crop, expanded for contact edges and nearby context.
    with Image.open(member["mask"]) as im:
        bbox = im.convert("L").getbbox()
        width, height = im.size
    if bbox is None:
        raise ValueError(f"empty review mask: {member['mask']}")
    margin = max(24, round(max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * .25))
    crop_box = (max(0,bbox[0]-margin), max(0,bbox[1]-margin),
                min(width,bbox[2]+margin), min(height,bbox[3]+margin))
    for x, path, label in [(30,source,"SOURCE CROP"), (555,target,"RESULT CROP"),
                            (1080,member["mask"],"MASK: WHITE = SELECTOR")]:
        draw.text((x, 730), label, font=_font(21), fill=(40, 40, 40))
        if path:
            with Image.open(path) as im:
                tile=im.convert("RGB").crop(crop_box)
                scale = min(490 / tile.width, 450 / tile.height)
                tile = tile.resize((max(1, round(tile.width*scale)), max(1, round(tile.height*scale))),
                                   Image.Resampling.NEAREST if path == member["mask"] else Image.Resampling.LANCZOS)
                canvas.paste(tile,(x+(490-tile.width)//2,775+(450-tile.height)//2))
    draw.text((30, 1270), "Inspect reference, requested change, other targets, edges and naturalness.",
              font=_font(20), fill=(60,60,60))
    return canvas


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--kind", choices=["twins", "multi"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.output_dir / "human_review.jsonl"
    existing = {r["group_id"]: r for r in
                (json.loads(line) for line in ledger.read_text().splitlines() if line.strip())} if ledger.exists() else {}
    with ledger.open("w", encoding="utf-8") as stream:
        for group in rows:
            gid = group.get("twin_id") or group.get("multi_id")
            stream.write(json.dumps(existing.get(gid, {"group_id": gid, "reviewer": "assistant", "status": "pending",
                                     "reference_correct": None, "target_done": None,
                                     "non_target_preserved": None, "natural": None,
                                     "decision": "pending", "notes": ""}), ensure_ascii=False) + "\n")
    pages = []
    page_index = []
    for group in rows:
        members = group.get("members", []) if args.kind == "twins" else group.get("units", [])
        if args.kind == "multi":
            by_id = {m["unit_id"]: m for m in members}
            members = [by_id[uid] for uid in group["generation_order"]]
        for index, member in enumerate(members):
            page = _member_page(group, member, args.kind, index)
            page_path = args.output_dir / f"page_{len(pages)+1:04d}.jpg"
            page.save(page_path, quality=92)
            pages.append(page)
            page_index.append({"page": len(pages), "group_id": group.get("twin_id") or group.get("multi_id"),
                               "unit_id": member["unit_id"], "image": str(page_path)})
    (args.output_dir / "page_index.json").write_text(json.dumps(page_index, indent=2))
    pdf = args.output_dir / "human_review.pdf"
    if pages:
        pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=120)
    (args.output_dir / "README.txt").write_text(
        "Review each page at readable zoom. Fill human_review.jsonl only after checking the clean source/result views. "
        "A pending or missing field blocks publication.\n", encoding="utf-8")
    print(json.dumps({"groups": len(rows), "pdf": str(pdf), "ledger": str(ledger)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
