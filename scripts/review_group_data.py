#!/usr/bin/env python3
"""Create a visual human-review PDF and an explicit decision ledger.

The ledger is deliberately initialized as ``pending``. A group is publishable
only after every requested field has been filled and the reviewer signature is
recorded; automatic 27B decisions never masquerade as human confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _image(path: str, size=(620, 420)):
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _page(group: dict, kind: str, page_size=(1300, 900)):
    canvas = Image.new("RGB", page_size, (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    gid = group.get("twin_id") or group.get("multi_id")
    draw.text((30, 22), f"{kind.upper()}  {gid}", fill=(20, 20, 20))
    draw.text((30, 50), f"source={group.get('source_id')} split={group.get('split')}", fill=(60, 60, 60))
    source = _image(group["source_image"], (390, 370))
    canvas.paste(source, (30, 90))
    if kind == "twins":
        members = group.get("members", [])
        for i, member in enumerate(members[:3]):
            x = 450 + (i % 2) * 420
            y = 90 + (i // 2) * 390
            path = member.get("edited_image") or member.get("mask")
            if path and Path(path).is_file():
                canvas.paste(_image(path, (390, 300)), (x, y))
            draw.text((x, y + 305), f"{member.get('unit_id')}  {member.get('ref')}", fill=(20, 20, 20))
            draw.text((x, y + 328), member.get("instruction", ""), fill=(40, 40, 40))
    else:
        for i, member in enumerate(group.get("units", [])[:4]):
            x = 450 + (i % 2) * 420
            y = 90 + (i // 2) * 390
            path = member.get("step_output_image") or member.get("mask")
            if path and Path(path).is_file():
                canvas.paste(_image(path, (390, 300)), (x, y))
            draw.text((x, y + 305), f"step {i+1} {member.get('unit_id')}  {member.get('ref')}", fill=(20, 20, 20))
            draw.text((x, y + 328), member.get("instruction", ""), fill=(40, 40, 40))
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
    with ledger.open("w", encoding="utf-8") as stream:
        for group in rows:
            gid = group.get("twin_id") or group.get("multi_id")
            stream.write(json.dumps({"group_id": gid, "reviewer": "assistant", "status": "pending",
                                     "reference_correct": None, "target_done": None,
                                     "non_target_preserved": None, "natural": None,
                                     "decision": "pending", "notes": ""}, ensure_ascii=False) + "\n")
    pages = [_page(group, args.kind) for group in rows]
    pdf = args.output_dir / "human_review.pdf"
    if pages:
        pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=120)
    (args.output_dir / "README.txt").write_text(
        "Review each page at readable zoom. Fill human_review.jsonl only after checking the clean source/result views. "
        "A pending or missing field blocks publication.\n", encoding="utf-8")
    print(json.dumps({"groups": len(rows), "pdf": str(pdf), "ledger": str(ledger)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
