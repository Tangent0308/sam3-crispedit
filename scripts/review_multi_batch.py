#!/usr/bin/env python3
"""Create a complete diagnostic PDF, including rejected multi-target proposals."""

from __future__ import annotations
import argparse
import json
import textwrap
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sa1b_pipeline.group_planning import group_views
from scripts.run_group_pipeline import _group_id
from scripts.preview_group_cases import page



def append_retention_evidence(canvas, group, font_path):
    """Show failed attempts separately; the main TAR is the committed image."""
    retention = group.get("retention")
    if not retention:
        return canvas
    decisions = retention["decisions"]
    dropped = [x for x in decisions if not x["accepted"]]
    width = canvas.width
    out = Image.new("RGB", (width, canvas.height + 170 + 650 * len(dropped)), "#f2f5fa")
    out.paste(canvas, (0, 0)); draw = ImageDraw.Draw(out)
    font = lambda size: ImageFont.truetype(font_path, size)
    y = canvas.height + 20
    draw.text((52,y), f"逐步保留：原计划 {retention['planned_n_targets']} 项 → 实际保留 {group['n_targets']} 项", font=font(32),fill="#182b43")
    draw.text((52,y+48), "保留：" + ", ".join(retention['kept_unit_ids']) + "；丢弃：" + ", ".join(retention['dropped_unit_ids']),font=font(28),fill="#33445b")
    draw.text((52,y+90), "下列失败尝试没有进入上面的最终 TAR；后续编辑从最近的合格图继续。",font=font(26),fill="#a73836")
    col=(width-140)//3
    for index, decision in enumerate(dropped):
        member=json.loads(Path(decision['checkpoint']).read_text())['member']
        y=canvas.height+170+index*650
        draw.text((52,y), f"丢弃 {member['unit_id']} / {member['operation']}",font=font(30),fill="#a73836")
        instruction=member['instruction']
        draw.multiline_text((52,y+42), "\n".join(textwrap.wrap(instruction,145)),font=font(24),fill="#33445b")
        with Image.open(member['mask']) as im:mask=im.convert('L')
        box=mask.getbbox();pad=max(32,round(max(box[2]-box[0],box[3]-box[1])*.35))
        box=(max(0,box[0]-pad),max(0,box[1]-pad),min(mask.width,box[2]+pad),min(mask.height,box[3]+pad))
        for j,(label,path) in enumerate([('SRC / LAST COMMITTED INPUT',member['step_input_image']),('MASK',member['mask']),('REJECTED ATTEMPT / NOT FINAL',member['step_output_image'])]):
            x=52+j*(col+15)
            draw.text((x,y+115),label,font=font(22),fill="#33445b")
            with Image.open(path) as raw:tile=ImageOps.contain(raw.convert('RGB').crop(box),(col,440))
            out.paste(tile,(x+(col-tile.width)//2,y+155+(440-tile.height)//2))
        reason='; '.join(decision['reasons'])
        draw.text((52,y+604),reason[:160],font=font(22),fill="#a73836")
    return out


def build(root: Path, font_path: str, audit_manifest: Path | None = None):
    chosen = [
        json.loads(l)
        for l in (root / "inputs/selected.jsonl").read_text().splitlines()
        if l.strip()
    ]
    # Keep deterministic failures: generated.jsonl intentionally contains
    # only groups eligible for semantic audit, not every rendered case.
    results = {}
    for path in sorted((root / "generated").glob("*/group.json")):
        group = json.loads(path.read_text())
        results[group["multi_id"]] = group
    audit_manifest = audit_manifest or root / "audited.jsonl"
    if audit_manifest.exists():
        for line in audit_manifest.read_text().splitlines():
            if line.strip():
                group = json.loads(line)
                results[group["multi_id"]] = group
    review_path = root / "visual_review.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    out = root / "inspection"
    out.mkdir(exist_ok=True)
    files = []
    index = []
    font = lambda n: ImageFont.truetype(font_path, n)
    for i, selection in enumerate(chosen, 1):
        gid = _group_id("multi", selection)
        review = reviews.get(gid, {})
        if gid in results:
            group = results[gid]
            group["preview_review"] = {
                "decision": review.get("decision", "pending"),
                "note_en": review.get("note_en", "Pending visual review"),
            }
            canvas = page(group, "multi", i, len(chosen), root)
            canvas = append_retention_evidence(canvas, group, font_path)
        else:
            rejection = root / "rejected" / f"{gid}.json"
            reason = (
                json.loads(rejection.read_text()).get("reason", "unknown")
                if rejection.exists()
                else "No complete generated group"
            )
            views, _, _ = group_views(
                selection["candidate"],
                selection["objects"],
                root / "inspection/views",
                locator_outline=True,
            )
            canvas = Image.new("RGB", (2400, 3000), "#f2f5fa")
            d = ImageDraw.Draw(canvas)
            d.text(
                (60, 40),
                f"{i:02d} / {len(chosen):02d}   {gid}",
                font=font(42),
                fill="#182b43",
            )
            d.text(
                (60, 110),
                "规划 / 几何检查拒绝：未生成目标图，不计为生图失败",
                font=font(37),
                fill="#a73836",
            )
            d.multiline_text(
                (60, 180),
                "\n".join(textwrap.wrap(reason, 100)),
                font=font(29),
                fill="#33445b",
                spacing=10,
            )
            source = Image.open(selection["candidate"]["image_path"]).convert("RGB")
            source = ImageOps.contain(source, (2260, 1000))
            canvas.paste(source, ((2400 - source.width) // 2, 350))
            sheet = Image.open(views[1]).convert("RGB")
            sheet = ImageOps.contain(sheet, (2260, 1500))
            canvas.paste(sheet, ((2400 - sheet.width) // 2, 1400))
        # A separate footer retains complete Chinese findings without truncation.
        note = review.get("note_zh", "尚无完整视觉复核记录。")
        lines = []
        for line in note.split("\n"):
            lines.extend(textwrap.wrap(line, 70) or [""])
        footer = 200 + len(lines) * 40
        full = Image.new("RGB", (canvas.width, canvas.height + footer), "#f2f5fa")
        full.paste(canvas, (0, 0))
        d = ImageDraw.Draw(full)
        d.text(
            (52, canvas.height + 20),
            "视觉复核：" + review.get("decision", "pending"),
            font=font(30),
            fill="#182b43",
        )
        d.multiline_text(
            (52, canvas.height + 70),
            "\n".join(lines),
            font=font(28),
            fill="#33445b",
            spacing=10,
        )
        path = out / f"case_{i:02d}.jpg"
        full.save(path, quality=94, subsampling=0)
        files.append(path)
        index.append(
            {
                "case": i,
                "group_id": gid,
                "generated": gid in results,
                "audit_manifest": str(audit_manifest),
                "automatic_pass": results.get(gid, {})
                .get("joint_audit", {})
                .get("pass", False),
                "review": review,
                "image": str(path),
            }
        )
    images = [Image.open(p).convert("RGB") for p in files]
    pdf = out / "multi_batch_src_mask_tar.pdf"
    images[0].save(
        pdf,
        save_all=True,
        append_images=images[1:],
        resolution=160,
        quality=94,
        subsampling=0,
    )
    for image in images:
        image.close()
    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    )
    print(pdf)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--font", default="/tmp/NotoSansCJKsc-Regular.otf")
    p.add_argument("--audit-manifest", type=Path, help="Optional frozen-image audit rerun; keep original results intact")
    args = p.parse_args()
    build(args.root, args.font, args.audit_manifest)
