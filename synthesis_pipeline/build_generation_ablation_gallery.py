"""Build aligned generation comparisons, without showing VLM decisions."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--variants", default="mirage_relaxed,official_full,context_edit")
    p.add_argument("--gallery-name", default="generation_gallery")
    args = p.parse_args()
    variants = args.variants.split(",")
    rows = [
        json.loads(x)
        for x in (args.data_root / "annotations.jsonl").read_text().splitlines()
        if x.strip()
    ]
    review_path = args.experiment_root / "generation_review.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    labels_by_variant = {v: [] for v in variants}
    completed = []
    out = args.experiment_root / args.gallery_name
    (out / "cases").mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        paths = [args.data_root / "edited" / row["image"]] + [
            args.experiment_root / v / "edited" / row["image"] for v in variants
        ]
        if not all(x.exists() for x in paths):
            continue
        completed.append(row)
        source = Image.open(args.data_root / "sources" / row["source_image"]).convert(
            "RGB"
        )
        mask = mask_array(source.size, row["mask"])
        crops = []
        fulls = []
        for path in paths:
            edited = Image.open(path).convert("RGB")
            before, after = audit_two_image_inputs(
                source, edited, mask, longest_side=1280, scope="context_crop"
            )
            full_before, full_after = audit_two_image_inputs(
                source, edited, mask, longest_side=max(source.size), scope="full"
            )
            if not crops:
                crops.append(before)
                fulls.append(full_before)
            crops.append(after)
            fulls.append(full_after)
        labels = ["Source + mask", "Baseline MIRAGE"] + variants
        canvas = Image.new("RGB", (480 * len(crops), 800), "white")
        draw = ImageDraw.Draw(canvas)
        for j, (crop, full, label) in enumerate(zip(crops, fulls, labels)):
            draw.text((j * 480 + 8, 4), label, fill="black")
            for y, picture in ((24, full), (414, crop)):
                tile = ImageOps.contain(picture, (474, 378))
                canvas.paste(
                    tile,
                    (j * 480 + (480 - tile.width) // 2, y + (378 - tile.height) // 2),
                )
        name = Path(row["image"]).stem + ".jpg"
        canvas.save(out / "cases" / name, quality=95)
        reasons = []
        for variant in variants:
            review = reviews.get(variant, {}).get(str(int(row["image"].split("_")[0])))
            if review:
                visual, match, reason = review
                labels_by_variant[variant].append(
                    dict(
                        image=row["image"],
                        task_type=row["task_type"],
                        quality="pass" if visual == match == "pass" else "fail",
                        visual_quality=visual,
                        instruction_match=match,
                        reason=reason,
                        reviewed_by=reviews.get("reviewer"),
                    )
                )
                reasons.append(
                    f"<p><b>{variant}</b> — visual: {visual}; instruction: {match}. {html.escape(reason)}</p>"
                )
        cards.append(
            f'<article><h3>{html.escape(row["image"])} | {row["task_type"]}</h3><p>{html.escape(row["editing_instruction"])}</p><a href="cases/{name}"><img src="cases/{name}"></a>'
            + "".join(reasons)
            + "</article>"
        )
    (out / "index.html").write_text(
        '<meta charset="utf-8"><title>Generation ablations</title><style>body{font-family:sans-serif;margin:24px}img{width:100%}article{margin-bottom:45px}</style><h1>Same input, instruction, seed and steps; full view above / context below</h1>'
        + "".join(cards)
    )
    summary = {}
    for variant, labels in labels_by_variant.items():
        if labels:
            (args.experiment_root / variant / "manual_review.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in labels)
            )
            summary[variant] = {
                "reviewed": len(labels),
                **{
                    f: sum(r[f] == "pass" for r in labels)
                    for f in ["quality", "visual_quality", "instruction_match"]
                },
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "annotations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in completed)
    )
    print(json.dumps({"cases": len(cards), "gallery": str(out / "index.html")}))


if __name__ == "__main__":
    main()
