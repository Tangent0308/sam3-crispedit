"""Make a portable-in-workspace gallery/manifest from explicitly verified runs."""

import argparse
import html
import json
import os
from pathlib import Path

from synthesis_pipeline.assemble_edit_validation import link
from synthesis_pipeline.evaluate_audit27 import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, action="append", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    args = p.parse_args()
    rows, cards, seen = [], [], set()
    for directory in ["sources", "edited"]:
        (args.out_root / directory).mkdir(parents=True, exist_ok=True)
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
            stem = Path(image).stem
            views = []
            for suffix in ["source", "edited"]:
                choices = [
                    root / prefix / f"inputs_overview/{stem}_{suffix}.png"
                    for prefix in ["audit", "thinking_low", "thinking_audit"]
                ]
                choices.append(root / "audit/inputs_crop" / f"{stem}_{suffix}.png")
                view = next(path for path in choices if path.exists())
                views.append(
                    f'<a href="{os.path.relpath(view,args.out_root)}"><img loading="lazy" src="{os.path.relpath(view,args.out_root)}"></a>'
                )
            e = lambda x: html.escape(str(x))
            cards.append(
                f'<article><h2>{e(image)} | {e(row["task_type"])}</h2><p><b>New instruction:</b> {e(row["editing_instruction"])}</p><p>Original: {e(revision["original_instruction"])}</p><div class="pair">'
                + "".join(views)
                + f'</div><p><b>Assistant verification:</b> {e(revision["reason"])}</p><p><a href="{os.path.relpath(root/"report/index.html",args.out_root)}">Complete model decisions, prompts and rejected alternatives</a> · <a href="sources/{row["source_image"]}">Original full image</a> · <a href="edited/{image}">Edited full image</a></p></article>'
            )
    manifest = args.out_root / "annotations.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError("Use a fresh release directory; manifest differs")
    manifest.write_text(content)
    (args.out_root / "index.html").write_text(
        '<meta charset="utf-8"><title>Verified fine-grained edits</title><style>body{font-family:sans-serif;margin:24px}.pair{display:flex}.pair a{width:50%}img{width:100%}article{border-top:3px solid #777;margin-top:36px}</style>'
        + f"<h1>{len(rows)} visually verified edit pairs with reconstructed instructions</h1><p>Individually curated across recorded experiments, not an automatic acceptance-rate claim. Original target masks are preserved in annotations.jsonl as COCO RLE.</p>"
        + "".join(cards)
    )
    print(
        json.dumps(
            {"verified_cases": len(rows), "gallery": str(args.out_root / "index.html")}
        )
    )


if __name__ == "__main__":
    main()
