"""Keep every audit/rewrite reason and exact model input accessible in HTML."""

import argparse
import html
import json
import os
from pathlib import Path

from synthesis_pipeline.evaluate_audit27 import confusion, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--experiment-root", type=Path, required=True)
    args = p.parse_args()
    root = args.experiment_root
    out = root / "audit_report"
    out.mkdir(exist_ok=True)
    manual = read_jsonl(args.data_root / "manual_review.jsonl")
    rows = read_jsonl(args.data_root / "annotations.jsonl")
    audit_paths = {
        "legacy_crop": args.data_root / "audit27_legacy_all100/edit_audit.jsonl",
        **{
            x.parent.name: x
            for x in (root / "audit_ablation").glob("*/edit_audit.jsonl")
        },
        **{
            x.parent.name: x
            for x in (root / "audit_grounded").glob("*/edit_audit.jsonl")
        },
        **{
            "thinking_" + x.parent.name: x
            for x in (root / "thinking_audit").glob("*/edit_audit.jsonl")
        },
        **{
            "thinking_" + x.parent.name: x
            for x in (root / "thinking_quality").glob("*/edit_audit.jsonl")
        },
        **{
            "thinking_low_" + x.parent.name: x
            for x in (root / "thinking_low_audit").glob("*/edit_audit.jsonl")
        },
    }
    audits = {k: read_jsonl(v) for k, v in audit_paths.items() if v.exists()}
    rewrite_paths = {
        "rewrite_full_v1": root / "relabel_baseline/rewrite_full/rewrites.jsonl",
        "rewrite_crop_v2": root / "relabel_v2/rewrite_crop/rewrites.jsonl",
        "rewrite_overview_v3": root / "relabel_v3/rewrite_overview/rewrites.jsonl",
        "rewrite_overview_thinking": root
        / "thinking_rewrite/rewrite_overview/rewrites.jsonl",
    }
    rewrites = {k: read_jsonl(v) for k, v in rewrite_paths.items() if v.exists()}
    review_path = root / "rewrite_review.json"
    rewrite_reviews = (
        json.loads(review_path.read_text()) if review_path.exists() else {}
    )
    scores = {
        name: {
            field: confusion(manual, audit, field)
            for field in ["quality", "visual_quality", "instruction_match"]
        }
        for name, audit in audits.items()
        if len(audit) == len(manual)
    }
    scores["incomplete_experiments"] = {
        name: {
            "attempted": len(audit),
            "requested": len(manual),
            "status": "partial_diagnostic_not_a_full_benchmark",
        }
        for name, audit in audits.items()
        if len(audit) != len(manual)
    }
    for split in ["dev", "holdout"]:
        subset_path = args.data_root / f"evaluation/{split}_annotations.jsonl"
        if subset_path.exists():
            subset = {k: manual[k] for k in read_jsonl(subset_path)}
            scores[f"{split}_visual_quality"] = {
                name: confusion(subset, audit, "visual_quality")
                for name, audit in audits.items()
                if len(audit) == len(manual)
            }
    (out / "metrics.json").write_text(json.dumps(scores, indent=2))
    escape = lambda value: html.escape(str(value))
    cards = []
    for image, row in rows.items():
        stem = Path(image).stem
        views = []
        for scope in ["crop", "full"]:
            pictures = [
                root / "audit_ablation" / f"inputs_{scope}" / f"{stem}_{suffix}.png"
                for suffix in ["source", "edited"]
            ]
            if all(x.exists() for x in pictures):
                views.append(
                    f'<details {"open" if scope=="crop" else ""}><summary>{scope}: exact two-image input</summary><div class="pair">'
                    + "".join(
                        f'<a href="{os.path.relpath(x,out)}"><img loading="lazy" src="{os.path.relpath(x,out)}"></a>'
                        for x in pictures
                    )
                    + "</div></details>"
                )
        decisions = []
        for name, audit in audits.items():
            result = audit.get(image)
            if not result:
                continue
            val = result.get("audit") or {}
            decisions.append(
                f'<section><h4>{name}: overall={escape(result.get("quality"))}; visual={escape(val.get("visual_quality"))}; instruction={escape(val.get("instruction_match"))}</h4><p>{escape(val.get("reason"))}</p><details><summary>Complete response and prompt</summary><pre>{escape(result.get("raw_response"))}</pre><pre>{escape(result.get("prompt","See saved baseline audit file"))}</pre></details></section>'
            )
        for name, rewrite in rewrites.items():
            result = rewrite.get(image)
            if result:
                review = rewrite_reviews.get(name, {}).get(
                    str(int(image.split("_")[0]))
                )
                decisions.append(
                    f'<section><h4>{name}: {escape(result.get("status"))} / {escape(result.get("task_type"))}</h4><p>{escape(result.get("editing_instruction"))}</p><p>Assistant rewrite review: {escape(review)}</p><details><summary>Complete response and prompt</summary><pre>{escape(result.get("raw_response"))}</pre><pre>{escape(result.get("prompt"))}</pre></details></section>'
                )
        expected = manual[image]
        cards.append(
            f'<article id="{stem}"><h2>{escape(image)}</h2><p>Original instruction: {escape(row["editing_instruction"])}</p>'
            + "".join(views)
            + f'<p><b>Frozen original assistant review:</b> visual={expected["visual_quality"]}, instruction={expected["instruction_match"]}. {escape(expected["reason"])}</p>'
            + "".join(decisions)
            + "</article>"
        )
    (out / "index.html").write_text(
        '<meta charset="utf-8"><title>Audit and rewrite iteration</title><style>body{font-family:sans-serif;margin:24px}.pair{display:flex}.pair a{width:50%}img{width:100%}pre{white-space:pre-wrap}article{border-top:3px solid #777;margin-top:36px}section{padding:6px 14px;background:#f3f3f3;margin:10px 0}</style><h1>100 pairs: exact inputs, all decisions, reasons, prompts and rewrites</h1><p>Original assistant labels remain frozen for comparison; disagreements and revised-instruction reviews are shown separately.</p>'
        + "".join(cards)
    )
    print(out / "index.html")


if __name__ == "__main__":
    main()
