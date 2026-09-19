"""Join frozen generation reviews, model decisions and independently reviewed labels."""

import argparse
import html
import json
import os
from pathlib import Path

from synthesis_pipeline.evaluate_audit27 import confusion, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--validation-name", default="validation_v2")
    args = parser.parse_args()
    root = args.experiment_root
    data = root / args.validation_name
    rows = read_jsonl(data / "annotations.jsonl")
    reviews = {}
    review_sources = [
        (root / "generation_review.json", "context_edit"),
        (root / "holdout/generation_review.json", "context_edit"),
    ]
    if args.validation_name == "validation_v2":
        review_sources.append(
            (root / "remove_iteration/generation_review.json", "context_removewide_v2")
        )
    elif args.validation_name == "validation_adaptive":
        review_sources.append(
            (
                root / "guided_iteration/generation_review.json",
                "context_guided_attribute",
            )
        )
    for path, variant in review_sources:
        reviews.update(json.loads(path.read_text())[variant])
    manual = {}
    for image, row in rows.items():
        visual, match, reason = reviews[str(int(image.split("_")[0]))]
        manual[image] = dict(
            image=image,
            task_type=row["task_type"],
            visual_quality=visual,
            instruction_match=match,
            quality="pass" if visual == match == "pass" else "fail",
            reason=reason,
            reviewed_by="assistant_direct_visual_review",
        )
    (data / "manual_review.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manual.values())
    )
    audits = {
        str(p.relative_to(data)): read_jsonl(p)
        for p in data.glob("*/**/edit_audit.jsonl")
    }
    rewrites = {
        str(p.relative_to(data)): read_jsonl(p)
        for p in data.glob("*/**/rewrites.jsonl")
    }
    metrics = {
        name: {
            field: confusion(manual, audit, field)
            for field in ["quality", "visual_quality", "instruction_match"]
        }
        for name, audit in audits.items()
    }
    (data / "audit_comparison.json").write_text(json.dumps(metrics, indent=2))
    review_file = data / "rewrite_manual_review.json"
    instruction_reviews = (
        json.loads(review_file.read_text()) if review_file.exists() else {}
    )
    out = data / "report"
    out.mkdir(exist_ok=True)
    cards = []
    verified_cards = []
    verified = []
    escape = lambda x: html.escape(str(x))
    for image, row in rows.items():
        stem = Path(image).stem
        images = [
            data / "audit/inputs_overview" / f"{stem}_{suffix}.png"
            for suffix in ["source", "edited"]
        ]
        if not all(p.exists() for p in images):
            images = [
                data / "audit/inputs_crop" / f"{stem}_{suffix}.png"
                for suffix in ["source", "edited"]
            ]
        pictures = "".join(
            f'<a href="{os.path.relpath(p, out)}"><img src="{os.path.relpath(p, out)}" loading="lazy"></a>'
            for p in images
        )
        decisions = []
        for name, audit in audits.items():
            result = audit.get(image)
            if result:
                decisions.append(
                    f'<h4>{escape(name)}</h4><p>{escape(result.get("audit"))}</p><details><summary>Complete response / prompt</summary><pre>{escape(result.get("raw_response"))}</pre><pre>{escape(result.get("prompt"))}</pre></details>'
                )
        for name, rewrite in rewrites.items():
            result = rewrite.get(image)
            if result:
                decisions.append(
                    f'<h4>{escape(name)}</h4><p>{escape(result.get("task_type"))}: {escape(result.get("editing_instruction"))}</p><details><summary>Complete response / prompt</summary><pre>{escape(result.get("raw_response"))}</pre><pre>{escape(result.get("prompt"))}</pre></details>'
                )
        reviewed = instruction_reviews.get(str(int(image.split("_")[0])))
        if reviewed:
            decisions.append(
                f"<h4>Assistant instruction review</h4><p>{escape(reviewed)}</p>"
            )
            if reviewed.get("accepted") and manual[image]["visual_quality"] == "pass":
                final = dict(row)
                final["task_type"] = reviewed["task_type"]
                final["editing_instruction"] = reviewed["instruction"]
                final["instruction_revision"] = dict(
                    original_instruction=row["editing_instruction"],
                    original_task_type=row["task_type"],
                    verification="assistant_visually_verified",
                    instruction_origin=reviewed.get("instruction_origin"),
                    reason=reviewed["reason"],
                )
                verified.append(final)
        m = manual[image]
        cards.append(
            f'<article><h2>{escape(image)}</h2><p>Original: {escape(row["editing_instruction"])}</p><div class="pair">{pictures}</div><h4>Assistant generation review</h4><p>visual={m["visual_quality"]}; instruction={m["instruction_match"]}. {escape(m["reason"])}</p>'
            + "".join(decisions)
            + "</article>"
        )
        if (
            reviewed
            and reviewed.get("accepted")
            and manual[image]["visual_quality"] == "pass"
        ):
            verified_cards.append(cards[-1])
    (out / "index.html").write_text(
        '<meta charset="utf-8"><title>New-pair validation</title><style>body{font-family:sans-serif;margin:24px}.pair{display:flex}.pair a{width:50%}img{width:100%}pre{white-space:pre-wrap}article{border-top:3px solid #777;margin-top:36px}</style><h1>New generation: exact inputs, complete decisions and instruction review</h1>'
        + "".join(cards)
    )
    (data / "assistant_verified_annotations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in verified)
    )
    (out / "verified.html").write_text(
        '<meta charset="utf-8"><title>Assistant-verified edit pairs</title>'
        "<style>body{font-family:sans-serif;margin:24px}.pair{display:flex}.pair a{width:50%}img{width:100%}pre{white-space:pre-wrap}article{border-top:3px solid #777;margin-top:36px}</style>"
        f"<h1>{len(verified)} assistant-verified pairs</h1>"
        "<p>Model-generated instructions selected after direct visual review. This is a curated release, not an automatic pipeline acceptance-rate claim.</p>"
        + "".join(verified_cards)
    )
    print(
        json.dumps(
            {
                "cases": len(rows),
                "verified": len(verified),
                "report": str(out / "index.html"),
                "metrics": metrics,
            }
        )
    )


if __name__ == "__main__":
    main()
