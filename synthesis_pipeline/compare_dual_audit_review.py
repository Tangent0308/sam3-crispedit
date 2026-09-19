"""Compare two VLM audits against an independently reviewed edit-pair set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {
        row["image"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def confusion(manual: dict[str, dict], predicted: dict[str, dict], field: str) -> dict:
    counts = Counter()
    for image, human in manual.items():
        expected = human.get(field)
        audit = predicted[image]
        observed = audit.get(field) if field == "quality" else (audit.get("audit") or {}).get(field)
        if expected not in {"pass", "fail"} or observed not in {"pass", "fail"}:
            counts["unscored"] += 1
        elif expected == observed == "pass":
            counts["true_pass"] += 1
        elif expected == observed == "fail":
            counts["true_fail"] += 1
        elif observed == "pass":
            counts["false_accept"] += 1
        else:
            counts["false_reject"] += 1
    evaluated = sum(counts[key] for key in (
        "true_pass", "true_fail", "false_accept", "false_reject"
    ))
    return {
        **dict(counts),
        "evaluated": evaluated,
        "agreement_fraction": round(
            (counts["true_pass"] + counts["true_fail"]) / evaluated, 4
        ) if evaluated else None,
    }


def candidate_score(
    manual: dict[str, dict], predicted: dict[str, dict], manual_field: str
) -> dict:
    counts = Counter()
    cases = []
    for image, row in predicted.items():
        candidate = row.get("rewrite_candidate")
        if not candidate:
            continue
        judgement = manual[image].get(manual_field)
        if judgement is True:
            counts["valid"] += 1
        elif judgement is False:
            counts["invalid"] += 1
        else:
            counts["unreviewed"] += 1
        cases.append({
            "image": image,
            "candidate": candidate,
            "human_accepted": judgement,
            "human_quality": manual[image].get("quality"),
            "human_visual_quality": manual[image].get("visual_quality"),
            "human_instruction_match": manual[image].get("instruction_match"),
        })
    return {"offered": len(cases), **dict(counts), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-jsonl", type=Path, required=True)
    parser.add_argument("--audit8-jsonl", type=Path, required=True)
    parser.add_argument("--audit27-jsonl", type=Path, required=True)
    parser.add_argument("--summary8-json", type=Path, required=True)
    parser.add_argument("--summary27-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    manual = load(args.manual_jsonl)
    audit8 = load(args.audit8_jsonl)
    audit27 = load(args.audit27_jsonl)
    if set(manual) != set(audit8) or set(manual) != set(audit27):
        raise ValueError("Manual and model audit case sets must match exactly")
    if any(row.get("quality") not in {"pass", "fail"} for row in manual.values()):
        raise ValueError("All manual cases must have a pass/fail overall quality")

    joint = Counter(
        f"8b_{audit8[image]['quality']}__27b_{audit27[image]['quality']}"
        for image in manual
    )
    result = {
        "cases": len(manual),
        "model_agreement": {
            "same_count": sum(
                audit8[image]["quality"] == audit27[image]["quality"]
                for image in manual
            ),
            "joint_counts": dict(joint),
        },
        "against_manual": {
            label: {
                field: confusion(manual, audit, field)
                for field in ("quality", "visual_quality", "instruction_match")
            }
            for label, audit in (("8b", audit8), ("27b", audit27))
        },
        "rewrite_candidates": {
            "8b": candidate_score(manual, audit8, "rewrite_8b_valid"),
            "27b": candidate_score(manual, audit27, "rewrite_27b_valid"),
        },
        "timing": {
            "8b": json.loads(args.summary8_json.read_text(encoding="utf-8")),
            "27b": json.loads(args.summary27_json.read_text(encoding="utf-8")),
        },
        "case_details": [
            {
                "image": image,
                "manual": manual[image]["quality"],
                "manual_reason": manual[image].get("reason"),
                "8b": audit8[image]["quality"],
                "27b": audit27[image]["quality"],
                "8b_rewrite": audit8[image].get("rewrite_candidate"),
                "27b_rewrite": audit27[image].get("rewrite_candidate"),
            }
            for image in manual
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "cases": result["cases"],
        "model_agreement": result["model_agreement"],
        "against_manual": result["against_manual"],
        "rewrite_candidates": {
            label: {key: value for key, value in summary.items() if key != "cases"}
            for label, summary in result["rewrite_candidates"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
