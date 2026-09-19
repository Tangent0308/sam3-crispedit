"""Export only explicitly reviewed model instructions, with exact provenance."""

import argparse
import json
from pathlib import Path

from synthesis_pipeline.evaluate_audit27 import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--decisions-json", type=Path, required=True)
    args = p.parse_args()
    rows = read_jsonl(args.data_root / "annotations.jsonl")
    manual = read_jsonl(args.data_root / "manual_review.jsonl")
    decisions = json.loads(args.decisions_json.read_text())
    review = {}
    for image, row in rows.items():
        key = str(int(image.split("_")[0]))
        decision = decisions[key]
        item = dict(accepted=False, reason=decision["reason"])
        if decision.get("rewrite_file"):
            if manual[image]["visual_quality"] != "pass":
                raise ValueError(f"Cannot accept a failed generation: {image}")
            path = args.data_root / decision["rewrite_file"]
            candidate = read_jsonl(path)[image]
            if candidate["status"] != "candidate":
                raise ValueError(f"No usable model instruction: {image}")
            item.update(
                accepted=True,
                task_type=candidate["task_type"],
                instruction=candidate["editing_instruction"],
                instruction_origin=str(path),
            )
        review[key] = item
    (args.data_root / "rewrite_manual_review.json").write_text(
        json.dumps(review, indent=2)
    )
    print(
        json.dumps(
            {
                "reviewed": len(review),
                "accepted": sum(x["accepted"] for x in review.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
