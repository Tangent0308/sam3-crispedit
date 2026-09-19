"""Prepare the exact two-image audit views for completed edits without VLM calls.

This permits blind visual monitoring while an edit worker pool is still running.
The generated subset manifest can be passed to build_two_image_audit_gallery.py
as both --annotations-jsonl and --audit8-jsonl with --hide-audits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from synthesis_pipeline.audit_edit_pairs import mask_array, write_jsonl
from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--longest-side", type=int, default=1280)
    parser.add_argument("--image-scope", choices=("full", "context_crop"), default="context_crop")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.annotations_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = [row for row in rows if (args.edited_dir / str(row["image"])).is_file()]
    inputs_dir = args.out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for row in tqdm(completed, desc="blind audit inputs"):
        stem = Path(str(row["image"])).stem
        before_path = inputs_dir / f"{stem}_source.png"
        after_path = inputs_dir / f"{stem}_edited.png"
        if before_path.is_file() and after_path.is_file():
            continue
        with Image.open(args.source_dir / str(row.get("source_image") or row["image"])) as handle:
            source = handle.convert("RGB")
        with Image.open(args.edited_dir / str(row["image"])) as handle:
            edited = handle.convert("RGB")
        mask = mask_array(source.size, row.get("mask", []))
        before, after = audit_two_image_inputs(source, edited, mask, args.longest_side, args.image_scope)
        before.save(before_path)
        after.save(after_path)
    write_jsonl(args.out_dir / "completed_annotations.jsonl", completed)
    print(json.dumps({"completed": len(completed), "requested": len(rows), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
