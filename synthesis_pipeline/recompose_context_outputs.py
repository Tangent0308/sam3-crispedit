"""Recompose cached diffusion outputs; no new model call or image synthesis."""

import argparse
import json
from pathlib import Path

from PIL import Image

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox
from utils.context_edit import compose_context_crop, compose_attribute_crop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--diagnostics-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--composition",
        choices=["remove_wide", "attribute_mask"],
        default="remove_wide",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in map(
        json.loads, (args.data_root / "annotations.jsonl").read_text().splitlines()
    ):
        if args.composition == "attribute_mask" and row["task_type"] != "attribute":
            continue
        raw = args.diagnostics_dir / Path(row["image"]).stem / "raw_edited_crop.png"
        if not raw.exists():
            continue
        source = Image.open(args.data_root / "sources" / row["source_image"]).convert(
            "RGB"
        )
        mask = mask_array(source.size, row["mask"])
        bbox = padded_mask_bbox(mask, 0.75, 128)
        edited = Image.open(raw).convert("RGB")
        if args.composition == "attribute_mask":
            result, _ = compose_attribute_crop(source, edited, mask, bbox)
        else:
            result, _ = compose_context_crop(
                source, edited, mask, row["task_type"], bbox, True
            )
        result.save(args.out_dir / row["image"])
        count += 1
    print(
        json.dumps(
            {"recomposed": count, "diffusion_calls": 0, "out_dir": str(args.out_dir)}
        )
    )


if __name__ == "__main__":
    main()
