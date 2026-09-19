"""Debug true mask interiors and holes without changing model input images."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.visual_prompt_utils import clean_target_cutout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ids", required=True)
    args = p.parse_args()
    wanted = {int(x) for x in args.ids.split(",")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for row in map(
        json.loads, (args.data_root / "annotations.jsonl").read_text().splitlines()
    ):
        if int(row["image"].split("_")[0]) not in wanted:
            continue
        source = Image.open(args.data_root / "sources" / row["source_image"]).convert(
            "RGB"
        )
        mask = mask_array(source.size, row["mask"])
        binary = Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB")
        cutout = clean_target_cutout(source, mask, (0, 0, source.width, source.height))
        canvas = Image.new("RGB", (1536, 560), "white")
        draw = ImageDraw.Draw(canvas)
        for j, (image, label) in enumerate(
            [
                (source, "SOURCE"),
                (binary, f"WHITE = TARGET ({mask.mean():.1%})"),
                (cutout, "ACTUAL TARGET PIXELS; DEBUG ONLY"),
            ]
        ):
            draw.text((j * 512 + 8, 8), label, fill="black")
            tile = ImageOps.contain(image, (504, 520))
            canvas.paste(
                tile, (j * 512 + (512 - tile.width) // 2, 32 + (520 - tile.height) // 2)
            )
        canvas.save(args.out_dir / (Path(row["image"]).stem + ".jpg"), quality=95)


if __name__ == "__main__":
    main()
