"""Render four aligned pairs per sheet for case-by-case instruction review."""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rewrites", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.rewrites.read_text().splitlines() if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    for offset in range(0, len(rows), 4):
        canvas = Image.new("RGB", (1600, 4 * 520), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(rows[offset : offset + 4]):
            y = j * 520
            label = (
                f'{row["image"]} | {row["task_type"]} | {row["editing_instruction"]}'
            )
            draw.multiline_text(
                (8, y + 4),
                "\n".join(textwrap.wrap(label, 125)),
                fill="black",
                font=font,
            )
            stem = Path(row["image"]).stem
            for k, suffix in enumerate(["source", "edited"]):
                picture = Image.open(args.inputs / f"{stem}_{suffix}.png").convert(
                    "RGB"
                )
                tile = ImageOps.contain(picture, (792, 442))
                canvas.paste(
                    tile,
                    (
                        k * 800 + (800 - tile.width) // 2,
                        y + 75 + (442 - tile.height) // 2,
                    ),
                )
        canvas.save(args.out_dir / f"sheet_{offset//4:02d}.jpg", quality=95)
    print(f"{len(rows)} cases; {(len(rows)+3)//4} sheets: {args.out_dir}")


if __name__ == "__main__":
    main()
