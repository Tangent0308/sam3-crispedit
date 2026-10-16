"""Fonts shipped with the code snapshot; no system font lookup is required.

The checksum preflight uses only the standard library and runs before runtime
setup. Pillow is imported only when a worker needs to render a label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
FONT_NAMES = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")


def font_path(*, bold: bool = False) -> Path:
    path = FONT_DIR / FONT_NAMES[int(bold)]
    if not path.is_file():
        raise FileNotFoundError(f"Bundled font missing: {path}; restore the complete code bundle")
    return path


def load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype(str(font_path(bold=bold)), size)


def verify_fonts(*, render: bool = False) -> None:
    manifest = json.loads((FONT_DIR / "sha256.json").read_text())
    for bold, name in enumerate(FONT_NAMES):
        path = font_path(bold=bool(bold))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]:
            raise RuntimeError(f"Bundled font checksum mismatch: {path}")
        if render:
            from PIL import Image, ImageDraw
            canvas = Image.new("L", (320, 64), 0)
            ImageDraw.Draw(canvas).text((4, 4), "U1 | MEMBER 1", font=load_font(24, bold=bool(bold)), fill=255)
            if canvas.getbbox() is None:
                raise RuntimeError(f"Bundled font rendered no labels: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Also test Pillow label rendering")
    args = parser.parse_args()
    verify_fonts(render=args.render)
    print("Bundled fonts OK" + (" (Pillow rendering verified)" if args.render else " (SHA-256 verified)"))
