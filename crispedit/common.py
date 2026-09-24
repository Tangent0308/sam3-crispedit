"""Dataset helpers shared by the current quality and scene filters."""

import io
import re
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EDIT_TYPES = frozenset({"add", "color", "motion", "remove", "replace"})


def canonical_edit_type(raw_type: object) -> str:
    value = re.sub(r"\s+", " ", str(raw_type or "").strip().lower().replace("-", " "))
    aliases = {
        "addition": "add", "removal": "remove", "replacement": "replace",
        "background change": "background", "background_change": "background",
        "color change": "color", "colour": "color", "colour change": "color",
        "motion change": "motion", "motion_change": "motion", "style change": "style",
    }
    value = aliases.get(value, value)
    return value if value in {"add", "remove", "replace", "color", "motion", "background", "style"} else "unknown"


def raw_type_from_filename(path: Path) -> str:
    match = re.match(r"(.+)_\d+\.parquet$", path.name)
    return match.group(1) if match else path.stem


def supported_shard(path: Path) -> bool:
    return canonical_edit_type(raw_type_from_filename(path)) in SUPPORTED_EDIT_TYPES


def decode_image(cell: dict) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(cell["bytes"]))).convert("RGB")
