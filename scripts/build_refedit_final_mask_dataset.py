#!/usr/bin/env python3
"""Build the strict final RefEdit mask dataset after quality and mask QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from refedit.finalize import build_final_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prefilter-manifest-dir", type=Path, required=True)
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_final_dataset(
        args.input_dir,
        args.prefilter_manifest_dir,
        args.grounding_dir,
        args.mask_dir,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
