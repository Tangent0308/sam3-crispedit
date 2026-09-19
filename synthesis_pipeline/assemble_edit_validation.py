"""Assemble a non-destructive validation set from prioritized edit directories."""

import argparse
import json
from pathlib import Path


def link(path, target):
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--edited-dir",
        type=Path,
        action="append",
        required=True,
        help="First existing output wins; give latest iteration first",
    )
    args = p.parse_args()
    (args.out_root / "edited").mkdir(parents=True, exist_ok=True)
    for name in ["sources", "masks"]:
        link(args.out_root / name, args.data_root / name)
    rows = []
    for row in map(
        json.loads, (args.data_root / "annotations.jsonl").read_text().splitlines()
    ):
        selected = next(
            (d / row["image"] for d in args.edited_dir if (d / row["image"]).exists()),
            None,
        )
        if selected is None:
            continue
        link(args.out_root / "edited" / row["image"], selected)
        row["generation_method"] = selected.parent.parent.name
        row["generation_output_path"] = str(selected.resolve())
        rows.append(row)
    manifest = args.out_root / "annotations.jsonl"
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError("Validation manifest differs; use a new output directory")
    manifest.write_text(content)
    print(json.dumps({"cases": len(rows), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
