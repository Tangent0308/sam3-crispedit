#!/usr/bin/env python3
"""Resolve old experiment paths or create relocated JSON/JSONL read views.

Read views are NOT resumable checkpoints: original fingerprints and frozen
contracts intentionally remain untouched. Use new output roots for new runs.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.archive_relocation import ArchivePaths, DEFAULT_MAP


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Old or current absolute path")
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--output", type=Path, help="Create a separate relocated JSON/JSONL view")
    p.add_argument("--jsonl", action="store_true")
    args = p.parse_args()
    paths = ArchivePaths.load(args.map)
    if args.output:
        print(paths.materialize(args.path, args.output, jsonl=args.jsonl))
    else:
        print(paths.path(args.path))


if __name__ == "__main__":
    main()
