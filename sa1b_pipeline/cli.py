"""Command line entry point for the SA-1B edit-data pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sa1b-edit")
    sub = p.add_subparsers(dest="command", required=True)
    norm = sub.add_parser("normalize", help="normalize SA-1B JSON/TFRecord records")
    norm.add_argument("--input", type=Path, required=True); norm.add_argument("--output-dir", type=Path, required=True)
    sel = sub.add_parser("select", help="select difficult scenes and plan four-type edits with Qwen3.8-27B")
    sel.add_argument("--manifest", type=Path, required=True); sel.add_argument("--output", type=Path, required=True)
    sel.add_argument("--model-id", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B")
    sel.add_argument("--tensor-parallel-size", type=int, default=1); sel.add_argument("--max-cases", type=int)
    gen = sub.add_parser("generate", help="generate Qwen-Image-2.1 targets")
    gen.add_argument("--manifest", type=Path, required=True); gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--model-id", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1"); gen.add_argument("--steps", type=int, default=40)
    audit = sub.add_parser("validate", help="run deterministic and optional semantic validation")
    audit.add_argument("--manifest", type=Path, required=True); audit.add_argument("--output", type=Path, required=True); audit.add_argument("--model-id")
    args, rest = p.parse_known_args(argv)
    if args.command == "normalize":
        from .normalize import main as run; run(["--input", str(args.input), "--output-dir", str(args.output_dir)])
    elif args.command == "select":
        from .select import main as run
        run(["--manifest", str(args.manifest), "--output", str(args.output), "--model-id", args.model_id] + (["--max-cases", str(args.max_cases)] if args.max_cases else []) + rest)
    elif args.command == "generate":
        from .editor import main as run; run(["--manifest", str(args.manifest), "--output-dir", str(args.output_dir), "--model-id", args.model_id, "--steps", str(args.steps)] + rest)
    else:
        from .validate import main as run; run(["--manifest", str(args.manifest), "--output", str(args.output)] + (["--model-id", args.model_id] if args.model_id else []) + rest)


if __name__ == "__main__":
    main()
