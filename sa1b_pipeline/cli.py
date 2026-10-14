"""Command line entry point for the SA-1B edit-data pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plan":
        from .per_mask import main as run
        run(argv[1:])
        return
    p = argparse.ArgumentParser(prog="sa1b-edit")
    sub = p.add_subparsers(dest="command", required=True)
    norm = sub.add_parser("normalize", help="normalize SA-1B JSON/TFRecord records")
    norm.add_argument("--input", type=Path); norm.add_argument("--image-records-dir", type=Path); norm.add_argument("--dense-records-dir", type=Path); norm.add_argument("--hdfs-image-uri"); norm.add_argument("--hdfs-dense-uri"); norm.add_argument("--scan-records", type=int, default=5000); norm.add_argument("--minimum-matches", type=int, default=100); norm.add_argument("--output-dir", type=Path, required=True)
    sub.add_parser("plan", help="design and audit an independent instruction for EVERY source mask")
    sel = sub.add_parser("select", help="alias for per-mask planning; old scene selection requires --planning-mode legacy-scene")
    sel.add_argument("--manifest", type=Path, required=True); sel.add_argument("--output", type=Path, required=True)
    sel.add_argument("--model-id", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B")
    sel.add_argument("--tensor-parallel-size", type=int, default=1); sel.add_argument("--max-cases", type=int); sel.add_argument("--max-candidates", type=int); sel.add_argument("--max-new-tokens", type=int, default=1000); sel.add_argument("--min-targets", type=int, default=2); sel.add_argument("--max-targets", type=int, default=5); sel.add_argument("--enforce-operation-mix", action="store_true")
    gen = sub.add_parser("generate", help="generate Qwen-Image-2.1 targets")
    gen.add_argument("--manifest", type=Path, required=True); gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--model-id", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1"); gen.add_argument("--steps", type=int, default=40)
    audit = sub.add_parser("validate", help="run deterministic and optional semantic validation")
    audit.add_argument("--manifest", type=Path, required=True); audit.add_argument("--output", type=Path, required=True); audit.add_argument("--model-id")
    mat = sub.add_parser("materialize", help="expand validated pairs into edit_ntp/edit_umt/edit rows")
    mat.add_argument("--generated", type=Path, required=True); mat.add_argument("--mask-codes", type=Path, required=True); mat.add_argument("--output", type=Path, required=True)
    doc = sub.add_parser("doctor", help="check SA-1B mounts and Qwen3.8/vLLM compatibility")
    doc.add_argument("--model", type=Path); doc.add_argument("--handoff")
    args, rest = p.parse_known_args(argv)
    if args.command == "plan":
        from .per_mask import main as run
        run(rest)
    elif args.command == "normalize":
        from .normalize import main as run
        argv2 = ["--output-dir", str(args.output_dir), "--scan-records", str(args.scan_records), "--minimum-matches", str(args.minimum_matches)]
        if args.hdfs_image_uri: argv2 += ["--hdfs-image-uri", args.hdfs_image_uri, "--hdfs-dense-uri", args.hdfs_dense_uri]
        elif args.image_records_dir: argv2 += ["--image-records-dir", str(args.image_records_dir), "--dense-records-dir", str(args.dense_records_dir)]
        else: argv2 += ["--input", str(args.input)]
        run(argv2 + rest)
    elif args.command == "select":
        from .select import main as run
        run(["--manifest", str(args.manifest), "--output", str(args.output), "--model-id", args.model_id, "--tensor-parallel-size", str(args.tensor_parallel_size), "--max-new-tokens", str(args.max_new_tokens), "--min-targets", str(args.min_targets), "--max-targets", str(args.max_targets)] + (["--enforce-operation-mix"] if args.enforce_operation_mix else []) + (["--max-cases", str(args.max_cases)] if args.max_cases else []) + (["--max-candidates", str(args.max_candidates)] if args.max_candidates else []) + rest)
    elif args.command == "generate":
        from .editor import main as run; run(["--manifest", str(args.manifest), "--output-dir", str(args.output_dir), "--model-id", args.model_id, "--steps", str(args.steps)] + rest)
    elif args.command == "materialize":
        from .training import main as run; run(["--generated", str(args.generated), "--mask-codes", str(args.mask_codes), "--output", str(args.output)] + rest)
    elif args.command == "doctor":
        from .doctor import main as run
        argv2 = []
        if args.model: argv2 += ["--model", str(args.model)]
        if args.handoff: argv2 += ["--handoff", str(args.handoff)]
        run(argv2 + rest)
    else:
        from .validate import main as run; run(["--manifest", str(args.manifest), "--output", str(args.output)] + (["--model-id", args.model_id] if args.model_id else []) + rest)


if __name__ == "__main__":
    main()
