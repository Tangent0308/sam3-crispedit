"""Run Qwen-Image-Edit workers with a shared dynamic work queue.

The inference configuration is intentionally identical to
``inference_mydemo_qwen2511.py``.  Dynamic scheduling only changes which GPU
processes an image; every image still receives a fresh generator with the same
seed, so output does not depend on queue order.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = REPO_ROOT / "inference_mydemo_qwen2511.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--instruction-jsonl", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--results-full-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated physical GPU ids, one persistent worker per id.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for worker processes.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--cpu-offload",
        choices=["none", "model", "sequential"],
        default="none",
    )
    parser.add_argument("--patch-ratio", type=float, default=0.2)
    parser.add_argument("--edit-method", default="mirage",
                        choices=["mirage", "mirage_relaxed", "official_full", "context_edit", "context_edit_v2", "context_adaptive"])
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=None,
        help="Optional persistent claim directory. A temporary directory is used by default.",
    )
    parser.add_argument("--claim-timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Worker logs; defaults to RESULTS_FULL_DIR.worker_logs.",
    )
    return parser.parse_args()


def load_image_names(path: Path) -> list[str]:
    names = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                names.append(str(json.loads(line)["image"]))
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image names in {path}")
    return names


def worker_command(args: argparse.Namespace, queue_dir: Path, worker_id: str) -> list[str]:
    return [
        args.python,
        str(INFERENCE_SCRIPT),
        "--image-root",
        str(args.image_root),
        "--instruction-jsonl",
        str(args.instruction_jsonl),
        "--crop-dir",
        str(args.crop_dir),
        "--results-full-dir",
        str(args.results_full_dir),
        "--model-id",
        args.model_id,
        "--device",
        "cuda",
        "--dtype",
        args.dtype,
        "--cpu-offload",
        args.cpu_offload,
        "--patch-ratio",
        str(args.patch_ratio),
        "--edit-method",
        args.edit_method,
        "--num-steps",
        str(args.num_steps),
        "--true-cfg-scale",
        str(args.true_cfg_scale),
        "--guidance-scale",
        str(args.guidance_scale),
        "--negative-prompt",
        args.negative_prompt,
        "--seed",
        str(args.seed),
        "--work-queue-dir",
        str(queue_dir),
        "--worker-id",
        worker_id,
        "--claim-timeout-seconds",
        str(args.claim_timeout_seconds),
    ]


def run_pool(args: argparse.Namespace, queue_dir: Path) -> dict:
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")

    image_names = load_image_names(args.instruction_jsonl)
    args.results_full_dir.mkdir(parents=True, exist_ok=True)
    pending = [name for name in image_names if not (args.results_full_dir / name).is_file()]
    if not pending:
        return {
            "requested_cases": len(image_names),
            "generated_cases": 0,
            "workers": 0,
            "wall_seconds": 0.0,
            "status": "already_complete",
        }

    worker_count = min(len(gpu_ids), len(pending))
    log_dir = args.log_dir or Path(str(args.results_full_dir) + ".worker_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, object, str, Path]] = []
    started = time.perf_counter()
    try:
        for index, gpu_id in enumerate(gpu_ids[:worker_count]):
            worker_id = f"worker-{index:02d}-gpu-{gpu_id}"
            log_path = log_dir / f"{worker_id}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            process = subprocess.Popen(
                worker_command(args, queue_dir, worker_id),
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, log_handle, worker_id, log_path))

        failures = []
        for process, _, worker_id, log_path in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append(
                    {
                        "worker_id": worker_id,
                        "return_code": return_code,
                        "log": str(log_path),
                    }
                )
        elapsed = time.perf_counter() - started
        if failures:
            raise RuntimeError(json.dumps({"worker_failures": failures}, indent=2))
    finally:
        for process, log_handle, _, _ in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()

    missing = [name for name in image_names if not (args.results_full_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Pool completed with missing outputs: {missing[:10]}")
    return {
        "requested_cases": len(image_names),
        "generated_cases": len(pending),
        "workers": worker_count,
        "wall_seconds": round(elapsed, 3),
        "cases_per_minute": round(len(pending) / elapsed * 60.0, 3),
        "status": "complete",
        "log_dir": str(log_dir),
    }


def main() -> None:
    args = parse_args()
    if args.queue_dir is not None:
        args.queue_dir.mkdir(parents=True, exist_ok=True)
        summary = run_pool(args, args.queue_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="mirage_qwen_edit_queue_") as temporary:
            summary = run_pool(args, Path(temporary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
