#!/usr/bin/env python3
"""Display elapsed time and ETA for a full referential-filter run.

The watcher is intentionally read-only.  It consumes the atomic worker progress
JSON files written by ``run_referential_filter_full.py`` and can therefore be
started, stopped, or restarted without affecting inference or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from tqdm import tqdm


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/"
    "ScaleEdit-CrispEdit-mask-referential-filter"
)


def _load_reports(
    output_root: Path,
    stage: str,
    cache: MutableMapping[Path, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate reports while retaining the last valid atomic snapshot."""
    for path in sorted((output_root / "progress").glob(f"{stage}-worker-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("stage") == stage:
                cache[path] = payload
        except (OSError, json.JSONDecodeError):
            # A writer uses atomic rename, but keeping the last valid report also
            # makes the monitor robust on network filesystems.
            continue

    reports = list(cache.values())
    return {
        "workers_reporting": len(reports),
        "workers_complete": sum(row.get("state") == "complete" for row in reports),
        "processed_rows": sum(int(row.get("processed_rows", 0)) for row in reports),
        "total_rows": sum(int(row.get("total_rows", 0)) for row in reports),
        "completed_shards": sum(
            int(row.get("completed_shards", 0)) for row in reports
        ),
        "total_shards": sum(int(row.get("total_shards", 0)) for row in reports),
        "model_calls": sum(int(row.get("model_calls", 0)) for row in reports),
        "parse_errors": sum(int(row.get("parse_errors", 0)) for row in reports),
        "input_errors": sum(int(row.get("input_errors", 0)) for row in reports),
        "sam_calls": sum(int(row.get("sam_calls", 0)) for row in reports),
        "sam_errors": sum(int(row.get("sam_errors", 0)) for row in reports),
        "states": Counter(str(row.get("state", "unknown")) for row in reports),
    }


def _detect_stage(output_root: Path) -> str:
    """Prefer SAM once its reports exist; otherwise monitor MLLM."""
    sam_reports = list((output_root / "progress").glob("sam-worker-*.json"))
    return "sam" if sam_reports else "mllm"


def _stage_started_at(progress_log: Path, stage: str) -> Optional[float]:
    """Read the most recent supervisor launch timestamp for a stage."""
    try:
        lines = progress_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    pattern = re.compile(
        rf"^\[(?P<timestamp>[^]]+)] launched \d+ {re.escape(stage)} workers\b"
    )
    for line in reversed(lines):
        match = pattern.search(line)
        if not match:
            continue
        try:
            return datetime.fromisoformat(match.group("timestamp")).timestamp()
        except ValueError:
            return None
    return None


def _format_interval(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "?"
    return tqdm.format_interval(seconds)


def _snapshot_line(
    stage: str,
    report: Mapping[str, Any],
    started_at: float,
    now: Optional[float] = None,
) -> str:
    now = time.time() if now is None else now
    processed = int(report["processed_rows"])
    total = int(report["total_rows"])
    elapsed = max(0.0, now - started_at)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = (total - processed) / rate if rate > 0 and total >= processed else None
    percent = 100.0 * processed / total if total else 0.0
    errors = (
        int(report["parse_errors"]) + int(report["input_errors"])
        if stage == "mllm"
        else int(report["sam_errors"])
    )
    calls = int(report["model_calls"] if stage == "mllm" else report["sam_calls"])
    timestamp = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
    return (
        f"[{timestamp}] stage={stage} rows={processed}/{total} ({percent:.2f}%) "
        f"elapsed={_format_interval(elapsed)} eta={_format_interval(remaining)} "
        f"rate={rate:.2f} rows/s shards={report['completed_shards']}/"
        f"{report['total_shards']} workers={report['workers_reporting']} "
        f"states={dict(report['states'])} calls={calls} errors={errors}"
    )


def _bar_postfix(
    stage: str, report: Mapping[str, Any], started_at: float
) -> str:
    now = time.time()
    processed = int(report["processed_rows"])
    total = int(report["total_rows"])
    elapsed = max(0.0, now - started_at)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = (total - processed) / rate if rate > 0 and total >= processed else None
    errors = (
        int(report["parse_errors"]) + int(report["input_errors"])
        if stage == "mllm"
        else int(report["sam_errors"])
    )
    return (
        f"elapsed={_format_interval(elapsed)} eta={_format_interval(remaining)} "
        f"rate={rate:.2f}row/s shards={report['completed_shards']}/"
        f"{report['total_shards']} workers={report['workers_reporting']} errors={errors}"
    )


def _make_bar(stage: str, report: Mapping[str, Any], interval: float) -> tqdm:
    return tqdm(
        total=int(report["total_rows"]),
        initial=int(report["processed_rows"]),
        desc=f"{stage.upper()} filter",
        unit="row",
        dynamic_ncols=True,
        mininterval=max(1.0, interval),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
    )


def _supervisor_alive(session: str) -> bool:
    if not session:
        return True
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def watch(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    progress_log = args.progress_log or output_root / "logs" / "progress.log"
    snapshot_log = args.snapshot_log or output_root / "logs" / "tqdm-progress.log"
    snapshot_log.parent.mkdir(parents=True, exist_ok=True)

    stage = _detect_stage(output_root)
    caches: Dict[str, Dict[Path, Mapping[str, Any]]] = {"mllm": {}, "sam": {}}
    report = _load_reports(output_root, stage, caches[stage])
    if not report["workers_reporting"]:
        raise RuntimeError(f"no {stage} progress reports found under {output_root}")

    started_at = _stage_started_at(progress_log, stage) or time.time()
    bar = _make_bar(stage, report, float(args.interval))

    try:
        with snapshot_log.open("a", encoding="utf-8", buffering=1) as clean_log:
            while True:
                if (output_root / "_SUCCESS").is_file():
                    report = _load_reports(output_root, stage, caches[stage])
                    bar.update(max(0, int(report["processed_rows"]) - bar.n))
                    bar.set_postfix_str(
                        _bar_postfix(stage, report, started_at) + " complete"
                    )
                    bar.refresh()
                    clean_log.write(
                        _snapshot_line(stage, report, started_at) + " status=complete\n"
                    )
                    return 0

                current_stage = _detect_stage(output_root)
                if current_stage != stage:
                    bar.close()
                    stage = current_stage
                    report = _load_reports(output_root, stage, caches[stage])
                    started_at = _stage_started_at(progress_log, stage) or time.time()
                    bar = _make_bar(stage, report, float(args.interval))

                report = _load_reports(output_root, stage, caches[stage])
                current = int(report["processed_rows"])
                if current > bar.n:
                    bar.update(current - bar.n)
                bar.set_postfix_str(_bar_postfix(stage, report, started_at))
                bar.refresh()
                clean_log.write(_snapshot_line(stage, report, started_at) + "\n")

                if not _supervisor_alive(args.supervisor_session):
                    clean_log.write(
                        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
                        "status=stopped supervisor_session_missing\n"
                    )
                    return 1
                time.sleep(args.interval)
    finally:
        bar.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--snapshot-log", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument(
        "--supervisor-session",
        default="referential_filter_full",
        help="tmux session checked for liveness; pass an empty value to disable",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    return watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
