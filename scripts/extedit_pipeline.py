#!/usr/bin/env python3
"""Protocol 2.0 labeling of CrispEdit-2M / ScaleEdit pairs (docs/EXTEDIT_PROTOCOL2.md).

Stages, each resumable (a stage skips ids it already wrote; failed records are retried):
  ingest   --crispedit add=25,color=25 --scaleedit object_addition=15,...   rows.jsonl + images
  precheck --workers 32                                                     CPU
  plan     --url http://127.0.0.1:8000/v1/chat/completions                  Qwen3.8-27B
  segment  --device cuda:0 [--shard 0/8]                                    SAM3 + SAMTok
  gate     --workers 32                                                     CPU
  audit    --url ... [--votes 2]                                            Qwen3.8-27B
  export                                                                    final/
  review   [--accept 40 --reject 40]                                        review sheets
  serve-command                                                             prints the vLLM command
Run with the labeling repo on PYTHONPATH (SAM3 and crispedit are imported from it), PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extedit import PIPELINE_VERSION, PROTOCOL_SHA256, STYLE_SHA256  # noqa: E402
from extedit import export as ex  # noqa: E402
from extedit import masks as mk  # noqa: E402
from extedit import pixels as px  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def quota(text):
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            key, value = item.split("=")
            out[key.strip()] = int(value)
    return out


def commit():
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "extedit", "scripts"],
                           capture_output=True, text=True).stdout.strip()
    return out + ("+dirty" if dirty else "")


def log_config(run, stage, args):
    path = Path(run) / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    config.setdefault("stages", {})[stage] = {"args": {k: v for k, v in vars(args).items() if k != "func"},
                                              "commit": commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                              "pipeline_version": PIPELINE_VERSION}
    config.update(protocol_sha256=PROTOCOL_SHA256, style_sha256=STYLE_SHA256)
    path.write_text(json.dumps(config, indent=1))

# ---------------------------------------------------------------- ingest


def cmd_ingest(args):
    from extedit import sources
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    bench = sources.benchmark_index() if not args.no_benchmark_check else None
    counts = sources.ingest(run, quota(args.crispedit), quota(args.scaleedit), seed=args.seed, benchmark=bench)
    log_config(run, "ingest", args)
    print(json.dumps(counts, indent=1))

# ---------------------------------------------------------------- precheck


def precheck_one(job):
    run, row = job
    try:
        src = px.load_rgb(run / row["source_image"])
        tgt = px.load_rgb(run / row["target_image"])
        align, aligned, valid, reason = px.choose_alignment(src, tgt)
        record = {"id": row["id"], "size": [src.shape[1], src.shape[0]], "target_size": [tgt.shape[1], tgt.shape[0]],
                  "align": align}
        if reason:
            return {**record, "decision": "drop_" + reason}
        change, stats, _ = px.change_map(src, aligned, None if valid.all() else valid)
        record["change_stats"] = stats
        if stats["noise"] > px.PRECHECK["drop_noise"]:
            return {**record, "decision": "drop_global_change"}
        if stats["area_frac"] < px.PRECHECK["min_change"]:
            return {**record, "decision": "drop_no_change"}
        if stats["area_frac"] > px.PRECHECK["max_change"]:
            return {**record, "decision": "drop_global_change"}
        areas, more = px.change_areas(change)
        flags = ["noisy"] if stats["noise"] > px.PRECHECK["max_noise"] else []
        return {**record, "decision": "keep", "flags": flags, "change": mk.encode(change), "areas": areas,
                "more_areas": more}
    except Exception as exc:  # noqa: BLE001 - recorded and retried on the next run
        return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"[:500]}


def cmd_precheck(args):
    run = Path(args.run)
    (run / "precheck").mkdir(exist_ok=True)
    done = ex.read_stage(run, "precheck")
    jobs = [(run, r) for r in ex.rows(run).values() if r["id"] not in done]
    print(f"precheck: {len(jobs)} rows", flush=True)
    with Pool(args.workers) as pool, open(run / "precheck" / "records.jsonl", "a") as out:
        for k, record in enumerate(pool.imap_unordered(precheck_one, jobs, chunksize=2), 1):
            out.write(json.dumps(record) + "\n")
            out.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    log_config(run, "precheck", args)

# ---------------------------------------------------------------- plan


def cmd_plan(args):
    import aiohttp
    from extedit import llm
    from extedit import plan as pl
    run = Path(args.run)
    (run / "plan").mkdir(exist_ok=True)
    rows, pre = ex.rows(run), ex.read_stage(run, "precheck")
    done = ex.read_stage(run, "plan")
    jobs = [{"id": i, "row": rows[i], "pre": p} for i, p in pre.items() if p["decision"] == "keep" and i not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"plan: {len(jobs)} rows", flush=True)
    pool = ThreadPoolExecutor(args.view_workers)

    def views(job):
        source, target = pl.open_images(run, job["row"], job["pre"]["align"])
        return pl.plan_parts(source, target, job["pre"]["areas"]), pl.plan_prompt(job["row"], job["pre"]["areas"],
                                                                                   job["pre"].get("more_areas", 0))

    async def main():
        timeout = aiohttp.ClientTimeout(total=3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            client = llm.Client(session, args.url, args.model)

            async def worker(job):
                start = time.monotonic()
                parts, _ = await asyncio.get_running_loop().run_in_executor(pool, views, job)
                record = await pl.plan_pair_with_parts(client, job["row"], parts, job["pre"]["areas"],
                                                       job["pre"].get("more_areas", 0))
                record["seconds"] = round(time.monotonic() - start, 1)
                return record
            await llm.run_jobs(jobs, worker, args.concurrency, run / "plan" / "records.jsonl")
    asyncio.run(main())
    log_config(run, "plan", args)

# ---------------------------------------------------------------- segment


def cmd_segment(args):
    from PIL import Image
    from extedit import segment as sg
    run = Path(args.run)
    (run / "segment").mkdir(exist_ok=True)
    index, count = (int(v) for v in args.shard.split("/"))
    rows, pre, plans = ex.rows(run), ex.read_stage(run, "precheck"), ex.read_stage(run, "plan")
    done = ex.read_stage(run, "segment")
    jobs = sorted(i for i, p in plans.items() if p.get("plan") and p["plan"]["quality"] == "clean" and i not in done)
    jobs = [i for k, i in enumerate(jobs) if k % count == index]
    print(f"segment shard {args.shard}: {len(jobs)} rows", flush=True)
    if not jobs:
        return
    seg = sg.Segmenter(args.device, v3_repo=args.v3_repo)
    with open(run / "segment" / f"records.{index}.jsonl", "a") as out:
        for k, row_id in enumerate(jobs, 1):
            start = time.monotonic()
            row = rows[row_id]
            try:
                image = Image.open(run / row["source_image"]).convert("RGB")
                change = mk.decode(pre[row_id]["change"])
                units = sg.segment_row(seg, image, row_id, plans[row_id]["plan"], pre[row_id]["areas"], change)
                record = {"id": row_id, "units": units, "seconds": round(time.monotonic() - start, 1)}
            except Exception as exc:  # noqa: BLE001 - recorded and retried on the next run
                record = {"id": row_id, "error": f"{type(exc).__name__}: {exc}"[:500]}
            out.write(json.dumps(record) + "\n")
            out.flush()
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    log_config(run, f"segment_{index}", args)

# ---------------------------------------------------------------- gate


def gate_one(job):
    from extedit import gate as gt
    run, row, pre, plan, seg = job
    try:
        return gt.gate_row(run, row, pre, plan, seg["units"], run / "pasted")
    except Exception as exc:  # noqa: BLE001
        return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"[:500]}


def cmd_gate(args):
    run = Path(args.run)
    (run / "gate").mkdir(exist_ok=True)
    rows, pre, plans, segs = (ex.rows(run), ex.read_stage(run, "precheck"), ex.read_stage(run, "plan"),
                              ex.read_stage(run, "segment"))
    done = ex.read_stage(run, "gate")
    jobs = [(run, rows[i], pre[i], plans[i]["plan"], s) for i, s in segs.items() if i not in done]
    print(f"gate: {len(jobs)} rows", flush=True)
    with Pool(args.workers) as pool, open(run / "gate" / "records.jsonl", "a") as out:
        for k, record in enumerate(pool.imap_unordered(gate_one, jobs, chunksize=1), 1):
            out.write(json.dumps(record) + "\n")
            out.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    log_config(run, "gate", args)

# ---------------------------------------------------------------- audit


def cmd_audit(args):
    import aiohttp
    from PIL import Image
    from extedit import audit as au
    from extedit import llm
    run = Path(args.run)
    (run / "audit").mkdir(exist_ok=True)
    rows, plans, segs, gates = (ex.rows(run), ex.read_stage(run, "plan"), ex.read_stage(run, "segment"),
                                ex.read_stage(run, "gate"))
    done = ex.read_stage(run, "audit")
    jobs = [{"id": i} for i, g in gates.items() if g["status"] == "passed" and i not in done]
    print(f"audit: {len(jobs)} rows", flush=True)
    pool = ThreadPoolExecutor(args.view_workers)

    def load(row_id):
        row, gate = rows[row_id], gates[row_id]
        return (Image.open(run / row["source_image"]).convert("RGB"),
                Image.open(run / gate["target_pasted"]).convert("RGB"))

    async def main():
        timeout = aiohttp.ClientTimeout(total=3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            client = llm.Client(session, args.url, args.model)

            async def worker(job):
                row_id = job["id"]
                source, target = await asyncio.get_running_loop().run_in_executor(pool, load, row_id)
                plan, seg, gate = plans[row_id]["plan"], segs[row_id], gates[row_id]
                many = len(plan["units"]) > 1
                results = []
                for unit, s, g in zip(plan["units"], seg["units"], gate["units"]):
                    whole = (unit["spec"]["operation"] == "add" and unit["slots"].get("granularity") == "object"
                             and not (s.get("localized") or {}).get("cut") and not s.get("site_spot"))
                    results.append(await au.audit_unit(client, source, target, unit["spec"], mk.decode(s["selection"]),
                                                       mk.decode(g["execution_region"]), whole, many, args.votes))
                return {"id": row_id, "units": results}
            await llm.run_jobs(jobs, worker, args.concurrency, run / "audit" / "records.jsonl")
    asyncio.run(main())
    log_config(run, "audit", args)

# ---------------------------------------------------------------- export, review


def cmd_export(args):
    summary = ex.export(args.run, args.out)
    log_config(args.run, "export", args)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def cmd_review(args):
    paths = ex.review(args.run, Path(args.run) / "review", args.accept, args.reject, args.seed)
    print(f"{len(paths)} sheets in {Path(args.run) / 'review'}")


def cmd_serve(args):
    from extedit import llm
    print(" ".join(llm.serve_command(args.vllm, gpus=args.gpus, port=args.port)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("ingest")
    s.add_argument("--run", required=True)
    s.add_argument("--crispedit", default="")
    s.add_argument("--scaleedit", default="")
    s.add_argument("--seed", type=int, default=20261020)
    s.add_argument("--no-benchmark-check", action="store_true")
    s.set_defaults(func=cmd_ingest)
    s = sub.add_parser("precheck")
    s.add_argument("--run", required=True)
    s.add_argument("--workers", type=int, default=32)
    s.set_defaults(func=cmd_precheck)
    for name, func in (("plan", cmd_plan), ("audit", cmd_audit)):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
        s.add_argument("--model", default="qwen38-27b")
        s.add_argument("--concurrency", type=int, default=64)
        s.add_argument("--view-workers", type=int, default=16)
        s.add_argument("--limit", type=int, default=0)
        if name == "audit":
            s.add_argument("--votes", type=int, default=2)
        s.set_defaults(func=func)
    s = sub.add_parser("segment")
    s.add_argument("--run", required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--shard", default="0/1")
    s.add_argument("--v3-repo", default="/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3")
    s.set_defaults(func=cmd_segment)
    s = sub.add_parser("gate")
    s.add_argument("--run", required=True)
    s.add_argument("--workers", type=int, default=32)
    s.set_defaults(func=cmd_gate)
    s = sub.add_parser("export")
    s.add_argument("--run", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_export)
    s = sub.add_parser("review")
    s.add_argument("--run", required=True)
    s.add_argument("--accept", type=int, default=40)
    s.add_argument("--reject", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_review)
    s = sub.add_parser("serve-command")
    s.add_argument("--vllm", default="vllm")
    s.add_argument("--gpus", type=int, default=8)
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
