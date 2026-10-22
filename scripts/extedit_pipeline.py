#!/usr/bin/env python3
"""Protocol 2.0 labeling of CrispEdit-2M / ScaleEdit pairs (docs/EXTEDIT_PROTOCOL2.md).

Stages, each resumable (a stage skips ids it already wrote; failed records are retried on the next run):
  ingest    --crispedit add=25,color=25 --scaleedit object_addition=15,...   rows + images; --whole-shards --node k/n
  precheck  --workers 32                                                     CPU
  plan      --url http://127.0.0.1:8000/v1/chat/completions                  Qwen3.8-27B
  segment   --device cuda:0 [--shard 0/8]                                    SAM3 + SAMTok
  gate      --workers 32                                                     CPU
  audit     --url ... [--votes 2]                                            Qwen3.8-27B
  textcheck --url ...                                                        Qwen3.8-27B reads back new text
  export                                                                     final/
  review    [--accept 40 --reject 40]                                        review sheets
  run       all of the above on one node, the stages running side by side (each one --follow)
  status    --runs <run> ...: rows per stage (progress of the nodes of a production run)
  publish   --runs <run> ... --out <dataset folder>: merge the final/ of several runs (nodes)
  serve-command                                                              prints the vLLM command
With --follow a stage keeps taking the rows its upstream stage writes until that stage is done (its DONE marker), so
the stages overlap: the 27B server answers plan and audit calls at the same time while SAM3 and the CPU stages work.
Run with the labeling repo on PYTHONPATH (SAM3 and crispedit are imported from it), PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extedit import PIPELINE_VERSION, PROTOCOL_SHA256, STYLE_SHA256  # noqa: E402
from extedit import export as ex  # noqa: E402
from extedit import llm  # noqa: E402
from extedit import masks as mk  # noqa: E402
from extedit import pixels as px  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POLL = 15.0


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
    path = Path(run) / f"config.{stage}.json"
    path.write_text(json.dumps({"stage": stage, "args": {k: v for k, v in vars(args).items() if k != "func"},
                                "commit": commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "pipeline_version": PIPELINE_VERSION, "protocol_sha256": PROTOCOL_SHA256,
                                "style_sha256": STYLE_SHA256}, indent=1))


class Rows:
    """The ingested rows, read incrementally (ingest may still be writing them)."""

    def __init__(self, run):
        self.tail, self.rows = llm.Tail(Path(run) / "ingest", "rows*.jsonl"), {}

    def update(self):
        for row in self.tail.read():
            self.rows[row["id"]] = row
        return self.rows


class Stage:
    """Records of a stage as they appear: `fresh()` reads the records written since the last call, keeping the
    latest record of every id (a record with an error is dropped so that the row can be retried)."""

    def __init__(self, run, stage):
        self.dir = Path(run) / stage
        self.tail, self.records = llm.Tail(self.dir), {}

    def fresh(self):
        for record in self.tail.read():
            if "error" in record:
                self.records.pop(record.get("id"), None)
            else:
                self.records[record["id"]] = record
        return self.records

    def done(self):
        return llm.is_done(self.dir)


def views_pool(workers):
    """Processes that draw the views and encode the images of the 27B requests: in threads they hold the GIL and
    starve the event loop that keeps the server busy (validation 2026-10-21: one core at 100 %, 3 requests per
    engine)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


def follow_loop(run, upstream, take, process, follow):
    """Run `process(jobs)` on what `take()` finds; with --follow keep going until the upstream stage is done and
    nothing new came."""
    while True:
        finished = (not follow) or llm.is_done(Path(run) / upstream)
        jobs = take()
        if jobs:
            process(jobs)
        if finished and not jobs:
            break
        if not jobs:
            time.sleep(POLL)

# ---------------------------------------------------------------- ingest


def cmd_ingest(args):
    from extedit import sources
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    bench = sources.benchmark_index() if not args.no_benchmark_check else None
    node = tuple(int(v) for v in args.node.split("/"))
    counts = sources.ingest(run, quota(args.crispedit), quota(args.scaleedit), seed=args.seed, benchmark=bench,
                            whole_shards=args.whole_shards, node=node)
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
    rows, seen = Rows(run), set(ex.read_stage(run, "precheck"))

    def take():
        new = [r for i, r in rows.update().items() if i not in seen]
        seen.update(r["id"] for r in new)
        return [(run, r) for r in new]

    with Pool(args.workers) as pool, open(run / "precheck" / "records.jsonl", "a") as out:
        def process(jobs):
            for record in pool.imap_unordered(precheck_one, jobs, chunksize=2):
                out.write(json.dumps(record) + "\n")
                out.flush()
        follow_loop(run, "ingest", take, process, args.follow)
    llm.mark_done(run / "precheck")
    log_config(run, "precheck", args)

# ---------------------------------------------------------------- plan and audit (27B)


def cmd_plan(args):
    import aiohttp
    from extedit import plan as pl
    run = Path(args.run)
    (run / "plan").mkdir(exist_ok=True)
    rows, pre = Rows(run), Stage(run, "precheck")
    seen = set(ex.read_stage(run, "plan"))
    pool = views_pool(args.view_workers)
    left = [args.limit or None]

    def new_jobs():
        rows.update()
        pre.fresh()
        jobs = [{"id": i, "row": rows.rows[i], "pre": p} for i, p in pre.records.items()
                if p["decision"] == "keep" and i not in seen and i in rows.rows]
        if left[0] is not None:
            jobs = jobs[:left[0]]
            left[0] -= len(jobs)
        seen.update(j["id"] for j in jobs)
        return jobs

    async def main():
        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=args.concurrency + 8)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            client = llm.Client(session, args.url, args.model)

            async def worker(job):
                start = time.monotonic()
                parts, base = await asyncio.get_running_loop().run_in_executor(
                    pool, pl.plan_request, str(run), job["row"], job["pre"])
                record = await pl.plan_pair_with_parts(client, job["row"], parts, job["pre"]["areas"],
                                                       job["pre"].get("more_areas", 0), base)
                record["seconds"] = round(time.monotonic() - start, 1)
                return record
            await llm.stream_jobs(new_jobs, worker, args.concurrency, run / "plan" / "records.jsonl",
                                  upstream_done=pre.done if args.follow else None, poll=POLL)
    asyncio.run(main())
    llm.mark_done(run / "plan")
    log_config(run, "plan", args)


def cmd_audit(args):
    import aiohttp
    from extedit import audit as au
    run = Path(args.run)
    (run / "audit").mkdir(exist_ok=True)
    rows, plans, segs, gates = Rows(run), Stage(run, "plan"), Stage(run, "segment"), Stage(run, "gate")
    seen = set(ex.read_stage(run, "audit"))
    pool = views_pool(args.view_workers)

    def new_jobs():
        rows.update()
        plans.fresh()
        segs.fresh()
        gates.fresh()
        jobs = [{"id": i} for i, g in gates.records.items()
                if g["status"] == "passed" and i not in seen and i in segs.records and i in plans.records]
        seen.update(j["id"] for j in jobs)
        return jobs

    async def main():
        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=args.concurrency + 8)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            client = llm.Client(session, args.url, args.model)

            async def worker(job):
                row_id = job["id"]
                requests = await asyncio.get_running_loop().run_in_executor(
                    pool, au.audit_requests, str(run), rows.rows[row_id], plans.records[row_id]["plan"],
                    segs.records[row_id], gates.records[row_id])
                results = await asyncio.gather(*(au.audit_request(client, r, args.votes) for r in requests))
                return {"id": row_id, "units": list(results)}
            await llm.stream_jobs(new_jobs, worker, args.concurrency, run / "audit" / "records.jsonl",
                                  upstream_done=gates.done if args.follow else None, poll=POLL)
    asyncio.run(main())
    llm.mark_done(run / "audit")
    log_config(run, "audit", args)

def cmd_textcheck(args):
    """Read back the written text of every text-change unit of the rows that passed the audit (extedit/textcheck)."""
    import aiohttp
    from extedit import textcheck as tc
    run = Path(args.run)
    (run / "textcheck").mkdir(exist_ok=True)
    rows, plans, segs, gates, audits = (Rows(run), Stage(run, "plan"), Stage(run, "segment"), Stage(run, "gate"),
                                        Stage(run, "audit"))
    seen = set(ex.read_stage(run, "textcheck"))
    pool = views_pool(args.view_workers)

    def new_jobs():
        rows.update()
        for stage in (plans, segs, gates, audits):
            stage.fresh()
        jobs = [{"id": i} for i, a in audits.records.items()
                if i not in seen and all(u.get("passed") is True for u in a["units"])
                and i in plans.records and i in segs.records and i in gates.records
                and tc.text_units(plans.records[i]["plan"])]
        seen.update(j["id"] for j in jobs)
        return jobs

    async def main():
        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=args.concurrency + 8)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            client = llm.Client(session, args.url, args.model)

            async def worker(job):
                row_id = job["id"]
                requests = await asyncio.get_running_loop().run_in_executor(
                    pool, tc.text_requests, str(run), rows.rows[row_id], plans.records[row_id]["plan"],
                    segs.records[row_id], gates.records[row_id])
                units = list(await asyncio.gather(*(tc.check_unit(client, r) for r in requests)))
                return {"id": row_id, "units": units, "passed": all(u["passed"] for u in units)}
            await llm.stream_jobs(new_jobs, worker, args.concurrency, run / "textcheck" / "records.jsonl",
                                  upstream_done=audits.done if args.follow else None, poll=POLL)
    asyncio.run(main())
    llm.mark_done(run / "textcheck")
    log_config(run, "textcheck", args)

# ---------------------------------------------------------------- segment


def cmd_segment(args):
    import zlib
    import cv2
    import torch
    from PIL import Image
    from extedit import segment as sg
    # Several workers may share one GPU and the CPU: each keeps to its own share of threads.
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    run = Path(args.run)
    (run / "segment").mkdir(exist_ok=True)
    index, count = (int(v) for v in args.shard.split("/"))
    rows, pre, plans = Rows(run), Stage(run, "precheck"), Stage(run, "plan")
    seen = set(ex.read_stage(run, "segment"))
    state = {"seg": None}

    def take():
        rows.update()
        pre.fresh()
        plans.fresh()
        jobs = sorted(i for i, p in plans.records.items() if p.get("plan") and p["plan"]["quality"] == "clean"
                      and i not in seen and i in pre.records and i in rows.rows
                      and zlib.crc32(i.encode()) % count == index)
        seen.update(jobs)
        return jobs

    with open(run / "segment" / f"records.{index}.jsonl", "a") as out:
        def process(jobs):
            if state["seg"] is None:
                state["seg"] = sg.Segmenter(args.device, v3_repo=args.v3_repo)
            for row_id in jobs:
                start = time.monotonic()
                row = rows.rows[row_id]
                try:
                    image = Image.open(run / row["source_image"]).convert("RGB")
                    change = mk.decode(pre.records[row_id]["change"])
                    units = sg.segment_row(state["seg"], image, row_id, plans.records[row_id]["plan"],
                                           pre.records[row_id]["areas"], change)
                    record = {"id": row_id, "units": units, "seconds": round(time.monotonic() - start, 1)}
                except Exception as exc:  # noqa: BLE001 - recorded and retried on the next run
                    record = {"id": row_id, "error": f"{type(exc).__name__}: {exc}"[:500]}
                out.write(json.dumps(record) + "\n")
                out.flush()
        follow_loop(run, "plan", take, process, args.follow)
    llm.mark_done(run / "segment", f"DONE.{index}-of-{count}")
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
    rows, pre, plans, segs = Rows(run), Stage(run, "precheck"), Stage(run, "plan"), Stage(run, "segment")
    seen = set(ex.read_stage(run, "gate"))

    def take():
        rows.update()
        pre.fresh()
        plans.fresh()
        segs.fresh()
        jobs = [(run, rows.rows[i], pre.records[i], plans.records[i]["plan"], s) for i, s in segs.records.items()
                if i not in seen and i in pre.records and i in plans.records and i in rows.rows]
        seen.update(job[1]["id"] for job in jobs)
        return jobs

    with Pool(args.workers) as pool, open(run / "gate" / "records.jsonl", "a") as out:
        def process(jobs):
            for record in pool.imap_unordered(gate_one, jobs, chunksize=1):
                out.write(json.dumps(record) + "\n")
                out.flush()
        follow_loop(run, "segment", take, process, args.follow)
    llm.mark_done(run / "gate")
    log_config(run, "gate", args)

# ---------------------------------------------------------------- run: all stages side by side on one node


def cmd_run(args):
    """Ingest (unless it is done), then every stage as its own process following its upstream; export when the
    audit is done.  The 27B server must already answer at --url."""
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-B", str(Path(__file__).resolve())]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    logs = run / "logs"
    logs.mkdir(exist_ok=True)
    procs = []

    def start(name, argv):
        stream = open(logs / f"{name}.log", "a")
        procs.append((name, subprocess.Popen(py + argv, env=env, stdout=stream, stderr=subprocess.STDOUT)))

    if not llm.is_done(run / "ingest"):
        start("ingest", ["ingest", "--run", str(run), "--crispedit", args.crispedit, "--scaleedit", args.scaleedit,
                         "--seed", str(args.seed), "--node", args.node]
              + (["--whole-shards"] if args.whole_shards else []))
    start("precheck", ["precheck", "--run", str(run), "--workers", str(args.workers), "--follow"])
    start("plan", ["plan", "--run", str(run), "--url", args.url, "--concurrency", str(args.plan_concurrency),
                   "--follow"])
    devices = args.segment_devices.split(",")
    for k, device in enumerate(devices):
        start(f"segment{k}", ["segment", "--run", str(run), "--device", f"cuda:{device}",
                              "--shard", f"{k}/{len(devices)}", "--v3-repo", args.v3_repo, "--follow"])
    start("gate", ["gate", "--run", str(run), "--workers", str(args.workers), "--follow"])
    start("audit", ["audit", "--run", str(run), "--url", args.url, "--concurrency", str(args.audit_concurrency),
                    "--votes", str(args.votes), "--follow"])
    start("textcheck", ["textcheck", "--run", str(run), "--url", args.url, "--concurrency", "64", "--follow"])
    started = time.time()
    failed = []
    for name, proc in procs:
        code = proc.wait()
        print(f"{name} exited {code} after {time.time() - started:.0f} s", flush=True)
        if code:
            failed.append(name)
    if failed:
        raise SystemExit(f"stages failed: {failed}; see {logs}")
    summary = ex.export(run)
    (run / "timing.json").write_text(json.dumps({"seconds": round(time.time() - started), "rows": summary["rows"]}))
    log_config(run, "run", args)
    print(json.dumps(summary, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- export, review


def cmd_export(args):
    summary = ex.export(args.run, args.out)
    log_config(args.run, "export", args)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def cmd_status(args):
    """Rows per stage of each run (a node of a production run), and how long it has been running."""
    total = {}
    for run in map(Path, args.runs):
        counts = {"ingest": len(ex.rows(run)) if (run / "ingest" / "rows.jsonl").exists() else 0}
        for stage in ("precheck", "plan", "segment", "gate", "audit", "textcheck"):
            records = ex.read_stage(run, stage) if (run / stage).exists() else {}
            counts[stage] = len(records)
            if stage == "precheck":
                counts["precheck_keep"] = sum(r.get("decision") == "keep" for r in records.values())
            if stage == "gate":
                counts["gate_passed"] = sum(r.get("status") == "passed" for r in records.values())
            if stage == "audit":
                counts["audit_passed"] = sum(all(u.get("passed") is True for u in r["units"]) for r in records.values())
            if stage == "textcheck":
                counts["textcheck_passed"] = sum(r.get("passed") is True for r in records.values())
            if llm.is_done(run / stage):
                counts[stage + "_done"] = True
        counts["finished"] = (run / "final" / "summary.json").exists()
        print(run.name, json.dumps(counts))
        for k, v in counts.items():
            if isinstance(v, int) and not isinstance(v, bool):
                total[k] = total.get(k, 0) + v
    if len(args.runs) > 1:
        print("total", json.dumps(total))


def cmd_publish(args):
    total = ex.publish(args.runs, args.out)
    print(json.dumps({k: v for k, v in total.items() if k != "runs"}, indent=1))


def cmd_review(args):
    out = Path(args.out) if args.out else Path(args.run) / "review"
    paths = ex.review(args.run, out, args.accept, args.reject, args.seed)
    print(f"{len(paths)} sheets in {out}")


def cmd_serve(args):
    print(" ".join(llm.serve_command(args.vllm, gpus=args.gpus, port=args.port, max_num_seqs=args.max_num_seqs,
                                     gpu_memory_utilization=args.gpu_memory_utilization)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("ingest")
    s.add_argument("--run", required=True)
    s.add_argument("--crispedit", default="")
    s.add_argument("--scaleedit", default="")
    s.add_argument("--seed", type=int, default=20261020)
    s.add_argument("--whole-shards", action="store_true", help="production: read whole shards in order")
    s.add_argument("--node", default="0/1", help="k/n: this node takes every n-th shard starting at k")
    s.add_argument("--no-benchmark-check", action="store_true")
    s.set_defaults(func=cmd_ingest)
    s = sub.add_parser("precheck")
    s.add_argument("--run", required=True)
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--follow", action="store_true")
    s.set_defaults(func=cmd_precheck)
    for name, func in (("plan", cmd_plan), ("audit", cmd_audit), ("textcheck", cmd_textcheck)):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
        s.add_argument("--model", default="qwen38-27b")
        s.add_argument("--concurrency", type=int, default=128)
        s.add_argument("--view-workers", type=int, default=16)
        s.add_argument("--follow", action="store_true")
        if name == "plan":
            s.add_argument("--limit", type=int, default=0)
        elif name == "audit":
            s.add_argument("--votes", type=int, default=2)
        s.set_defaults(func=func)
    s = sub.add_parser("segment")
    s.add_argument("--run", required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--shard", default="0/1")
    s.add_argument("--v3-repo", default="/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3")
    s.add_argument("--threads", type=int, default=16, help="CPU threads of this worker (torch and OpenCV)")
    s.add_argument("--follow", action="store_true")
    s.set_defaults(func=cmd_segment)
    s = sub.add_parser("gate")
    s.add_argument("--run", required=True)
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--follow", action="store_true")
    s.set_defaults(func=cmd_gate)
    s = sub.add_parser("run")
    s.add_argument("--run", required=True)
    s.add_argument("--crispedit", default="")
    s.add_argument("--scaleedit", default="")
    s.add_argument("--seed", type=int, default=20261020)
    s.add_argument("--whole-shards", action="store_true")
    s.add_argument("--node", default="0/1")
    s.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    s.add_argument("--plan-concurrency", type=int, default=160)
    s.add_argument("--audit-concurrency", type=int, default=96)
    s.add_argument("--votes", type=int, default=2)
    s.add_argument("--segment-devices", default="7",
                   help="one SAM3 + SAMTok worker per entry, e.g. 6,7 or 7,7,7 (three workers sharing GPU 7)")
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--v3-repo", default="/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("export")
    s.add_argument("--run", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_export)
    s = sub.add_parser("status", help="rows per stage of one or more runs")
    s.add_argument("--runs", nargs="+", required=True)
    s.set_defaults(func=cmd_status)
    s = sub.add_parser("publish", help="merge the final/ of several runs into one dataset folder")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_publish)
    s = sub.add_parser("review")
    s.add_argument("--run", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--accept", type=int, default=40)
    s.add_argument("--reject", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_review)
    s = sub.add_parser("serve-command")
    s.add_argument("--vllm", default="vllm")
    s.add_argument("--gpus", type=int, default=8)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-num-seqs", type=int, default=96)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    s.set_defaults(func=cmd_serve)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
