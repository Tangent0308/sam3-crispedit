"""Export: D3 / D4 records, the token cache, rejections with their stage and reason, a summary and review sheets.

Records have the fields of the D3 manifest (datasets/SAMTokEdit_Cleaned_20261009/protocol_adapt_qwen38_27b_v1/
protocol_v2_blind/manifest_protocol_v2.jsonl) and of the D4 manifest (manifest_multi_protocol_v2.jsonl, units with the
D3 unit fields); image paths are relative to the run (or dataset) root.  The token cache has the fields of the round-2
caches v1_tokens.jsonl / d4_tokens.jsonl (span, decoded mask, round trip, size check, change outside the edit).
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from . import PIPELINE_VERSION, PROTOCOL_SHA256
from . import instruction_protocol as ip
from . import masks as mk
from . import views as vw
from .gate import PASTE_BAND
from .textcheck import text_units

STAGES = ("ingest", "precheck", "plan", "segment", "gate", "audit", "textcheck")


def read_stage(run, stage):
    """{id: record} of every JSONL file of a stage directory (shards merged, the last line of an id wins, records
    that failed with an error left out)."""
    from .llm import read_records
    out = {}
    for path in sorted((Path(run) / stage).glob("records*.jsonl")):
        out.update(read_records(path))
    return out


def rows(run):
    return {r["id"]: r for r in map(json.loads, open(Path(run) / "ingest" / "rows.jsonl"))}


def drift_stats(src, tgt, allowed):
    """Change of the stored AFTER outside the edit (round2_data.drifts reads out_diff and out_share8)."""
    d = np.abs(tgt.astype(np.float32) - src.astype(np.float32)).mean(-1)
    outside = ~allowed
    return {"out_diff": round(float(d[outside].mean()), 3) if outside.any() else 0.0,
            "out_share8": round(float((d[outside] > 8).mean()), 4) if outside.any() else 0.0,
            "in_diff": round(float(d[allowed].mean()), 2) if allowed.any() else 0.0}


def row_drift(job):
    """Change of one stored AFTER outside its edit (drift_stats), for a process pool.  job: (run, source image,
    pasted AFTER, RLEs of S / execution / edited regions, [(paste region RLE, operation)])."""
    run, source, target, regions, pastes = job
    src = np.asarray(Image.open(Path(run) / source).convert("RGB"))
    tgt = np.asarray(Image.open(Path(run) / target).convert("RGB"))
    allowed = np.zeros(src.shape[:2], bool)
    for rle in regions:
        allowed |= mk.decode(rle)
    for rle, op in pastes:  # the paste band and its soft edge belong to the edit
        band = max(2, int(PASTE_BAND[op] * min(src.shape[:2]))) + 8
        allowed |= mk.grow(mk.decode(rle), band)
    return {"size_ok": tgt.shape == src.shape, **drift_stats(src, tgt, allowed)}


def drift_key(job):
    import hashlib
    return hashlib.sha1(json.dumps(job[1:], sort_keys=True).encode()).hexdigest()


def drifts(jobs, cache_path, workers):
    """row_drift of every job, in parallel; results are cached by their inputs, so that a second export (the
    pipeline's second pass) only computes new or changed rows."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[entry["key"]] = entry["drift"]
    keys = [drift_key(job) for job in jobs]
    todo = [(k, job) for k, job in zip(keys, jobs) if k not in cache]
    if todo:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool, \
                open(cache_path, "a") as stream:
            for (key, _), drift in zip(todo, pool.map(row_drift, [job for _, job in todo], chunksize=16)):
                cache[key] = drift
                stream.write(json.dumps({"key": key, "drift": drift}) + "\n")
    return [cache[k] for k in keys]


def unit_record(row_id, unit, seg, gate_unit, audit, text_check=None):
    spec = unit["spec"]
    flags = list(gate_unit.get("flags", [])) + [f"fix:{f}" for f in unit.get("fixes", [])]
    selection = {"source": seg.get("source", "sam3"), **seg["selection"],
                 "granularity": unit["slots"].get("granularity") or ("part" if "part_as_s" in unit["fixes"] else "object"),
                 **({"anchor": seg["localized"]} if seg.get("localized") else {}),
                 **({"site_spot": seg["site_spot"]} if seg.get("site_spot") else {})}
    return {"id": row_id, "instruction_spec": spec, "instruction_ref": audit["instruction_ref"],
            "selection": selection,
            "execution_region": {"source": "change_support", **{k: gate_unit["execution_region"][k]
                                                                 for k in ("rle_size", "rle_counts", "box_1000")}},
            "edited_region": gate_unit["edited_region"], "flags": flags, "mask_semantics": ip.mask_contract(spec),
            "instruction_protocol_version": ip.VERSION, "adapter_version": PIPELINE_VERSION,
            "instruction_noref": audit["instruction_noref"], "noref_ineligible": audit["noref_ineligible"],
            "protocol_v2_change": None, "comparable": unit.get("comparable", False),
            "audit": {k: audit[k] for k in ("passed", "problems")},
            **({"text_check": {"read_after": text_check["after"], "read_before": text_check["before"]}}
               if text_check else {})}


def export(run, out=None, workers=None):
    """Write final/ (or `out`): manifests, tokens, rejections and summary; return the summary.  The change of every
    accepted AFTER outside its edit is measured in `workers` processes (default: up to 48).  When the run has a
    textcheck stage, a row with a text-change unit is kept only if its text check passed."""
    import os
    run = Path(run)
    check_text = (run / "textcheck").is_dir()
    out = Path(out) if out else run / "final"
    out.mkdir(parents=True, exist_ok=True)
    data = {s: (rows(run) if s == "ingest" else read_stage(run, s)) for s in STAGES}
    singles, multis, tokens, multi_tokens, rejected = [], [], [], [], []
    accepted = []
    for row_id, row in data["ingest"].items():
        pre, plan = data["precheck"].get(row_id), data["plan"].get(row_id)
        seg, gate, audit = data["segment"].get(row_id), data["gate"].get(row_id), data["audit"].get(row_id)
        base = {"id": row_id, "dataset": row["dataset"], "native_type": row["native_type"]}
        if pre is None:
            rejected.append({**base, "stage": "precheck", "reason": "missing"})
            continue
        if pre["decision"] != "keep":
            rejected.append({**base, "stage": "precheck", "reason": pre["decision"]})
            continue
        if plan is None or plan.get("plan") is None:
            rejected.append({**base, "stage": "plan", "reason": "missing" if plan is None else "no_valid_plan"})
            continue
        if plan["plan"]["quality"] != "clean":
            rejected.append({**base, "stage": "plan", "reason": "quality:" + plan["plan"]["problem"],
                             "observed": plan["plan"]["observed"]})
            continue
        if gate is None or seg is None:
            rejected.append({**base, "stage": "segment" if seg is None else "gate", "reason": "missing"})
            continue
        if gate["status"] != "passed":
            reasons = [h for u in gate["units"] for h in u.get("hard", [])] or [gate.get("reason")]
            rejected.append({**base, "stage": "gate", "reason": ",".join(sorted(set(map(str, reasons))))})
            continue
        if audit is None:
            rejected.append({**base, "stage": "audit", "reason": "missing"})
            continue
        failed = [a for a in audit["units"] if a["passed"] is not True]
        if failed:
            keys = sorted({k for a in failed for k in ("edit", "relation", "target", "outline", "noref", "wording",
                                                       "result") if a.get(k) != "pass"})
            rejected.append({**base, "stage": "audit", "reason": ",".join(keys),
                             "problems": " || ".join(a.get("problems", "") for a in failed)})
            continue
        checked = {}
        if check_text and text_units(plan["plan"]):
            text = data["textcheck"].get(row_id)
            if text is None:
                rejected.append({**base, "stage": "textcheck", "reason": "missing"})
                continue
            if not text["passed"]:
                bad = [u for u in text["units"] if not u["passed"]]
                rejected.append({**base, "stage": "textcheck", "reason": ",".join(sorted({u["reason"] for u in bad})),
                                 "problems": " || ".join(f'asked "{u["expected"]}", read "{u.get("after", "")}"'
                                                         for u in bad)})
                continue
            checked = {u["unit"]: u for u in text["units"]}
        units = [unit_record(row_id, u, s, g, a, checked.get(k)) for k, (u, s, g, a) in
                 enumerate(zip(plan["plan"]["units"], seg["units"], gate["units"], audit["units"]))]
        regions = [{k: u[key][k] for k in ("rle_size", "rle_counts")} for u in units
                   for key in ("selection", "execution_region", "edited_region")]
        pastes = [({k: g["paste_region"][k] for k in ("rle_size", "rle_counts")}, u["spec"]["operation"])
                  for u, g in zip(plan["plan"]["units"], gate["units"])]
        accepted.append((row_id, row, plan, seg, gate, units,
                         (str(run), row["source_image"], gate["target_pasted"], regions, pastes)))
    workers = workers or min(48, os.cpu_count() or 1)
    measured = drifts([a[-1] for a in accepted], out / "drift_cache.jsonl", workers) if accepted else []
    for (row_id, row, plan, seg, gate, units, _), drift in zip(accepted, measured):
        common = {"source_image": row["source_image"], "target_image": gate["target_pasted"], "id": row_id,
                  "dataset": row["dataset"], "type_before": row["native_type"], "instruction_before": row["instruction"],
                  "observed": plan["plan"]["observed"], "treatment": gate["treatment"], "provenance": row["provenance"],
                  "instruction_protocol_version": ip.VERSION, "adapter_version": PIPELINE_VERSION}
        if len(units) == 1:
            u = units[0]
            singles.append({**common, **{k: v for k, v in u.items() if k not in ("id",)}, "host_view": None})
            s = seg["units"][0]
            tokens.append({"id": row_id, "span": s["span"], "dec": {"size": s["dec"]["rle_size"],
                                                                     "counts": s["dec"]["rle_counts"]},
                           "roundtrip": s["roundtrip"], **drift})
        else:
            for k, u in enumerate(units, 1):
                u["id"] = f"{row_id}#U{k}"
            multis.append({**common, "multi_version": PIPELINE_VERSION, "units": units,
                           "flags": [f"U{k}:{f}" for k, u in enumerate(units, 1) for f in u["flags"]]})
            multi_tokens.append({"id": row_id, "units": [{"uid": f"U{k}", "span": s["span"],
                                                          "dec": {"size": s["dec"]["rle_size"],
                                                                  "counts": s["dec"]["rle_counts"]},
                                                          "roundtrip": s["roundtrip"]}
                                                         for k, s in enumerate(seg["units"], 1)], **drift})
    for name, records in (("manifest_protocol_v2.jsonl", singles), ("manifest_multi_protocol_v2.jsonl", multis),
                          ("tokens.jsonl", tokens), ("multi_tokens.jsonl", multi_tokens),
                          ("rejected.jsonl", rejected)):
        with open(out / name, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(data, singles, multis, rejected)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def summarize(data, singles, multis, rejected):
    by_type = Counter(f"{r['dataset']}/{r['native_type']}" for r in data["ingest"].values())
    kept_type = Counter(f"{r['dataset']}/{r['type_before']}" for r in singles + multis)
    ops = Counter(r["instruction_spec"]["operation"] for r in singles)
    ops_multi = Counter(u["instruction_spec"]["operation"] for r in multis for u in r["units"])
    stage = Counter(r["stage"] for r in rejected)
    reasons = Counter(f"{r['stage']}:{part}" for r in rejected for part in str(r["reason"]).split(","))
    return {"pipeline_version": PIPELINE_VERSION, "protocol_sha256": PROTOCOL_SHA256,
            "rows": len(data["ingest"]), "accepted_single": len(singles), "accepted_multi": len(multis),
            "accepted_multi_units": sum(len(r["units"]) for r in multis), "rejected": len(rejected),
            "by_source_type": {k: {"rows": n, "accepted": kept_type.get(k, 0)} for k, n in sorted(by_type.items())},
            "single_operations": dict(ops), "multi_unit_operations": dict(ops_multi),
            "rejected_by_stage": dict(stage), "rejection_reasons": dict(reasons.most_common()),
            "noref_ineligible": sum(r["instruction_noref"] is None for r in singles)}


def review(run, out_dir, n_accept=40, n_reject=40, seed=0):
    """Review sheets of a random sample of accepted and rejected rows."""
    run, out_dir = Path(run), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = run / "final"
    singles = [json.loads(l) for l in open(final / "manifest_protocol_v2.jsonl")]
    multis = [json.loads(l) for l in open(final / "manifest_multi_protocol_v2.jsonl")]
    rejected = [json.loads(l) for l in open(final / "rejected.jsonl")]
    ingest = rows(run)
    plan = read_stage(run, "plan")
    pre = read_stage(run, "precheck")
    rng = random.Random(seed)
    written = []
    for record in rng.sample(singles + multis, min(n_accept, len(singles) + len(multis))):
        units = record["units"] if "units" in record else [record]
        src = Image.open(run / record["source_image"]).convert("RGB")
        tgt = Image.open(run / record["target_image"]).convert("RGB")
        masks = []
        for u in units:
            masks.append((mk.decode(u["execution_region"]), vw.EXEC_COLOR))
            masks.append((mk.decode(u["selection"]), vw.S_COLOR))
        box = vw.union_box([u["selection"]["box_1000"] for u in units] + [u["execution_region"]["box_1000"] for u in units])
        panels = [("BEFORE, S green / exec orange", vw.outlined(src, masks, 10 ** 6, numbers=False)),
                  ("AFTER (pasted)", vw.resized(tgt, 10 ** 6)), ("close-up", vw.close_up(src, tgt, masks, box))]
        lines = [f"{record['id']}  {record['dataset']}/{record['type_before']}  {record.get('treatment')}",
                 f"original: {record['instruction_before']}", f"observed: {record.get('observed')}"]
        for k, u in enumerate(units, 1):
            lines += [f"U{k} ref: {u['instruction_ref']}", f"U{k} noref: {u['instruction_noref']}",
                      f"U{k} flags: {', '.join(u['flags']) or '-'}"]
        path = out_dir / f"accept_{record['id'].replace('#', '_')}.jpg"
        vw.sheet(panels, lines).save(path, quality=88)
        written.append(path)
    for record in rng.sample(rejected, min(n_reject, len(rejected))):
        row = ingest[record["id"]]
        src = Image.open(run / row["source_image"]).convert("RGB")
        tgt = Image.open(run / row["target_image"]).convert("RGB")
        if tgt.size != src.size:
            tgt = tgt.resize(src.size, Image.Resampling.LANCZOS)
        p = pre.get(record["id"]) or {}
        change = [(mk.decode(a), vw.PALETTE[(a["n"] - 1) % len(vw.PALETTE)][1]) for a in p.get("areas", [])]
        panels = [("BEFORE, changed areas", vw.outlined(src, change, 10 ** 6)), ("AFTER", vw.resized(tgt, 10 ** 6))]
        pl = (plan.get(record["id"]) or {}).get("plan") or {}
        lines = [f"{record['id']}  {record['dataset']}/{record['native_type']}  rejected at {record['stage']}: "
                 f"{record['reason']}", f"original: {row['instruction']}", f"observed: {pl.get('observed', '-')}",
                 f"problems: {record.get('problems', '-')}"]
        path = out_dir / f"reject_{record['stage']}_{record['id']}.jpg"
        vw.sheet(panels, lines).save(path, quality=88)
        written.append(path)
    return written


def publish(runs, out, workers=32):
    """Merge the final records of several runs (the nodes of a production run) into one dataset folder: the source
    images and the pasted AFTER copied (in `workers` threads) under images/<id[-2:]>/, paths relative to the folder,
    manifests, token caches, rejections (with their run) and a summary."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ("manifest_protocol_v2.jsonl", "manifest_multi_protocol_v2.jsonl", "tokens.jsonl", "multi_tokens.jsonl",
             "rejected.jsonl")
    merged = {name: [] for name in names}
    summaries, copies = [], []
    for run in map(Path, runs):
        final = run / "final"
        summaries.append({"run": str(run), **json.loads((final / "summary.json").read_text())})
        for name in names:
            for record in map(json.loads, open(final / name)):
                if name.startswith("manifest"):
                    for key, kind in (("source_image", "source"), ("target_image", "target")):
                        src = run / record[key]
                        dst = out / "images" / record["id"][-2:] / f"{record['id']}.{kind}{src.suffix}"
                        copies.append((src, dst))
                        record[key] = str(dst.relative_to(out))
                elif name == "rejected.jsonl":
                    record["run"] = run.name
                merged[name].append(record)

    def copy(job):
        src, dst = job
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, part)
        part.replace(dst)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(copy, copies))
    for name, records in merged.items():
        with open(out / name, "w") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    total = {"runs": summaries, "rows": sum(s["rows"] for s in summaries),
             "accepted_single": len(merged["manifest_protocol_v2.jsonl"]),
             "accepted_multi": len(merged["manifest_multi_protocol_v2.jsonl"]),
             "rejected": len(merged["rejected.jsonl"]), "images": len(copies), "pipeline_version": PIPELINE_VERSION}
    (out / "summary.json").write_text(json.dumps(total, indent=1, ensure_ascii=False))
    return total
