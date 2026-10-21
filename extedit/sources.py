"""Ingest: rows of the local CrispEdit-2M and ScaleEdit shards in one format, with quotas per type.

A row is {id, dataset, native_type, native_op, instruction, source_image, target_image, provenance}; the image bytes
are written as they are (no re-encode) under <run>/images/<id[-2:]>/.  Exclusions (doc 07 section 1.3): types out of
scope, rows the old pipeline already processed (exclusion lists kept next to the source shards), short side under
512 px, more than `per_source` rows sharing one source image, and source images of the benchmark.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

DATASETS = "/mnt/bn/strategy-mllm-train/user/tanyue/datasets"
CRISPEDIT = f"{DATASETS}/CrispEdit_Labeling/source/CrispEdit-2M"
SCALEEDIT = f"{DATASETS}/ScaleEdit_Labeling/source/ScaleEdit-filtered-source"
CRISPEDIT_USED = f"{DATASETS}/CrispEdit_Labeling/source/USED_SHARDS_before_20261020.json"
SCALEEDIT_USED = f"{DATASETS}/ScaleEdit_Labeling/source/USED_ROWS_before_20261020.parquet"
CRISPEDIT_REVISION = "764723d87e765bbceeff7e08bb0890f89abecf6b"

# native type -> operation the instruction asks for ("text": written text, attribute.text / add / remove)
CRISPEDIT_TYPES = {"add": "add", "remove": "remove", "replace": "replace", "color": "attribute"}
SCALEEDIT_TYPES = {"object_addition": "add", "object_removal": "remove", "object_replacement": "replace",
                   "color_change": "attribute", "material_change": "attribute",
                   "object_surface_text_editing": "text", "building_surface_text_editing": "text"}
MIN_SHORT_SIDE = 512


def crispedit_shards(root=CRISPEDIT):
    """{type: [shard paths]} of the shards not in the old pipeline's exclusion list."""
    used = set()
    if Path(CRISPEDIT_USED).exists():
        used = {Path(u).name for u in json.load(open(CRISPEDIT_USED))["files"]}
    out = defaultdict(list)
    for path in sorted(Path(root).glob("*.parquet")):
        kind = path.stem.rsplit("_", 1)[0]
        if kind in CRISPEDIT_TYPES and path.name not in used:
            out[kind].append(path)
    return out


def scaleedit_index(root=SCALEEDIT, pattern="expand-dl300k20261020*.parquet"):
    """{final_task: [(shard path, row index, sample_id)]} of the in-scope rows of the new download."""
    used = set()
    if Path(SCALEEDIT_USED).exists():
        used = set(pq.read_table(SCALEEDIT_USED, columns=["sample_id"]).column("sample_id").to_pylist())
    out = defaultdict(list)
    for path in sorted(Path(root).glob(pattern)):
        table = pq.read_table(path, columns=["sample_id", "final_task"])
        for k, (sid, task) in enumerate(zip(table.column("sample_id").to_pylist(),
                                            table.column("final_task").to_pylist())):
            if task in SCALEEDIT_TYPES and sid not in used:
                out[task].append((path, k, sid))
    return out


def image_format(data):
    with Image.open(io.BytesIO(data)) as im:
        return im.format, im.size


EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def write_image(data, folder, stem):
    """The bytes as they are, as <folder>/<stem><extension of the format>."""
    fmt, size = image_format(data)
    dst = Path(folder) / (stem + EXT.get(fmt, ".img"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(data)
    return dst, size


def sha1(data):
    return hashlib.sha1(data).hexdigest()


def sample_crispedit(quota, rng):
    """[(id, shard, row index, type)] spread over shards: rows drawn at random from shards drawn at random."""
    shards, picks = crispedit_shards(), []
    for kind, n in quota.items():
        paths = shards.get(kind, [])
        if not n or not paths:
            continue
        chosen = rng.sample(paths, min(len(paths), max(1, -(-n // 8))))  # about 8 rows per shard
        pool = [(p, k) for p in chosen for k in range(pq.ParquetFile(p).metadata.num_rows)]
        for path, k in rng.sample(pool, min(n, len(pool))):
            picks.append((f"crispedit-{path.stem}-{k:03d}", path, k, kind))
    return picks


def sample_scaleedit(quota, rng):
    index, picks = scaleedit_index(), []
    for task, n in quota.items():
        rows = index.get(task, [])
        for path, k, sid in rng.sample(rows, min(n, len(rows))):
            picks.append((f"scaleedit-{hashlib.sha1(sid.encode()).hexdigest()[:20]}", path, k, task, sid))
    return picks


def read_rows(path, indices, columns):
    table = pq.read_table(path, columns=columns)
    return {k: {c: table.column(c)[k].as_py() for c in columns} for k in indices}


def whole_shard_jobs(crispedit_quota, scaleedit_quota, node=(0, 1)):
    """Shard-by-shard plan for production: every row of a shard is read in one pass.  Shards are dealt to nodes
    round-robin by their position in the sorted list; a type stops when its quota is met (0 = no limit)."""
    k, n = node
    jobs = []
    shards = crispedit_shards()
    for kind in crispedit_quota:
        for i, path in enumerate(shards.get(kind, [])):
            if i % n == k:
                jobs.append(("crispedit", kind, path))
    if scaleedit_quota:
        paths = sorted(Path(SCALEEDIT).glob("expand-dl300k20261020*.parquet"))
        jobs += [("scaleedit", None, path) for i, path in enumerate(paths) if i % n == k]
    return jobs


def ingest(run, crispedit_quota, scaleedit_quota, seed=20261020, per_source=3, benchmark=None, whole_shards=False,
           node=(0, 1)):
    """Write rows.jsonl (row by row, so that a following precheck can start at once) and the images of a run, then
    the DONE marker; return the counts of kept and dropped rows.
    benchmark: (sha1 set, dHash list) of the benchmark's source images (benchmark_index); exact or near copies
    (dHash distance <= NEAR_DHASH) are dropped.
    whole_shards: production mode (whole_shard_jobs), else rows sampled at random over the shards (validation)."""
    from .llm import mark_done
    run = Path(run)
    rng = random.Random(seed)
    (run / "ingest").mkdir(parents=True, exist_ok=True)
    stream = open(run / "ingest" / "rows.jsonl", "a")
    kept, dropped, per_image = Counter(), Counter(), Counter()

    def keep(row_id, dataset, native_type, native_op, instruction, src_bytes, tgt_bytes, provenance):
        digest = sha1(src_bytes)
        if benchmark is not None and (digest in benchmark[0] or near(dhash(src_bytes), benchmark[1])):
            dropped["benchmark_image"] += 1
            return
        per_image[digest] += 1
        if per_image[digest] > per_source:
            dropped["same_source_image"] += 1
            return
        folder = run / "images" / row_id[-2:]
        src, src_size = write_image(src_bytes, folder, row_id + ".source")
        tgt, tgt_size = write_image(tgt_bytes, folder, row_id + ".target")
        if min(src_size) < MIN_SHORT_SIDE:
            dropped["short_side"] += 1
            src.unlink()
            tgt.unlink()
            return
        row = {"id": row_id, "dataset": dataset, "native_type": native_type, "native_op": native_op,
               "instruction": (instruction or "").strip(), "source_image": str(src.relative_to(run)),
               "target_image": str(tgt.relative_to(run)), "source_size": list(src_size),
               "target_size": list(tgt_size), "source_sha1": digest, "provenance": provenance}
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        kept[f"{dataset}/{native_type}"] += 1

    def crispedit_row(path, k, kind, r):
        keep(f"crispedit-{path.stem}-{k:03d}", "crispedit", kind, CRISPEDIT_TYPES[kind], r["instruction"],
             r["input_img"]["bytes"], r["output_img"]["bytes"],
             {"repo": "WeiChow/CrispEdit-2M", "revision": CRISPEDIT_REVISION, "shard": path.name, "row": k})

    def scaleedit_row(path, k, task, sid, r):
        keep(f"scaleedit-{hashlib.sha1(sid.encode()).hexdigest()[:20]}", "scaleedit", task, SCALEEDIT_TYPES[task],
             r["final_instruction"], r["source_image"], r["edited_image"],
             {"repo": "InternVL-U/ScaleEdit-12M via QingyuShi/scaleedit-filtered-6m", "shard": path.name, "row": k,
              "sample_id": sid})

    if whole_shards:
        used = set()
        if Path(SCALEEDIT_USED).exists():
            used = set(pq.read_table(SCALEEDIT_USED, columns=["sample_id"]).column("sample_id").to_pylist())
        full = lambda dataset, kind: 0 < ({**crispedit_quota, **scaleedit_quota}.get(kind) or 0) <= kept[f"{dataset}/{kind}"]
        for dataset, kind, path in whole_shard_jobs(crispedit_quota, scaleedit_quota, node):
            if dataset == "crispedit":
                if full("crispedit", kind):
                    continue
                table = pq.read_table(path, columns=["input_img", "output_img", "instruction", "type"])
                for k, r in enumerate(table.to_pylist()):
                    if not full("crispedit", kind):
                        crispedit_row(path, k, kind, r)
            else:
                index = pq.read_table(path, columns=["sample_id", "final_task"]).to_pylist()
                wanted = [k for k, r in enumerate(index) if r["final_task"] in scaleedit_quota
                          and r["final_task"] in SCALEEDIT_TYPES and r["sample_id"] not in used]
                if not wanted or all(full("scaleedit", index[k]["final_task"]) for k in wanted):
                    continue  # no image bytes are read for a shard without a wanted row
                table = pq.read_table(path, columns=["source_image", "edited_image", "final_instruction"])
                for k in wanted:
                    task = index[k]["final_task"]
                    if not full("scaleedit", task):
                        scaleedit_row(path, k, task, index[k]["sample_id"],
                                      {c: table.column(c)[k].as_py() for c in table.column_names})
    else:
        by_shard = defaultdict(list)
        for row_id, path, k, kind in sample_crispedit(crispedit_quota, rng):
            by_shard[path].append((k, kind))
        for path, items in sorted(by_shard.items()):
            data = read_rows(path, [k for k, _ in items], ["input_img", "output_img", "instruction", "type"])
            for k, kind in items:
                crispedit_row(path, k, kind, data[k])
        by_shard = defaultdict(list)
        for row_id, path, k, task, sid in sample_scaleedit(scaleedit_quota, rng):
            by_shard[path].append((k, task, sid))
        for path, items in sorted(by_shard.items()):
            data = read_rows(path, [k for k, _, _ in items], ["source_image", "edited_image", "final_instruction"])
            for k, task, sid in items:
                scaleedit_row(path, k, task, sid, data[k])
    stream.close()
    counts = {"rows": sum(kept.values()), "by_type": dict(kept), "dropped": dict(dropped)}
    (run / "ingest" / "summary.json").write_text(json.dumps(counts, indent=1))
    mark_done(run / "ingest")
    return counts


BENCHMARK_ASSETS = f"{DATASETS}/samtok_edit_benchmark_v2/assets"
NEAR_DHASH = 6


def dhash(data, size=8):
    """64-bit difference hash of an image (gray, (size+1) x size)."""
    with Image.open(io.BytesIO(data) if isinstance(data, bytes) else data) as im:
        g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return bits


def near(h, hashes, limit=NEAR_DHASH):
    return any(bin(h ^ other).count("1") <= limit for other in hashes)


def benchmark_index(root=BENCHMARK_ASSETS):
    """(SHA-1 set, dHash list) of the benchmark's source images (assets/<case>/source.jpg, 1,084 in 2.6.0)."""
    digests, hashes = set(), []
    for path in sorted(Path(root).glob("*/source.*")):
        data = path.read_bytes()
        digests.add(sha1(data))
        hashes.append(dhash(data))
    return frozenset(digests), hashes
