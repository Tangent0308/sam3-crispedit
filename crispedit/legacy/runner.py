"""Legacy CrispEdit pixel-difference-assisted mask runner.

This module is retained only for regression comparison.  Run it explicitly as
``python -m crispedit.legacy.runner``; it is not part of the production path.

特点:
1) 一个进程绑定一张 GPU，尽可能并行地处理多个 parquet shard
2) 每个 worker 只加载一次 SAM3 模型，循环处理自己分到的 shards
3) 输出与输入 shard 对齐的 parquet，按 row_idx 对应回原始数据
4) 支持 tqdm 总进度条、skip-existing、overwrite、limit-rows、小规模预览图
5) 适合后台跑: `nohup python ... > run.log 2>&1 &`

输出 parquet（每个输入 shard 对应一个输出 shard）主要字段:
- row_idx
- raw_type / canonical_type
- instruction
- phrases_json
- qc_flag / qc_status / diff_iou / diff_precision / diff_recall / area_frac
- mask_height / mask_width / mask_sum
- mask_png (binary PNG bytes)

解码示例:
    from PIL import Image
    import io
    mask = Image.open(io.BytesIO(row['mask_png'])).convert('L')

小规模验证建议:
    python -m crispedit.legacy.runner \
      --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
      --output-dir /tmp/crispedit_mask_parallel_test \
      --devices 0,1 \
      --max-shards-per-type 1 \
      --limit-rows-per-shard 2 \
      --preview-dir /tmp/crispedit_mask_parallel_test/previews \
      --preview-rows-per-shard 2
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from tqdm import tqdm

from crispedit.legacy.pipeline import annotate_one, canonicalize_type


@dataclass
class ShardJob:
    raw_type: str
    input_path: str
    output_path: str
    manifest_path: Optional[str]
    num_rows: int
    limit_rows: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrispEdit mask labeling over parquet shards")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing CrispEdit parquet shards")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write aligned output parquet shards")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.environ.get("CRISPEDIT_SAM3_CHECKPOINT_PATH"),
        help="Optional SAM3 checkpoint path; default uses HF cache/download",
    )
    parser.add_argument("--devices", type=str, default="auto", help="Comma-separated CUDA device ids, or 'auto', or 'cpu'")
    parser.add_argument("--include-types", type=str, default=None, help="Comma-separated raw types to include (e.g. 'add,color')")
    parser.add_argument("--max-shards-per-type", type=int, default=None, help="For testing: cap number of shards per type")
    parser.add_argument("--limit-rows-per-shard", type=int, default=None, help="For testing: only process the first N rows of each shard")
    parser.add_argument("--batch-size", type=int, default=8, help="Parquet row batch size for iteration")
    parser.add_argument("--keep-manifest-dir", type=Path, default=None, help="Optional directory of per-shard prefilter manifest parquet files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output shard files")
    parser.add_argument("--fail-fast", action="store_true", help="Abort worker on first row/shard failure")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional directory to save preview thumbnails")
    parser.add_argument("--preview-rows-per-shard", type=int, default=0, help="Save previews for the first N processed rows per shard")
    parser.add_argument("--compression", type=str, default="zstd", help="Parquet compression codec")
    parser.add_argument("--progress-mininterval", type=float, default=2.0, help="tqdm mininterval")
    return parser.parse_args()


def parse_devices(spec: str) -> List[str]:
    spec = (spec or "auto").strip().lower()
    if spec == "cpu":
        return ["cpu"]
    if spec == "auto":
        import torch

        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
    ids = [part.strip() for part in spec.split(",") if part.strip()]
    devices = []
    for part in ids:
        if part.startswith("cuda:"):
            devices.append(part)
        else:
            devices.append(f"cuda:{int(part)}")
    return devices or ["cpu"]


def raw_type_from_filename(path: Path) -> str:
    m = re.match(r"(.+)_\d+\.parquet$", path.name)
    return m.group(1) if m else path.stem


def iter_input_shards(input_dir: Path, include_types: Optional[Sequence[str]]) -> Dict[str, List[Path]]:
    include = None
    if include_types:
        include = {t.strip() for t in include_types if t.strip()}
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(input_dir.glob("*.parquet")):
        raw_type = raw_type_from_filename(path)
        if include is not None and raw_type not in include:
            continue
        grouped.setdefault(raw_type, []).append(path)
    return grouped


def count_rows(path: Path, limit_rows: Optional[int]) -> int:
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    return min(total, limit_rows) if limit_rows is not None else total


def build_jobs(args: argparse.Namespace) -> List[ShardJob]:
    grouped = iter_input_shards(args.input_dir, args.include_types.split(",") if args.include_types else None)
    jobs: List[ShardJob] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for raw_type, paths in sorted(grouped.items()):
        selected = paths[: args.max_shards_per_type] if args.max_shards_per_type is not None else paths
        for input_path in selected:
            out_path = args.output_dir / input_path.name
            manifest_path = None
            if args.keep_manifest_dir is not None:
                candidate = args.keep_manifest_dir / input_path.name
                if not candidate.exists():
                    raise FileNotFoundError(f"keep manifest not found for shard: {candidate}")
                manifest_path = str(candidate)
            rows = count_rows(input_path, args.limit_rows_per_shard)
            if rows <= 0:
                continue
            jobs.append(
                ShardJob(
                    raw_type=raw_type,
                    input_path=str(input_path),
                    output_path=str(out_path),
                    manifest_path=manifest_path,
                    num_rows=rows,
                    limit_rows=args.limit_rows_per_shard,
                )
            )
    return jobs


def assign_jobs(jobs: List[ShardJob], devices: List[str]) -> List[Tuple[str, List[ShardJob]]]:
    buckets = [{"device": dev, "rows": 0, "jobs": []} for dev in devices]
    for job in sorted(jobs, key=lambda j: j.num_rows, reverse=True):
        bucket = min(buckets, key=lambda b: b["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += job.num_rows
    return [(bucket["device"], bucket["jobs"]) for bucket in buckets if bucket["jobs"]]


def encode_mask_png(mask: np.ndarray) -> bytes:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def decode_image(cell: Dict) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def draw_label(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 42), "white")
    canvas.paste(img, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), text, fill="black")
    return canvas


def overlay_on(img: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> Image.Image:
    if mask.shape != (img.height, img.width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(img.size, Image.NEAREST)) > 0
        mask = mask.astype(np.uint8)
    base = img.convert("RGBA")
    rgba = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = mask * int(255 * alpha)
    overlay = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_preview(preview_dir: Path, shard_name: str, row_idx: int, sample: Dict, out: Dict) -> Path:
    mask = out["mask"].astype(np.uint8)
    qc = out["qc"]
    debug = out.get("debug", {})
    if qc["etype"] == "motion":
        diff_map = debug.get("motion_diff")
    elif qc["etype"] in ("background", "style"):
        diff_map = debug.get("global_diff")
    else:
        diff_map = debug.get("local_diff")

    src = sample["input_img"].resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
    tgt = sample["output_img"].resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
    diff_img = Image.fromarray((diff_map.astype(np.uint8) * 255), mode="L").convert("RGB") if diff_map is not None else Image.new("RGB", src.size, "black")
    mask_img = Image.fromarray((mask * 255), mode="L").convert("RGB")
    overlay = overlay_on(tgt, mask)

    tiles = [
        draw_label(src, "source"),
        draw_label(tgt, "target"),
        draw_label(diff_img, "diff"),
        draw_label(mask_img, "mask"),
        draw_label(overlay, "overlay"),
    ]
    canvas = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles) + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), f"{shard_name} row={row_idx} | {qc['etype']} | {qc['status']} | {qc['flag']}", fill="black")
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 28))
        x += tile.width

    out_path = preview_dir / f"{Path(shard_name).stem}__row{row_idx}.png"
    canvas.save(out_path)
    return out_path


def rows_to_table(rows: List[Dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def load_manifest_rows(manifest_path: Optional[str], limit_rows: Optional[int]) -> Dict[int, Dict]:
    if manifest_path is None:
        return {}
    rows = pq.read_table(manifest_path).to_pylist()
    if limit_rows is not None:
        rows = [row for row in rows if int(row.get("row_idx", -1)) < limit_rows]
    return {int(row["row_idx"]): row for row in rows}


def base_prefilter_fields(manifest_row: Optional[Dict]) -> Dict:
    if not manifest_row:
        return {
            "prefilter_verdict": "NOT_RUN",
            "prefilter_confidence": math.nan,
            "prefilter_method": "",
            "prefilter_evidence_schema": "",
            "prefilter_model_name": "",
            "prefilter_run_id": "",
            "prefilter_reason": "",
            "filter_decision": "keep",
            "filter_reason_codes": "",
            "filter_mismatch_score": 0.0,
        }
    return {
        "prefilter_verdict": manifest_row.get("prefilter_verdict", "NOT_RUN"),
        "prefilter_confidence": float(manifest_row.get("prefilter_confidence", math.nan)),
        "prefilter_method": manifest_row.get("prefilter_method", ""),
        "prefilter_evidence_schema": manifest_row.get("prefilter_evidence_schema", ""),
        "prefilter_model_name": manifest_row.get("prefilter_model_name", ""),
        "prefilter_run_id": manifest_row.get("prefilter_run_id", ""),
        "prefilter_reason": manifest_row.get("prefilter_reason", ""),
        "filter_decision": manifest_row.get("filter_decision", manifest_row.get("prefilter_decision", "keep")),
        "filter_reason_codes": manifest_row.get("filter_reason_codes", ""),
        "filter_mismatch_score": float(manifest_row.get("filter_mismatch_score", 0.0)),
    }


def make_prefilter_skip_row(row_idx: int, record: Dict, manifest_row: Dict) -> Dict:
    pre = base_prefilter_fields(manifest_row)
    verdict = str(pre["prefilter_verdict"] or "SKIP").upper()
    return {
        "row_idx": row_idx,
        "raw_type": record.get("type", ""),
        "canonical_type": "PREFILTER_SKIP",
        "instruction": record.get("instruction", ""),
        "phrases_json": "{}",
        "qc_flag": "PREFILTER_SKIP",
        "qc_status": f"PREFILTER_{verdict}",
        "diff_iou": math.nan,
        "diff_precision": math.nan,
        "diff_recall": math.nan,
        "area_frac": math.nan,
        "mask_height": 0,
        "mask_width": 0,
        "mask_sum": 0,
        "mask_png": b"",
        **pre,
    }


def process_shard(
    job: ShardJob,
    processor,
    args: argparse.Namespace,
    preview_dir: Optional[Path],
    use_cuda_autocast: bool,
    progress_queue,
    worker_idx: int,
) -> Tuple[List[Dict], Dict]:
    input_path = Path(job.input_path)
    output_path = Path(job.output_path)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    summary = {"rows": 0, "errors": 0, "flags": {}, "statuses": {}, "kept": 0, "prefilter_skipped": 0}

    manifest_rows = load_manifest_rows(job.manifest_path, job.limit_rows)
    pf = pq.ParquetFile(input_path)
    writer = None
    row_idx = 0
    preview_saved = 0
    pending_rows = 0
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            records = batch.to_pylist()
            out_rows: List[Dict] = []
            for record in records:
                if job.limit_rows is not None and row_idx >= job.limit_rows:
                    break
                manifest_row = manifest_rows.get(row_idx)
                if job.manifest_path is not None and manifest_row is None:
                    raise KeyError(f"missing keep-manifest row for {input_path.name} row_idx={row_idx}")
                prefilter_fields = base_prefilter_fields(manifest_row)
                should_skip = bool(manifest_row) and prefilter_fields["filter_decision"] != "keep"
                try:
                    if should_skip:
                        out_rows.append(make_prefilter_skip_row(row_idx, record, manifest_row))
                        summary["flags"]["PREFILTER_SKIP"] = summary["flags"].get("PREFILTER_SKIP", 0) + 1
                        skip_status = f"PREFILTER_{str(prefilter_fields['prefilter_verdict'] or 'SKIP').upper()}"
                        summary["statuses"][skip_status] = summary["statuses"].get(skip_status, 0) + 1
                        summary["prefilter_skipped"] += 1
                    else:
                        sample = {
                            "input_img": decode_image(record["input_img"]),
                            "output_img": decode_image(record["output_img"]),
                            "instruction": record["instruction"],
                            "type": record["type"],
                        }
                        if use_cuda_autocast:
                            import torch

                            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                                out = annotate_one(processor, sample)
                        else:
                            out = annotate_one(processor, sample)
                        mask = out["mask"].astype(np.uint8)
                        qc = out["qc"]
                        phrases_json = json.dumps(out["phrases"], ensure_ascii=False)
                        out_rows.append(
                            {
                                "row_idx": row_idx,
                                "raw_type": record["type"],
                                "canonical_type": qc["etype"],
                                "instruction": record["instruction"],
                                "phrases_json": phrases_json,
                                "qc_flag": qc["flag"],
                                "qc_status": qc["status"],
                                "diff_iou": float(qc["diff_iou"]),
                                "diff_precision": float(qc["diff_precision"]),
                                "diff_recall": float(qc["diff_recall"]),
                                "area_frac": float(qc["area_frac"]),
                                "mask_height": int(mask.shape[0]),
                                "mask_width": int(mask.shape[1]),
                                "mask_sum": int(mask.sum()),
                                "mask_png": encode_mask_png(mask),
                                **prefilter_fields,
                            }
                        )
                        summary["flags"][qc["flag"]] = summary["flags"].get(qc["flag"], 0) + 1
                        summary["statuses"][qc["status"]] = summary["statuses"].get(qc["status"], 0) + 1
                        summary["kept"] += 1
                        if preview_dir is not None and args.preview_rows_per_shard > 0 and preview_saved < args.preview_rows_per_shard:
                            save_preview(preview_dir, input_path.name, row_idx, sample, out)
                            preview_saved += 1
                except Exception as exc:
                    summary["errors"] += 1
                    out_rows.append(
                        {
                            "row_idx": row_idx,
                            "raw_type": record.get("type", job.raw_type),
                            "canonical_type": "ERROR",
                            "instruction": record.get("instruction", ""),
                            "phrases_json": json.dumps({"error": repr(exc)}, ensure_ascii=False),
                            "qc_flag": "ERROR",
                            "qc_status": "ERROR",
                            "diff_iou": math.nan,
                            "diff_precision": math.nan,
                            "diff_recall": math.nan,
                            "area_frac": math.nan,
                            "mask_height": 0,
                            "mask_width": 0,
                            "mask_sum": 0,
                            "mask_png": b"",
                            **prefilter_fields,
                        }
                    )
                    if args.fail_fast:
                        raise
                finally:
                    row_idx += 1
                    summary["rows"] += 1
                    pending_rows += 1
            if out_rows:
                table = rows_to_table(out_rows)
                if writer is None:
                    tmp_output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(tmp_output_path, table.schema, compression=args.compression)
                writer.write_table(table)
                progress_queue.put({
                    "kind": "rows",
                    "count": pending_rows,
                    "worker": worker_idx,
                    "shard": input_path.name,
                    "done_rows": summary["rows"],
                    "total_rows": job.num_rows,
                })
                pending_rows = 0
            if job.limit_rows is not None and row_idx >= job.limit_rows:
                break
    finally:
        if writer is not None:
            writer.close()
        if tmp_output_path.exists():
            tmp_output_path.replace(output_path)
    return [], summary


def worker_main(worker_idx: int, device: str, jobs: List[ShardJob], args_dict: Dict, progress_queue):
    args = argparse.Namespace(**args_dict)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import torch

        runtime_device = device
        builder_device = device
        processor_device = device
        use_cuda_autocast = False
        if device.startswith("cuda"):
            device_index = int(device.split(":", 1)[1]) if ":" in device else 0
            torch.cuda.set_device(device_index)
            # model_builder 只识别精确字符串 "cuda" 才会 .cuda()，因此这里显式设置当前卡后用 cuda。
            builder_device = "cuda"
            processor_device = "cuda"
            runtime_device = f"cuda:{device_index}"
            use_cuda_autocast = True

        progress_queue.put({"kind": "log", "message": f"worker-{worker_idx} starting on {runtime_device} with {len(jobs)} shards"})
        model = build_sam3_image_model(device=builder_device, checkpoint_path=args.checkpoint_path, load_from_HF=args.checkpoint_path is None)
        processor = Sam3Processor(model, device=processor_device, confidence_threshold=0.5)

        for job in jobs:
            output_path = Path(job.output_path)
            if output_path.exists() and not args.overwrite:
                progress_queue.put({"kind": "log", "message": f"worker-{worker_idx} skip existing {output_path.name}"})
                progress_queue.put({"kind": "rows", "count": job.num_rows})
                progress_queue.put({"kind": "shard_done", "worker": worker_idx, "shard": job.input_path, "summary": {"skipped": True, "rows": job.num_rows}})
                continue

            _, summary = process_shard(job, processor, args, preview_dir, use_cuda_autocast, progress_queue, worker_idx)
            progress_queue.put({"kind": "shard_done", "worker": worker_idx, "shard": job.input_path, "summary": summary})
    except Exception as exc:
        progress_queue.put({"kind": "worker_error", "worker": worker_idx, "device": device, "error": repr(exc)})
        if args.fail_fast:
            raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_idx, "device": device})


def build_preview_collage(preview_dir: Path, output_path: Path, limit: int = 8) -> Optional[Path]:
    images = sorted(preview_dir.glob("*.png"))[:limit]
    if not images:
        return None
    panels = []
    for path in images:
        img = Image.open(path).convert("RGB")
        width = 900
        height = max(1, int(img.height * (width / img.width)))
        panels.append(img.resize((width, height), Image.BILINEAR))

    cols = 2
    rows = math.ceil(len(panels) / cols)
    tile_w = max(img.width for img in panels)
    tile_h = max(img.height for img in panels)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "CrispEdit small-scale parallel labeling previews", fill="black")
    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        canvas.paste(panel, (c * tile_w, 44 + r * tile_h))
    canvas.save(output_path)
    return output_path


def aggregate_run_summary(shard_summaries: List[Dict]) -> Dict:
    flags: Dict[str, int] = {}
    statuses: Dict[str, int] = {}
    errors = 0
    rows = 0
    kept = 0
    prefilter_skipped = 0
    for item in shard_summaries:
        summary = item.get("summary", {})
        rows += int(summary.get("rows", 0))
        errors += int(summary.get("errors", 0))
        kept += int(summary.get("kept", 0))
        prefilter_skipped += int(summary.get("prefilter_skipped", 0))
        for key, value in summary.get("flags", {}).items():
            flags[key] = flags.get(key, 0) + int(value)
        for key, value in summary.get("statuses", {}).items():
            statuses[key] = statuses.get(key, 0) + int(value)
    return {
        "rows": rows,
        "errors": errors,
        "kept": kept,
        "prefilter_skipped": prefilter_skipped,
        "flags": flags,
        "statuses": statuses,
    }


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.keep_manifest_dir is not None:
        args.keep_manifest_dir = args.keep_manifest_dir.resolve()
    if args.preview_dir is not None:
        args.preview_dir = args.preview_dir.resolve()
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    if not jobs:
        print("No shards matched the requested filters.")
        return

    devices = parse_devices(args.devices)
    assignments = assign_jobs(jobs, devices)
    total_rows = sum(job.num_rows for job in jobs)

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []

    run_manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "devices": devices,
        "job_count": len(jobs),
        "total_rows": total_rows,
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "jobs": [asdict(job) for job in jobs],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for worker_idx, (device, worker_jobs) in enumerate(assignments):
        proc = ctx.Process(
            target=worker_main,
            args=(worker_idx, device, worker_jobs, {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, progress_queue),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    active_workers = len(processes)
    shard_summaries = []
    pbar = tqdm(total=total_rows, dynamic_ncols=True, mininterval=args.progress_mininterval, desc="mask rows")
    try:
        while active_workers > 0:
            try:
                msg = progress_queue.get(timeout=0.2)
            except queue.Empty:
                for proc in processes:
                    if not proc.is_alive() and proc.exitcode not in (None, 0):
                        tqdm.write(f"worker process failed with exit code {proc.exitcode}")
                continue

            kind = msg.get("kind")
            if kind == "rows":
                pbar.update(int(msg.get("count", 0)))
                done_rows = msg.get("done_rows")
                total_for_shard = msg.get("total_rows")
                shard_name = msg.get("shard")
                if shard_name is not None and done_rows is not None and total_for_shard is not None:
                    pbar.set_postfix_str(f"{shard_name} {done_rows}/{total_for_shard}")
            elif kind == "log":
                tqdm.write(msg.get("message", ""))
            elif kind == "shard_done":
                shard_summaries.append(msg)
                summary = msg.get("summary", {})
                tqdm.write(f"done {Path(msg['shard']).name}: rows={summary.get('rows', 0)} errors={summary.get('errors', 0)}")
            elif kind == "worker_error":
                tqdm.write(f"worker-{msg.get('worker')} on {msg.get('device')} error: {msg.get('error')}")
                if args.fail_fast:
                    raise RuntimeError(msg.get("error"))
            elif kind == "worker_done":
                active_workers -= 1
    finally:
        pbar.close()
        for proc in processes:
            proc.join()

    summary = aggregate_run_summary(shard_summaries)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.preview_dir is not None:
        collage_path = build_preview_collage(args.preview_dir, args.output_dir / "preview_collage.png")
        if collage_path:
            print(f"Preview collage: {collage_path}")

    print(f"Run summary: {args.output_dir / 'run_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
