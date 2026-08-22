#!/usr/bin/env python3
"""
CrispEdit-2M 本地 MLLM 预筛选 Runner
=================================

在 SAM3 打标前，先用本地 Qwen3-VL 对 raw `(source, target, instruction, type)`
样本做语义有效性判断，输出：

1) audit parquet: 每行一个 MLLM verdict / reason / confidence
2) keep manifest parquet: 供后续 SAM3 runner 按 row_idx 决定 keep / skip

默认决策策略:
- PASS -> keep
- FAIL -> drop
- UNSURE -> drop
- ERROR -> drop

适合后台跑: `nohup python ... > run.log 2>&1 &`
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
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm import tqdm

from crispedit_mask_pipeline import canonicalize_type, parse_instruction


PROMPT_VERSION = "qwen3vl_edit_prefilter_v4_add_strict"
DEFAULT_MODEL = os.environ.get(
    "CRISPEDIT_QWEN_MODEL_PATH",
    "/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct",
)

PROMPT_TEMPLATE = """You are verifying whether an image editing request was successfully achieved.

You will receive:
- Image 1: source image BEFORE editing
- Image 2: target image AFTER editing
- One editing instruction
- One raw edit type label

Main task:
Judge whether Image 2 achieves the requested target state RELATIVE TO Image 1.

Follow this checklist carefully:
1. Identify the object, region, attribute, or style named in the instruction.
2. Compare that relevant part between Image 1 and Image 2.
3. First decide whether there is any relevant visible change there:
   - NONE = no meaningful relevant difference
   - SUBTLE = some relevant difference, but weak / partial / hard to verify
   - CLEAR = obvious relevant difference
4. Then decide whether the requested end state is achieved:
   - YES = clearly achieved
   - MOSTLY = almost achieved, minor residual issues
   - PARTIAL = some relevant progress, but not enough to confirm success
   - NO = clearly not achieved
   - UNCLEAR = evidence is mixed or hard to judge

Verdict mapping:
- PASS if the requested end state is clearly achieved (YES or MOSTLY) and there is no major contradiction.
- UNSURE if there is any relevant visible change (SUBTLE or CLEAR) but the requested end state is only PARTIAL or UNCLEAR.
- FAIL only if there is truly no meaningful relevant change, or the result clearly changes the wrong thing / wrong target.

Important rules:
1. The raw edit type is only a coarse hint for most types. For ADD, the stricter ADD-specific rules below override this.
2. Do NOT say "unchanged" unless change_presence is NONE.
3. If you notice some relevant change but it does not fully satisfy the instruction, explicitly acknowledge that change in the reason and prefer UNSURE over FAIL.
4. Judge final-state achievement, not whether the raw type label is perfectly pure, except that ADD must still show truly new target content.
5. Do NOT give PASS just because the image looks different. The visible change must match the instruction specifically.
6. For remove edits, deleting the whole image or wiping unrelated content is not a correct PASS unless the instruction truly asked for that.
7. Keep each observation short (max 12 words) and keep the reason short (max 28 words).
8. Before finalizing, check that your verdict, structured fields, and written reason all agree.

Return JSON only with exactly these keys:
{{
  "verdict": "PASS" | "FAIL" | "UNSURE",
  "confidence": <float 0-1>,
  "source_observation": "<one sentence, max 12 words>",
  "target_observation": "<one sentence, max 12 words>",
  "reason": "<max 28 words>",
  "change_presence": "NONE" | "SUBTLE" | "CLEAR",
  "instruction_achievement": "NO" | "PARTIAL" | "MOSTLY" | "YES" | "UNCLEAR",
  "failure_mode": "NO_RELEVANT_CHANGE" | "PARTIAL_RELEVANT_CHANGE" | "WRONG_TARGET" | "AMBIGUOUS" | "OTHER"
}}

Raw edit type (coarse hint only): {raw_type}
Instruction: {instruction}
"""

ADD_PROMPT_APPEND = """ADD-specific strict rules:
1. PASS only if the requested target content is newly added in Image 2 relative to Image 1.
2. If Image 1 already contains the requested target and Image 2 mainly makes that same existing instance larger, closer, more centered, more prominent, cleaner, sharper, or redrawn, do NOT PASS.
3. If Image 1 already contains a similar object, PASS only when Image 2 clearly adds an extra matching instance or an extra requested detail that was absent before.
4. For detail additions attached to an existing object (for example beads, rhinestones, decorations, accessories, or text), PASS only if that detail is visibly absent in Image 1 and present in Image 2.
5. Reframing, zooming, recentering, cropping, or style cleanup alone is not a valid ADD success.
6. If the evidence of newly added target content is weak, ambiguous, or only partial, prefer UNSURE or FAIL over PASS.

Deterministic ADD parsing hints:
- Parsed target phrase: {target_phrase}
- Parsed location hint: {location_hint}
- Parsed multiplicity hint: {multiplicity_hint}
"""

ADD_DETAIL_TOKENS = {
    "accessory",
    "accessories",
    "badge",
    "badges",
    "bead",
    "beads",
    "bow",
    "bows",
    "caption",
    "captions",
    "chain",
    "chains",
    "decal",
    "decals",
    "decoration",
    "decorations",
    "detail",
    "details",
    "jewelry",
    "label",
    "labels",
    "logo",
    "logos",
    "necklace",
    "necklaces",
    "pattern",
    "patterns",
    "print",
    "prints",
    "ribbon",
    "ribbons",
    "rhinestone",
    "rhinestones",
    "sticker",
    "stickers",
    "stripe",
    "stripes",
    "text",
    "trim",
}
ADD_EXISTING_TARGET_PATTERNS = (
    r"\balready(?:\s+\w+){0,3}\s+(?:present|there|visible|exists?)\b",
    r"\bexisting\b",
    r"\bsource already\b",
    r"\balready in (?:image 1|the source)\b",
    r"\bpresent before\b",
)
ADD_REFRAME_ONLY_PATTERNS = (
    r"\b(?:centered|recentered|recentred|larger|bigger|zoomed|zoom in|close up|close-up|cropped|reframed|more prominent|redrawn|restyled|cleaner|sharper|spans most of the image)\b",
)
ADD_POSITIVE_NOVELTY_PATTERNS = (
    r"\b(?:new|newly added|added|extra|another|appears|inserted|placed|put|restored|now has)\b",
)


@dataclass
class ShardJob:
    raw_type: str
    input_path: str
    audit_path: str
    manifest_path: str
    num_rows: int
    limit_rows: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local MLLM prefilter over CrispEdit parquet shards")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw CrispEdit parquet shards")
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory to write per-row MLLM audit parquet shards")
    parser.add_argument("--keep-manifest-dir", type=Path, required=True, help="Directory to write row-selection manifest parquet shards")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL, help="Local Qwen3-VL model path")
    parser.add_argument("--devices", type=str, default="auto", help="Comma-separated CUDA device ids, or 'auto', or 'cpu'")
    parser.add_argument("--include-types", type=str, default=None, help="Comma-separated raw types to include")
    parser.add_argument("--max-shards-per-type", type=int, default=None, help="For testing: cap number of shards per type")
    parser.add_argument("--limit-rows-per-shard", type=int, default=None, help="For testing: only process first N rows of each shard")
    parser.add_argument("--batch-size", type=int, default=1, help="Rows per parquet/model inference batch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output shard files")
    parser.add_argument("--fail-fast", action="store_true", help="Abort worker on first row/shard failure")
    parser.add_argument("--compression", type=str, default="zstd", help="Parquet compression codec")
    parser.add_argument("--progress-mininterval", type=float, default=2.0, help="tqdm mininterval")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Generation cap per row")
    parser.add_argument("--keep-verdicts", type=str, default="PASS", help="Comma-separated verdicts to keep, e.g. 'PASS' or 'PASS,UNSURE'")
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
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    args.keep_manifest_dir.mkdir(parents=True, exist_ok=True)
    for raw_type, paths in sorted(grouped.items()):
        selected = paths[: args.max_shards_per_type] if args.max_shards_per_type is not None else paths
        for input_path in selected:
            rows = count_rows(input_path, args.limit_rows_per_shard)
            if rows <= 0:
                continue
            jobs.append(
                ShardJob(
                    raw_type=raw_type,
                    input_path=str(input_path),
                    audit_path=str(args.audit_dir / input_path.name),
                    manifest_path=str(args.keep_manifest_dir / input_path.name),
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


def rows_to_table(rows: List[Dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def decode_image(cell: Dict) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(cell["bytes"]))).convert("RGB")


def _safe_canonical_type(raw_type: str) -> Optional[str]:
    try:
        return canonicalize_type(raw_type)
    except Exception:
        return None


def _build_prefilter_context(raw_type: str, instruction: str) -> Dict:
    raw_type = str(raw_type or "")
    instruction = str(instruction or "")
    canonical_type = _safe_canonical_type(raw_type)
    phrases = None
    if canonical_type is not None:
        try:
            phrases = parse_instruction(instruction, canonical_type)
        except Exception:
            phrases = None
    return {
        "raw_type": raw_type,
        "instruction": instruction,
        "canonical_type": canonical_type,
        "phrases": phrases or {},
    }


def _format_add_location_hint(location_hint: Optional[Dict]) -> str:
    if not location_hint:
        return "none"
    fields = []
    if "x" in location_hint:
        fields.append(f"x={float(location_hint['x']):.2f}")
    if "y" in location_hint:
        fields.append(f"y={float(location_hint['y']):.2f}")
    if "radius" in location_hint:
        fields.append(f"radius={float(location_hint['radius']):.2f}")
    if location_hint.get("region"):
        fields.append(f"region={location_hint['region']}")
    return ", ".join(fields) if fields else "none"


def build_prompt(raw_type: str, instruction: str) -> str:
    context = _build_prefilter_context(raw_type, instruction)
    prompt = PROMPT_TEMPLATE.format(raw_type=raw_type, instruction=instruction)
    if context.get("canonical_type") != "add":
        return prompt
    phrases = context.get("phrases") or {}
    target_phrase = str(phrases.get("target") or "").strip() or "<none parsed>"
    location_hint = _format_add_location_hint(phrases.get("location_hint"))
    multiplicity_hint = "multiple-or-counted target likely" if phrases.get("allow_multiple") else "single-or-unspecified target"
    return prompt + "\n\n" + ADD_PROMPT_APPEND.format(
        target_phrase=target_phrase,
        location_hint=location_hint,
        multiplicity_hint=multiplicity_hint,
    )


def extract_json(text: str) -> Dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON found in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def normalize_model_output(parsed: Dict) -> Dict:
    verdict = str(parsed.get("verdict", "UNSURE")).strip().upper()
    if verdict not in {"PASS", "FAIL", "UNSURE"}:
        verdict = "UNSURE"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    change_presence = str(parsed.get("change_presence", "SUBTLE")).strip().upper()
    if change_presence not in {"NONE", "SUBTLE", "CLEAR"}:
        change_presence = "SUBTLE"

    instruction_achievement = str(parsed.get("instruction_achievement", "UNCLEAR")).strip().upper()
    if instruction_achievement not in {"NO", "PARTIAL", "MOSTLY", "YES", "UNCLEAR"}:
        instruction_achievement = "UNCLEAR"

    failure_mode = str(parsed.get("failure_mode", "OTHER")).strip().upper()
    if failure_mode not in {
        "NO_RELEVANT_CHANGE",
        "PARTIAL_RELEVANT_CHANGE",
        "WRONG_TARGET",
        "AMBIGUOUS",
        "OTHER",
    }:
        failure_mode = "OTHER"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "source_observation": str(parsed.get("source_observation", "")).strip(),
        "target_observation": str(parsed.get("target_observation", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
        "change_presence": change_presence,
        "instruction_achievement": instruction_achievement,
        "failure_mode": failure_mode,
    }


def verdict_to_decision(verdict: str, keep_verdicts: Sequence[str]) -> str:
    return "keep" if verdict in keep_verdicts else "drop"


def filter_score(decision: str, verdict: str, confidence: float) -> float:
    if decision == "keep":
        return 0.0
    if verdict == "UNSURE":
        return max(0.5, confidence)
    return confidence


def _is_add_detail_phrase(target_phrase: Optional[str], instruction: str) -> bool:
    text = f"{target_phrase or ''} {instruction or ''}".lower()
    return any(tok in text for tok in ADD_DETAIL_TOKENS)


def _text_matches_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    s = str(text or "")
    return any(re.search(pattern, s, flags=re.I) for pattern in patterns)


def apply_add_post_policy(parsed: Dict, context: Dict) -> Dict:
    canonical_type = context.get("canonical_type")
    if canonical_type != "add":
        return parsed

    out = dict(parsed)
    phrases = context.get("phrases") or {}
    target_phrase = str(phrases.get("target") or "")
    reason = str(out.get("reason", "")).strip()
    source_obs = str(out.get("source_observation", "")).strip()
    target_obs = str(out.get("target_observation", "")).strip()
    joined = " ".join(part for part in [reason, source_obs, target_obs] if part).lower()

    is_detail = _is_add_detail_phrase(target_phrase, context.get("instruction", ""))
    mentions_existing = _text_matches_any_pattern(joined, ADD_EXISTING_TARGET_PATTERNS)
    mentions_reframe = _text_matches_any_pattern(joined, ADD_REFRAME_ONLY_PATTERNS)
    mentions_novelty = _text_matches_any_pattern(joined, ADD_POSITIVE_NOVELTY_PATTERNS)

    if out.get("verdict") == "PASS":
        if mentions_existing and not mentions_novelty:
            out["verdict"] = "FAIL"
            out["instruction_achievement"] = "NO"
            out["failure_mode"] = "NO_RELEVANT_CHANGE" if mentions_reframe else "WRONG_TARGET"
            out["change_presence"] = "SUBTLE" if out.get("change_presence") == "NONE" else out.get("change_presence", "SUBTLE")
            detail_text = "detail already present before" if is_detail else "target already present before"
            out["reason"] = f"ADD invalid: {detail_text}; no clearly new added content."
        elif mentions_reframe and not mentions_novelty:
            out["verdict"] = "FAIL"
            out["instruction_achievement"] = "NO"
            out["failure_mode"] = "NO_RELEVANT_CHANGE"
            out["change_presence"] = "SUBTLE" if out.get("change_presence") == "NONE" else out.get("change_presence", "SUBTLE")
            out["reason"] = "ADD invalid: change is reframing/enlargement, not new target content."

    if out.get("verdict") in {"FAIL", "UNSURE"}:
        if out.get("failure_mode") == "OTHER":
            out["failure_mode"] = "PARTIAL_RELEVANT_CHANGE" if out.get("change_presence") in {"SUBTLE", "CLEAR"} else "NO_RELEVANT_CHANGE"
        if not out.get("reason"):
            out["reason"] = "ADD not clearly supported by visible newly added target content."

    return normalize_model_output(out)


def write_table(rows: List[Dict], path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = rows_to_table(rows)
    pq.write_table(table, path, compression=compression)


def process_shard(job: ShardJob, runner, args: argparse.Namespace, progress_queue, worker_idx: int, run_id: str) -> Dict:
    input_path = Path(job.input_path)
    audit_path = Path(job.audit_path)
    manifest_path = Path(job.manifest_path)
    tmp_audit_path = audit_path.with_suffix(audit_path.suffix + ".tmp")
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    summary = {
        "rows": 0,
        "errors": 0,
        "kept": 0,
        "dropped": 0,
        "verdicts": {},
        "decisions": {},
    }

    pf = pq.ParquetFile(input_path)
    audit_writer = None
    manifest_writer = None
    row_idx = 0
    pending_rows = 0
    model_name = Path(args.model_path).name

    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            records = batch.to_pylist()
            if job.limit_rows is not None and row_idx >= job.limit_rows:
                break
            if job.limit_rows is not None:
                remaining = job.limit_rows - row_idx
                if remaining <= 0:
                    break
                records = records[:remaining]
            if not records:
                continue

            batch_results = runner.infer_batch(records, input_path.name, row_idx)
            audit_rows: List[Dict] = []
            manifest_rows: List[Dict] = []
            for batch_offset, (record, result) in enumerate(zip(records, batch_results)):
                current_row_idx = row_idx + batch_offset
                raw_text = result.get("raw_text", "")
                parsed = result["model_output"]
                parse_ok = bool(result.get("parse_ok", False))
                context = _build_prefilter_context(record.get("type", job.raw_type), record.get("instruction", ""))
                if parse_ok:
                    parsed = apply_add_post_policy(parsed, context)
                if not parse_ok:
                    summary["errors"] += 1
                    if args.fail_fast:
                        raise RuntimeError(f"prefilter parse/runtime error at {input_path.name}:{current_row_idx}: {parsed.get('reason', '')}")

                verdict = parsed["verdict"]
                decision = verdict_to_decision(verdict, runner.keep_verdicts) if verdict in {"PASS", "FAIL", "UNSURE"} else "drop"
                confidence = float(parsed["confidence"])
                reason = parsed.get("reason", "")
                summary["verdicts"][verdict] = summary["verdicts"].get(verdict, 0) + 1
                summary["decisions"][decision] = summary["decisions"].get(decision, 0) + 1
                if decision == "keep":
                    summary["kept"] += 1
                else:
                    summary["dropped"] += 1

                audit_row = {
                    "row_idx": current_row_idx,
                    "raw_type": record.get("type", job.raw_type),
                    "instruction": record.get("instruction", ""),
                    "prefilter_verdict": verdict,
                    "prefilter_decision": decision,
                    "prefilter_confidence": confidence,
                    "prefilter_source_observation": parsed.get("source_observation", ""),
                    "prefilter_target_observation": parsed.get("target_observation", ""),
                    "prefilter_reason": reason,
                    "prefilter_change_presence": parsed.get("change_presence", "SUBTLE"),
                    "prefilter_instruction_achievement": parsed.get("instruction_achievement", "UNCLEAR"),
                    "prefilter_failure_mode": parsed.get("failure_mode", "OTHER"),
                    "prefilter_parse_ok": parse_ok,
                    "prefilter_model_name": model_name,
                    "prefilter_prompt_version": PROMPT_VERSION,
                    "prefilter_run_id": run_id,
                    "filter_decision": decision,
                    "filter_reason_codes": "" if decision == "keep" else f"PREFILTER_{verdict}",
                    "filter_mismatch_score": float(filter_score(decision, verdict, confidence)),
                    "filter_version": PROMPT_VERSION,
                    "raw_text": raw_text,
                }
                manifest_row = {
                    "row_idx": current_row_idx,
                    "prefilter_verdict": verdict,
                    "prefilter_decision": decision,
                    "prefilter_confidence": confidence,
                    "prefilter_reason": reason,
                    "prefilter_change_presence": audit_row["prefilter_change_presence"],
                    "prefilter_instruction_achievement": audit_row["prefilter_instruction_achievement"],
                    "prefilter_failure_mode": audit_row["prefilter_failure_mode"],
                    "prefilter_parse_ok": parse_ok,
                    "prefilter_model_name": model_name,
                    "prefilter_prompt_version": PROMPT_VERSION,
                    "prefilter_run_id": run_id,
                    "filter_decision": decision,
                    "filter_reason_codes": audit_row["filter_reason_codes"],
                    "filter_mismatch_score": audit_row["filter_mismatch_score"],
                    "filter_version": PROMPT_VERSION,
                }
                audit_rows.append(audit_row)
                manifest_rows.append(manifest_row)

            row_idx += len(records)
            summary["rows"] += len(records)
            pending_rows += len(records)

            if audit_rows:
                audit_table = rows_to_table(audit_rows)
                manifest_table = rows_to_table(manifest_rows)
                if audit_writer is None:
                    tmp_audit_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    audit_writer = pq.ParquetWriter(tmp_audit_path, audit_table.schema, compression=args.compression)
                    manifest_writer = pq.ParquetWriter(tmp_manifest_path, manifest_table.schema, compression=args.compression)
                audit_writer.write_table(audit_table)
                manifest_writer.write_table(manifest_table)
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
        if audit_writer is not None:
            audit_writer.close()
        if manifest_writer is not None:
            manifest_writer.close()
        if tmp_audit_path.exists():
            tmp_audit_path.replace(audit_path)
        if tmp_manifest_path.exists():
            tmp_manifest_path.replace(manifest_path)
    return summary


class QwenPrefilterRunner:
    def __init__(self, model_path: str, device: str, max_new_tokens: int, keep_verdicts: Sequence[str]):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.keep_verdicts = tuple(v.strip().upper() for v in keep_verdicts if v.strip())
        self._torch = torch
        model_kwargs = {
            "trust_remote_code": True,
        }
        if device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs).to("cpu").eval()
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = {"": device}
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "padding_side"):
            self.processor.tokenizer.padding_side = "left"

    def _build_messages(self, record: Dict) -> List[Dict]:
        src = decode_image(record["input_img"])
        tgt = decode_image(record["output_img"])
        prompt = build_prompt(str(record.get("type", "")), str(record.get("instruction", "")))
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1 is the source before editing."},
                {"type": "image", "image": src},
                {"type": "text", "text": "Image 2 is the target after editing."},
                {"type": "image", "image": tgt},
                {"type": "text", "text": prompt},
            ],
        }]

    def _error_result(self, reason: str) -> Dict:
        return {
            "model_output": {
                "verdict": "ERROR",
                "confidence": 1.0,
                "source_observation": "",
                "target_observation": "",
                "reason": reason,
                "change_presence": "SUBTLE",
                "instruction_achievement": "UNCLEAR",
                "failure_mode": "OTHER",
            },
            "raw_text": "",
            "parse_ok": False,
        }

    def infer_batch(self, records: Sequence[Dict], shard_name: str, row_idx_start: int) -> List[Dict]:
        if not records:
            return []

        try:
            conversations = [self._build_messages(record) for record in records]
        except Exception as exc:
            return [self._error_result(repr(exc)) for _ in records]

        try:
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            inputs = inputs.to(self.model.device)
            prompt_length = int(inputs.input_ids.shape[1])
            with self._torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            trimmed = generated_ids[:, prompt_length:]
            texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception as exc:
            reason = repr(exc)
            return [self._error_result(reason) for _ in records]

        results: List[Dict] = []
        for batch_offset, text in enumerate(texts):
            try:
                parsed = normalize_model_output(extract_json(text))
                results.append({
                    "model_output": parsed,
                    "raw_text": text,
                    "parse_ok": True,
                })
            except Exception as exc:
                results.append(self._error_result(repr(exc)))

        if len(results) != len(records):
            reason = f"batch size mismatch for {Path(shard_name).stem} starting at row {row_idx_start}"
            return [self._error_result(reason) for _ in records]
        return results

    def infer(self, record: Dict, shard_name: str, row_idx: int) -> Dict:
        return self.infer_batch([record], shard_name, row_idx)[0]


def worker_main(worker_idx: int, device: str, jobs: List[ShardJob], args_dict: Dict, progress_queue, run_id: str):
    args = argparse.Namespace(**args_dict)
    keep_verdicts = [v.strip().upper() for v in args.keep_verdicts.split(",") if v.strip()]
    try:
        progress_queue.put({"kind": "log", "message": f"worker-{worker_idx} starting on {device} with {len(jobs)} shards"})
        runner = QwenPrefilterRunner(args.model_path, device, args.max_new_tokens, keep_verdicts)
        for job in jobs:
            audit_path = Path(job.audit_path)
            manifest_path = Path(job.manifest_path)
            if audit_path.exists() and manifest_path.exists() and not args.overwrite:
                progress_queue.put({"kind": "log", "message": f"worker-{worker_idx} skip existing {audit_path.name}"})
                progress_queue.put({"kind": "rows", "count": job.num_rows})
                progress_queue.put({"kind": "shard_done", "worker": worker_idx, "shard": job.input_path, "summary": {"skipped": True, "rows": job.num_rows}})
                continue
            summary = process_shard(job, runner, args, progress_queue, worker_idx, run_id)
            progress_queue.put({"kind": "shard_done", "worker": worker_idx, "shard": job.input_path, "summary": summary})
    except Exception as exc:
        progress_queue.put({"kind": "worker_error", "worker": worker_idx, "device": device, "error": repr(exc)})
        if args.fail_fast:
            raise
    finally:
        progress_queue.put({"kind": "worker_done", "worker": worker_idx, "device": device})


def aggregate_run_summary(shard_summaries: List[Dict]) -> Dict:
    verdicts: Dict[str, int] = {}
    decisions: Dict[str, int] = {}
    rows = 0
    errors = 0
    kept = 0
    dropped = 0
    for item in shard_summaries:
        summary = item.get("summary", {})
        rows += int(summary.get("rows", 0))
        errors += int(summary.get("errors", 0))
        kept += int(summary.get("kept", 0))
        dropped += int(summary.get("dropped", 0))
        for key, value in summary.get("verdicts", {}).items():
            verdicts[key] = verdicts.get(key, 0) + int(value)
        for key, value in summary.get("decisions", {}).items():
            decisions[key] = decisions.get(key, 0) + int(value)
    return {
        "rows": rows,
        "errors": errors,
        "keep": kept,
        "drop": dropped,
        "drop_rate": round(dropped / max(rows, 1), 6),
        "verdict_counts": verdicts,
        "decision_counts": decisions,
        "prompt_version": PROMPT_VERSION,
    }


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.audit_dir = args.audit_dir.resolve()
    args.keep_manifest_dir = args.keep_manifest_dir.resolve()

    jobs = build_jobs(args)
    if not jobs:
        print("No shards matched the requested filters.")
        return

    devices = parse_devices(args.devices)
    assignments = assign_jobs(jobs, devices)
    total_rows = sum(job.num_rows for job in jobs)
    run_id = time.strftime("prefilter_%Y%m%d_%H%M%S")

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []

    run_manifest = {
        "input_dir": str(args.input_dir),
        "audit_dir": str(args.audit_dir),
        "keep_manifest_dir": str(args.keep_manifest_dir),
        "devices": devices,
        "job_count": len(jobs),
        "total_rows": total_rows,
        "run_id": run_id,
        "prompt_version": PROMPT_VERSION,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "jobs": [asdict(job) for job in jobs],
    }
    (args.audit_dir / "run_config.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for worker_idx, (device, worker_jobs) in enumerate(assignments):
        proc = ctx.Process(
            target=worker_main,
            args=(worker_idx, device, worker_jobs, {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, progress_queue, run_id),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    active_workers = len(processes)
    shard_summaries = []
    pbar = tqdm(total=total_rows, dynamic_ncols=True, mininterval=args.progress_mininterval, desc="prefilter rows")
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
                tqdm.write(
                    f"done {Path(msg['shard']).name}: rows={summary.get('rows', 0)} keep={summary.get('kept', 0)} drop={summary.get('dropped', 0)} errors={summary.get('errors', 0)}"
                )
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
    (args.audit_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run summary: {args.audit_dir / 'run_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
