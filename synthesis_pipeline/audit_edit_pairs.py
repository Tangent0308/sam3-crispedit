"""Audit edited pairs visually and measure whether pixel changes stay localized."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.vlm_utils as vlm
from synthesis_pipeline.visual_prompt_utils import audit_visual_inputs


COMMON_AUDIT_PROMPT = """Judge one localized image edit as PASS or FAIL. Use a practical training-data standard: pass a broadly correct, natural-looking edit; fail a clear error or an edit that is too weak to verify.

IMAGE 1 is a tight aligned SOURCE/EDITED comparison. In both halves the SAME thin black/white edge traces the exact target footprint; judge only photographic content INSIDE that edge when deciding whether the selected instance remains. Same-category content outside the edge is a distractor that should normally remain. The black/white line is geometry, not an object color. IMAGE 2 shows, from left to right, a wider aligned SOURCE crop, black/white target mask, aligned EDITED crop, and a false-color absolute-difference map whose white contour is the target boundary. Red in the difference map means strong change and blue means weak change; it is not an object color. IMAGE 3 shows FULL SOURCE beside FULL EDITED. No semantic-looking mask color is painted over photographic pixels.

Requested edit: <<<INSTRUCTION>>>
Named target: <<<TARGETS>>>
Planner's literal masked-content inventory: <<<MASKED_CONTENT>>>
Edit type: <<<TASK_TYPE>>>

Type-specific criteria:
<<<TYPE_CRITERIA>>>

Write `source_inventory` from photographic content inside the SOURCE edge in IMAGE 1 alone before reasoning about the request, and `edited_inventory` from content inside the EDITED edge. Exclude nearby objects outside the edge and anything visible only in IMAGE 2 context or IMAGE 3. Then independently verify both the Named target and planner inventory; if either points to a nearby object outside the SOURCE edge, record source_mismatch. Never infer appearance from the edge, binary shape, difference colors, or instruction. A pixel difference alone is not proof: the requested result must be recognizable in FULL EDITED at useful training scale.

Then fill five failure slots. Each slot must be JSON null only after visually verifying that failure is absent; otherwise write concise visible evidence:
- source_mismatch: a pre-edit descriptor/precondition in the instruction is false in SOURCE, or requested added content already exists.
- completion_failure: requested semantic change is missing, too weak to recognize, partial, or leaves old-target residue.
- target_or_count_failure: wrong instance/part changed, or requested count/extent is not satisfied.
- dependency_failure: the result leaves an obviously floating or impossible attached/held/supported object, or incorrectly removes such content outside the requested scope.
- preservation_or_artifact_failure: a non-target instance or scene structure changes materially, or the edit has visible residue, duplicated parts, halos, hard cutout edges, mask-shaped stains, blur patches, texture discontinuities, implausible fill, overlaps, or malformed content. Small natural rendering variation is acceptable.

Compare counts of repeated scene objects before and after, including any unrequested object newly appearing inside the edited footprint. There is no review class. If the edit is ambiguous, difficult to see, or impossible to verify, use `fail` with the relevant evidence. Use `pass` when the requested change is clearly present and broadly natural even if it is not pixel-perfect. Any non-null failure slot requires `fail`. Do not merely restate the request as evidence.

Return one compact JSON object only with exactly these keys:
`source_inventory` (string), `edited_inventory` (string), `source_mismatch` (null or string), `completion_failure` (null or string), `target_or_count_failure` (null or string), `dependency_failure` (null or string), `preservation_or_artifact_failure` (null or string), `quality` (`pass` or `fail`), and `reason` (string)."""

TYPE_CRITERIA = {
    "add": (
        "Verify that the new item is absent in SOURCE, clearly recognizable at the "
        "requested anchor in EDITED, and visually integrated rather than pasted on."
    ),
    "remove": (
        "First compare content inside the two aligned edges in IMAGE 1. Judge "
        "disappearance at that exact footprint, not by asking whether any "
        "same-category object remains elsewhere in the crop or full image. A similar "
        "instance whose SOURCE pixels are outside IMAGE 1 should remain and is not "
        "target residue. The requested instance and all of its masked visible extent "
        "must disappear. "
        "If a same-shaped person/object, recognizable face, clothing, silhouette, or "
        "pose remains at that location, the target was altered rather than removed. "
        "Fail residue, ghost parts, duplicate remnants, floating dependencies, or an "
        "unnatural reconstruction behind it."
    ),
    "replace": (
        "The old identity must be gone and the new identity clearly recognizable at "
        "the same location. Fail mixtures, recolors posing as replacement, residue, "
        "or a new object with visibly pasted or incoherent boundaries."
    ),
    "attribute": (
        "The requested attribute must visibly change on the intended masked content "
        "without changing its identity or spilling across its boundary. Fail dirty "
        "edges, partial recoloring, shape replacement, or changes to another instance."
    ),
}

AUDIT_FAILURE_KEYS = (
    "source_mismatch",
    "completion_failure",
    "target_or_count_failure",
    "dependency_failure",
    "preservation_or_artifact_failure",
)

AUDIT_VERSION = "edit_pair_exact_edge_magnified_v15_cached"

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}

MINIMUM_INSIDE_CHANGED_FRACTION = {
    "add": 0.08,
    "remove": 0.75,
    "replace": 0.45,
    "attribute": 0.10,
}
MAXIMUM_OUTSIDE_GUARD_CHANGED_FRACTION = 0.12
HARD_MINIMUM_REMOVE_CHANGED_FRACTION = 0.50
HIGH_CONFIDENCE_REMOVE_CHANGED_FRACTION = 0.95
MAXIMUM_REMOVE_OVERRIDE_OUTSIDE_FRACTION = 0.05
UNCHANGED_REMOVE_CLAIM_PATTERN = re.compile(
    r"\b(?:not (?:performed|executed|applied|completed)|no change|identical|"
    r"same (?:content|object|person|target)|remain\w*(?:\s+\w+){0,3}\s+visible)\b",
    flags=re.IGNORECASE,
)
DEPENDENT_OBJECT_TERMS = {
    "backpack",
    "bag",
    "bicycle",
    "bike",
    "cart",
    "leash",
    "stroller",
    "suitcase",
    "umbrella",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--guard-pixels", type=int, default=24)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--vlm",
        choices=[
            "gemma4",
            "qwen8b",
            "qwen4b",
            "qwen8b-vllm",
            "qwen4b-vllm",
            "qwen35",
        ],
        default="qwen8b-vllm",
    )
    parser.add_argument("--vlm-model-id", default=None)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--vlm-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse audit rows whose source, edited image, annotation, prompt, "
            "and model settings have the same content fingerprint."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(
    row: dict[str, Any], source_path: Path, edited_path: Path, args: argparse.Namespace
) -> str:
    payload = {
        "audit_version": AUDIT_VERSION,
        "audit_prompt": build_audit_prompt(row),
        "image": row.get("image"),
        "source_image": row.get("source_image"),
        "editing_instruction": row.get("editing_instruction"),
        "refer_object": row.get("refer_object"),
        "planning_visual_input": row.get("planning_visual_input"),
        "mask": row.get("mask"),
        "source_sha256": file_sha256(source_path),
        "edited_sha256": file_sha256(edited_path),
        "guard_pixels": args.guard_pixels,
        "minimum_inside_changed_fraction": MINIMUM_INSIDE_CHANGED_FRACTION,
        "maximum_outside_guard_changed_fraction": (
            MAXIMUM_OUTSIDE_GUARD_CHANGED_FRACTION
        ),
        "hard_minimum_remove_changed_fraction": HARD_MINIMUM_REMOVE_CHANGED_FRACTION,
        "high_confidence_remove_changed_fraction": (
            HIGH_CONFIDENCE_REMOVE_CHANGED_FRACTION
        ),
        "maximum_remove_override_outside_fraction": (
            MAXIMUM_REMOVE_OVERRIDE_OUTSIDE_FRACTION
        ),
        "vlm": args.vlm,
        "vlm_model_id": args.vlm_model_id,
        "vlm_dtype": args.vlm_dtype,
        "max_new_tokens": args.max_new_tokens,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = str(text).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_polygons(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if all(isinstance(item, (int, float)) for item in value):
        return [[float(item) for item in value]] if len(value) >= 6 else []
    return [
        [float(number) for number in polygon]
        for polygon in value
        if isinstance(polygon, list)
        and len(polygon) >= 6
        and all(isinstance(number, (int, float)) for number in polygon)
    ]


def decode_coco_rle(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, dict) or "counts" not in value or "size" not in value:
        return None
    rle = {"counts": value["counts"], "size": value["size"]}
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def mask_array(size: tuple[int, int], masks: Any) -> np.ndarray:
    values = masks if isinstance(masks, list) else [masks]
    rle_canvas = np.zeros((size[1], size[0]), dtype=bool)
    found_rle = False
    for value in values:
        decoded = decode_coco_rle(value)
        if decoded is None:
            continue
        found_rle = True
        if decoded.shape != rle_canvas.shape:
            decoded = cv2.resize(
                decoded.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        rle_canvas |= decoded
    if found_rle:
        return rle_canvas

    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    if isinstance(values, list):
        for mask in values:
            for polygon in normalize_polygons(mask):
                draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=255)
    return np.asarray(canvas, dtype=np.uint8) > 0


def locality_metrics(
    source: Image.Image, edited: Image.Image, mask: np.ndarray, guard_pixels: int
) -> dict[str, float]:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    if edited.size != source.size:
        edited = edited.resize(source.size, Image.Resampling.LANCZOS)
    edited_array = np.asarray(edited.convert("RGB"), dtype=np.float32) / 255.0
    delta = np.abs(source_array - edited_array).mean(axis=2)
    kernel_size = max(1, guard_pixels * 2 + 1)
    dilated = cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    ).astype(bool)
    outside = ~dilated

    def mean_where(selection: np.ndarray) -> float:
        return float(delta[selection].mean()) if selection.any() else 0.0

    def changed_where(selection: np.ndarray) -> float:
        return float((delta[selection] >= 0.05).mean()) if selection.any() else 0.0

    return {
        "mask_area_fraction": round(float(mask.mean()), 8),
        "inside_mean_abs_diff": round(mean_where(mask), 8),
        "inside_changed_fraction": round(changed_where(mask), 8),
        "outside_guard_mean_abs_diff": round(mean_where(outside), 8),
        "outside_guard_changed_fraction": round(changed_where(outside), 8),
        "global_mean_abs_diff": round(float(delta.mean()), 8),
    }


def normalized_task_type(row: dict[str, Any]) -> str:
    task_type = str(row.get("task_type", "")).strip().lower()
    if task_type not in TYPE_CRITERIA:
        raise ValueError(f"Unsupported task_type for audit: {task_type!r}")
    return task_type


def refer_object_text(row: dict[str, Any]) -> str:
    value = row.get("refer_object", [])
    values = value if isinstance(value, list) else [value]
    return "; ".join(str(item) for item in values if str(item).strip())


def build_audit_prompt(row: dict[str, Any]) -> str:
    task_type = normalized_task_type(row)
    prompt = COMMON_AUDIT_PROMPT.replace(
        "INSTRUCTION", str(row.get("editing_instruction", ""))
    )
    targets = refer_object_text(row)
    return (
        prompt.replace("TARGETS", targets)
        .replace("MASKED_CONTENT", str(row.get("masked_content", "not provided")))
        .replace("TASK_TYPE", task_type)
        .replace("TYPE_CRITERIA", TYPE_CRITERIA[task_type])
    )


def audit_messages(
    source_localization: Image.Image,
    detail_comparison: Image.Image,
    full_comparison: Image.Image,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_localization},
                {"type": "image", "image": detail_comparison},
                {"type": "image", "image": full_comparison},
                {"type": "text", "text": build_audit_prompt(row)},
            ],
        }
    ]


def normalize_audit_result(value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Validate failure-first evidence and enforce a binary verdict."""
    if not value:
        return None
    normalized = dict(value)
    model_quality = str(normalized.get("quality", "")).strip().lower()
    if model_quality not in {"pass", "fail"}:
        return None
    for key in ("source_inventory", "edited_inventory", "reason"):
        if not str(normalized.get(key, "")).strip():
            return None
    detected_failures = []
    for key in AUDIT_FAILURE_KEYS:
        if key not in normalized:
            return None
        evidence = normalized.get(key)
        if evidence is None:
            continue
        if not isinstance(evidence, str):
            return None
        stripped = evidence.strip()
        if not stripped or stripped.lower() in {"none", "null", "n/a", "no"}:
            normalized[key] = None
            continue
        normalized[key] = stripped
        detected_failures.append(key)
    normalized["model_quality"] = model_quality
    normalized["failure_tags"] = detected_failures
    # A generated `pass` must never override explicit failure evidence.
    normalized["quality"] = "fail" if detected_failures else model_quality
    return normalized


def deterministic_diagnostic_warnings(
    row: dict[str, Any], metrics: dict[str, float], audit: Optional[dict[str, Any]]
) -> list[str]:
    """Record pixel-level diagnostics without overriding the visual verdict."""
    task_type = normalized_task_type(row)
    inside_changed = metrics["inside_changed_fraction"]
    outside_changed = metrics["outside_guard_changed_fraction"]
    warnings = []
    minimum_inside_change = MINIMUM_INSIDE_CHANGED_FRACTION[task_type]
    if inside_changed < minimum_inside_change:
        warnings.append(f"weak_{task_type}_pixel_change")
    if outside_changed > MAXIMUM_OUTSIDE_GUARD_CHANGED_FRACTION:
        warnings.append("broad_change_outside_target_guard")

    # A color used by refer_object describes the source target, not the desired
    # result. Keep disagreements visible for analysis, but let the edge-aware
    # visual audit decide the binary verdict.
    if audit:
        refer_text = refer_object_text(row)
        refer_colors = {
            token
            for token in re.findall(r"[a-z]+", refer_text.lower())
            if token in COLOR_WORDS
        }
        source_tokens = set(
            re.findall(r"[a-z]+", str(audit.get("source_inventory", "")).lower())
        )
        normalized_source_colors = {
            "gray" if token == "grey" else token for token in source_tokens
        }
        normalized_refer_colors = {
            "gray" if token == "grey" else token for token in refer_colors
        }
        planning_visual_input = str(row.get("planning_visual_input", ""))
        if planning_visual_input.startswith("clean_crop_cutout_binary"):
            for color in sorted(normalized_refer_colors - normalized_source_colors):
                warnings.append(f"source_color_not_confirmed:{color}")
        elif "red" in normalized_refer_colors:
            warnings.append("legacy_red_source_descriptor_unconfirmed")

        if task_type == "remove" and outside_changed < 0.05:
            source_terms = source_tokens & DEPENDENT_OBJECT_TERMS
            instruction_tokens = set(
                re.findall(
                    r"[a-z]+", str(row.get("editing_instruction", "")).lower()
                )
            )
            for term in sorted(source_terms - instruction_tokens):
                warnings.append(f"dependent_scope_unverified:{term}")
    return warnings


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(array.min()), 8),
        "p25": round(float(np.quantile(array, 0.25)), 8),
        "median": round(float(np.median(array)), 8),
        "p75": round(float(np.quantile(array, 0.75)), 8),
        "max": round(float(array.max()), 8),
    }


def main() -> None:
    audit_started = time.perf_counter()
    args = parse_args()
    rows = load_jsonl(args.annotations_jsonl)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    existing_path = args.out_dir / "edit_audit.jsonl"
    existing_rows = load_jsonl(existing_path) if args.resume and existing_path.exists() else []
    existing_by_image = {
        str(row.get("image")): row
        for row in existing_rows
        if isinstance(row.get("image"), str) and row.get("input_fingerprint")
    }
    tasks = []
    output_by_image: dict[str, dict[str, Any]] = {}
    reused_cases = 0
    backend_load_seconds = 0.0
    inference_seconds = 0.0
    for row in rows:
        image_name = str(row.get("image", ""))
        source_path = args.source_dir / str(row.get("source_image") or image_name)
        edited_path = args.edited_dir / image_name
        fingerprint = input_fingerprint(row, source_path, edited_path, args)
        existing = existing_by_image.get(image_name)
        if existing and existing.get("input_fingerprint") == fingerprint:
            output_by_image[image_name] = existing
            reused_cases += 1
            continue
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")
        with Image.open(edited_path) as handle:
            edited = handle.convert("RGB")
        mask = mask_array(source.size, row.get("mask", []))
        source_localization, detail_comparison, full_comparison = audit_visual_inputs(
            source, edited, mask
        )
        tasks.append(
            {
                "row": row,
                "source": source,
                "edited": edited,
                "source_localization": source_localization,
                "detail_comparison": detail_comparison,
                "full_comparison": full_comparison,
                "metrics": locality_metrics(source, edited, mask, args.guard_pixels),
                "input_fingerprint": fingerprint,
            }
        )

    if tasks:
        vlm.configure_backend(
            name=args.vlm,
            model_id=args.vlm_model_id,
            device=args.vlm_device,
            dtype=args.vlm_dtype,
        )
        backend_started = time.perf_counter()
        backend = vlm.get_backend()
        backend_load_seconds = time.perf_counter() - backend_started
        batch_size = max(1, args.batch_size)
        for start in tqdm(range(0, len(tasks), batch_size), desc="edit-pair audit"):
            batch = tasks[start : start + batch_size]
            inference_started = time.perf_counter()
            outputs = backend.chat_batch(
                [
                    audit_messages(
                        task["source_localization"],
                        task["detail_comparison"],
                        task["full_comparison"],
                        task["row"],
                    )
                    for task in batch
                ],
                max_new_tokens=args.max_new_tokens,
            )
            inference_seconds += time.perf_counter() - inference_started
            for task, raw in zip(batch, outputs):
                parsed = normalize_audit_result(parse_json_object(raw))
                quality = (
                    str(parsed.get("quality", "parse_error"))
                    if parsed
                    else "parse_error"
                )
                metric_warnings = deterministic_diagnostic_warnings(
                    task["row"], task["metrics"], parsed
                )
                if parsed:
                    parsed["metric_warnings"] = metric_warnings
                    if (
                        normalized_task_type(task["row"]) == "remove"
                        and task["metrics"]["inside_changed_fraction"]
                        < HARD_MINIMUM_REMOVE_CHANGED_FRACTION
                    ):
                        evidence = (
                            "Too little of the masked target changed for a complete "
                            "removal; the target likely remains or was only altered."
                        )
                        if not parsed.get("completion_failure"):
                            parsed["completion_failure"] = evidence
                        if "completion_failure" not in parsed["failure_tags"]:
                            parsed["failure_tags"].append("completion_failure")
                        parsed["quality"] = "fail"
                        parsed["reason"] = (
                            str(parsed.get("reason", "")).rstrip(". ")
                            + ". "
                            + evidence
                        ).strip()
                        quality = "fail"
                    elif (
                        normalized_task_type(task["row"]) == "remove"
                        and task["metrics"]["inside_changed_fraction"]
                        >= HIGH_CONFIDENCE_REMOVE_CHANGED_FRACTION
                        and task["metrics"]["outside_guard_changed_fraction"]
                        <= MAXIMUM_REMOVE_OVERRIDE_OUTSIDE_FRACTION
                        and parsed.get("failure_tags") == ["completion_failure"]
                        and UNCHANGED_REMOVE_CLAIM_PATTERN.search(
                            str(parsed.get("completion_failure", ""))
                        )
                    ):
                        # The exact-edge VLM occasionally calls an almost entirely
                        # changed footprint "unchanged" after noticing a preserved
                        # same-class distractor. Resolve only that narrow, measurable
                        # contradiction; residue/artifact/count failures still fail.
                        changed = task["metrics"]["inside_changed_fraction"]
                        parsed["completion_failure"] = None
                        parsed["failure_tags"] = []
                        parsed["quality"] = "pass"
                        parsed["verdict_override"] = (
                            "high_pixel_change_contradicts_unchanged_claim"
                        )
                        parsed["reason"] = (
                            "Passed by the remove contradiction guard: the model's "
                            f"unchanged claim conflicts with {changed:.1%} changed "
                            "pixels inside the exact target and no other failure was "
                            "reported."
                        )
                        quality = "pass"
                output_by_image[str(task["row"].get("image", ""))] = {
                    "image": task["row"].get("image"),
                    "editing_instruction": task["row"].get("editing_instruction"),
                    "refer_object": task["row"].get("refer_object"),
                    "quality": quality,
                    "audit": parsed,
                    "locality_metrics": task["metrics"],
                    "raw_response": raw,
                    "input_fingerprint": task["input_fingerprint"],
                }
        vlm.shutdown_backend()

    output_rows = [output_by_image[str(row.get("image", ""))] for row in rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "edit_audit.jsonl", output_rows)
    metric_names = list(output_rows[0]["locality_metrics"]) if output_rows else []
    summary = {
        "cases": len(output_rows),
        "quality_counts": dict(sorted(Counter(row["quality"] for row in output_rows).items())),
        "metric_warning_counts": dict(
            sorted(
                Counter(
                    warning
                    for row in output_rows
                    for warning in (row.get("audit") or {}).get("metric_warnings", [])
                ).items()
            )
        ),
        "metric_quantiles": {
            name: quantiles([row["locality_metrics"][name] for row in output_rows])
            for name in metric_names
        },
        "guard_pixels": args.guard_pixels,
        "minimum_inside_changed_fraction": MINIMUM_INSIDE_CHANGED_FRACTION,
        "maximum_outside_guard_changed_fraction": (
            MAXIMUM_OUTSIDE_GUARD_CHANGED_FRACTION
        ),
        "hard_minimum_remove_changed_fraction": HARD_MINIMUM_REMOVE_CHANGED_FRACTION,
        "high_confidence_remove_changed_fraction": (
            HIGH_CONFIDENCE_REMOVE_CHANGED_FRACTION
        ),
        "maximum_remove_override_outside_fraction": (
            MAXIMUM_REMOVE_OVERRIDE_OUTSIDE_FRACTION
        ),
        "verdict_override_counts": dict(
            sorted(
                Counter(
                    str(row.get("audit", {}).get("verdict_override"))
                    for row in output_rows
                    if row.get("audit", {}).get("verdict_override")
                ).items()
            )
        ),
        "audit_version": AUDIT_VERSION,
        "vlm_calls": len(tasks),
        "calls_per_audited_case": 1.0 if tasks else 0.0,
        "audited_cases": len(tasks),
        "reused_cases": reused_cases,
        "resume": args.resume,
        "backend_load_seconds": round(backend_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "inference_cases_per_minute": (
            round(len(tasks) / inference_seconds * 60.0, 3)
            if inference_seconds > 0
            else 0.0
        ),
        "audit_wall_seconds": round(time.perf_counter() - audit_started, 3),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
