"""Package the four formal SAMTok edit types into one portable dataset.

Only model-pass cases are included. Source and edited PNGs are hard-linked into
the delivery directory, so the package owns its image paths without copying
large image payloads or depending on symlinks to the experiment directories.
The original manifests and audit records are linked under ``provenance``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any


TASK_TYPES = ("remove", "add", "replace", "attribute")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def one_mask(value: Any, *, case: str) -> dict[str, Any]:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{case}: expected one mask, got {len(value)}")
        value = value[0]
    if not isinstance(value, dict) or not {"size", "counts"} <= value.keys():
        raise ValueError(f"{case}: missing COCO RLE mask")
    return value


def add_link(links: dict[Path, Path], destination: Path, source: Path) -> None:
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"missing or empty source file: {source}")
    previous = links.setdefault(destination, source)
    if previous != source:
        raise ValueError(f"two different files map to {destination}: {previous} and {source}")


def hardlink(pair: tuple[Path, Path]) -> None:
    destination, source = pair
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        existing, expected = destination.stat(), source.stat()
        if (existing.st_dev, existing.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError(f"existing destination is not the expected hard link: {destination}")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_audits(path: Path) -> dict[str, dict[str, Any]]:
    audits = {}
    for row in read_jsonl(path):
        name = row["image"]
        if name in audits:
            raise ValueError(f"duplicate audit case {name} in {path}")
        audits[name] = row
    return audits


def packaged_row(
    row: dict[str, Any], audit: dict[str, Any], origin: str,
    run_root: Path, out_root: Path, links: dict[Path, Path],
) -> dict[str, Any]:
    name = row["image"]
    if Path(name).name != name or not name.endswith(".png"):
        raise ValueError(f"unsafe case image name: {name}")
    task_type = row["task_type"]
    if task_type not in TASK_TYPES or (origin == "remove") != (task_type == "remove"):
        raise ValueError(f"unexpected task type {task_type} from {origin}")
    mask = one_mask(row.get("mask"), case=name)
    source_name = row["source_image"]
    if Path(source_name).name != source_name:
        raise ValueError(f"unsafe source image name: {source_name}")
    source_file = (
        Path(row["source_path"]) if origin == "remove"
        else run_root / "data/add_replace_attribute/sources" / source_name
    )
    edited_file = Path(row["edited_path"])
    source_relative = Path("images/sources") / origin / source_name
    edited_relative = Path("images/edited") / task_type / name
    add_link(links, out_root / source_relative, source_file)
    add_link(links, out_root / edited_relative, edited_file)

    verdict = audit.get("decision") if origin == "remove" else audit.get("quality")
    if verdict != "pass":
        raise ValueError(f"{name}: model-pass manifest disagrees with audit ({verdict})")
    parsed = audit.get("parsed") if origin == "remove" else audit.get("audit")
    if not isinstance(parsed, dict) or not parsed.get("reason"):
        raise ValueError(f"{name}: missing parsed audit reason")
    original_mask = row.get("original_mask")
    if original_mask is not None:
        original_mask = one_mask(original_mask, case=name)
    instruction = row.get("editing_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{name}: empty editing instruction")
    output = {
        "case_id": name[:-4],
        "task_type": task_type,
        "editing_instruction": instruction,
        "new_instruction": row.get("new_instruction", instruction),
        "refer_object": row.get("refer_object"),
        "source_image": source_relative.as_posix(),
        "edited_image": edited_relative.as_posix(),
        "mask_rle": mask,
        "original_mask_rle": original_mask if original_mask != mask else None,
        "source_subset": row.get("source_subset"),
        "parquet_row_index": row.get("parquet_row_index"),
        "mask_index": row.get("mask_index"),
        "num_masks": row.get("num_masks"),
        "source_problem": row.get("problem"),
        "source_answer": row.get("answer"),
        "reference_binding": row.get("reference_binding"),
        "region_contract": row.get("region_contract"),
        "planning_status": row.get("resolution_status", row.get("plan_status", "accepted")),
        "planning_details": row.get("relation_plan") if origin == "remove" else {
            "masked_content": row.get("masked_content"),
            "mask_compatibility": row.get("mask_compatibility"),
            "visual_grounding": row.get("visual_grounding"),
            "mask_refinement": row.get("mask_refinement"),
        },
        "audit_status": "pass",
        "audit_reason": parsed["reason"],
        "audit_details": parsed,
        "audit_pixel_metrics": audit.get("pixel_evidence") if origin == "remove" else audit.get("locality_metrics"),
        "quality_label": "model_pass_not_human_verified",
        "provenance": {
            "origin": origin,
            "run_id": run_root.name,
            "original_case_image": name,
            "original_source_path": str(source_file),
            "original_edited_path": str(edited_file),
            "original_manifest": f"provenance/{origin}_model_pass.jsonl",
            "original_audit": f"provenance/{origin}_audit.jsonl",
        },
    }
    if row.get("sam_target_mask") is not None:
        output["sam_target_mask"] = row["sam_target_mask"]
    if row.get("execution_region") is not None:
        output["execution_region"] = row["execution_region"]
    return output


def materialize(remove_run: Path, multitype_run: Path, out_root: Path, workers: int) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    remove_run, multitype_run, out_root = remove_run.resolve(), multitype_run.resolve(), out_root.resolve()
    run_specs = (("remove", remove_run), ("multitype", multitype_run))
    for origin, run_root in run_specs:
        report = json.loads((run_root / "reports/final.json").read_text())
        if origin == "remove" and not report.get("all_nodes_completed"):
            raise ValueError("remove run has not completed")
        if origin == "multitype" and not (run_root / "attempts/resume-012/control/finalize.ok.json").is_file():
            raise ValueError("multitype resume-012 has not finalized")

    links: dict[Path, Path] = {}
    records: list[dict[str, Any]] = []
    seen_cases = set()
    source_keys = {"remove": set(), "multitype": set()}
    for origin, run_root in run_specs:
        result_root = run_root / "results"
        audits = load_audits(result_root / "audit.jsonl")
        for row in read_jsonl(result_root / "model_pass.jsonl"):
            name = row["image"]
            if name in seen_cases:
                raise ValueError(f"duplicate cross-run case: {name}")
            seen_cases.add(name)
            if name not in audits:
                raise ValueError(f"missing audit for {name}")
            records.append(packaged_row(row, audits[name], origin, run_root, out_root, links))
            source_keys[origin].add(row["source_image"])
        for name in ("model_pass.jsonl", "audit.jsonl", "all_cases.jsonl"):
            add_link(links, out_root / "provenance" / f"{origin}_{name}", result_root / name)
        add_link(links, out_root / "provenance" / f"{origin}_final.json", run_root / "reports/final.json")
    records.sort(key=lambda row: (TASK_TYPES.index(row["task_type"]), row["case_id"]))
    counts = Counter(row["task_type"] for row in records)
    expected = {"remove": 7990, "add": 6269, "replace": 5016, "attribute": 8682}
    if dict(counts) != expected:
        raise ValueError(f"unexpected official model-pass counts: {dict(counts)} != {expected}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Hard-linking {len(links)} distinct files with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(hardlink, pair): pair for pair in links.items()}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 2000 == 0 or index == len(futures):
                print(f"linked {index}/{len(futures)}", flush=True)

    atomic_jsonl(out_root / "manifest.jsonl", records)
    for task_type in TASK_TYPES:
        atomic_jsonl(out_root / "by_type" / task_type / "manifest.jsonl",
                     [row for row in records if row["task_type"] == task_type])
    summary = {
        "status": "complete",
        "dataset_version": "combined-v1",
        "quality_label": "model_pass_not_human_verified",
        "total_cases": len(records),
        "task_type_counts": dict(counts),
        "unique_source_images_by_origin": {key: len(value) for key, value in source_keys.items()},
        "hardlinked_files": len(links),
        "image_storage": "regular hard links; independent of experiment path removal, shared bytes until modified",
        "source_parquet": "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet",
        "origin_runs": {origin: str(path) for origin, path in run_specs},
        "manifest": "manifest.jsonl",
        "mask_format": "COCO compressed RLE in mask_rle (height, width order)",
    }
    atomic_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-run", type=Path, required=True)
    parser.add_argument("--multitype-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    materialize(args.remove_run, args.multitype_run, args.out_root, args.workers)


if __name__ == "__main__":
    main()
