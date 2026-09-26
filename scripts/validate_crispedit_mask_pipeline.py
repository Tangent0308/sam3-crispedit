#!/usr/bin/env python3
"""Validate two-stage selection, row alignment and mask payloads after labeling."""

import argparse
import io
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from crispedit.mask.grounding_runner import load_selection, prefilter_fields
from crispedit.common import supported_shard
from crispedit.mask.selection import apply_scene, load_filters, SCENE_FIELDS


def recoverable_ground_parse_error(ground, mask):
    """Return whether a grounding parse error is safely reviewable row-wise."""

    return (
        not ground.get("ground_parse_ok", False)
        and ground.get("grounding_status") == "PARSE_ERROR"
        and ground.get("qc_flag") == "GROUND_FAIL"
        and mask.get("qc_flag") == "GROUND_FAIL"
    )


def recoverable_observation_parse_error(observation, mask):
    """An unparsed observation is recoverable only when no mask was promoted."""

    return (
        bool(observation)
        and not observation.get("parse_ok", False)
        and mask.get("qc_flag") == "GROUND_FAIL"
    )


def validate(args):
    selection = load_selection(args.selection_file)
    names = set(selection) if selection is not None else {
        p.name for p in args.input_dir.glob("*.parquet") if supported_shard(p)}
    for directory in (args.run_dir / "grounding", args.run_dir / "mask"):
        actual = {p.name for p in directory.glob("*.parquet")}
        if actual != names:
            raise ValueError(f"shard set mismatch: {directory}")
    counts, flags, types, sources = Counter(), Counter(), Counter(), Counter()
    cases = []
    for name in tqdm(sorted(names), desc="validate shards"):
        raw_count = pq.ParquetFile(args.input_dir / name).metadata.num_rows
        quality, scene = load_filters(args.quality_dir / name, args.difficulty_dir / name, raw_count)
        ground = pq.read_table(args.run_dir / "grounding" / name).to_pylist()
        masks = pq.read_table(args.run_dir / "mask" / name).to_pylist()
        wanted = selection[name] if selection is not None else list(range(raw_count))
        if [r["row_idx"] for r in ground] != wanted or [r["row_idx"] for r in masks] != wanted:
            raise ValueError(f"row alignment failed: {name}")
        for g, m in zip(ground, masks):
            index = g["row_idx"]
            expected = apply_scene(prefilter_fields(quality[index]), scene.get(index), True)
            for row in (g, m):
                for field in ["filter_decision", "prefilter_verdict", "prefilter_run_id", *[f for f,_ in SCENE_FIELDS]]:
                    if row[field] != expected[field]:
                        raise ValueError(f"selection metadata mismatch: {name}:{index}: {field}")
            counts["rows"] += 1
            counts[expected["mask_selection_reason"]] += 1
            flags[m["qc_flag"]] += 1
            if expected["filter_decision"] == "drop":
                if g["qc_flag"] != "PREFILTER_SKIP" or m["qc_flag"] != "PREFILTER_SKIP" or m["mask_png"] or m["mask_sum"]:
                    raise ValueError(f"drop row was labeled: {name}:{index}")
            else:
                types[m["raw_type"]] += 1
                sources[m["mask_source"]] += 1
                counts["usable_masks"] += m["qc_flag"] == "OK"
                counts["review_required"] += m["qc_flag"] != "OK"
                expected_side = "target" if m["canonical_type"] == "add" else "source"
                for instance in m["instance_masks"]:
                    if instance["grounding_image"] != expected_side or bool(instance["mapped_from_target"]) != (expected_side == "target"):
                        raise ValueError(f"mask canvas contract violated: {name}:{index}")
                if m["qc_flag"] == "PREFILTER_SKIP":
                    raise ValueError(f"selected row was skipped: {name}:{index}")
                ground_parse_error = not g["ground_parse_ok"]
                counts["ground_parse_errors"] += ground_parse_error
                ground_parse_recoverable = recoverable_ground_parse_error(g, m)
                counts["recoverable_ground_parse_errors"] += ground_parse_recoverable
                counts["nonrecoverable_ground_parse_errors"] += (
                    ground_parse_error and not ground_parse_recoverable
                )
                payload = json.loads(g["ground_json"])
                observation = payload.get("observation")
                counts['no_realized_changes'] += bool(payload.get('no_realized_changes'))
                if payload.get('no_realized_changes') and (m['qc_flag'] == 'OK' or m['mask_sum']):
                    raise ValueError(f'no-edit observation produced a successful mask: {name}:{index}')
                observation_parse_error = bool(observation) and not observation.get("parse_ok", False)
                counts["observation_parse_errors"] += observation_parse_error
                observation_parse_recoverable = recoverable_observation_parse_error(observation, m)
                counts["recoverable_observation_parse_errors"] += observation_parse_recoverable
                counts["nonrecoverable_observation_parse_errors"] += (
                    observation_parse_error and not observation_parse_recoverable
                )
                counts["observation_retries"] += max(0, len((observation or {}).get("attempts", []))-1)
                scope = (observation or {}).get('scope_review', {})
                counts['scope_requests'] += len(scope.get('attempts', []))
                counts['scope_corrections'] += len(scope.get('corrections', []))
                counts['scope_parse_errors'] += bool(scope) and not scope.get('parse_ok', False)
                if payload.get('scope_review_failed') and m['qc_flag'] == 'OK':
                    raise ValueError(f'failed scope review promoted to OK: {name}:{index}')
                if scope:
                    original = observation['independent_parsed']['changes']
                    revised = observation['parsed']['changes']
                    if [(c['change_id'], c['target_ref']) for c in original] != [(c['change_id'], c['target_ref']) for c in revised]:
                        raise ValueError(f'scope review changed checklist identity: {name}:{index}')
                coverage = (observation or {}).get('coverage_review', {})
                counts['coverage_requests'] += len(coverage.get('attempts', []))
                counts['coverage_parse_errors'] += bool(coverage) and not coverage.get('parse_ok')
                if payload.get('coverage_review_failed') and m['qc_flag'] == 'OK':
                    raise ValueError(f'failed coverage review promoted to OK: {name}:{index}')
                counts["unresolved_changes"] += sum(len(item.get("unresolved", [])) for item in payload.get("requests", []))
                counts["runtime_errors"] += bool(payload.get("runtime_error")) or m["qc_flag"] == "ERROR"
                for request in payload.get("requests", []):
                    for audit in request.get('evidence_verification', []):
                        counts['evidence_requests'] += len(audit.get('attempts', []))
                        counts['evidence_parse_errors'] += not audit.get('parse_ok', False)
                        counts['evidence_' + audit.get('parsed', {}).get('decision','error')] += 1
                    if request.get('evidence_issues') and m['qc_flag'] == 'OK':
                        raise ValueError(f'evidence conflict promoted to OK: {name}:{index}')
                    refinement = request.get("bbox_refinement")
                    conflicts = request.get('refinement_identity_issues', [])
                    counts['refinement_identity_conflicts'] += len(conflicts)
                    if conflicts and m['qc_flag'] == 'OK':
                        raise ValueError('identity-conflicting refinement cannot produce OK mask')
                    if refinement:
                        counts["bbox_refinement_failures"] += not refinement.get("parse_ok", False)
                        counts["retained_initial_boxes"] += sum(c.get("status") == "KEPT_ORIGINAL" for c in refinement.get("candidates", []))
                counts['source_canvas_issues'] += len(payload.get('canvas_issues', []))
                if payload.get('canvas_issues') and m['qc_flag'] == 'OK':
                    raise ValueError(f'incomplete source canvas promoted to OK: {name}:{index}')
                if m["mask_png"]:
                    image = np.asarray(Image.open(io.BytesIO(m["mask_png"]))) > 0
                    if image.shape != (g["source_height"], g["source_width"]):
                        raise ValueError(f"mask not in source coordinates: {name}:{index}")
                    if image.shape != (m["mask_height"], m["mask_width"]):
                        raise ValueError(f"mask dimensions mismatch: {name}:{index}")
                    if int(image.sum()) != m["mask_sum"] or abs(float(image.mean()) - m["area_frac"]) > 1e-8:
                        raise ValueError(f"mask counts mismatch: {name}:{index}")
                    if m["qc_flag"] == "OK" and not image.any():
                        raise ValueError(f"OK row has an empty mask: {name}:{index}")
                    for instance in m["instance_masks"]:
                        decoded = mask_utils.decode({"size": instance["rle_size"],
                                                     "counts": instance["rle_counts"].encode("ascii")})
                        if decoded.shape != image.shape or int(decoded.sum()) != instance["area"]:
                            raise ValueError(f"instance RLE mismatch: {name}:{index}")
                        counts["validated_instances"] += 1
                    counts["nonempty_masks"] += bool(image.any())
                elif m["qc_flag"] == "OK":
                    raise ValueError(f"OK row has no mask: {name}:{index}")
                if selection is not None:
                    cases.append({"shard":name,"row_idx":index,"type":m["raw_type"],
                                  "qc_flag":m["qc_flag"],"mask_source":m["mask_source"],"area_frac":m["area_frac"]})
    return {"shards":len(names),"counts":dict(counts),"flags":dict(flags),
            "selected_types":dict(types),"selected_sources":dict(sources),"cases":cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["input-dir", "quality-dir", "difficulty-dir", "run-dir"]:
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    args = parser.parse_args()
    summary = validate(args)
    (args.run_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fatal_counts = (
        "runtime_errors",
        "nonrecoverable_ground_parse_errors",
        "nonrecoverable_observation_parse_errors",
    )
    if any(summary["counts"].get(key) for key in fatal_counts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
