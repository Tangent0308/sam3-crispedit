"""Import the benchmark-v2 SA-1B source shortlist as MLLM-review candidates.

The benchmark stores independently reviewed SA-1B images and immutable dense
visible-region masks in ``sa1b_source_selection``.  This adapter preserves
that provenance while exposing the same Candidate JSONL consumed by
``sa1b_pipeline.select``.  It does not import benchmark instructions or edit
outputs: a new MLLM selection/planning pass remains responsible for training
units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .geometry import sha256
from .io import write_jsonl
from .schema import Candidate, CandidateObject


def import_sources(selected_sources: Path, asset_root: Path, output: Path, max_cases: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with selected_sources.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            image = (asset_root / row["image"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            objects = []
            for target in row.get("targets", []):
                mask = (asset_root / target["mask"]).resolve()
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                geometry = target.get("segmentation", {})
                objects.append(CandidateObject(
                    object_id=str(target.get("object_id", target.get("unit_id"))),
                    category=str(target.get("source_category") or target.get("category") or "object"),
                    description=str(target.get("source_description") or target.get("label_zh") or ""),
                    mask_path=str(mask),
                    bbox=list(target.get("bbox_xyxy") or geometry.get("bbox_xyxy") or []),
                    area_fraction=target.get("area_fraction"),
                    metadata={
                        "benchmark_unit_id": target.get("unit_id"),
                        "physical_parent": target.get("physical_parent_zh"),
                        "mask_sha256": target.get("mask_sha256"),
                        "annotation_origin": target.get("annotation_origin"),
                    },
                ))
            candidate = Candidate(
                source_id=str(row["id"]),
                image_path=str(image),
                objects=objects,
                caption=str(row.get("caption") or ""),
                tags=["benchmark_v2_source_shortlist", "sa1b"],
                provenance={
                    "source_manifest": str(selected_sources.resolve()),
                    "selection_index": row.get("selection_index"),
                    "review_image": str((asset_root / row["review_image"]).resolve()) if row.get("review_image") else None,
                    "benchmark_source_sha256": row.get("image_sha256"),
                    "selection_status": "benchmark_shortlist_requires_mllm_reselection",
                },
                image_sha256=row.get("image_sha256") or sha256(image),
            )
            rows.append(candidate.to_dict())
            if max_cases and len(rows) >= max_cases:
                break
    write_jsonl(output, rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-sources", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True,
                        help="directory containing the relative image/mask paths in selected_sources.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args(argv)
    rows = import_sources(args.selected_sources, args.asset_root, args.output, args.max_cases)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
