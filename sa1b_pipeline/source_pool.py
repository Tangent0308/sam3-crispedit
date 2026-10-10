"""Import retrieved SA-1B dense records while excluding frozen evaluation images.

The retrieval cache is a source of raw SA-1B images and annotations, not an
acceptance list. Every retained candidate still requires MLLM visual review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry import sha256
from .io import read_jsonl, write_jsonl


def benchmark_keys(manifest: Path) -> tuple[set[str], set[str]]:
    ids, hashes = set(), set()
    for row in read_jsonl(manifest):
        source = row.get("source", row)
        for key in ("source_id", "image_id", "family_id", "id"):
            if source.get(key):
                ids.add(str(source[key]).removeprefix("v2-").removeprefix("sa1b-"))
        for key in ("image_sha256", "sha256"):
            if source.get(key):
                hashes.add(source[key])
    if not ids or not hashes:
        raise ValueError("benchmark exclusion manifest must contain source IDs and image hashes")
    return ids, hashes


def build_pool(cache: Path, benchmark: Path, output: Path, sample_index: Path | None = None) -> dict:
    excluded_ids, excluded_hashes = benchmark_keys(benchmark)
    benchmark_digest = sha256(benchmark)
    index = json.loads(sample_index.read_text()).get("selection", {}) if sample_index else {}
    accepted, rejected = [], []
    seen = set()
    for directory in sorted(cache.iterdir()):
        if not directory.is_dir():
            continue
        image, dense = directory / "image.jpg", directory / "dense.json"
        source_id = directory.name
        reason = None
        if source_id in excluded_ids:
            reason = "benchmark_source_id"
        elif not image.is_file() or not dense.is_file():
            reason = "missing_image_or_dense"
        if reason:
            rejected.append({"source_id": source_id, "reason": reason})
            continue
        digest = sha256(image)
        if digest in excluded_hashes or digest in seen:
            rejected.append({"source_id": source_id, "reason": "benchmark_image_hash" if digest in excluded_hashes else "duplicate_image_hash"})
            continue
        record = json.loads(dense.read_text())
        if record.get("key") != source_id:
            raise ValueError(f"dense/image identity mismatch at {directory}")
        objects = [{"object_id": str(obj["object_id"]), "category": obj.get("category", "object"),
                    "description": obj.get("gar_caption", ""), "rle": obj["segmentation"],
                    "bbox": obj.get("bbox"), "metadata": {"score": obj.get("score")}}
                   for obj in record.get("objects", []) if obj.get("segmentation")]
        if len(objects) < 2:
            rejected.append({"source_id": source_id, "reason": "fewer_than_two_masks"})
            continue
        seen.add(digest)
        accepted.append({"source_id": source_id, "image_path": str(image.resolve()), "image_sha256": digest,
                         "caption": record.get("caption", ""),
                         "tags": [tag.strip() for tag in record.get("merged_tags", "").split(",") if tag.strip()],
                         "objects": objects, "provenance": {
                             "dense_record": str(dense.resolve()), "dense_sha256": sha256(dense),
                             "tfrecord_index": index.get(source_id),
                             "benchmark_exclusion_manifest": str(benchmark.resolve()),
                             "benchmark_exclusion_sha256": benchmark_digest,
                             "selection_status": "unreviewed_requires_mllm_selection"}})
    write_jsonl(output, accepted)
    write_jsonl(output.with_suffix(".exclusions.jsonl"), rejected)
    report = {"candidate_images": len(accepted), "candidate_masks": sum(len(r["objects"]) for r in accepted),
              "excluded": len(rejected), "benchmark_source_ids": len(excluded_ids),
              "benchmark_id_overlap": sum(r["reason"] == "benchmark_source_id" for r in rejected),
              "benchmark_byte_hash_overlap": sum(r["reason"] == "benchmark_image_hash" for r in rejected),
              "near_duplicate_check": "not_performed", "manifest": str(output)}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--exclude-benchmark", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_pool(args.cache, args.exclude_benchmark, args.output, args.sample_index)))


if __name__ == "__main__":
    main()
