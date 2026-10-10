import json
from pathlib import Path

import pytest
from PIL import Image

from sa1b_pipeline.geometry import sha256
from sa1b_pipeline.io import read_jsonl, write_jsonl
from sa1b_pipeline.source_pool import build_pool
from sa1b_pipeline.training import materialize


def source(root, name, color):
    directory = root / name
    directory.mkdir(parents=True)
    Image.new("RGB", (8, 8), color).save(directory / "image.jpg")
    (directory / "dense.json").write_text(json.dumps({
        "key": name, "objects": [{"object_id": i, "category": "cup", "gar_caption": "a cup",
                                  "segmentation": {"size": [8, 8], "counts": "dummy"}}
                                 for i in range(2)]}))
    return directory / "image.jpg"


def test_pool_excludes_id_and_renamed_identical_image(tmp_path):
    cache = tmp_path / "cache"
    heldout = source(cache, "sa_1", "red")
    source(cache, "sa_2", "red")  # Different ID, exact same image bytes.
    source(cache, "sa_3", "blue")
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl(benchmark, [{"source": {"image_id": "sa_1", "sha256": sha256(heldout)}}])
    output = tmp_path / "pool.jsonl"
    report = build_pool(cache, benchmark, output)
    assert report["candidate_images"] == 1
    assert [row["source_id"] for row in read_jsonl(output)] == ["sa_3"]
    assert {row["reason"] for row in read_jsonl(output.with_suffix(".exclusions.jsonl"))} == {
        "benchmark_source_id", "benchmark_image_hash"}
    assert read_jsonl(output)[0]["objects"][0]["description"] == "a cup"


def test_training_rejects_benchmark_calibration_before_writing(tmp_path):
    pairs, codes, output = [tmp_path / name for name in ("pairs.jsonl", "codes.jsonl", "out.jsonl")]
    write_jsonl(pairs, [{"source_provenance": {"selection_status": "benchmark_shortlist_requires_mllm_reselection"}}])
    write_jsonl(codes, [])
    with pytest.raises(ValueError, match="calibration"):
        materialize(pairs, codes, output)
    assert not output.exists()


def test_training_hash_gate_does_not_trust_claimed_id(tmp_path):
    image = source(tmp_path / "cache", "sa_1", "red")
    benchmark, pairs, codes, output = [tmp_path / name for name in (
        "benchmark.jsonl", "pairs.jsonl", "codes.jsonl", "out.jsonl")]
    write_jsonl(benchmark, [{"source": {"image_id": "sa_1", "sha256": sha256(image)}}])
    write_jsonl(pairs, [{"case_id": "sa1b-renamed", "source_id": "renamed", "source_image": str(image)}])
    write_jsonl(codes, [])
    with pytest.raises(ValueError, match="leakage"):
        materialize(pairs, codes, output, benchmark)
    assert not output.exists()
