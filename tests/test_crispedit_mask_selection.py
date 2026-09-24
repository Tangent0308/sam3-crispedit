import argparse
import io
import json
from queue import Queue

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from crispedit.mask import grounding_runner as grounding
from crispedit.mask import runner as mask
from crispedit.mask.artifacts import reusable_table, signature, signed_schema, check_output_location
from crispedit.mask.selection import apply_scene, load_filters


def fixtures(tmp_path):
    quality = [
        {"row_idx": i, "prefilter_verdict": "PASS" if i else "FAIL",
         "filter_decision": "keep" if i else "drop", "prefilter_run_id": "quality-run"}
        for i in range(3)
    ]
    scene = [
        {"row_idx": i, "source_prefilter_verdict": "PASS", "source_prefilter_run_id": "quality-run",
         "scene_decision": "PASS" if i == 2 else "DROP", "scene_pass": i == 2,
         "scene_parse_ok": True, "scene_run_id": "scene-run", "scene_reason": "selected instance"}
        for i in (1, 2)
    ]
    qp, sp = tmp_path / "quality.parquet", tmp_path / "scene.parquet"
    pq.write_table(pa.Table.from_pylist(quality), qp)
    pq.write_table(pa.Table.from_pylist(scene), sp)
    return qp, sp, quality, scene


def test_join_preserves_quality_and_filters_scene(tmp_path):
    qp, sp, _, _ = fixtures(tmp_path)
    q, s = load_filters(qp, sp, 3)
    joined = [apply_scene(grounding.prefilter_fields(q[i]), s.get(i), True) for i in range(3)]
    assert [r["filter_decision"] for r in joined] == ["drop", "drop", "keep"]
    assert [r["mask_selection_reason"] for r in joined] == ["QUALITY_DROP", "SCENE_DROP", "SELECTED"]
    assert joined[1]["prefilter_verdict"] == "PASS"
    assert mask._copy_metadata(joined[2])["scene_run_id"] == "scene-run"


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "wrong_run", "conflicting_pass", "unparsed_pass", "quality_missing", "quality_conflict"])
def test_invalid_manifests_fail_closed(tmp_path, defect):
    qp, sp, q, s = fixtures(tmp_path)
    if defect == "missing": s.pop()
    if defect == "extra": s.append({**s[0], "row_idx": 0})
    if defect == "duplicate": s.append(s[0])
    if defect == "wrong_run": s[0]["source_prefilter_run_id"] = "other"
    if defect == "conflicting_pass": s[0]["scene_pass"] = True
    if defect == "unparsed_pass": s[1]["scene_parse_ok"] = False
    if defect == "quality_missing": q.pop()
    if defect == "quality_conflict": q[0]["filter_decision"] = "keep"
    pq.write_table(pa.Table.from_pylist(q), qp)
    pq.write_table(pa.Table.from_pylist(s), sp)
    with pytest.raises(ValueError): load_filters(qp, sp, 3)


def test_process_only_double_pass_and_propagates_skip_metadata(tmp_path):
    qp, sp, _, _ = fixtures(tmp_path)
    img = io.BytesIO()
    Image.new("RGB", (16, 12)).save(img, format="PNG")
    records = [{"type": "remove", "instruction": f"remove instance {i}",
                "input_img": {"bytes": img.getvalue()}, "output_img": {"bytes": img.getvalue()}} for i in range(3)]
    raw, out = tmp_path / "raw.parquet", tmp_path / "out.parquet"
    pq.write_table(pa.Table.from_pylist(records), raw)
    job = grounding.GroundingJob("remove", str(raw), str(out), str(qp), None, 3, str(sp), 1)
    args = argparse.Namespace(model_path="test-model", batch_size=3, grounding_mode="two-pass", fail_fast=True, compression="zstd")
    class FakeGrounder:
        prompt_version = "test"
        def infer(self, samples):
            assert [s["instruction"] for s in samples] == ["remove instance 2"]
            return [{"boxes": {"source": [{"ref": "instance", "bbox_2d": [0,0,1000,1000]}], "target": []}, "requests": []}]
    result = grounding.process_job(job, FakeGrounder(), args, Queue(), 0)
    assert result["rows"] == 3 and result["prefilter_skipped"] == 2
    table = pq.read_table(out)
    assert table["row_idx"].to_pylist() == [0,1,2]
    assert table["scene_decision"].to_pylist() == ["", "DROP", "PASS"]
    assert grounding.summarize_grounding(table)["prefilter_skipped"] == 2


def test_resume_checks_identity_rows_schema_and_errors(tmp_path):
    raw = tmp_path / "raw"
    raw.write_bytes(b"raw")
    side = tmp_path / "manifest"
    side.write_bytes(b"PASS")
    path = tmp_path / "out.parquet"
    schema = pa.schema([("row_idx", pa.int64()), ("qc_flag", pa.string())])
    digest = signature(raw, [side], {"mode": "current"})
    def save(indices, flags):
        pq.write_table(pa.Table.from_pylist([dict(row_idx=i, qc_flag=f) for i,f in zip(indices, flags)],
                                           schema=signed_schema(schema,digest)), path)
    save([1,3], ["OK", "PREFILTER_SKIP"])
    assert reusable_table(path,digest,schema,[1,3]) is not None
    assert reusable_table(path,digest,schema,[1,2,3]) is None
    side.write_bytes(b"DROP")
    assert reusable_table(path,signature(raw,[side],{"mode":"current"}),schema,[1,3]) is None
    save([1,3],["OK","ERROR"])
    assert reusable_table(path,digest,schema,[1,3]) is None
    path.write_bytes(b"interrupted parquet")
    assert reusable_table(path,digest,schema,[1,3]) is None


def test_output_cannot_overwrite_inputs(tmp_path):
    with pytest.raises(ValueError): check_output_location(tmp_path, tmp_path / "raw")
    with pytest.raises(ValueError): check_output_location(tmp_path / "raw" / "out", tmp_path / "raw")
    check_output_location(tmp_path / "out", tmp_path / "raw")


def test_resume_does_not_reuse_unparsed_grounding(tmp_path):
    path = tmp_path / "ground.parquet"
    schema = pa.schema([("row_idx", pa.int64()), ("qc_flag", pa.string()), ("ground_parse_ok", pa.bool_())])
    def save(flag, parsed):
        table = pa.Table.from_pylist([{"row_idx": 0, "qc_flag": flag, "ground_parse_ok": parsed}],
                                    schema=signed_schema(schema, "digest"))
        pq.write_table(table, path)
    save("GROUND_FAIL", False)
    assert reusable_table(path, "digest", schema, [0]) is None
    save("PREFILTER_SKIP", False)
    assert reusable_table(path, "digest", schema, [0]) is not None
    save("OK", True)
    assert reusable_table(path, "digest", schema, [0]) is not None


def test_mask_sparse_selection_scope(tmp_path):
    raw, ground = tmp_path / "raw", tmp_path / "ground"
    raw.mkdir(); ground.mkdir()
    for name in ["add_00000.parquet", "remove_00000.parquet"]:
        pq.write_table(pa.table({"value":[0,1,2]}),raw/name)
    pq.write_table(pa.table({"row_idx":[2],"raw_type":["add"]}),ground/"add_00000.parquet")
    selected = tmp_path / "selection.json"
    selected.write_text(json.dumps([{"shard":"add_00000.parquet","row_idx":2}]))
    args = argparse.Namespace(input_dir=raw, grounding_dir=ground, output_dir=tmp_path/"out", include_types=None, selection_file=selected)
    assert len(mask.build_jobs(args)) == 1
    args.selection_file = None
    with pytest.raises(ValueError, match="sparse grounding"):
        mask.build_jobs(args)
    args.selection_file = selected
    selected.write_text(json.dumps([{"shard":"add_00000.parquet","row_idx":1}]))
    with pytest.raises(ValueError): mask.build_jobs(args)
