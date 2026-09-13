import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.build_unified_mask_dataset import (
    DatasetSpec,
    TRAIN_SCHEMA,
    build_dataset,
)


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(stream, format="PNG")
    return stream.getvalue()


def _mask_bytes() -> bytes:
    stream = io.BytesIO()
    array = np.zeros((3, 4), dtype=np.uint8)
    array[0, :3] = 255
    Image.fromarray(array, mode="L").save(stream, format="PNG")
    return stream.getvalue()


def _mask_row(row_idx: int, **overrides):
    row = {
        "row_idx": row_idx,
        "raw_type": "add",
        "canonical_type": "add",
        "instruction": "Add a red cup.",
        "sample_id": f"scale-{row_idx}",
        "edit_task": "object_addition",
        "final_task": "object_addition",
        "original_instruction": "Add a cup.",
        "final_instruction": "Add a red cup.",
        "ground_json": json.dumps({"mask_mode": "regions"}),
        "mask_png": _mask_bytes(),
        "instance_masks": [],
        "mask_source": "pcs",
        "area_frac": 0.25,
        "qc_flag": "OK",
        "qc_flags_json": '["OK"]',
        "mask_height": 3,
        "mask_width": 4,
        "mask_sum": 3,
        "ar_delta": 0.0,
        "grounding_status": "OK",
        "mllm_model": "test-mllm",
        "prompt_version": "test-prompt",
        "sam_version": "test-sam",
        "mask_policy_version": "test-mask-policy",
        "prefilter_verdict": "PASS",
        "prefilter_confidence": 0.99,
        "prefilter_method": "fact",
        "prefilter_evidence_schema": "fact_evidence",
        "prefilter_model_name": "test-prefilter",
        "prefilter_run_id": "test-run",
        "filter_decision": "keep",
        "prefilter_reason": "",
        "filter_reason_codes": "",
        "filter_mismatch_score": 0.01,
    }
    row.update(overrides)
    return row


def test_build_dataset_joins_sparse_sidecars_and_unifies_schema(tmp_path: Path):
    crisp_raw = tmp_path / "crisp-raw"
    crisp_mask = tmp_path / "crisp-mask"
    scale_raw = tmp_path / "scale-raw"
    scale_mask = tmp_path / "scale-mask"
    for path in (crisp_raw, crisp_mask, scale_raw, scale_mask):
        path.mkdir()

    source = _image_bytes((10, 20, 30))
    edited = _image_bytes((40, 50, 60))
    crisp_raw_rows = [
        {
            "input_img": {"bytes": source, "path": None},
            "instruction": "Add a red cup.",
            "output_img": {"bytes": edited, "path": None},
            "type": "add",
        },
        {
            "input_img": {"bytes": source, "path": None},
            "instruction": "Add a blue cup.",
            "output_img": {"bytes": edited, "path": None},
            "type": "add",
        },
    ]
    pq.write_table(pa.Table.from_pylist(crisp_raw_rows), crisp_raw / "add_00000.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                _mask_row(0),
                _mask_row(
                    1,
                    instruction="Add a blue cup.",
                    qc_flag="PREFILTER_SKIP",
                    qc_flags_json='["PREFILTER_SKIP"]',
                    mask_png=b"",
                    mask_sum=0,
                    area_frac=float("nan"),
                    prefilter_verdict="FAIL",
                    filter_decision="drop",
                ),
            ]
        ),
        crisp_mask / "add_00000.parquet",
    )

    scale_raw_rows = [
        {
            "sample_id": "scale-0",
            "split": "train",
            "source_relative_path": "source.parquet",
            "manifest_row_index": 10,
            "public_source_row_index": 9,
            "edit_task": "object_addition",
            "final_task": "object_addition",
            "original_instruction": "Add a cup.",
            "final_instruction": "Add a red cup.",
            "instruction_action": "REWRITE",
            "category_action": "KEEP",
            "confidence": 0.9,
            "source_image": source,
            "edited_image": edited,
            "source_image_url": "",
            "source_image_origin": "test",
            "source_image_width": 4,
            "source_image_height": 3,
            "edited_image_width": 4,
            "edited_image_height": 3,
        },
        {
            "sample_id": "scale-1",
            "split": "train",
            "source_relative_path": "source.parquet",
            "manifest_row_index": 11,
            "public_source_row_index": 10,
            "edit_task": "object_removal",
            "final_task": "object_removal",
            "original_instruction": "Remove a cup.",
            "final_instruction": "Remove a cup.",
            "instruction_action": "KEEP",
            "category_action": "KEEP",
            "confidence": 0.9,
            "source_image": source,
            "edited_image": edited,
            "source_image_url": "",
            "source_image_origin": "test",
            "source_image_width": 4,
            "source_image_height": 3,
            "edited_image_width": 4,
            "edited_image_height": 3,
        },
    ]
    pq.write_table(pa.Table.from_pylist(scale_raw_rows), scale_raw / "part-00000.parquet")
    scale_row = _mask_row(0)
    for key in list(scale_row):
        if key.startswith("prefilter_") or key in {
            "filter_decision",
            "filter_reason_codes",
            "filter_mismatch_score",
        }:
            scale_row.pop(key)
    pq.write_table(
        pa.Table.from_pylist([scale_row]), scale_mask / "part-00000.parquet"
    )

    output = tmp_path / "unified"
    manifest = build_dataset(
        [
            DatasetSpec("crispedit", crisp_raw, crisp_mask),
            DatasetSpec("scaleedit", scale_raw, scale_mask),
        ],
        output,
        None,
        workers=2,
        resume=False,
    )

    assert manifest["accepted_rows"] == 2
    assert manifest["rejected_rows"] == 2
    assert manifest["datasets"]["crispedit"]["primary_reasons"] == {
        "prefilter_not_pass": 1
    }
    assert manifest["datasets"]["scaleedit"]["primary_reasons"] == {
        "missing_mask_record": 1
    }
    crisp_out = pq.read_table(output / "crispedit/data/add_00000.parquet")
    scale_out = pq.read_table(output / "scaleedit/data/part-00000.parquet")
    assert crisp_out.schema == TRAIN_SCHEMA
    assert scale_out.schema == TRAIN_SCHEMA
    assert crisp_out.column("sample_id").to_pylist() == [
        "crispedit:add_00000.parquet#0"
    ]
    assert scale_out.column("sample_id").to_pylist() == ["scaleedit:scale-0"]
    assert crisp_out.column("edit_type").to_pylist() == ["object_addition"]
    assert scale_out.column("mask_area").to_pylist() == [3]
    assert (output / "_SUCCESS").is_file()
    assert pq.read_table(output / "shards.parquet").num_rows == 2

    resumed = build_dataset(
        [
            DatasetSpec("crispedit", crisp_raw, crisp_mask),
            DatasetSpec("scaleedit", scale_raw, scale_mask),
        ],
        output,
        None,
        workers=2,
        resume=True,
    )
    assert resumed["accepted_rows"] == 2
    assert resumed["validation"] == {
        "checked_shards": 2,
        "checked_rows": 2,
        "checked_rejections": 2,
        "unique_sample_ids": 2,
        "schema_consistent": True,
        "all_rows_strict_pass": True,
    }
