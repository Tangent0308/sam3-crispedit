import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from refedit import GROUND_PROMPT_VERSION, MASK_POLICY_VERSION
from refedit.finalize import build_final_dataset
from scaleedit.grounding_runner import GROUND_SCHEMA
from scaleedit.mask_runner import MASK_SCHEMA
from tests.test_refedit_pipeline import _tiny_refedit


def _write_prefilter(path: Path, source_name: str) -> None:
    rows = []
    for row_idx, img_id, instruction, task in (
        (0, 12, "Remove the left cup", "object_removal"),
        (1, 13, "Change the right vase to green", "color_change"),
    ):
        rows.append(
            {
                "row_idx": row_idx,
                "sample_id": f"refedit:{img_id}",
                "img_id": img_id,
                "source_relative_path": f"data/{source_name}",
                "task": task,
                "instruction": instruction,
                "prefilter_verdict": "PASS",
                "prefilter_confidence": 0.95,
                "prefilter_reason_codes_json": "[]",
                "prefilter_model_name": "Qwen3.8-27B",
                "prefilter_prompt_version": "test-prefilter",
            }
        )
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _ground_row(row_idx: int, img_id: int, instruction: str) -> dict:
    return {
        "row_idx": row_idx,
        "sample_id": f"refedit:{img_id}",
        "source_relative_path": "data/train-00000-of-00001.parquet",
        "final_instruction": instruction,
        "grounding_status": "OK",
        "prompt_version": GROUND_PROMPT_VERSION,
    }


def _png_mask(mask_sum: int = 6, size: tuple[int, int] = (8, 6)) -> bytes:
    array = np.zeros((size[1], size[0]), dtype=np.uint8)
    flat = array.reshape(-1)
    flat[:mask_sum] = 255
    stream = io.BytesIO()
    Image.fromarray(array, mode="L").save(stream, format="PNG")
    return stream.getvalue()


def _mask_row(
    row_idx: int, img_id: int, instruction: str, qc_flag: str
) -> dict:
    mask_sum = 6 if qc_flag == "OK" else 3
    return {
        "row_idx": row_idx,
        "sample_id": f"refedit:{img_id}",
        "source_relative_path": "data/train-00000-of-00001.parquet",
        "final_instruction": instruction,
        "final_task": "object_removal" if row_idx == 0 else "color_change",
        "ground_json": '{"mask_mode":"regions"}',
        "mask_png": _png_mask(mask_sum=mask_sum),
        "mask_source": "pcs",
        "area_frac": mask_sum / 48.0,
        "qc_flag": qc_flag,
        "qc_flags_json": f'["{qc_flag}"]',
        "mask_height": 6,
        "mask_width": 8,
        "mask_sum": mask_sum,
        "ar_delta": 0.0,
        "grounding_status": "OK",
        "prompt_version": GROUND_PROMPT_VERSION,
        "mask_policy_version": MASK_POLICY_VERSION,
    }


def test_final_dataset_keeps_only_prefilter_pass_and_mask_ok(tmp_path):
    source = _tiny_refedit(tmp_path / "dataset")
    quality = tmp_path / "quality" / "manifest" / source.name
    grounding = tmp_path / "grounding" / source.name
    masks = tmp_path / "masks" / source.name
    _write_prefilter(quality, source.name)
    grounding.parent.mkdir(parents=True)
    masks.parent.mkdir(parents=True)
    ground_rows = [
        _ground_row(0, 12, "Remove the left cup"),
        _ground_row(1, 13, "Change the right vase to green"),
    ]
    mask_rows = [
        _mask_row(0, 12, "Remove the left cup", "OK"),
        _mask_row(1, 13, "Change the right vase to green", "SEMANTIC_QC"),
    ]
    pq.write_table(pa.Table.from_pylist(ground_rows, schema=GROUND_SCHEMA), grounding)
    pq.write_table(pa.Table.from_pylist(mask_rows, schema=MASK_SCHEMA), masks)

    output = tmp_path / "final"
    summary = build_final_dataset(
        tmp_path / "dataset",
        tmp_path / "quality" / "manifest",
        tmp_path / "grounding",
        tmp_path / "masks",
        output,
    )
    assert summary["prefilter_pass_rows"] == 2
    assert summary["final_rows"] == 1
    assert summary["rejected_after_mask"] == 1
    final_rows = pq.read_table(output / "data" / source.name).to_pylist()
    assert [row["sample_id"] for row in final_rows] == ["refedit:12"]
    rejected = pq.read_table(
        output / "audit" / "rejected_mask_qc.parquet"
    ).to_pylist()
    assert rejected[0]["sample_id"] == "refedit:13"
    assert rejected[0]["rejection_reason"] == "mask_qc_flag=SEMANTIC_QC"
