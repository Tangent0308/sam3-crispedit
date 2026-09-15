import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from refedit import GROUND_PROMPT_VERSION
from refedit.grounding_runner import Qwen35RefEditGrounder, build_jobs
from refedit.io import iter_row_batches, sample_id
from refedit.policy import TaskInference, apply_refedit_contract, infer_task
from scaleedit.grounding_runner import Qwen35ScaleEditGrounder
from scaleedit.policy import build_observation_prompt


def _png(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _tiny_refedit(root: Path) -> Path:
    path = root / "data" / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "img_id": 12,
            "source_img": {"bytes": _png("red"), "path": None},
            "instruction": "Remove the left cup",
            "target_img": {"bytes": _png("white"), "path": None},
        },
        {
            "img_id": 13,
            "source_img": {"bytes": _png("blue"), "path": None},
            "instruction": "Change the right vase to green",
            "target_img": {"bytes": _png("green"), "path": None},
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_task_inference_is_deterministic():
    assert infer_task("Add a cup on the right").final_task == "object_addition"
    assert infer_task("Remove the middle bird").final_task == "object_removal"
    assert infer_task("Move the left chair forward").final_task == "action_editing"
    assert infer_task("Change the vase material to glass").final_task == "material_change"
    assert infer_task("Change the right vase to blue").final_task == "color_change"
    assert infer_task("Replace the first sign with a clock").final_task == "object_replacement"


def test_refedit_contract_fails_closed_for_nonlocal_route():
    task = TaskInference("color_change", "test")
    result = apply_refedit_contract(
        {
            "prompt_version": "base",
            "mask_mode": "full_image",
            "source": [],
            "target": [],
            "ground_parse_ok": True,
        },
        task,
    )
    assert result["prompt_version"] == GROUND_PROMPT_VERSION
    assert result["base_prompt_version"] == "base"
    assert result["ground_parse_ok"] is False
    assert "NON_LOCAL_ROUTE:full_image" in result["refedit_policy_flags"]
    assert "MISSING_REQUIRED_SIDE:source" in result["refedit_policy_flags"]


def test_native_reader_and_selected_jobs(tmp_path):
    path = _tiny_refedit(tmp_path / "dataset")
    rows = [item for batch in iter_row_batches(path, 1) for item in batch]
    assert [index for index, _ in rows] == [0, 1]
    assert sample_id(rows[1][1]["img_id"]) == "refedit:13"
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"img_ids": [13]}), encoding="utf-8")
    args = argparse.Namespace(
        input_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        img_id=[],
        selection_file=selection,
        limit_rows_per_shard=None,
        limit_shards=None,
    )
    jobs = build_jobs(args)
    assert len(jobs) == 1
    assert jobs[0].num_rows == 1
    assert jobs[0].selected_sample_ids == ("refedit:13",)


def test_prefilter_manifest_selects_only_pass_rows(tmp_path):
    source = _tiny_refedit(tmp_path / "dataset")
    manifest_dir = tmp_path / "quality" / "manifest"
    manifest_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_idx": 1,
                    "sample_id": "refedit:13",
                    "img_id": 13,
                    "source_relative_path": f"data/{source.name}",
                    "task": "color_change",
                    "instruction": "Change the right vase to green",
                    "prefilter_verdict": "PASS",
                    "prefilter_confidence": 0.95,
                    "prefilter_reason_codes_json": "[]",
                    "prefilter_model_name": "Qwen3.8-27B",
                    "prefilter_prompt_version": "test",
                }
            ]
        ),
        manifest_dir / source.name,
    )
    args = argparse.Namespace(
        input_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        img_id=[],
        selection_file=None,
        prefilter_manifest_dir=manifest_dir,
        limit_rows_per_shard=None,
        limit_shards=None,
    )
    jobs = build_jobs(args)
    assert len(jobs) == 1
    assert jobs[0].num_rows == 1
    assert jobs[0].selected_sample_ids == ("refedit:13",)


def test_prompt_override_does_not_change_scaleedit_default():
    task, instruction = "color_change", "Change the right vase to blue"
    scale = Qwen35ScaleEditGrounder.__new__(Qwen35ScaleEditGrounder)
    refedit = Qwen35RefEditGrounder.__new__(Qwen35RefEditGrounder)
    base = build_observation_prompt(task, instruction)
    assert scale._build_observation_prompt(task, instruction) == base
    overridden = refedit._build_observation_prompt(task, instruction)
    assert "RefEdit source/result image pair" in overridden
    assert "Prefer a precise regions plan" in overridden
    assert "ScaleEdit source/result image pair" not in overridden
