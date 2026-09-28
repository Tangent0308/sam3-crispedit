from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_refedit_self_contained_dataset import (
    SELF_CONTAINED_SCHEMA,
    build_self_contained_dataset,
)
from scripts.validate_refedit_self_contained_dataset import (
    validate_self_contained_dataset,
)
from tests.test_refedit_finalization import (
    _ground_row,
    _mask_row,
    _write_prefilter,
)
from tests.test_refedit_pipeline import _tiny_refedit
from scaleedit.grounding_runner import GROUND_SCHEMA
from scaleedit.mask_runner import MASK_SCHEMA


def _prepare_source_and_final(tmp_path: Path, qc_flag: str = "OK") -> tuple[Path, Path]:
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
        _mask_row(1, 13, "Change the right vase to green", qc_flag),
    ]
    pq.write_table(pa.Table.from_pylist(ground_rows, schema=GROUND_SCHEMA), grounding)
    pq.write_table(pa.Table.from_pylist(mask_rows, schema=MASK_SCHEMA), masks)

    from refedit.finalize import build_final_dataset

    final_dir = tmp_path / "final"
    build_final_dataset(
        tmp_path / "dataset",
        tmp_path / "quality" / "manifest",
        tmp_path / "grounding",
        tmp_path / "masks",
        final_dir,
    )
    return tmp_path / "dataset", final_dir


def test_self_contained_dataset_embeds_images_and_mask(tmp_path: Path):
    input_dir, final_dir = _prepare_source_and_final(tmp_path, qc_flag="SEMANTIC_QC")
    output_dir = tmp_path / "self-contained"
    summary = build_self_contained_dataset(input_dir, final_dir, output_dir, workers=2)

    assert summary["rows"] == 1
    assert summary["source_final_rows"] == 1
    assert summary["validation"] == {
        "checked_shards": 1,
        "checked_rows": 1,
        "unique_sample_ids": 1,
        "schema_consistent": True,
        "all_rows_strict_pass": True,
    }
    table = pq.read_table(output_dir / "data" / "train-00000-of-00001.parquet")
    assert table.schema == SELF_CONTAINED_SCHEMA
    row = table.to_pylist()[0]
    assert row["sample_id"] == "refedit:12"
    assert row["prefilter_verdict"] == "PASS"
    assert row["grounding_status"] == "OK"
    assert row["qc_flag"] == "OK"
    assert row["quality_status"] == "strict_pass"
    assert row["source_img"]["bytes"]
    assert row["target_img"]["bytes"]
    assert row["mask_png"]
    assert row["source_width"] == row["mask_width"]
    assert row["source_height"] == row["mask_height"]
    assert (output_dir / "_SUCCESS").is_file()

    report = validate_self_contained_dataset(output_dir)
    assert report["validation_error_count"] == 0
    assert report["rows"] == 1


def test_self_contained_dataset_fails_closed_on_instruction_mismatch(tmp_path: Path):
    input_dir, final_dir = _prepare_source_and_final(tmp_path)
    shard = final_dir / "data" / "train-00000-of-00001.parquet"
    rows = pq.read_table(shard).to_pylist()
    rows[0]["final_instruction"] = "Wrong instruction"
    pq.write_table(pa.Table.from_pylist(rows, schema=MASK_SCHEMA), shard)

    output_dir = tmp_path / "self-contained"
    try:
        build_self_contained_dataset(input_dir, final_dir, output_dir, workers=1)
    except ValueError as exc:
        assert "instruction mismatch" in str(exc)
    else:
        raise AssertionError("expected strict mismatch failure")


def test_self_contained_dataset_resume_reuses_complete_reports(tmp_path: Path):
    input_dir, final_dir = _prepare_source_and_final(tmp_path)
    output_dir = tmp_path / "self-contained"
    first = build_self_contained_dataset(input_dir, final_dir, output_dir, workers=1)
    second = build_self_contained_dataset(
        input_dir,
        final_dir,
        output_dir,
        workers=1,
        resume=True,
    )
    assert first["rows"] == second["rows"] == 2
    assert second["validation"]["checked_rows"] == 2
