from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.build_category_previews import _load_selection, _selected_mask_paths


def test_mask_and_validator_skip_historical_background_style(tmp_path):
    from types import SimpleNamespace
    from crispedit.mask.runner import build_jobs
    from scripts.validate_crispedit_mask_pipeline import validate
    source, run = tmp_path / 'source', tmp_path / 'run'
    source.mkdir()
    for kind in ['style', 'background change', 'unknown']:
        (source / f'{kind}_00000.parquet').touch()  # Must not even be read.
    for kind in ['grounding', 'mask']:
        (run / kind).mkdir(parents=True)
    args = SimpleNamespace(input_dir=source, run_dir=run, grounding_dir=run / 'grounding',
                           output_dir=run / 'mask', selection_file=None, include_types=None)
    assert build_jobs(args) == []
    assert validate(args)['shards'] == 0


def test_validator_allows_only_row_level_recoverable_parse_errors():
    from scripts.validate_crispedit_mask_pipeline import (
        recoverable_ground_parse_error,
        recoverable_observation_parse_error,
    )

    ground = {"ground_parse_ok": False, "grounding_status": "PARSE_ERROR", "qc_flag": "GROUND_FAIL"}
    failed_mask = {"qc_flag": "GROUND_FAIL"}
    review_mask = {"qc_flag": "MASK_REVIEW"}
    observation = {"parse_ok": False}

    assert recoverable_ground_parse_error(ground, failed_mask)
    assert not recoverable_ground_parse_error(ground, review_mask)
    assert recoverable_observation_parse_error(observation, failed_mask)
    assert not recoverable_observation_parse_error(observation, review_mask)






def test_category_preview_selection_is_keyed_by_shard(tmp_path: Path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        '{"cases":[{"shard":"add_00001.parquet","row_idx":7}]}',
        encoding="utf-8",
    )
    selected = _load_selection(selection_path)
    assert selected == {"add_00001.parquet": {7}}
    assert "add_00002.parquet" not in selected

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "add_00001.parquet").touch()
    (mask_dir / "add_00002.parquet").touch()
    assert [path.name for path in _selected_mask_paths(mask_dir, selected)] == [
        "add_00001.parquet"
    ]
