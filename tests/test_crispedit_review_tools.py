import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.compare_grounded_mask_runs import _mask_metrics
from scripts.evaluate_grounded_mask_bad_cases import read_area_by_row_idx


def test_sparse_area_lookup_uses_explicit_original_row_idx(tmp_path):
    path = tmp_path / "sparse.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"row_idx": 104, "area_frac": 0.25},
                {"row_idx": 211, "area_frac": 0.5},
            ]
        ),
        path,
    )
    assert read_area_by_row_idx(path, 211) == pytest.approx(0.5)


def test_mask_comparison_reports_directional_added_and_lost_fractions():
    import numpy as np

    baseline = np.asarray([[1, 1], [0, 0]], dtype=bool)
    candidate = np.asarray([[1, 0], [1, 0]], dtype=bool)
    metrics = _mask_metrics(baseline, candidate)
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["candidate_added_frac"] == pytest.approx(1 / 2)
    assert metrics["baseline_lost_frac"] == pytest.approx(1 / 2)
