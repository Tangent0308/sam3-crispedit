import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_referential_edits import METADATA_COLUMNS  # noqa: E402
from run_referential_filter_full import (  # noqa: E402
    _assigned_shards,
    _evidence_path,
    _valid_evidence,
    discover_shards,
    stage_fuse_full,
)
from watch_referential_filter_progress import (  # noqa: E402
    _load_reports,
    _snapshot_line,
    _stage_started_at,
)

from difficulty_filter.referential import (  # noqa: E402
    MLLM_PROMPT_VERSION,
    SAM_COUNT_POLICY_VERSION,
)


def _write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _source_row(sample_id, edit_type="color_change", mask_mode="regions"):
    values = {
        "sample_id": sample_id,
        "source_dataset": "crispedit",
        "source_shard": "source.parquet",
        "source_row_idx": 0,
        "edit_type": edit_type,
        "raw_edit_type": edit_type,
        "instruction": "Change the left orange to green.",
        "mask_mode": mask_mode,
        "mask_area_fraction": 0.1,
    }
    return {column: values.get(column) for column in METADATA_COLUMNS}


def _mllm_row(sample_id, status="ok"):
    return {
        "sample_id": sample_id,
        "mllm_status": status,
        "prompt_version": MLLM_PROMPT_VERSION,
        "edited_subject_phrase": "left orange" if status == "ok" else "",
        "object_category": "orange" if status == "ok" else "",
        "visible_same_class_count": 3 if status == "ok" else None,
        "selected_instance_count": 1 if status == "ok" else None,
        "subset_relation": "yes" if status == "ok" else "",
        "reference_cues": ["spatial"] if status == "ok" else [],
        "fine_grained_referential": "yes" if status == "ok" else "",
        "mllm_confidence": 0.9 if status == "ok" else None,
        "mllm_reason": "one of three" if status == "ok" else "excluded",
    }


def _sam_row(sample_id, status="ok"):
    return {
        "sample_id": sample_id,
        "sam_policy_version": SAM_COUNT_POLICY_VERSION,
        "sam_status": status,
        "sam_count": 3 if status == "ok" else None,
        "sam_selected_count": 1 if status == "ok" else None,
        "sam_unselected_count": 2 if status == "ok" else None,
        "sam_high_confidence_count": 3 if status == "ok" else None,
    }


def test_shard_assignment_is_disjoint_and_complete(tmp_path):
    data_dir = tmp_path / "input" / "crispedit" / "data"
    for index in range(5):
        _write([{"sample_id": str(index)}], data_dir / f"part-{index}.parquet")
    shards = discover_shards(tmp_path / "input", ["crispedit"])
    first = _assigned_shards(shards, 0, 2)
    second = _assigned_shards(shards, 1, 2)
    assert set(first).isdisjoint(second)
    assert set(first + second) == set(shards)


def test_zero_row_evidence_is_a_valid_checkpoint(tmp_path):
    output = tmp_path / "empty.parquet"
    _write([], output)
    assert _valid_evidence(output, 0, "prompt_version", MLLM_PROMPT_VERSION)


def test_tqdm_watcher_aggregates_reports_and_formats_timing(tmp_path):
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    for worker, rows in enumerate((10, 15)):
        (progress_dir / f"sam-worker-{worker:02d}.json").write_text(
            json.dumps(
                {
                    "stage": "sam",
                    "state": "running",
                    "processed_rows": rows,
                    "total_rows": 50,
                    "completed_shards": worker + 1,
                    "total_shards": 5,
                    "sam_calls": rows - 2,
                    "sam_errors": 0,
                }
            ),
            encoding="utf-8",
        )
    report = _load_reports(tmp_path, "sam", {})
    assert report["processed_rows"] == 25
    assert report["total_rows"] == 100
    assert report["workers_reporting"] == 2
    line = _snapshot_line("sam", report, started_at=100.0, now=110.0)
    assert "elapsed=00:10" in line
    assert "eta=00:30" in line
    assert "rate=2.50 rows/s" in line


def test_tqdm_watcher_uses_latest_stage_launch(tmp_path):
    log = tmp_path / "progress.log"
    log.write_text(
        "[2026-09-14T01:00:00+00:00] launched 8 sam workers pids=[]\n"
        "[2026-09-14T03:00:00+00:00] launched 8 sam workers pids=[]\n",
        encoding="utf-8",
    )
    expected = datetime.fromisoformat("2026-09-14T03:00:00+00:00").timestamp()
    assert _stage_started_at(log, "sam") == expected


def test_full_fusion_writes_aligned_strict_and_loose_manifests(tmp_path):
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source_path = dataset_root / "crispedit" / "data" / "part-0.parquet"
    source_rows = [
        _source_row("keep"),
        _source_row("global", edit_type="background_replacement"),
    ]
    _write(source_rows, source_path)
    mllm_path = _evidence_path(output_root, "mllm", "crispedit", source_path)
    sam_path = _evidence_path(output_root, "sam", "crispedit", source_path)
    _write(
        [_mllm_row("keep"), _mllm_row("global", "skipped_deterministic")],
        mllm_path,
    )
    _write(
        [_sam_row("keep"), _sam_row("global", "skipped_no_mllm_category")],
        sam_path,
    )
    assert _valid_evidence(
        mllm_path, 2, "prompt_version", MLLM_PROMPT_VERSION
    )
    assert _valid_evidence(
        sam_path, 2, "sam_policy_version", SAM_COUNT_POLICY_VERSION
    )

    stage_fuse_full(
        argparse.Namespace(
            dataset_root=dataset_root,
            output_root=output_root,
            datasets="crispedit",
            shard_glob="*.parquet",
            max_shards=0,
            max_rows_per_shard=0,
            all_selected_fraction=0.9,
            progress_every_shards=1,
        )
    )
    audit = pq.read_table(output_root / "audit/crispedit/part-0.parquet").to_pylist()
    strict = pq.read_table(output_root / "final/selected_manifest.parquet").to_pylist()
    loose = pq.read_table(
        output_root / "final/loose_selected_manifest.parquet"
    ).to_pylist()
    assert [row["decision"] for row in audit] == ["keep", "drop"]
    assert [row["sample_id"] for row in strict] == ["keep"]
    assert [row["sample_id"] for row in loose] == ["keep"]
