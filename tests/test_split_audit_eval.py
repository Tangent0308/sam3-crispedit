from synthesis_pipeline.split_audit_eval import split_rows


def test_split_keeps_two_regions_together_and_balances_strata():
    rows = []
    for subset in ("gres", "ver"):
        for pair in (("add", "remove"), ("replace", "attribute")):
            for source_id in range(10):
                for mask_index, task_type in enumerate(pair):
                    rows.append({
                        "source_subset": subset,
                        "parquet_row_index": source_id + (100 if pair[0] == "replace" else 0),
                        "mask_index": mask_index,
                        "task_type": task_type,
                    })
    dev, holdout = split_rows(rows, 123, 0.6)
    assert len(dev) == 48
    assert len(holdout) == 32
    for subset in ("gres", "ver"):
        for task_type in ("add", "remove", "replace", "attribute"):
            assert sum(r["source_subset"] == subset and r["task_type"] == task_type for r in dev) == 6
            assert sum(r["source_subset"] == subset and r["task_type"] == task_type for r in holdout) == 4
    dev_keys = {(r["source_subset"], r["parquet_row_index"]) for r in dev}
    holdout_keys = {(r["source_subset"], r["parquet_row_index"]) for r in holdout}
    assert not dev_keys & holdout_keys
