import json
from pathlib import Path

import pytest

from scripts.verify_group_inputs import main


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _candidate(source_id, operation="add", n=2):
    return {
        "candidate": {"source_id": source_id, "image_sha256": f"hash-{source_id}"},
        "objects": [{"object_id": str(i)} for i in range(n)],
        "operation": operation,
        "scope_level": "object",
        "gate": {},
    }


def _multi(source_id):
    return {
        "candidate": {"source_id": source_id, "image_sha256": f"hash-{source_id}"},
        "objects": [{"object_id": "0"}, {"object_id": "1"}],
        "operations": ["add", "replace"],
        "adjacent_pairs": [{"object_ids": ["0", "1"], "eligible": True}],
    }


def _args(tmp_path, *, twin_eval_source="eval-twin"):
    frozen = tmp_path / "frozen"
    (frozen / "inputs").mkdir(parents=True)
    (frozen / "inputs/node0.jsonl").write_text(json.dumps({"source_id": "frozen"}) + "\n")
    exclusion = tmp_path / "benchmark.jsonl"
    _write_jsonl(exclusion, [{"source": {"image_id": "benchmark", "sha256": "benchmark-hash"}}])
    reservation = tmp_path / "reservation.json"
    reservation.write_text(json.dumps({"reserved_ids": [twin_eval_source, "eval-multi"],
                                       "reserved_image_sha256": [f"hash-{twin_eval_source}", "hash-eval-multi"]}))
    twin_train = tmp_path / "twin_train.jsonl"
    _write_jsonl(twin_train, [_candidate("train-a", op) for train, op in
                              zip(("a", "b", "c", "d"), ("add", "replace", "remove", "attribute"))])
    twin_eval = tmp_path / "twin_eval.jsonl"
    _write_jsonl(twin_eval, [_candidate(twin_eval_source, "add"),
                             _candidate("eval-multi", "replace"),
                             _candidate("eval-c", "remove"),
                             _candidate("eval-d", "attribute")])
    multi_train = tmp_path / "multi_train.jsonl"
    _write_jsonl(multi_train, [_multi("train-c")])
    multi_eval = tmp_path / "multi_eval.jsonl"
    _write_jsonl(multi_eval, [_multi("eval-multi")])
    output = tmp_path / "report.json"
    return ["--frozen-root", str(frozen), "--benchmark-exclusion", str(exclusion),
            "--reservation", str(reservation), "--twin-train", str(twin_train),
            "--twin-eval", str(twin_eval), "--multi-train", str(multi_train),
            "--multi-eval", str(multi_eval), "--twin-train-count", "4",
            "--twin-eval-count", "4", "--multi-train-count", "1",
            "--multi-eval-count", "1", "--output", str(output)]


def test_verifier_accepts_selection_manifests_and_records_isolation(tmp_path):
    args = _args(tmp_path)
    assert main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "verified"
    assert report["eval_vs_all_train"] == {"source_id_overlap": 0, "image_hash_overlap": 0}
    assert report["datasets"]["twin_train"]["member_counts"] == {"2": 4}


def test_verifier_rejects_eval_source_in_frozen_training(tmp_path):
    args = _args(tmp_path, twin_eval_source="frozen")
    with pytest.raises(ValueError, match="evaluation source leaked"):
        main(args)
