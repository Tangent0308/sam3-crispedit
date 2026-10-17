import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from sa1b_pipeline.group_production import (
    accept,
    choose,
    describe,
    proposal_key,
    split_cells,
    target_cells,
)
from scripts.run_group_pipeline import _validate_multi_selection_operations
from scripts.run_group_production_4node import command, owner_lock, run_wave


def proposal(sid, op="add", rel="cross", n=2):
    return {
        "candidate": {"source_id": sid},
        "operation": op,
        "category_relation": rel,
        "objects": [{"object_id": str(i)} for i in range(n)],
    }


def good(sid, op="add", rel="cross", n=2):
    return {
        "twin_id": sid,
        "source_id": sid,
        "operation": op,
        "category_relation": rel,
        "members": [{} for _ in range(n)],
        "group_checks": {"group_pass": True},
    }


def test_global_and_rank_quotas_preserve_requested_scale():
    q = target_cells("twins", 15000)
    assert sum(q.values()) == 15000
    assert sum(v for k, v in q.items() if k.endswith(":3")) == 2250
    assert sum(v for k, v in q.items() if ":same:" in k) == 4500
    for op in ("add", "remove", "replace", "attribute"):
        assert sum(v for k, v in q.items() if k.startswith(op + ":")) == 3750
    for kind, n in [("twins", 15000), ("multi", 5000)]:
        q = target_cells(kind, n)
        shards = split_cells(q)
        assert [sum(x.values()) for x in shards] == [n // 4] * 4
        assert {k: sum(x[k] for x in shards) for k in q} == q
    q = target_cells("multi", 5000)
    assert {
        n: sum(v for k, v in q.items() if k.startswith(str(n) + ":"))
        for n in (2, 3, 4, 5)
    } == {2: 1000, 3: 1500, 4: 1250, 5: 1250}
    ops = Counter()
    for k, v in q.items():
        for op in k.split(":")[1].split("+"):
            ops[op] += v
    assert sum(ops.values()) == 17750 and max(ops.values()) - min(ops.values()) <= 4


def test_no_retries_no_source_overflow_and_only_missing_cells():
    q = {"add:cross:2": 2, "remove:cross:2": 1}
    done = [good("r", "remove")]
    old = [proposal("old")]
    pool = [
        proposal("old"),
        proposal("old", "remove"),
        proposal("a"),
        proposal("b"),
        proposal("r", "remove"),
    ]
    picked = choose(pool, old, done, q, "twins", 10)
    assert {x["candidate"]["source_id"] for x in picked} == {"a", "b"}
    assert all(x["operation"] == "add" for x in picked)
    assert proposal_key(proposal("old")) == proposal_key(proposal("old", "remove"))


def test_accept_does_not_use_failed_or_overquota_groups():
    q = {"add:cross:2": 2}
    bad = good("bad")
    bad["group_checks"]["group_pass"] = False
    values = [good("a"), good("a"), bad, good("b"), good("c")]
    assert len(accept(values, q, "twins")) == 2
    assert describe(values, q, "twins")["complete"]
    one = good("x")
    two = {**good("y"), "source_id": "x"}
    three = {**good("z"), "source_id": "x"}
    assert len(accept([one, two, three], {"add:cross:2": 3}, "twins")) == 2


def test_dynamic_multi_retains_per_group_invariants():
    rows = [{"objects": [{}, {}], "operations": ["add", "remove"]} for _ in range(100)]
    with pytest.raises(RuntimeError, match="skew"):
        _validate_multi_selection_operations(rows)
    _validate_multi_selection_operations(rows, aggregate_balance=False)
    with pytest.raises(RuntimeError, match="repeats"):
        _validate_multi_selection_operations(
            [{"objects": [{}, {}], "operations": ["add", "add"]}],
            aggregate_balance=False,
        )


def test_stage_resume_skips_completed_work_and_rejects_changed_command(tmp_path):
    import sys

    target = tmp_path / "calls.txt"
    cmd = [
        sys.executable,
        "-c",
        f"from pathlib import Path; p=Path({str(target)!r});p.write_text(p.read_text()+'x' if p.exists() else 'x')",
    ]
    command(tmp_path, "planning", cmd, "0")
    command(tmp_path, "planning", cmd, "0")
    assert target.read_text() == "x"
    with pytest.raises(ValueError, match="config changed"):
        command(tmp_path, "planning", cmd + ["changed"], "0")
    assert len((tmp_path / "stage_attempts.jsonl").read_text().splitlines()) == 1


def test_owner_lock_prevents_concurrent_rank_and_cleans_after_failure(tmp_path):
    lock = tmp_path / "rank.lock.d"
    with pytest.raises(ValueError), owner_lock(lock):
        with pytest.raises(RuntimeError, match="active owner"), owner_lock(lock):
            pass
        raise ValueError("stage failed")
    assert not lock.exists()


def test_wave_with_no_plan_never_starts_editor(tmp_path, monkeypatch):
    args = SimpleNamespace(
        root=tmp_path, mllm_python="mllm", editor_python="editor", editor_devices="0,1"
    )
    c = {
        "planner_model": "p",
        "planner_tp": 4,
        "blind_model": "b",
        "editor_model": "e",
        "audit_model": "a",
        "site_model": "s",
    }
    wave = tmp_path / "wave"
    (wave / "inputs").mkdir(parents=True)
    called = []

    def fake(wave, name, cmd, devices):
        called.append(name)
        (wave / "planned.jsonl").write_text("")

    monkeypatch.setattr("scripts.run_group_production_4node.command", fake)
    run_wave(args, c, "twins", wave, [proposal("s")])
    assert called == ["planning", "blind"]
    assert (wave / "audited.jsonl").read_text() == ""


def test_prepare_owns_media_and_splits_by_image_hash(tmp_path):
    import hashlib

    from PIL import Image

    from scripts.run_group_production_4node import prepare

    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 16), "red").save(source)
    h = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source_id": "s1",
                "image_path": str(source),
                "image_sha256": h,
                "objects": [],
            }
        )
        + "\n"
        + json.dumps(
            {
                "source_id": "s2",
                "image_path": str(source),
                "image_sha256": h,
                "objects": [],
            }
        )
        + "\n"
    )
    reservation = tmp_path / "reservation.json"
    reservation.write_text(
        json.dumps(
            {
                "schema_version": "sa1b.samtok.source_isolation.v1",
                "reserved_ids": [],
                "reserved_image_sha256": [],
            }
        )
    )
    root = tmp_path / "run"
    prepare(
        SimpleNamespace(
            root=root,
            manifest=manifest,
            reservation=reservation,
            twins_target=8,
            multi_target=8,
        )
    )
    ready = json.loads((root / "inputs/ready.json").read_text())
    assert ready["sources"] == 1
    records = [
        json.loads(line)
        for p in (root / "inputs").glob("train_rank*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len(records) == 1 and Path(records[0]["image_path"]).is_relative_to(root)
    assert Path(records[0]["image_path"]).read_bytes() == source.read_bytes()


def test_prepare_fails_closed_on_eval_hash(tmp_path):
    import hashlib

    from scripts.run_group_production_4node import prepare

    source = tmp_path / "source.jpg"
    source.write_bytes(b"fake image payload")
    h = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        json.dumps({"source_id": "train-id", "image_path": str(source), "objects": []})
        + "\n"
    )
    reservation = tmp_path / "reservation.json"
    reservation.write_text(
        json.dumps(
            {
                "schema_version": "sa1b.samtok.source_isolation.v1",
                "reserved_ids": [],
                "reserved_image_sha256": [h],
            }
        )
    )
    with pytest.raises(ValueError, match="eval source leak"):
        prepare(
            SimpleNamespace(
                root=tmp_path / "run",
                manifest=manifest,
                reservation=reservation,
                twins_target=8,
                multi_target=8,
            )
        )
    assert not (tmp_path / "run/inputs/ready.json").exists()


def test_planner_loads_verified_local_copy_but_keeps_canonical_identity(monkeypatch):
    import sys

    from sa1b_pipeline.mllm import VLLMSelector

    seen = {}

    class Processor:
        @staticmethod
        def from_pretrained(model, **kwargs):
            seen["processor"] = model
            return object()

    def llm(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setenv("SA1B_FORCE_DIRECT_VLLM", "1")
    monkeypatch.setattr(
        "sa1b_pipeline.model_cache.resolve_model", lambda model: "/verified/local/model"
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoProcessor=Processor)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=llm, SamplingParams=lambda **kw: kw, __version__="0.31.1"),
    )
    selector = VLLMSelector("/canonical/model")
    assert selector.model_id == "/canonical/model"
    assert seen["processor"] == seen["model"] == "/verified/local/model"


def test_production_profile_requires_parts_and_caps_each_scope():
    values = []
    for i in range(100):
        g = good(str(i))
        g["members"] = [
            {"scope": f"kind{i // 5}", "granularity": "object", "area_fraction": 0.01}
            for _ in range(2)
        ]
        values.append(g)
    q = {"add:cross:2": 100}
    assert len(accept(values, q, "twins")) == 60
    for i, g in enumerate(values[60:]):
        for m in g["members"]:
            m["granularity"] = "part"
    assert len(accept(values, q, "twins")) == 100
    for g in values:
        for m in g["members"]:
            m["scope"] = "same category"
    assert len(accept(values, q, "twins")) == 5


def test_rank_coordinator_refills_rejects_and_resumes_without_generation(
    tmp_path, monkeypatch
):
    from sa1b_pipeline.production_state import write_rows
    from scripts.run_group_production_4node import run_kind

    (tmp_path / "inputs").mkdir()
    write_rows(
        tmp_path / "inputs/train_rank0.jsonl", [{"source_id": str(i)} for i in range(2)]
    )
    args = SimpleNamespace(root=tmp_path, rank=0, selection_workers=1)
    cfg = {
        "rank_targets": {"twins": [{"add:cross:2": 2}]},
        "max_attempts": {"twins": 40},
        "wave_size": 2,
        "source_chunk_size": 2,
    }

    class Inline:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def map(self, fn, jobs, **kwargs):
            return map(fn, jobs)

    def proposals(job):
        result = []
        for i in range(2):
            row = proposal(job[0]["source_id"])
            row["objects"] = [{"object_id": str(2 * i + j)} for j in range(2)]
            result.append(row)
        return result

    calls = []

    def fake_wave(args, c, kind, wave, chosen):
        calls.append(wave)
        outputs = []
        for i, row in enumerate(chosen):
            g = good(str(wave.name) + str(i))
            g["source_id"] = row["candidate"]["source_id"]
            if len(calls) == 1 and i == 0:
                g["group_checks"]["group_pass"] = False
            outputs.append(g)
        write_rows(wave / "audited.jsonl", outputs)

    monkeypatch.setattr(
        "scripts.run_group_production_4node.cf.ProcessPoolExecutor", Inline
    )
    monkeypatch.setattr("scripts.run_group_production_4node.propose", proposals)
    monkeypatch.setattr("scripts.run_group_production_4node.run_wave", fake_wave)
    monkeypatch.setattr(
        "sa1b_pipeline.group_schema.validate_twin_record", lambda *a, **k: None
    )
    first = run_kind(args, cfg, "twins")
    assert first["complete"] and first["accepted"] == 2 and len(calls) == 2
    before = (tmp_path / "nodes/node0/twins/accepted.jsonl").read_bytes()
    second = run_kind(args, cfg, "twins")
    assert second["complete"] and len(calls) == 2
    assert (tmp_path / "nodes/node0/twins/accepted.jsonl").read_bytes() == before


@pytest.mark.parametrize('policy,mode', [('keep-passed-v1','serial'), ('parallel-components-v1','parallel-components')])
def test_new_multi_wave_uses_transactional_workers_and_no_second_audit(tmp_path,monkeypatch,policy,mode):
    args=SimpleNamespace(root=tmp_path,mllm_python='mllm',editor_python='editor',editor_devices='0,1,6,7')
    c=dict(planner_model='p',planner_tp=4,editor_model='e',audit_model='a',site_model='s',
           multi_commit_policy=policy,multi_audit_devices='6,7')
    wave=tmp_path/'wave';(wave/'inputs').mkdir(parents=True)
    calls=[]
    def fake(wave,name,cmd,devices):
        calls.append((name,cmd,devices))
        if name=='planning':(wave/'planned.jsonl').write_text('{"multi_id":"x"}\n')
    monkeypatch.setattr('scripts.run_group_production_4node.command',fake)
    run_wave(args,c,'multi',wave,[proposal('s')])
    assert [x[0] for x in calls]==['planning','transactional_generation_audit']
    cmd=calls[-1][1]
    assert cmd[cmd.index('--editor-devices')+1]=='0,1'
    assert cmd[cmd.index('--audit-devices')+1]=='6,7'
    assert cmd[cmd.index('--generation-mode')+1]==mode
