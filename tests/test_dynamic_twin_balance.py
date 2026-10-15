from sa1b_pipeline.dynamic_twin_balance import (
    accepted_groups,
    choose_wave,
    progress,
    quotas,
)


def row(op, rel, sid):
    return {
        "operation": op,
        "category_relation": rel,
        "candidate": {"source_id": str(sid)},
    }


def good(op, rel, sid):
    return {
        "operation": op,
        "category_relation": rel,
        "twin_id": str(sid),
        "group_checks": {"group_pass": True},
    }


def test_quotas():
    for n in [1, 8, 10, 12, 5000]:
        q = quotas(n)
        assert sum(q.values()) == n
        assert sum(v for k, v in q.items() if k.endswith(":same")) == round(0.3 * n)
        ops = [
            sum(v for k, v in q.items() if k.startswith(op + ":"))
            for op in ("add", "remove", "replace", "attribute")
        ]
        assert max(ops) - min(ops) <= 1


def test_only_missing_classes_replenished_and_no_retries():
    q = quotas(8, 0.5)
    accepted = [
        good(op, rel, op + rel)
        for op in ("remove", "replace", "attribute")
        for rel in ("same", "cross")
    ]
    old = [row("add", "same", "old")]
    pool = [
        row(op, rel, op + rel + str(i))
        for op in ("add", "remove", "replace", "attribute")
        for rel in ("same", "cross")
        for i in range(5)
    ]
    wave = choose_wave(
        pool + [row("add", "same", "old")],
        old,
        accepted,
        q,
        wave_size=4,
        max_attempts=5,
    )
    assert len(wave) == 4 and all(r["operation"] == "add" for r in wave)
    assert len({r["candidate"]["source_id"] for r in wave}) == 4
    assert all(r["candidate"]["source_id"] != "old" for r in wave)


def test_caps_quality_duplicates_and_resume():
    q = quotas(8, 0.5)
    g = good("attribute", "same", "a")
    bad = good("add", "same", "bad")
    bad["group_checks"]["group_pass"] = False
    groups = [g, g, good("attribute", "same", "b"), bad]
    assert len(accepted_groups(groups, q)) == 1
    assert progress(groups, [], q)["accepted"] == 1
    assert choose_wave([], [], [], q, wave_size=4, max_attempts=1) == []
    assert (
        choose_wave(
            [row("add", "same", "new")],
            [row("add", "same", "old")],
            [],
            q,
            wave_size=4,
            max_attempts=1,
        )
        == []
    )


def test_low_yield_gets_more_attempts_without_relaxing_quota():
    q = {"add:cross": 4, "remove:cross": 4}
    attempted = [row("add", "cross", "old_a" + str(i)) for i in range(8)] + [
        row("remove", "cross", "old_r" + str(i)) for i in range(4)
    ]
    accepted = [good("remove", "cross", "r" + str(i)) for i in range(2)]
    pool = [
        row(op, "cross", op + str(i)) for op in ("add", "remove") for i in range(20)
    ]
    wave = choose_wave(pool, attempted, accepted, q, wave_size=10, max_attempts=22)
    assert sum(r["operation"] == "add" for r in wave) > sum(
        r["operation"] == "remove" for r in wave
    )


def test_coordinator_resumes_and_refills_only_rejected_cell(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from sa1b_pipeline.production_state import read_rows, write_rows
    from scripts.run_dynamic_twins import run

    root = tmp_path / "run"
    pool = [
        row(op, rel, op + rel + str(i))
        for op in ("add", "remove", "replace", "attribute")
        for rel in ("same", "cross")
        for i in range(3)
    ]
    for r in pool:
        r["objects"] = [{"category": "bag"}]
    write_rows(root / "inputs/candidate_pool.jsonl", pool)
    a = SimpleNamespace(
        output_root=root,
        target=4,
        same_fraction=0.3,
        manifest=tmp_path / "m.jsonl",
        reservation=tmp_path / "reserve.json",
        seed=1,
        pool_per_cell=3,
        selection_only=False,
        stop_after="planning",
        candidate_extension=None,
        max_attempts=8,
        wave_size=4,
        planner_python="python",
        editor_python="python",
        mllm_model="planner",
        editor_model="editor",
        site_model=None,
        mllm_devices="0",
        editor_devices="1",
        audit_model=None,
        audit_tp=1,
        audit_devices=None,
    )
    calls = []

    def fake(cmd, **kwargs):
        from pathlib import Path

        wave = Path(cmd[cmd.index("--output-root") + 1])
        stage = (
            "planning"
            if "--plan-only" in cmd
            else "generation"
            if "--generate-only" in cmd
            else "audit"
        )
        calls.append((wave.name, stage))
        selected = read_rows(wave / "inputs/selected.jsonl")
        groups = [
            good(r["operation"], r["category_relation"], r["candidate"]["source_id"])
            for r in selected
        ]
        if wave.name == "001":
            for g in groups:
                if g["operation"] == "add":
                    g["group_checks"]["group_pass"] = False
        write_rows(
            wave
            / (
                {"planning": "planned", "generation": "generated", "audit": "audited"}[
                    stage
                ]
                + ".jsonl"
            ),
            groups,
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.run_dynamic_twins.subprocess.run", fake)
    monkeypatch.setattr(
        "scripts.run_dynamic_twins.refresh_twin_add_checks", lambda group: True
    )
    run(a)
    assert calls == [("001", "planning")]
    run(a)
    assert calls == [
        ("001", "planning")
    ]  # Same boundary does not accidentally generate.
    a.stop_after = None
    run(a)
    assert calls.count(("001", "planning")) == 1
    accepted = read_rows(root / "accepted.jsonl")
    assert len(accepted) == 4
    assert {g["operation"] for g in accepted} == {
        "add",
        "remove",
        "replace",
        "attribute",
    }
    second = read_rows(root / "waves/002/inputs/selected.jsonl")
    assert all(r["operation"] == "add" for r in second)
    assert json.loads((root / "progress.json").read_text())["complete"]
