from pathlib import Path

import numpy as np
from PIL import Image

from sa1b_pipeline.group_geometry import precise_point, twin_pair_gate, adjacent_pair_gate
from sa1b_pipeline.group_planning import build_multi_group, build_twin_group, group_views
from sa1b_pipeline.group_audit import (_call_batched, _semantic_group_pass,
                                        MULTI_REQUIRED_AUDIT_FIELDS, TWIN_REQUIRED_AUDIT_FIELDS)
from scripts.run_group_pipeline import _require_manual_review
from sa1b_pipeline.group_schema import validate_twin_record, validate_multi_record
from sa1b_pipeline.group_selection import select_twin_candidates, select_multi_candidates
from sa1b_pipeline.source_isolation import assert_no_eval_leakage


def _source(tmp_path):
    image = tmp_path / "source.jpg"
    Image.new("RGB", (128, 96), (130, 130, 130)).save(image)
    masks = []
    for i, box in enumerate(((12, 20, 36, 48), (72, 24, 98, 52))):
        mask = np.zeros((96, 128), dtype=np.uint8)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 255
        path = tmp_path / f"mask{i}.png"
        Image.fromarray(mask).save(path)
        masks.append(path)
    objects = [{"object_id": str(i), "category": "backpack", "mask_path": str(p), "metadata": {}}
               for i, p in enumerate(masks)]
    return image, objects


def test_precise_point_is_inside_and_not_bbox_corner():
    mask = np.zeros((20, 20), bool)
    mask[4:16, 7:14] = True
    point = precise_point(mask)
    assert mask[point[1], point[0]]
    assert point[0] >= 8 and point[0] <= 12


def test_group_views_and_twin_record_contract(tmp_path):
    image, objects = _source(tmp_path)
    candidate = {"source_id": "sa_test", "image_path": str(image), "image_sha256": "x", "objects": objects,
                 "provenance": {"source": "test"}}
    paths, mask_paths, _ = group_views(candidate, objects, tmp_path / "views")
    assert len(paths) == 2 and all(Path(p).is_file() for p in paths)
    geom = []
    for p in mask_paths:
        arr = np.asarray(Image.open(p)) > 0
        ys, xs = np.where(arr)
        from sa1b_pipeline.group_geometry import precise_point
        geom.append({"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                     "point": precise_point(arr), "area_fraction": float(arr.mean())})
    plan = {"feasible": True, "operation": "attribute", "scope": "backpack", "payload": "bright yellow",
            "attributes": [{"property": "color", "value": "bright yellow"}], "target_part": "",
            "remove_mode": "whole", "placement": None, "qualifier": "",
            "members": [{"object_id": "0", "ref_phrase": "the backpack on the left", "evidence": "left"},
                        {"object_id": "1", "ref_phrase": "the backpack on the right", "evidence": "right"}]}
    record = build_twin_group(candidate, objects, plan, mask_paths, split="train", scope_level="object",
                              group_id="twin-test", geometry=geom)
    for m in record["members"]:
        m["support_mask"] = m["mask"]
        m["edited_image"] = str(image)
    validate_twin_record(record)
    assert record["members"][0]["instruction_noref"] == record["members"][1]["instruction_noref"]


def test_pair_gates_require_same_category_and_adjacency(tmp_path):
    image, objects = _source(tmp_path)
    gate = twin_pair_gate(objects[0], objects[1], image_size=(128, 96))
    assert gate["eligible"]
    objects[1]["category"] = "chair"
    assert not twin_pair_gate(objects[0], objects[1], image_size=(128, 96))["eligible"]


def test_three_member_twin_selection_keeps_distinct_instances(tmp_path):
    image = tmp_path / "triple.jpg"
    Image.new("RGB", (160, 96), (130, 130, 130)).save(image)
    objects = []
    for i, (x0, x1) in enumerate(((8, 32), (60, 84), (116, 140))):
        mask = np.zeros((96, 160), dtype=np.uint8)
        mask[28:58, x0:x1] = 255
        path = tmp_path / f"triple_mask{i}.png"
        Image.fromarray(mask).save(path)
        objects.append({"object_id": str(i), "category": "backpack", "mask_path": str(path),
                        "bbox": [x0, 28, x1, 58], "metadata": {}})
    rows = select_twin_candidates([{"source_id": "triple", "image_path": str(image), "objects": objects}],
                                  target=1, operation="attribute", member_count=3)
    assert len(rows) == 1
    assert len(rows[0]["objects"]) == 3
    assert len({o["object_id"] for o in rows[0]["objects"]}) == 3


def test_eval_leakage_rejected(tmp_path):
    reservation = tmp_path / "reservation.json"
    reservation.write_text('{"schema_version":"sa1b.samtok.source_isolation.v1","reserved_ids":["eval"],"reserved_image_sha256":[]}')
    try:
        assert_no_eval_leakage([{"source_id": "eval"}], reservation)
    except ValueError:
        pass
    else:
        raise AssertionError("eval source was not rejected")


def test_multi_selection_supports_fast_capacity_and_strict_modes(tmp_path):
    image = tmp_path / "multi.jpg"
    Image.new("RGB", (128, 96), (130, 130, 130)).save(image)
    objects = []
    for i, (x0, x1) in enumerate(((16, 36), (36, 56), (78, 102))):
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[30:50, x0:x1] = 255
        path = tmp_path / f"multi_mask{i}.png"
        Image.fromarray(mask).save(path)
        objects.append({"object_id": str(i), "category": "backpack", "mask_path": str(path),
                        "bbox": [x0, 30, x1, 50], "metadata": {}})
    candidate = {"source_id": "multi", "image_path": str(image), "objects": objects}
    fast = select_multi_candidates([candidate], target=1, seed=7, strict_geometry=False)
    strict = select_multi_candidates([candidate], target=1, seed=7, strict_geometry=True)
    assert len(fast) == len(strict) == 1
    assert fast[0]["adjacent_pairs"][0]["geometry_exact"] is False
    assert strict[0]["adjacent_pairs"][0]["geometry_exact"] is True


def test_multi_attribute_rejects_value_repeated_in_source_selector(tmp_path):
    image, objects = _source(tmp_path)
    objects[0]["category"] = "red SUV"
    candidate = {"source_id": "multi-attr", "image_path": str(image), "objects": objects}
    _, mask_paths, _ = group_views(candidate, objects[:1], tmp_path / "attr_views")
    mask = np.asarray(Image.open(mask_paths[0])) > 0
    ys, xs = np.where(mask)
    geometry = [{"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                 "point": precise_point(mask), "area_fraction": float(mask.mean())}]
    plan = {"feasible": True, "generation_order": ["U1"], "targets": [{
        "unit_id": "U1", "object_id": "0", "operation": "attribute",
        "scope": "red SUV", "ref_phrase": "the red SUV", "payload": "",
        "target_part": "", "remove_mode": "whole", "placement": None,
        "attributes": [{"property": "color", "value": "red"}], "qualifier": "",
        "evidence": "same red source"
    }]}
    try:
        build_multi_group(candidate, objects[:1], plan, mask_paths, split="train",
                          group_id="multi-attr-test", geometry=geometry,
                          assignments={"0": "attribute"})
    except ValueError as exc:
        assert "already present" in str(exc)
    else:
        raise AssertionError("source attribute no-op was accepted")


def test_ref_color_must_match_explicit_dense_object_color(tmp_path):
    image, objects = _source(tmp_path)
    objects[0].update({"category": "car,suv", "description": "A black SUV with a glossy finish."})
    candidate = {"source_id": "color-ref", "image_path": str(image), "objects": objects[:1]}
    _, mask_paths, _ = group_views(candidate, objects[:1], tmp_path / "color_views")
    mask = np.asarray(Image.open(mask_paths[0])) > 0
    ys, xs = np.where(mask)
    geometry = [{"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                 "point": precise_point(mask), "area_fraction": float(mask.mean())}]
    plan = {"feasible": True, "generation_order": ["U1"], "targets": [{
        "unit_id": "U1", "object_id": "0", "operation": "remove", "scope": "suv",
        "ref_phrase": "the red SUV", "payload": "", "target_part": "",
        "remove_mode": "whole", "placement": None, "attributes": [], "qualifier": "",
        "evidence": "wrong color"
    }]}
    try:
        build_multi_group(candidate, objects[:1], plan, mask_paths, split="train",
                          group_id="color-ref-test", geometry=geometry, assignments={"0": "remove"})
    except ValueError as exc:
        assert "reference color" in str(exc)
    else:
        raise AssertionError("contradictory ref color was accepted")


def test_multi_plan_requires_exact_unit_ids_and_unique_refs(tmp_path):
    image, objects = _source(tmp_path)
    candidate = {"source_id": "multi-id", "image_path": str(image), "objects": objects}
    _, mask_paths, _ = group_views(candidate, objects, tmp_path / "id_views")
    geometry = []
    for path in mask_paths:
        mask = np.asarray(Image.open(path)) > 0
        ys, xs = np.where(mask)
        geometry.append({"bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                         "point": precise_point(mask), "area_fraction": float(mask.mean())})
    targets = []
    for index, obj in enumerate(objects, 1):
        targets.append({"unit_id": "U1", "object_id": str(obj["object_id"]),
                        "operation": "remove", "scope": "backpack", "ref_phrase": "the backpack",
                        "payload": "", "target_part": "", "remove_mode": "whole", "placement": None,
                        "attributes": [], "qualifier": "", "evidence": "same"})
    plan = {"feasible": True, "generation_order": ["U1", "U2"], "targets": targets}
    try:
        build_multi_group(candidate, objects, plan, mask_paths, split="train", group_id="multi-id-test",
                          geometry=geometry, assignments={"0": "remove", "1": "remove"})
    except ValueError as exc:
        assert "mismatch" in str(exc) or "reference" in str(exc)
    else:
        raise AssertionError("invalid unit ids/references were accepted")


def test_group_publish_manual_review_gate_requires_signed_fields(tmp_path):
    ledger = tmp_path / "human_review.jsonl"
    ledger.write_text(
        '{"group_id":"g1","reviewer":"reviewer","status":"manual_pass",'
        '"decision":"manual_pass","reference_correct":true,"target_done":true,'
        '"non_target_preserved":true,"natural":true}\n', encoding="utf-8")
    groups = [{"twin_id": "g1"}]
    summary = _require_manual_review(groups, ledger, kind="twins", split="eval")
    assert summary["passing_groups"] == 1 and summary["all_eval_groups"]


def test_group_audit_requires_all_fields_before_recomputing_pass():
    twin = {"decision": "pass", **{field: True for field in TWIN_REQUIRED_AUDIT_FIELDS}}
    multi = {"decision": "pass", **{field: True for field in MULTI_REQUIRED_AUDIT_FIELDS}}
    assert _semantic_group_pass(twin, TWIN_REQUIRED_AUDIT_FIELDS)
    assert _semantic_group_pass(multi, MULTI_REQUIRED_AUDIT_FIELDS)
    multi["no_crosstalk"] = False
    assert not _semantic_group_pass(multi, MULTI_REQUIRED_AUDIT_FIELDS)


def test_group_audit_batches_without_reordering():
    class Fake:
        batch_size = 2

        def __init__(self):
            self.batch_sizes = []

        def call_batch(self, jobs):
            self.batch_sizes.append(len(jobs))
            return [{"job_id": job["job_id"]} for job in jobs]

    model = Fake()
    jobs = [{"job_id": str(i)} for i in range(5)]
    result = _call_batched(model, jobs)
    assert [row["job_id"] for row in result] == [str(i) for i in range(5)]
    assert model.batch_sizes == [2, 2, 1]



def test_clean_group_views_preserve_color_and_mask_geometry(tmp_path):
    image, objects = _source(tmp_path)
    candidate = {"source_id": "s", "image_path": str(image), "objects": objects}
    paths, masks, digest = group_views(candidate, objects, tmp_path / "views")
    assert np.array_equal(np.asarray(Image.open(paths[0])), np.asarray(Image.open(image)))
    sheet = np.asarray(Image.open(paths[1]))
    # The photographic half is never tinted with a guide color.
    photo = sheet[32:352, :512]
    assert np.all(photo[..., 0] == photo[..., 1])
    assert np.all(photo[..., 1] == photo[..., 2])
    locator = sheet[32:352, 512:]
    assert set(np.unique(locator)) <= {0, 35, 255}
    # Geometry updates invalidate the view cache even for the same object ID.
    changed = np.asarray(Image.open(objects[0]["mask_path"])).copy()
    changed[0:10, 0:10] = 255
    Image.fromarray(changed).save(objects[0]["mask_path"])
    _, _, next_digest = group_views(candidate, objects, tmp_path / "views")
    assert next_digest != digest


def test_twin_assignment_eval_size_and_direct_anchor_fail_closed():
    import pytest
    base = {"feasible": True, "operation": "remove"}
    with pytest.raises(ValueError, match="assigned operation"):
        build_twin_group({}, [], base, [], split="train", scope_level="object",
                         group_id="x", geometry=[], assigned_operation="add")
    with pytest.raises(ValueError, match="exactly two"):
        build_twin_group({}, [{}, {}, {}], base, [], split="eval", scope_level="object",
                         group_id="x", geometry=[])
    with pytest.raises(ValueError, match="direct support"):
        build_twin_group({}, [], {"operation": "add", "placement": {"anchor_mode": "reference"}},
                         [], split="train", scope_level="object", group_id="x", geometry=[])


def test_member_audit_failure_cannot_be_overridden_by_group_pass(tmp_path):
    from sa1b_pipeline.group_audit import audit_twin_groups
    image, objects = _source(tmp_path)
    members = [{"unit_id": f"U{i+1}", "object_id": obj["object_id"], "ref": f"bag {i}",
                "operation": "attribute", "instruction": "Change the bag color to blue.",
                "instruction_spec": {}, "mask": obj["mask_path"], "support_mask": obj["mask_path"],
                "edited_image": str(image)} for i, obj in enumerate(objects)]
    group = {"twin_id": "tw", "source_image": str(image), "operation": "attribute",
             "members": members, "group_checks": {k: {"pass": True} for k in
             ("deterministic", "cross_target_preservation", "support_gap")}}
    class Model:
        batch_size = 2
        def call_batch(self, jobs):
            results = []
            for job in jobs:
                if "::" in job["job_id"]:
                    assert "Expected operation: attribute" in job["prompt"]
                    audit = dict(decision="pass", target_done=True, non_target_preserved=True,
                                 natural=True, operation_correct=not job["job_id"].endswith("U2"))
                else:
                    audit = {"decision": "pass", **dict.fromkeys(TWIN_REQUIRED_AUDIT_FIELDS, True)}
                results.append({"job_id": job["job_id"], "semantic_audit": audit})
            return results
    audited = audit_twin_groups([group], Model(), tmp_path / "audit")[0]
    assert audited["group_checks"]["semantic_audit_pass"]
    assert not audited["group_checks"]["group_pass"]
    assert not audited["members"][1]["semantic_audit"]["operation_correct"]


def test_multi_step_audit_compares_each_actual_input_and_output(tmp_path):
    from sa1b_pipeline.group_audit import audit_multi_groups
    image, objects = _source(tmp_path)
    mid, final = tmp_path / "mid.png", tmp_path / "final.png"
    Image.open(image).save(mid)
    Image.open(image).save(final)
    units = [{"unit_id": f"U{i+1}", "object_id": obj["object_id"], "ref": f"bag {i}",
              "operation": "attribute", "instruction": "Change the bag color to blue.",
              "instruction_spec": {}, "mask": obj["mask_path"], "support_mask": obj["mask_path"],
              "step_input_image": str(image if i == 0 else mid),
              "step_output_image": str(mid if i == 0 else final)} for i, obj in enumerate(objects)]
    group = {"multi_id": "mu", "source_image": str(image), "units": units,
             "joint_target": str(final), "joint_audit": {"steps": [{"decision": "pending_semantic_audit"}]*2,
             "outside_union_support_fraction": 0.0}}
    class Model:
        batch_size = 2
        def call_batch(self, jobs):
            results = []
            for job in jobs:
                if "::" in job["job_id"]:
                    i = int(job["job_id"][-1])-1
                    assert job["images"][:2] == [units[i]["step_input_image"], units[i]["step_output_image"]]
                    audit = dict(decision="pass", target_done=True, non_target_preserved=True,
                                 natural=True, operation_correct=True)
                else:
                    audit = {"decision": "pass", **dict.fromkeys(MULTI_REQUIRED_AUDIT_FIELDS, True)}
                results.append({"job_id": job["job_id"], "semantic_audit": audit})
            return results
    assert audit_multi_groups([group], Model(), tmp_path / "audit")[0]["joint_audit"]["pass"]


def test_twin_noop_uses_source_fields_and_word_boundaries(tmp_path):
    image, objects = _source(tmp_path)
    for obj in objects:
        obj["category"] = "bag,backpack"
    geometry = [{"bbox": [1, 1, 20, 20], "point": [5, 5], "area_fraction": .05}] * 2
    plan = {"feasible": True, "operation": "attribute", "scope": "backpack", "payload": "",
            "attributes": [{"property": "color", "value": "red"}],
            "members": [{"object_id": str(i), "ref_phrase": f"the backpack on the {side}",
                         "source_color": "grey", "evidence": "Partially obscured; can become red."}
                        for i, side in enumerate(("left", "right"))]}
    group = build_twin_group({"source_id": "s", "image_path": str(image)}, objects, plan,
                            [o["mask_path"] for o in objects], split="train", scope_level="object",
                            group_id="tw", geometry=geometry, assigned_operation="attribute")
    assert "selected backpack" in group["instruction_noref"]
    import pytest
    plan["members"][0]["source_color"] = "red"
    with pytest.raises(ValueError, match="already present"):
        build_twin_group({}, objects, plan, [], split="train", scope_level="object",
                         group_id="tw", geometry=geometry)


def test_audit_only_preserves_generated_checkpoint_without_loading_editor(tmp_path, monkeypatch):
    import json
    import scripts.run_group_pipeline as runner
    image, objects = _source(tmp_path)
    root = tmp_path / "run"
    (root / "inputs").mkdir(parents=True)
    candidate = {"source_id": "s", "image_path": str(image), "objects": objects}
    selection = {"candidate": candidate, "objects": objects, "operation": "remove",
                 "scope_level": "object", "gate": {}}
    (root / "inputs/selected.jsonl").write_text(json.dumps(selection) + "\n")
    # Audit rejection remains available for diagnosis; it never enters final.
    group = {"twin_id": "tw", "group_checks": {"group_pass": False}}
    (root / "generated.jsonl").write_text(json.dumps(group) + "\n")
    before = (root / "generated.jsonl").read_bytes()
    monkeypatch.setattr(runner, "_candidate_rows", lambda *a: [candidate])
    def forbidden(*a):
        raise AssertionError("audit-only loaded editor")
    monkeypatch.setattr(runner, "_editor", forbidden)
    class Auditor:
        def close(self):
            pass
    monkeypatch.setattr(runner, "_auditor", lambda *a: Auditor())
    monkeypatch.setattr(runner, "audit_twin_groups", lambda groups, *a: groups)
    runner.main(["--kind", "twins", "--manifest", str(tmp_path / "manifest.jsonl"),
                 "--output-root", str(root), "--target", "1", "--audit-only", "--audit-model", "fake"])
    assert (root / "generated.jsonl").read_bytes() == before
    assert json.loads((root / "audited.jsonl").read_text())["twin_id"] == "tw"
    assert (root / "final/twins_train.jsonl").read_text() == ""
    assert not json.loads((root / "release.json").read_text())["published"]


def test_exact_twin_area_ratio_rejects_bbox_only_false_positive(tmp_path):
    import pytest
    image, objects = _source(tmp_path)
    plan = {"feasible": True, "operation": "remove", "scope": "backpack", "payload": "",
            "members": [{"object_id": str(i), "ref_phrase": f"the {side} backpack"}
                        for i, side in enumerate(("left", "right"))]}
    geometry = [{"bbox": [1, 1, 20, 20], "point": [5, 5], "area_fraction": area}
                for area in (.12, .02)]
    with pytest.raises(ValueError, match="exact mask area ratio"):
        build_twin_group({"source_id": "s", "image_path": str(image)}, objects, plan,
                         [o["mask_path"] for o in objects], split="train", scope_level="object",
                         group_id="tw", geometry=geometry)


def test_review_pdf_includes_fifth_step_and_preserves_decisions(tmp_path):
    import json
    from scripts.review_group_data import main
    image, objects = _source(tmp_path)
    units = [{"unit_id": f"U{i}", "ref": "the backpack", "instruction": "Remove the backpack.",
              "mask": objects[0]["mask_path"], "step_input_image": str(image),
              "step_output_image": str(image)} for i in range(1, 6)]
    group = {"multi_id": "m", "source_id": "s", "source_image": str(image), "units": units,
             "generation_order": ["U5", "U4", "U3", "U2", "U1"]}
    manifest = tmp_path / "groups.jsonl"
    manifest.write_text(json.dumps(group) + "\n")
    out = tmp_path / "review"
    out.mkdir()
    decision = {"group_id": "m", "reviewer": "assistant", "status": "reviewed", "decision": "fail"}
    (out / "human_review.jsonl").write_text(json.dumps(decision) + "\n")
    main(["--manifest", str(manifest), "--kind", "multi", "--output-dir", str(out)])
    index = json.loads((out / "page_index.json").read_text())
    assert [r["unit_id"] for r in index] == group["generation_order"]
    assert all(Path(r["image"]).is_file() for r in index)
    assert json.loads((out / "human_review.jsonl").read_text()) == decision


def test_optional_contour_keeps_clean_pixels_and_stays_outside_mask(tmp_path):
    image, objects = _source(tmp_path)
    candidate = {"source_id": "s", "image_path": str(image), "objects": objects}
    base, _, base_digest = group_views(candidate, objects, tmp_path / "views")
    outlined, _, outline_digest = group_views(candidate, objects, tmp_path / "views", locator_outline=True)
    before = np.asarray(Image.open(base[1]))
    after = np.asarray(Image.open(outlined[1]))
    assert np.array_equal(before, after[:, :1024])
    assert base_digest != outline_digest
    for top in (32, 384):
        mask = np.asarray(Image.open(outlined[1]))[top:top+320,512:1024,0] == 255
        clean = after[top:top+320,:512]
        guided = after[top:top+320,1024:]
        assert np.array_equal(clean[mask], guided[mask])
        assert np.any(clean[~mask] != guided[~mask])


def test_precise_point_handles_mask_touching_image_border():
    mask = np.ones((9, 9), dtype=bool)
    assert precise_point(mask) == [4, 4]
    # A thin selector far from the image edge still needs its own centre.
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:10, 3:30] = True
    x, y = precise_point(mask)
    assert y == 7 and 5 <= x <= 27


def test_fast_twin_selection_checks_real_area_not_bbox_area(tmp_path):
    image, objects = _source(tmp_path)
    objects[0]['bbox'] = [12,20,36,48]
    objects[1]['bbox'] = [72,24,98,52]
    thin = np.zeros((96,128),dtype=np.uint8)
    thin[20:22,12:36] = 255
    Image.fromarray(thin).save(objects[0]['mask_path'])
    candidate = {'source_id':'s','image_path':str(image),'objects':objects}
    assert select_twin_candidates([candidate],target=1,operation='attribute',strict_geometry=False) == []


def test_group_edit_fields_reject_observed_semantic_and_grammar_failures():
    import pytest
    from sa1b_pipeline.group_planning import _validate_group_edit_fields
    with pytest.raises(ValueError, match="physical form"):
        _validate_group_edit_fields({"operation": "replace", "scope": "stripe", "payload": "a solid black stripe"})
    _validate_group_edit_fields({"operation": "replace", "scope": "wheel rim", "payload": "a gold five-spoke wheel rim"})
    for site in ("empty", "optional noun", "arm guard"):
        with pytest.raises(ValueError, match="placement site"):
            _validate_group_edit_fields({"operation": "add", "scope": "arm guard", "placement": {"site": site}})
    for site in ("", "top edge"):
        _validate_group_edit_fields({"operation": "add", "scope": "arm guard", "placement": {"site": site}})


def test_review_sample_quota_cannot_hide_reviewed_failure(tmp_path):
    import json
    import pytest
    groups = [{"twin_id": f"g{i}"} for i in range(201)]
    rows = [{"group_id": g["twin_id"], "status": "reviewed", "reviewer": "assistant", "decision": "pass",
             "reference_correct": True, "target_done": True, "non_target_preserved": True, "natural": True}
            for g in groups]
    rows[-1]["decision"] = "fail"
    path = tmp_path / "review.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(RuntimeError, match="explicitly rejected"):
        _require_manual_review(groups, path, kind="twins", split="train")


def test_render_resume_verifies_inputs_and_preserves_context(tmp_path):
    import pytest
    from sa1b_pipeline.group_generation import _render_member
    image, objects = _source(tmp_path)
    unit = {"mask_path": str(objects[0]["mask_path"]), "instruction_ref": "Change it to red."}
    class Editor:
        config = {"steps": 40}
        last_generation_context = {}
        calls = 0
        def generate(self, source, unit, seed):
            self.calls += 1
            self.last_generation_context = {"render_call": self.calls, "seed": seed}
            return Image.open(source).convert("RGB")
    editor = Editor()
    output = tmp_path / "edited.png"
    context = _render_member(str(image), unit, output, editor, 7)
    editor.last_generation_context = {"unrelated": True}
    assert _render_member(str(image), unit, output, editor, 7) == context
    assert editor.calls == 1 and context["seed"] == 7
    with pytest.raises(ValueError, match="input changed"):
        _render_member(str(image), unit, output, editor, 8)
    with pytest.raises(ValueError, match="input changed"):
        _render_member(str(image), {**unit, "instruction_ref": "Remove it."}, output, editor, 7)
    Image.new("RGB", (128, 96), "red").save(image)
    with pytest.raises(ValueError, match="input changed"):
        _render_member(str(image), unit, output, editor, 7)


def test_render_resume_rejects_unverified_legacy_image(tmp_path):
    import pytest
    from sa1b_pipeline.group_generation import _render_member
    image, objects = _source(tmp_path)
    class Editor:
        config = {}
    with pytest.raises(ValueError, match="no verifiable"):
        _render_member(str(image), {"mask_path": str(objects[0]["mask_path"])}, image, Editor(), 0)


def test_scene_locator_preserves_selected_pixels_and_crop_alignment(tmp_path):
    image, objects = _source(tmp_path)
    candidate = {"source_id": "s", "image_path": str(image)}
    plain, masks, old_digest = group_views(candidate, objects, tmp_path / "views")
    located, _, new_digest = group_views(candidate, objects, tmp_path / "views", locate_in_scene=True)
    assert old_digest != new_digest
    assert len(located) == 2
    assert Image.open(plain[1]).tobytes() == Image.open(located[1]).tobytes()
    before = np.asarray(Image.open(plain[0]))
    after = np.asarray(Image.open(located[0]))
    union = np.logical_or.reduce([np.asarray(Image.open(p)) > 0 for p in masks])
    assert np.array_equal(before[union], after[union])
    assert np.any(before[~union] != after[~union])


def test_twin_category_exclusions_cover_modified_nouns_not_materials():
    from sa1b_pipeline.group_geometry import category_allowed
    for name in ("artificial autumn leaves", "brick wall", "large grey boulder", "sandy embankment", "stone,rock"):
        assert not category_allowed(name)[0]
    for name in ("stone fountain", "wall clock", "leaf-shaped pendant", "brick chimney", "shirt sleeve"):
        assert category_allowed(name)[0]
