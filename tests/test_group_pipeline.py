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
