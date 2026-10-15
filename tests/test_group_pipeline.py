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

