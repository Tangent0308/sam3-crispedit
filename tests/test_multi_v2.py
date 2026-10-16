from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from sa1b_pipeline.group_generation import multi_support_preflight
from sa1b_pipeline.group_audit import multi_audit_comparison
from sa1b_pipeline.specialist_editors import writable_mask


def fixture_group(tmp_path, gap=6):
    src = tmp_path / "source.png"
    Image.new("RGB", (200, 120), (100, 100, 100)).save(src)
    members = []
    for i, box in enumerate(((20, 30, 70, 90), (70 + gap, 30, 120 + gap, 90))):
        mask = np.zeros((120, 200), np.uint8)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 255
        path = tmp_path / f"mask{i}.png"
        Image.fromarray(mask).save(path)
        members.append(
            {
                "unit_id": f"U{i + 1}",
                "mask": str(path),
                "operation": "remove" if i == 0 else "attribute",
                "unit": {
                    "mask_path": str(path),
                    "operation": "remove" if i == 0 else "attribute",
                },
            }
        )
    return {"source_image": str(src), "units": members}


def test_multi_small_support_overlap_is_protected(tmp_path):
    group = fixture_group(tmp_path)
    result = multi_support_preflight(group, tmp_path / "out")
    assert result["pass"]
    assert 0 < result["pairs"][0]["support_coverage_of_other"] <= 0.1
    support = writable_mask(
        group["units"][0]["unit"], (200, 120), require_add_placement=False
    )
    neighbor = np.asarray(Image.open(group["units"][1]["mask"])) > 127
    assert not (support & neighbor).any()
    own = np.asarray(Image.open(group["units"][0]["mask"])) > 127
    assert (support[own]).all()


def test_multi_large_collar_overlap_is_clipped_not_rejected(tmp_path):
    group = fixture_group(tmp_path, gap=0)
    result = multi_support_preflight(group, tmp_path / "out")
    assert result["pass"]
    assert result["pairs"][0]["support_coverage_of_other"] > 0.1
    assert result["pairs"][0]["effective_support_coverage_of_other"] == 0
    support = writable_mask(
        group["units"][0]["unit"], (200, 120), require_add_placement=False
    )
    neighbor = np.asarray(Image.open(group["units"][1]["mask"])) > 127
    assert not (support & neighbor).any()


def test_multi_overlapping_source_targets_rejected(tmp_path):
    group = fixture_group(tmp_path, gap=-2)
    result = multi_support_preflight(group, tmp_path / "out")
    assert not result["pass"]
    assert result["pairs"][0]["source_overlap_pixels"] > 0
    assert "execution_mask_path" not in group["units"][0]["unit"]


def test_multi_joint_panel_retains_last_target_and_actual_order(tmp_path):
    group = fixture_group(tmp_path)
    red = tmp_path / "red.png"
    Image.new("RGB", (200, 120), "red").save(red)
    blue = tmp_path / "blue.png"
    Image.new("RGB", (200, 120), "blue").save(blue)
    group["joint_target"] = str(blue)
    group["units"] = [deepcopy(group["units"][i % 2]) for i in range(5)]
    group["generation_order"] = ["U5", "U4", "U3", "U2", "U1"]
    for i, u in enumerate(group["units"]):
        u.update(
            unit_id=f"U{i + 1}", step_output_image=str(red), support_mask=u["mask"]
        )
    path = multi_audit_comparison(group, tmp_path / "panel.png")
    panel = np.asarray(Image.open(path))
    assert panel.shape[:2] == (1492, 1536)
    # Last row still shows its own step and final, not a duplicated full-image tile.
    y = 52 + 4 * 288 + 160
    assert tuple(panel[y, 576]) == (255, 0, 0)
    assert tuple(panel[y, 960]) == (0, 0, 255)


def test_five_target_schedules_allow_repeat_before_last_member():
    from sa1b_pipeline.multi_operation_balance import operation_schedules

    expected = ("add", "add", "remove", "replace", "attribute")
    objects = [{"i": i} for i in range(5)]
    schedules = operation_schedules(
        objects, (100, 100), lambda o, op, size: op == expected[o["i"]]
    )
    assert schedules == [expected]
    assert len(operation_schedules(objects, (100, 100), lambda *_: True)) == 240


def test_cutout_preserves_selected_colors_and_hides_excluded_content():
    from sa1b_pipeline.group_planning import _crop_sheet

    source = Image.new("RGB", (100, 100), "red")
    mask = np.ones((100, 100), dtype=bool)
    mask[25:75, 25:75] = False
    panel = _crop_sheet(source, [{"object_id": "x"}], [mask], locator_cutout=True)
    assert panel.size == (1536, 304)
    a = np.asarray(panel)
    # Center is the excluded hole, even though the outer boundary surrounds it.
    assert tuple(a[168, 1280]) == (100, 100, 100)
    assert tuple(a[48, 1280]) == (255, 0, 0)


def test_multi_soft_fit_ranks_assignments_inside_the_same_quota_cell():
    from sa1b_pipeline.group_production import multi_priority, multi_cell

    objects = [{"category": "chair"}, {"category": "panel"}]
    good = {"objects": objects, "operations": ["replace", "add"]}
    poor = {"objects": objects, "operations": ["add", "replace"]}
    assert multi_cell(good["operations"]) == multi_cell(poor["operations"])
    assert multi_priority(good) > multi_priority(poor)


def test_material_only_replacement_rejected_but_new_shape_allowed():
    import pytest
    from sa1b_pipeline.group_planning import _validate_group_edit_fields

    with pytest.raises(ValueError, match="physical form"):
        _validate_group_edit_fields(
            {
                "operation": "replace",
                "scope": "table",
                "ref_phrase": "the white table surface",
                "payload": "a dark wood table surface",
            }
        )
    _validate_group_edit_fields(
        {"operation": "replace", "scope": "table", "payload": "a round wooden table"}
    )


def test_audit_stage_retries_only_transient_contract_visibility(tmp_path, monkeypatch):
    import sys
    from scripts.group_production import stage

    (tmp_path / "generated.jsonl").write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage",
            "--stage",
            "audit",
            "--wave",
            str(tmp_path),
            "--kind",
            "multi",
            "--model",
            "test-model",
        ],
    )
    attempts = []

    def transient(path, value):
        attempts.append(value)
        if len(attempts) < 3:
            raise FileNotFoundError(path)

    monkeypatch.setattr(stage, "immutable_json", transient)
    monkeypatch.setattr(stage.time, "sleep", lambda _: None)
    stage.main()
    assert len(attempts) == 3
    assert (tmp_path / "audited_worker0.jsonl").read_text() == ""
    import pytest

    def changed(path, value):
        raise ValueError("checkpoint/config changed")

    monkeypatch.setattr(stage, "immutable_json", changed)
    with pytest.raises(ValueError, match="changed"):
        stage.main()
