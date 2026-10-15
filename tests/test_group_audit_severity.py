from sa1b_pipeline.group_audit import _acknowledged_hard_defects


def test_serious_warning_is_not_softened():
    assert _acknowledged_hard_defects(["Person appears to be floating without a seat"])
    assert _acknowledged_hard_defects(["The target is unchanged"])
    assert _acknowledged_hard_defects(["Wrong target was edited"])


def test_minor_and_negated_notes_remain_soft():
    assert not _acknowledged_hard_defects(
        [
            "Minor texture variation",
            "Style may be unusual",
            "No floating objects",
            "No obvious wrong target",
            "The person is not floating",
        ]
    )


def test_joint_gate_rejects_acknowledged_defect_but_keeps_minor_warning():
    from copy import deepcopy

    from sa1b_pipeline.group_audit import JOINT_MEMBER_FIELDS, _apply_joint_twin_audit

    group = {
        "members": [{"unit_id": "U1"}, {"unit_id": "U2"}],
        "group_checks": {
            key: {"pass": True}
            for key in (
                "blind_disambiguation",
                "deterministic",
                "cross_target_preservation",
                "support_gap",
            )
        },
    }
    result = {
        "semantic_audit": {
            "members": [
                {
                    "unit_id": uid,
                    "observed_entity": "chair",
                    "ref_entity": "chair",
                    "observed_change": "removed",
                    "checks": {key: "pass" for key in JOINT_MEMBER_FIELDS},
                    "warnings": [],
                    "failure_reasons": [],
                }
                for uid in ("U1", "U2")
            ],
            "comparable_granularity": "pass",
            "shared_edit_valid": "pass",
        }
    }
    for warning, expected in [
        ("Minor texture variation", True),
        ("Person appears to be floating without a seat", False),
    ]:
        g, r = deepcopy(group), deepcopy(result)
        r["semantic_audit"]["members"][0]["warnings"] = [warning]
        _apply_joint_twin_audit(g, r)
        assert g["group_checks"]["group_pass"] is expected
    g, r = deepcopy(group), deepcopy(result)
    r["semantic_audit"]["failure_reasons"] = ["wrong source reference"]
    _apply_joint_twin_audit(g, r)
    assert not g["group_checks"]["group_pass"]


def test_add_noop_noise_rejected_without_large_host_ratio(tmp_path):
    import numpy as np
    from PIL import Image

    from sa1b_pipeline.validate import deterministic_check

    source = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    support = np.ones((128, 128), dtype=np.uint8) * 255
    for name, pixels in [("src", source), ("mask", mask), ("support", support)]:
        Image.fromarray(pixels).save(tmp_path / (name + ".png"))
    row = {
        "source_image": str(tmp_path / "src.png"),
        "edited_image": str(tmp_path / "tar.png"),
        "mask": str(tmp_path / "mask.png"),
        "support_mask": str(tmp_path / "support.png"),
        "operation": "add",
    }
    noisy = source.copy()
    noisy[48, 48] = 255
    Image.fromarray(noisy).save(row["edited_image"])
    assert "no_visible_addition" in deterministic_check(row)["failure_reasons"]
    # Tiny real addition is outside the source anchor but inside the support.
    small = source.copy()
    small[29:31, 48:50] = 255
    Image.fromarray(small).save(row["edited_image"])
    result = deterministic_check(row)
    assert result["decision"] == "pending_semantic_audit"
    assert result["deterministic"]["add_changed_pixels"] == 4
