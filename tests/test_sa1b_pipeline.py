from pathlib import Path

import numpy as np
from PIL import Image

from sa1b_pipeline.geometry import mask_geometry
from sa1b_pipeline.instructions import build_unit, validate_instruction
from sa1b_pipeline.io import candidate_from_dict
from sa1b_pipeline.training import build_training_rows


def test_geometry_has_tight_box_and_interior_point():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:12, 8:17] = True
    geometry = mask_geometry(mask)
    assert geometry["bbox"] == [8, 5, 17, 12]
    assert mask[geometry["point"][1], geometry["point"][0]]
    assert geometry["components"] == 1


def test_candidate_normalizes_relative_paths(tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "source.jpg")
    Image.new("L", (20, 20), 255).save(tmp_path / "mask.png")
    candidate = candidate_from_dict({"source_id": "x", "image_path": "source.jpg", "objects": [{"object_id": "o", "category": "cup", "mask_path": "mask.png"}]}, tmp_path)
    assert Path(candidate.image_path).is_absolute()
    assert Path(candidate.objects[0].mask_path).is_absolute()


def test_instruction_contracts_are_operation_specific(tmp_path):
    geometry = {"bbox": [1, 2, 10, 12], "point": [4, 5]}
    obj = {"object_id": "o", "category": "strap", "description": "the thin strap"}
    unit = build_unit({"unit_index": 1, "operation": "replace", "ref_phrase": "the thin strap across the bag", "scope": "strap", "payload": "a silver chain strap", "evidence": "thin boundary"}, obj, str(tmp_path / "m.png"), geometry, [])
    assert unit.instruction_ref == "Replace the thin strap across the bag with a silver chain strap."
    assert not validate_instruction(unit)


def test_fragmented_small_mask_geometry():
    mask = np.zeros((50, 50), dtype=bool)
    mask[2:5, 2:5] = True
    mask[40:44, 40:44] = True
    geometry = mask_geometry(mask)
    assert geometry["components"] == 2
    assert geometry["area_fraction"] < 0.02


def test_training_materializer_emits_latest_four_row_contract():
    pair = {
        "case_id": "c", "unit_id": "U1", "operation": "attribute",
        "instruction": "Change the color of the red coat to navy blue.",
        "instruction_noref": "Change the color of the selected coat to navy blue.",
        "ref_phrase": "the red coat", "source_image": "s.png", "edited_image": "e.png", "mask": "m.png",
    }
    rows = build_training_rows(pair, "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>")
    assert [r["sample_type"] for r in rows] == ["edit_ntp", "edit_umt", "edit_umt", "edit"]
    assert "<|mt_start|>" in rows[1]["prompt"] and rows[2]["instr_variant"] == "noref"
    assert "image" not in rows[0] and "mt_cot" in rows[0]


def test_rle_selected_targets_materialize_masks(tmp_path):
    from pycocotools import mask as coco
    from sa1b_pipeline.select import select_and_plan
    source = tmp_path / 'image.png'
    Image.new('RGB', (40, 40), 'white').save(source)
    objects = []
    for index, x in enumerate((4, 24)):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:15, x:x + 8] = 1
        rle = coco.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode()
        objects.append({'object_id': str(index), 'category': 'cup', 'rle': rle})
    candidate = candidate_from_dict({'source_id': 'source', 'image_path': str(source), 'objects': objects})

    class Selector:
        calls = 0
        def generate(self, jobs):
            self.calls += 1
            if self.calls == 1:
                return [{'accept': True, 'selected_object_ids': ['0', '1']}]
            return [{'units': [
                {'object_id': '0', 'operation': 'attribute', 'ref_phrase': 'the left cup', 'scope': 'cup', 'payload': 'blue'},
                {'object_id': '1', 'operation': 'remove', 'ref_phrase': 'the right cup', 'scope': 'cup'},
            ]}]

    cases = select_and_plan([candidate], Selector(), audit_path=tmp_path / 'decisions.jsonl')
    assert len(cases) == 1
    assert len(cases[0].units) == 2
    for unit in cases[0].units:
        restored = np.asarray(Image.open(unit.mask_path)) > 0
        assert restored.sum() == 80
        assert restored[unit.point[1], unit.point[0]]
    assert len((tmp_path / 'decisions.jsonl').read_text().splitlines()) == 2


def test_instruction_payload_normalizes_model_verbatim_templates():
    from sa1b_pipeline.instructions import build_unit
    geometry = {"bbox": [1, 2, 10, 12], "point": [4, 5]}
    obj = {"object_id": "o", "category": "fence"}
    unit = build_unit({"operation": "attribute", "ref_phrase": "the fence", "scope": "fence", "payload": "Change the color/material of the fence to polished gold"}, obj, "m.png", geometry, [])
    assert unit.instruction_ref == "Change the color of the fence to polished gold."
    unit = build_unit({"operation": "replace", "ref_phrase": "the fence", "scope": "fence", "payload": "Replace the fence with a wooden gate"}, obj, "m.png", geometry, [])
    assert unit.instruction_ref == "Replace the fence with a wooden gate."
