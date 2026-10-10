from pathlib import Path

import numpy as np
from PIL import Image

from sa1b_pipeline.geometry import mask_geometry
from sa1b_pipeline.instructions import build_unit, validate_instruction
from sa1b_pipeline.io import candidate_from_dict


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
