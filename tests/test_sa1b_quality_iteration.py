from types import SimpleNamespace

import numpy as np
from PIL import Image

from sa1b_pipeline.editor import DiffusersQwenImageEditor, _target_guide, _target_hint
from sa1b_pipeline.geometry import mask_geometry, operation_quality_gate
from sa1b_pipeline.validate import _target_panel, deterministic_check


def test_operation_quality_gate_is_operation_specific():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:70, 30:80] = True
    feature = mask_geometry(mask)
    assert operation_quality_gate(feature, "attribute")["eligible"]
    assert operation_quality_gate(feature, "remove")["eligible"]

    tiny = np.zeros((100, 100), dtype=bool)
    tiny[10:12, 10:12] = True
    tiny_feature = mask_geometry(tiny)
    assert not operation_quality_gate(tiny_feature, "remove")["eligible"]
    assert "mask_too_small" in operation_quality_gate(tiny_feature, "remove")["reasons"]


def test_target_guide_is_aligned_and_prompt_contains_coordinates():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:12, 8:17] = True
    guide = _target_guide(Image.new("RGB", (30, 20), "gray"), mask)
    assert guide.size == (30, 20)
    assert _target_hint(mask) == "x=27%..57%, y=25%..60%"


def test_editor_uses_aligned_guide_and_regional_weights(tmp_path):
    source = Image.new("RGB", (80, 60), "gray")
    source_path = tmp_path / "source.png"
    source.save(source_path)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)

    captured = {}

    class FakeGenerator:
        def manual_seed(self, seed):
            captured["seed"] = seed
            return self

    class FakeTorch:
        @staticmethod
        def Generator(**kwargs):
            return FakeGenerator()

    class FakePipe:
        def region_callback_kwargs(self, crop, local_mask, operation, protected, policy):
            captured["regional_mask"] = local_mask.copy()
            captured["policy"] = policy
            return {"height": 64, "width": 96, "regional_weights": np.ones((4, 6)),
                    "output_resolution": 1024, "use_kv_cache": True}, None

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (96, 64), "blue")])

    editor = object.__new__(DiffusersQwenImageEditor)
    editor.torch = FakeTorch()
    editor.pipe = FakePipe()
    editor.steps = 4
    unit = {
        "operation": "attribute",
        "instruction_ref": "Change the color of the selected square to blue.",
        "mask_path": str(mask_path),
    }
    output = editor.generate(str(source_path), unit, seed=17)
    assert output.size == source.size
    assert isinstance(captured["image"], list) and len(captured["image"]) == 2
    assert captured["regional_weights"].shape == (4, 6)
    assert captured["policy"] == "guard-any-v1"
    assert "Image 2 is an aligned location guide" in captured["prompt"]
    assert "x=" in captured["prompt"] and "y=" in captured["prompt"]


def test_validation_vetoes_low_change_attribute_and_builds_target_panel(tmp_path):
    source = Image.new("RGB", (80, 60), "gray")
    edited = source.copy()
    source_path, edited_path = tmp_path / "source.png", tmp_path / "edited.png"
    source.save(source_path)
    edited.save(edited_path)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)
    row = {"source_image": str(source_path), "edited_image": str(edited_path),
           "mask": str(mask_path), "operation": "attribute", "instruction": "Make it blue."}
    check = deterministic_check(row)
    assert check["decision"] == "fail"
    assert "no_visible_target_change" in check["failure_reasons"]
    panel = Image.open(_target_panel(row, tmp_path / "audit"))
    assert panel.size == (1536, 512)
