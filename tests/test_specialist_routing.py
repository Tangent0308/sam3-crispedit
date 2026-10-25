import numpy as np
import pytest
from PIL import Image

from sa1b_pipeline.specialist_routing import (
    MaterialColorEditor, SpecialistRouter, choose_route, material_proposal,
)


def unit(tmp_path, operation="attribute", value="red"):
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(mask).save(p)
    return {"operation": operation, "mask_path": str(p), "execution_mask_path": str(p),
            "instruction_spec": {"operation": operation, "remove_mode": "whole",
                                 "attributes": [{"property": "color", "value": value}]}}


def test_material_preserves_pages_and_outside(tmp_path):
    u = unit(tmp_path)
    a = np.full((100, 100, 3), 120, np.uint8)
    a[20:80, 20:80] = [120, 75, 45]
    a[20:30, 20:80] = [240, 235, 220]  # distinct light page strip
    p = tmp_path / "source.png"
    Image.fromarray(a).save(p)
    e = MaterialColorEditor()
    result = np.asarray(e.generate(p, u))
    assert np.array_equal(result[:20], a[:20])
    assert np.array_equal(result[20:28, 20:80], a[20:28, 20:80])
    assert result[50, 50, 0] > result[50, 50, 1]
    assert not np.array_equal(result[50, 50], a[50, 50])


def test_dark_and_noncolor_fallback(tmp_path):
    u = unit(tmp_path)
    assert choose_route(Image.new("RGB", (100, 100), (3, 3, 3)), u)["backend"] == "qwen"
    u["instruction_spec"]["attributes"][0]["property"] = "material"
    assert choose_route(Image.new("RGB", (100, 100), "gray"), u)["backend"] == "qwen"


def test_remove_large_and_partial_fallback(tmp_path):
    u = unit(tmp_path, "remove")
    assert choose_route(Image.new("RGB", (100, 100), "gray"), u)["backend"] == "qwen"
    # A contents operation cannot be sent to a mask-only remover.
    u["instruction_spec"]["remove_mode"] = "contents"
    assert choose_route(Image.new("RGB", (100, 100), "gray"), u)["backend"] == "qwen"


def test_small_surface_fold_does_not_route_to_inpainting(tmp_path):
    u = unit(tmp_path, "remove")
    mask = np.zeros((100, 100), np.uint8)
    mask[45:55, 45:55] = 255
    Image.fromarray(mask).save(u["execution_mask_path"])
    src = Image.new("RGB", (100, 100), "gray")
    u["instruction_spec"]["ref"] = "the small bottle"
    assert choose_route(src, u)["backend"] == "lama"
    u["instruction_spec"]["ref"] = "the fold on the blue shoulder"
    assert choose_route(src, u)["backend"] == "qwen"


def test_router_calls_only_one_editor_and_keeps_audit(tmp_path):
    class Editor:
        def __init__(self): self.calls = 0
        def generate(self, image_path, unit, seed=0):
            self.calls += 1
            return Image.open(image_path).convert("RGB")
    qwen, lama = Editor(), Editor()
    router = SpecialistRouter(qwen, lama)
    p = tmp_path / "source.png"
    Image.new("RGB", (100, 100), (120, 75, 45)).save(p)
    router.generate(p, unit(tmp_path))
    assert qwen.calls == 1 and lama.calls == 0
    assert router.last_generation_context["requires_semantic_audit"]
