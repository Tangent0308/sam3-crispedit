import numpy as np
import pytest
from PIL import Image

from sa1b_pipeline.qwen_tool_refinement import (
    restore_color_texture, poisson_proposal, noop_remove_evidence, QwenWithTools,
)


def make_unit(tmp_path):
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 20:80] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(m).save(p)
    return {"operation": "attribute", "mask_path": str(p),
            "execution_mask_path": str(p),
            "instruction_spec": {"operation": "attribute",
                                 "attributes": [{"property": "color", "value": "red"}]}}


def test_texture_restores_small_detail_without_changing_outside(tmp_path):
    u = make_unit(tmp_path)
    y, x = np.mgrid[:100, :100]
    source = np.full((100, 100, 3), [120, 100, 80], dtype=np.float32)
    source += (np.sin(x*2)*2)[..., None]
    source = np.rint(source).astype("uint8")
    edited = source.copy()
    edited[20:80, 20:80] = [170, 65, 60]
    out, record = restore_color_texture(Image.fromarray(source), Image.fromarray(edited), u)
    assert record["applied"]
    result = np.asarray(out)
    assert np.array_equal(result[:20], edited[:20])
    assert np.any(result[30:70, 30:70] != edited[30:70, 30:70])
    assert np.max(np.abs(result.astype(int)-edited.astype(int))) < 12


def test_finish_request_and_structure_change_leave_proposal_unchanged(tmp_path):
    u = make_unit(tmp_path)
    a = np.full((100, 100, 3), 100, np.uint8)
    b = a.copy()
    b[30:70, 30:70] = 240
    out, record = restore_color_texture(Image.fromarray(a), Image.fromarray(b), u)
    assert not record["applied"]
    assert record["reason"] == "geometry/edge mismatch"
    assert np.array_equal(np.asarray(out), b)
    u["instruction_spec"]["attributes"][0]["value"] = "glossy red"
    _, record = restore_color_texture(Image.fromarray(a), Image.fromarray(b), u)
    assert record["reason"] == "finish/pattern request"


def test_poisson_proposal_keeps_every_outside_pixel(tmp_path):
    u = make_unit(tmp_path)
    a = np.full((100, 100, 3), 130, np.uint8)
    b = a.copy()
    b[30:70, 30:70] = [200, 70, 60]
    out, record = poisson_proposal(Image.fromarray(a), Image.fromarray(b), u)
    assert record["applied"]
    mask = np.asarray(Image.open(u["mask_path"])) > 127
    assert np.array_equal(np.asarray(out)[~mask], b[~mask])
    assert record["requires_semantic_audit"]


def test_misaligned_images_raise(tmp_path):
    with pytest.raises(ValueError, match="identical"):
        restore_color_texture(Image.new("RGB", (20, 20)), Image.new("RGB", (30, 30)), make_unit(tmp_path))


def test_noop_trigger_is_not_a_pass_and_never_repairs_fold(tmp_path):
    u = make_unit(tmp_path)
    u["operation"] = "remove"
    u["instruction_spec"] = {"operation": "remove", "remove_mode": "whole", "ref": "the object"}
    m = np.zeros((100, 100), np.uint8)
    m[45:55, 45:55] = 255
    Image.fromarray(m).save(u["mask_path"])
    a = Image.new("RGB", (100, 100), "gray")
    evidence = noop_remove_evidence(a, a, u)
    assert evidence["trigger"] and evidence["does_not_prove_semantic_failure"]
    u["instruction_spec"]["ref"] = "the fold"
    assert not noop_remove_evidence(a, a, u)["trigger"]


def test_wrapper_one_qwen_and_one_tool_then_requires_new_audit(tmp_path):
    class Editor:
        def __init__(self): self.calls = 0
        def generate(self, path, unit, seed=0):
            self.calls += 1
            return Image.open(path).convert("RGB")
    u = make_unit(tmp_path)
    u["operation"] = "remove"
    u["instruction_spec"] = {"operation": "remove", "remove_mode": "whole", "ref": "the object"}
    m = np.zeros((100, 100), np.uint8)
    m[45:55, 45:55] = 255
    Image.fromarray(m).save(u["mask_path"])
    src = tmp_path / "source.png"
    Image.new("RGB", (100, 100), "gray").save(src)
    qwen, lama = Editor(), Editor()
    wrapper = QwenWithTools(qwen, lama, repair_noop_remove=True)
    wrapper.generate(src, u)
    assert qwen.calls == 1 and lama.calls == 1
    assert wrapper.last_generation_context["requires_semantic_audit"]
    assert wrapper.last_generation_context["audit_cached_qwen_pixels_invalid"]


def test_saved_proposal_and_tool_config_prevent_stale_resume(tmp_path):
    from sa1b_pipeline.group_generation import _render_member
    class Editor:
        config = {"backend": "test"}
        def generate(self, path, unit, seed=0):
            return Image.open(path).convert("RGB")
    unit = make_unit(tmp_path)
    src = tmp_path / "source.png"
    Image.new("RGB", (100, 100), "gray").save(src)
    output = tmp_path / "target.png"
    wrapped = QwenWithTools(Editor(), diagnostic_directory=tmp_path / "proposals")
    context = _render_member(str(src), unit, output, wrapped, 0)
    assert Image.open(context["qwen_proposal_path"]).size == (100, 100)
    assert context["qwen_pixels_sha256"] == context["final_pixels_sha256"]
    assert _render_member(str(src), unit, output, wrapped, 0) == context
    changed = QwenWithTools(Editor(), restore_texture=True)
    with pytest.raises(ValueError, match="input changed"):
        _render_member(str(src), unit, output, changed, 0)
