import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from utils.context_edit import edit_context_crop, validate_refinement_execution
from utils.qwen_pipeline_loader import detect_qwen_family, is_qwen21_pipeline


def test_local_model_index_selects_qwen_image_21(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline"})
    )

    assert detect_qwen_family(str(tmp_path)) == "qwen21"
    assert detect_qwen_family("Qwen/Qwen-Image-Edit-2511") == "qwen2511"


def test_qwen21_context_call_uses_official_no_cfg_api():
    captured = {}

    class QwenImage21Pipeline:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            input_image = kwargs["image"]
            # Native 2.1 RGBA may use transparent pixels whose hidden RGB is
            # only a matte color.  The adapter must not leak it into the edit.
            layer = Image.new("RGBA", input_image.size, (255, 0, 255, 0))
            return SimpleNamespace(images=[layer])

    pipe = QwenImage21Pipeline()
    source = Image.new("RGB", (96, 96), "red")
    mask = np.zeros((96, 96), dtype=bool)
    mask[32:64, 32:64] = True
    row = {
        "task_type": "attribute",
        "editing_instruction": "Change the central square to blue.",
        "new_instruction": "Change the central square to blue.",
        "masked_content": "central square",
        "region_contract": {
            "status": "original",
            "source_size": [96, 96],
            "segmentation_target": "central square",
        },
    }

    result = edit_context_crop(
        pipe,
        source,
        row,
        mask,
        generator=None,
        true_cfg_scale=1.0,
        negative_prompt=None,
        grounded_composition=True,
    )

    assert is_qwen21_pipeline(pipe)
    assert captured["true_cfg_scale"] == 1.0
    assert captured["use_kv_cache"] is True
    assert captured["prompt"].startswith("Edit the provided photo in place.")
    assert "rectangle" not in captured["prompt"].lower()
    assert "guidance_scale" not in captured
    assert "negative_prompt" not in captured
    assert result.size == source.size
    assert np.array_equal(np.asarray(result), np.asarray(source))


def test_qwen21_grounded_variant_accepts_resolved_region_contract():
    row = {
        "task_type": "remove",
        "region_contract": {"status": "original"},
    }

    validate_refinement_execution(row, "context_grounded_v4_qwen21")
    validate_refinement_execution(row, "context_grounded_v3_qwen21")
