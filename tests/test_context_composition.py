import numpy as np
import pytest
from PIL import Image

from utils.context_edit import (
    compose_context_crop,
    compose_attribute_crop,
    edit_context_crop,
)


def test_image_boundary_does_not_restore_removed_target():
    source = Image.new("RGB", (100, 100), "red")
    edited = Image.new("RGB", (100, 100), "blue")
    mask = np.ones((100, 100), dtype=bool)
    result, alpha = compose_context_crop(
        source, edited, mask, "remove", (0, 0, 100, 100), True
    )
    assert np.asarray(alpha).min() == 255
    assert result.getpixel((99, 0)) == (0, 0, 255)


def test_internal_crop_feathers_but_keeps_distant_pixels_exact():
    source = Image.new("RGB", (200, 200), "red")
    edited = Image.new("RGB", (100, 100), "blue")
    mask = np.zeros((200, 200), dtype=bool)
    mask[90:110, 90:110] = True
    result, alpha = compose_context_crop(
        source, edited, mask, "remove", (50, 50, 150, 150), True
    )
    assert 0 < alpha.getpixel((0, 50)) < 255
    assert alpha.getpixel((50, 50)) == 255
    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert result.getpixel((100, 100)) == (0, 0, 255)


def test_empty_or_misaligned_mask_fails_explicitly():
    source = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError, match="nonempty"):
        compose_context_crop(
            source, source, np.zeros((100, 100)), "remove", (0, 0, 100, 100)
        )
    with pytest.raises(ValueError, match="shape"):
        compose_context_crop(
            source, source, np.ones((50, 50)), "remove", (0, 0, 100, 100)
        )


def test_attribute_composition_excludes_external_guides_and_other_instances():
    source = Image.new("RGB", (100, 100), "red")
    edited = Image.new("RGB", (100, 100), "blue")
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    result, alpha = compose_attribute_crop(source, edited, mask, (0, 0, 100, 100))
    assert np.all(np.asarray(alpha)[~mask] == 0)
    assert np.array_equal(np.asarray(result)[~mask], np.asarray(source)[~mask])
    assert result.getpixel((50, 50)) == (0, 0, 255)


def test_guided_attribute_keeps_clean_primary_input_and_short_training_label():
    from types import SimpleNamespace

    source = Image.new("RGB", (100, 100), "red")
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    row = {
        "task_type": "attribute",
        "editing_instruction": "Make the central square blue.",
    }
    captured = {}

    def pipe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (100, 100), "blue")])

    result = edit_context_crop(
        pipe,
        source,
        row,
        mask,
        None,
        target_guide=True,
        attribute_mask_composition=True,
        preserve_attribute_texture=True,
    )
    clean, guide = captured["image"]
    assert np.array_equal(np.asarray(clean), np.asarray(source))
    assert guide.size == source.size
    assert np.array_equal(np.asarray(guide)[mask], np.asarray(source)[mask])
    assert np.array_equal(np.asarray(result)[~mask], np.asarray(source)[~mask])
    assert "posterization" in captured["negative_prompt"]
    assert row["editing_instruction"] == "Make the central square blue."
