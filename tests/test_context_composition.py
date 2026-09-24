import numpy as np
import pytest
from PIL import Image

from utils.context_edit import (
    compose_context_crop,
    compose_attribute_crop,
    edit_context_crop,
    compose_guarded_crop,
    compose_grounded_crop,
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


def test_guarded_remove_protects_neighbor_and_never_restores_target():
    source = Image.new('RGB', (160, 160), 'red')
    edited = Image.new('RGB', source.size, 'blue')
    mask = np.zeros((160, 160), dtype=bool)
    mask[50:110, 50:80] = True
    neighbor = np.zeros_like(mask)
    neighbor[50:110, 82:100] = True
    result, alpha = compose_guarded_crop(source, edited, mask, 'remove', (0, 0, 160, 160), neighbor)
    assert np.all(np.asarray(alpha)[mask] == 255)
    assert np.all(np.asarray(alpha)[neighbor] == 0)
    assert np.array_equal(np.asarray(result)[neighbor], np.asarray(source)[neighbor])
    assert result.getpixel((0, 0)) == source.getpixel((0, 0))


def test_guarded_attribute_clean_single_input_protects_outside():
    from types import SimpleNamespace
    source = Image.new('RGB', (100, 100), 'red')
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    captured = {}
    def pipe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(images=[Image.new('RGB', (100, 100), 'blue')])
    result = edit_context_crop(pipe, source, {'task_type': 'attribute',
        'editing_instruction': 'Make the central square blue.'}, mask, None, guarded_composition=True)
    assert isinstance(captured['image'], Image.Image)
    assert np.array_equal(np.asarray(captured['image']), np.asarray(source))
    assert np.array_equal(np.asarray(result)[~mask], np.asarray(source)[~mask])


def test_guarded_overlap_target_takes_priority():
    source = Image.new('RGB', (100, 100), 'red')
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True
    result, alpha = compose_guarded_crop(source, Image.new('RGB', source.size, 'blue'),
        mask, 'replace', (0, 0, 100, 100), mask.copy())
    assert np.all(np.asarray(alpha)[mask] == 255)


def test_connected_support_follows_attached_change_not_isolated_neighbor():
    source = Image.new('RGB', (200, 200), 'white')
    pixels = np.asarray(source).copy()
    pixels[50:150, 80:100] = (0, 0, 0)
    pixels[60:90, 150:170] = (0, 0, 0)
    source = Image.fromarray(pixels)
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:70, 80:100] = True
    result, alpha = compose_guarded_crop(source, Image.new('RGB', source.size, 'white'),
        mask, 'remove', (0, 0, 200, 200), connected_support=True)
    assert result.getpixel((90, 140)) == (255, 255, 255)
    assert result.getpixel((160, 70)) == (0, 0, 0)
    assert np.all(np.asarray(alpha)[mask] == 255)


def test_poisson_boundary_fallback_does_not_restore_removed_image_edge():
    source = Image.new('RGB', (100, 100), 'red')
    edited = Image.new('RGB', source.size, 'blue')
    mask = np.ones((100, 100), dtype=bool)
    result, _ = compose_guarded_crop(source, edited, mask, 'remove', (0, 0, 100, 100),
                                    connected_support=True, poisson_blend=True)
    assert np.array_equal(np.asarray(result), np.asarray(edited))


def test_poisson_does_not_modify_protected_instance():
    source = Image.new('RGB', (200, 200), 'red')
    edited = source.copy()
    edited.paste('blue', (70, 70, 130, 130))
    mask = np.zeros((200, 200), dtype=bool)
    mask[80:120, 80:120] = True
    protected = np.zeros_like(mask)
    protected[85:115, 125:135] = True
    result, _ = compose_guarded_crop(source, edited, mask, 'remove', (0, 0, 200, 200),
        protected_mask=protected, connected_support=True, poisson_blend=True)
    assert np.array_equal(np.asarray(result)[protected], np.asarray(source)[protected])


def test_complete_remove_unit_includes_stick_and_preserves_neighbor():
    source = Image.new('RGB',(160,160),'red')
    raw = Image.new('RGB',source.size,'blue')
    complete = np.zeros((160,160),bool)
    complete[30:70,50:90] = True
    complete[70:125,67:73] = True
    protected = np.zeros_like(complete);protected[75:120,76:85] = True
    result, alpha = compose_grounded_crop(source,raw,complete,'remove',(0,0,160,160),protected)
    assert result.getpixel((70,120)) == (0,0,255)
    assert np.all(np.asarray(alpha)[complete] == 255)
    assert np.array_equal(np.asarray(result)[protected],np.asarray(source)[protected])


def test_replacement_support_follows_new_shape_without_writing_entire_crop():
    source = Image.new('RGB',(200,200),'red');raw=Image.new('RGB',source.size,'blue')
    old=np.zeros((200,200),bool);old[60:140,85:115]=True
    new=np.zeros_like(old);new[95:135,50:155]=True
    result,alpha=compose_grounded_crop(source,raw,old,'replace',(0,0,200,200),replacement_mask=new)
    assert np.all(np.asarray(alpha)[old|new] == 255)
    assert result.getpixel((150,120)) == (0,0,255)
    assert result.getpixel((10,10)) == (255,0,0)


def test_unresolved_semantic_contract_blocks_diffusion():
    def pipe(**kwargs):
        raise AssertionError('Unresolved geometry must never reach diffusion')
    source=Image.new('RGB',(20,20));mask=np.ones((20,20),bool)
    with pytest.raises(ValueError,match='resolved region contract'):
        edit_context_crop(pipe,source,{'region_contract':{'status':'unresolved'}},mask,None,grounded_composition=True)


def test_grounded_poisson_mask_mutation_cannot_restore_source(monkeypatch):
    import cv2
    source=Image.new('RGB',(120,120),'red')
    raw=Image.new('RGB',source.size,'blue')
    target=np.zeros((120,120),bool);target[45:75,45:75]=True
    def mutating_clone(src,dst,mask,center,flags):
        mask[:]=0
        return src
    monkeypatch.setattr(cv2,'seamlessClone',mutating_clone)
    result,alpha=compose_grounded_crop(source,raw,target,'remove',(0,0,120,120),poisson_blend=True)
    assert result.getpixel((60,60))==(0,0,255)
    assert result.getpixel((0,0))==(255,0,0)
