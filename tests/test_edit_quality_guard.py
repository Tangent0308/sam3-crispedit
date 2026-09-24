import numpy as np


def test_removal_pixel_veto_only_flags_very_weak_target_changes():
    from PIL import Image
    from utils.edit_quality_guard import removal_change_evidence
    source=Image.new('RGB',(80,80),(40,60,80));target=np.zeros((80,80),bool)
    target[20:60,20:60]=True
    before=target.copy()
    assert removal_change_evidence(source,source,target)['insufficient_change']
    pixels=np.asarray(source).copy();pixels[target]=180
    actual=removal_change_evidence(source,Image.fromarray(pixels),target)
    assert not actual['insufficient_change'] and actual['semantic_quality']=='not_checked'
    assert np.array_equal(target,before)
from PIL import Image
import pytest

from utils.edit_quality_guard import replacement_phrase, raw_locality, composition_retention
from utils.context_edit import qwen21_edit_prompt
from utils.context_edit import compose_grounded_crop
from utils.remove_support import adaptive_remove_support


@pytest.mark.parametrize('instruction,target,expected', [
    ('Replace the pot with the orange hose by a blue plastic bucket.', '', 'a blue plastic bucket'),
    ('Replace the man by the door with a woman.', '', 'a woman'),
    ('Replace the car with a truck with a blue roof.', 'car', 'a truck with a blue roof'),
])
def test_replacement_source_relations_are_not_new_target(instruction, target, expected):
    assert replacement_phrase(dict(editing_instruction=instruction, segmentation_target=target)) == expected


def scene():
    source = Image.new('RGB', (100, 100), (100,100,100))
    target = np.zeros((100,100),bool); target[30:70,30:70] = True
    raw = np.asarray(source).copy(); raw[40:50,40:50] = (240,20,20)
    return source, Image.fromarray(raw), target, np.zeros_like(target)


def test_localized_raw_can_survive_segmentation_failure_without_fake_semantic_mask():
    source,raw,target,protected=scene()
    info,support=raw_locality(source,raw,target,protected)
    assert info['locality_pass'] and info['semantic_quality']=='not_checked'
    assert not support[~target].any()


def test_raw_fallback_rejects_noop_and_distant_or_protected_damage():
    source,raw,target,protected=scene()
    assert not raw_locality(source,source,target,protected)[0]['locality_pass']
    pixels=np.asarray(raw).copy();pixels[5:15,5:15]=255
    damaged=Image.fromarray(pixels)
    assert not raw_locality(source,damaged,target,protected)[0]['locality_pass']
    protected[5:15,5:15]=True
    assert raw_locality(source,damaged,target,protected)[0]['protected_changed_pixels']>4


def test_composition_detects_erased_addition_but_accepts_retained_edit():
    source,raw,target,_=scene()
    assert composition_retention(source,raw,source,target)['collapsed']
    assert not composition_retention(source,raw,raw,target)['collapsed']


def test_add_retention_ignores_discarded_incidental_host_regeneration():
    source,final,target,_=scene()
    pixels=np.asarray(final).copy(); pixels[55:70,30:70]=220
    raw=Image.fromarray(pixels)
    inserted=np.zeros_like(target);inserted[40:50,40:50]=True
    assert not composition_retention(source,raw,final,inserted)['collapsed']


def test_remove_support_recovers_changed_small_hole_without_mutating_label():
    source=Image.new('RGB',(160,160),(80,80,80))
    raw=Image.new('RGB',source.size,(160,160,160))
    target=np.zeros((160,160),bool);target[40:120,40:120]=True
    target[72:88,72:88]=False;frozen=target.copy()
    support,info=adaptive_remove_support(source,raw,target)
    assert support[80,80] and not target[80,80]
    assert np.array_equal(target,frozen) and not info['source_mask_modified']
    final,alpha=compose_grounded_crop(source,raw,target,'remove',(0,0,160,160),
        remove_composition_policy='adaptive-remove-v1')
    assert np.asarray(final)[80,80,0]==160
    assert np.asarray(alpha)[0,0]==0


def test_remove_support_preserves_protected_neighbor_and_distant_pixels_exactly():
    source=Image.new('RGB',(160,160),(80,80,80))
    raw=Image.new('RGB',source.size,(160,160,160))
    target=np.zeros((160,160),bool);target[40:120,40:120]=True
    protect=np.zeros_like(target);protect[:,120:130]=True
    final,alpha=compose_grounded_crop(source,raw,target,'remove',(0,0,160,160),protect,
        remove_composition_policy='adaptive-remove-v1')
    assert np.array_equal(np.asarray(final)[protect],np.asarray(source)[protect])
    assert np.array_equal(np.asarray(final)[:15],np.asarray(source)[:15])


def test_remove_support_requires_real_change_and_does_not_fill_large_interiors():
    source=Image.new('RGB',(160,160),(80,80,80))
    target=np.zeros((160,160),bool);target[20:140,20:140]=True
    target[40:120,40:120]=False
    support,_=adaptive_remove_support(source,source,target)
    assert np.array_equal(support,target)
    raw=Image.new('RGB',source.size,(180,180,180))
    support,_=adaptive_remove_support(source,raw,target)
    assert not support[80,80]  # No broad hole filling / rectangular writeback.


def test_remove_support_rejects_empty_or_misaligned_masks():
    source=Image.new('RGB',(32,32))
    with pytest.raises(ValueError):adaptive_remove_support(source,source,np.zeros((32,32),bool))
    with pytest.raises(ValueError):adaptive_remove_support(source,source,np.ones((16,16),bool))


def test_shadow_prompt_only_changes_removal_and_never_requests_ambient_shadow_erasure():
    assert 'own cast shadow' in qwen21_edit_prompt('Remove the target.','remove','remove-shadow-v2')
    assert 'shadows belonging to other' in qwen21_edit_prompt('Remove the target.','remove','remove-shadow-v2')
    assert qwen21_edit_prompt('Add a patch.','add','remove-shadow-v2')==qwen21_edit_prompt('Add a patch.','add','typed-v1')


@pytest.mark.parametrize('policy', ['adaptive-remove-v1', 'adaptive-remove-v2'])
def test_snapshot_exports_write_support_separately_from_original_mask(tmp_path, policy):
    import json
    from synthesis_pipeline.assemble_fresh_snapshot import attach_remove_support
    from synthesis_pipeline.audit_edit_pairs import mask_array
    row={'image':'000_test_remove.png','task_type':'remove','mask':{'original':'unchanged'}}
    d=tmp_path/'diagnostics/000_test_remove';d.mkdir(parents=True)
    (d/'generation_request.json').write_text(json.dumps(dict(source_size=[32,32],crop_bbox=[4,5,20,21],
        remove_composition_policy=policy)))
    alpha=np.zeros((16,16),np.uint8);alpha[2:6,2:6]=255
    Image.fromarray(alpha).save(d/'composition_alpha.png')
    exported=attach_remove_support(row,tmp_path)
    assert exported['mask']==row['mask'] and 'edit_support_mask' not in row
    support=mask_array((32,32),exported['edit_support_mask'])
    assert support.sum()==16 and support[7,6]
    assert not exported['remove_composition']['source_mask_modified']
    assert exported['remove_composition']['policy'] == policy


def test_type_prompt_reaches_model_without_coordinate_protocol_or_label_mutation():
    action='Change the shirt to blue.'
    prompt=qwen21_edit_prompt(action,'attribute','typed-v1')
    assert action in prompt and 'fine detail' in prompt and 'illumination' in prompt
    assert 'rectangle' not in prompt and 'x=' not in prompt
    assert 'thin extremities' in qwen21_edit_prompt('Remove the chair.','remove','typed-v1')
