from copy import deepcopy
import pytest
from utils.context_edit import removal_execution_evidence, qwen21_edit_prompt


def accepted_row():
    return dict(task_type='remove', editing_instruction='Remove the selected device on the left.',
        visual_grounding=dict(decision='accept', target_description='device on the left',
            selected_surfaces=['main casing', 'thin lower extension'],
            outside_description='A curved pale supporting surface underneath and another device to the right.'),
        planning_revision=dict(decision='accept'))


def test_verified_parts_and_support_reach_editor_without_changing_label():
    row=accepted_row(); frozen=deepcopy(row)
    prompt=qwen21_edit_prompt(row['editing_instruction'],'remove','remove-evidence-v1',
        removal_execution_evidence(row))
    assert 'thin lower extension' in prompt and 'curved pale supporting surface' in prompt
    assert 'farther background layer' in prompt and 'neighboring instances' in prompt
    assert row == frozen
    assert 'TARGET 1' not in prompt and 'x=' not in prompt


@pytest.mark.parametrize('changed', ['visual_grounding','planning_revision'])
def test_unverified_context_is_not_used(changed):
    row=accepted_row(); row[changed]['decision']='reject'
    assert removal_execution_evidence(row)=={}


@pytest.mark.parametrize('policy', ['remove-evidence-v1', 'remove-parts-v1', 'remove-context-v1'])
def test_legacy_rows_have_exact_typed_fallback(policy):
    action='Remove the item on the left.'
    assert qwen21_edit_prompt(action,'remove',policy,{}) == qwen21_edit_prompt(action,'remove','typed-v1')


@pytest.mark.parametrize('field', ['visual_grounding', 'planning_revision'])
@pytest.mark.parametrize('value', [None, [], 'invalid'])
def test_malformed_evidence_containers_fall_back(field, value):
    row = accepted_row()
    row[field] = value
    assert removal_execution_evidence(row) == {}


@pytest.mark.parametrize('task',['add','replace','attribute'])
@pytest.mark.parametrize('policy', ['remove-evidence-v1', 'remove-parts-v1', 'remove-context-v1'])
def test_other_types_are_byte_identical(task, policy):
    evidence=removal_execution_evidence(accepted_row())
    assert qwen21_edit_prompt('Instruction.',task,policy,evidence)==qwen21_edit_prompt('Instruction.',task,'typed-v1')


def test_invalid_or_overlong_context_falls_back_instead_of_bloating_prompt():
    row=accepted_row();row['visual_grounding']['selected_surfaces']=['word '*25]
    assert removal_execution_evidence(row)=={}


def test_parts_and_context_ablation_are_separable():
    evidence=removal_execution_evidence(accepted_row())
    parts=qwen21_edit_prompt('Remove it.','remove','remove-parts-v1',evidence)
    context=qwen21_edit_prompt('Remove it.','remove','remove-context-v1',evidence)
    assert 'thin lower extension' in parts and 'curved pale supporting surface' not in parts
    assert 'thin lower extension' not in context and 'curved pale supporting surface' in context


def test_evidence_survives_qwen21_prompt_adapter_and_uses_one_clean_image(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from utils.context_edit import edit_context_crop
    captured={}

    class QwenImage21Pipeline:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[kwargs['image']])

    row=accepted_row()
    row['region_contract']=dict(status='original',source_size=[96,96],segmentation_target='device')
    source=Image.new('RGB',(96,96),'gray')
    mask=np.zeros((96,96),bool);mask[32:64,32:64]=True
    edit_context_crop(QwenImage21Pipeline(),source,row,mask,None,true_cfg_scale=1,
        grounded_composition=True,qwen21_prompt_policy='remove-evidence-v1',diagnostics_dir=tmp_path)
    assert isinstance(captured['image'],Image.Image)
    assert 'thin lower extension' in captured['prompt']
    assert 'curved pale supporting surface' in captured['prompt']
    request=json.loads((tmp_path/'generation_request.json').read_text())
    assert request['reference_images']==1
    assert request['removal_execution_evidence']['provenance']=='accepted_frozen_visual_grounding'
    assert row['editing_instruction']=='Remove the selected device on the left.'
