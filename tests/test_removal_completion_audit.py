import json
import pytest

from synthesis_pipeline.audit_removal_concise import parse_audit, admitted, selected_layout
from synthesis_pipeline.plan_removal_relations import parse_relation_response, PROMPT_V12
from utils.removal_relations import is_grounded_relation_policy
from synthesis_pipeline.rewrite_verified_removals import eligible, parse_instruction


@pytest.mark.parametrize('missing',['target_removed','quality','instruction_match','reason'])
@pytest.mark.parametrize('policy',['completion-v5','completion-v6'])
def test_completion_audit_requires_all_four_fields(missing,policy):
    record=dict(target_removed='pass',quality='pass',instruction_match='pass',reason='Observed complete removal.')
    del record[missing]
    assert parse_audit(json.dumps(record),policy) is None


def test_retained_subject_cannot_be_rescued_by_other_pass_fields():
    record=dict(target_removed='fail',quality='pass',instruction_match='pass',reason='Only an attachment changed.')
    assert not admitted(parse_audit(json.dumps(record),'completion-v5'))


def test_v12_compiles_only_main_target_and_keeps_general_rules():
    record=dict(target='the left subject',relations=[dict(description='exclusive attachment')])
    assert parse_relation_response(json.dumps(record),'relations-v12')['instruction']=='Remove the left subject.'
    assert is_grounded_relation_policy('relations-v12')
    assert 'OTHER living being' in PROMPT_V12
    assert 'respective depths' in PROMPT_V12


def test_rewrite_cannot_rescue_surviving_subject_or_pixel_veto():
    assert not eligible({'parsed':{'target_removed':'fail','quality':'pass'}})
    assert not eligible({'parsed':{'target_removed':'pass','quality':'pass'},'pixel_veto_applied':True})
    assert eligible({'parsed':{'target_removed':'pass','quality':'pass','instruction_match':'fail'}})
    assert parse_instruction('{"instruction":"Change the object to blue."}') is None
    assert parse_instruction('{"instruction":"Remove the left object."}')=='Remove the left object.'


def test_input_layout_depends_on_area_not_object_or_case():
    import numpy as np
    mask=np.zeros((100,100),bool);mask[20:30,20:30]=True
    assert selected_layout(mask,'adaptive')=='stacked'
    mask[20:80,20:80]=True
    assert selected_layout(mask,'adaptive')=='full'
    assert selected_layout(mask,'stacked')=='stacked'
