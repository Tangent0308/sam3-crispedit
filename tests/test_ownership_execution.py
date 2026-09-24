"""Generic ownership, target invariance and seam tests (no case vocabulary)."""
import json
import numpy as np
import pytest
from PIL import Image
from synthesis_pipeline.resolve_removal_relations import choose_retained
from synthesis_pipeline.resolve_removal_relations import choose_auxiliary
from synthesis_pipeline.resolve_removal_relations import conservative_keep_box
from synthesis_pipeline.plan_removal_relations import parse_relation_response, validate_relation_plan, PROMPT_V15
from utils.remove_support import boundary_seam_alpha


def test_keep_can_be_larger_than_target_and_cannot_consume_it():
    target=np.zeros((100,100),bool);target[40:50,40:50]=True
    neighbor=np.zeros_like(target);neighbor[20:80,52:80]=True
    kept,checks=choose_retained([(neighbor,.9)],[600,450],target,np.zeros_like(target),bbox=[520,200,800,800])
    assert np.array_equal(kept,neighbor) and checks[0]['accepted']
    assert choose_retained([(target|neighbor,.9)],[450,450],target,np.zeros_like(target),bbox=[400,200,800,800])[0] is None


def test_keep_rejects_wrong_point_and_large_target_overlap():
    target=np.zeros((100,100),bool);target[30:70,30:70]=True
    merged=target.copy();merged[40:60,70:80]=True
    assert choose_retained([(merged,.95)],[750,450],target,np.zeros_like(target),bbox=[300,300,800,700])[0] is None
    neighbor=np.zeros_like(target);neighbor[10:30,70:90]=True
    assert choose_retained([(neighbor,.95)],[750,800],target,np.zeros_like(target),bbox=[700,100,900,300])[0] is None


def test_strong_location_can_resolve_large_attachment_without_touching_guard():
    target=np.zeros((100,100),bool);target[40:60,40:50]=True
    item=np.zeros_like(target);item[30:70,50:65]=True
    guard=np.zeros_like(target);guard[30:34,50:65]=True
    kwargs=dict(bbox=[500,300,650,700])
    assert choose_auxiliary([(item,.95)],[550,500],target,guard,**kwargs)[0] is None
    found,checks=choose_auxiliary([(item,.95)],[550,500],target,guard,policy='ownership-v1',**kwargs)
    assert found is not None and not np.any(found&guard)
    assert checks[0]['strongly_located']
    assert choose_auxiliary([(item,.7)],[550,500],target,guard,policy='ownership-v1',**kwargs)[0] is None
    assert choose_auxiliary([(item,.95)],[550,320],target,guard,policy='ownership-v1',**kwargs)[0] is None


def test_box_fallback_only_preserves_outside_target_and_never_deletes():
    target=np.zeros((100,100),bool);target[40:60,40:60]=True
    relation=dict(action='keep',point=[700,500],bbox=[450,300,800,700])
    guard=conservative_keep_box(relation,target)
    assert guard is not None and not (guard&target).any() and guard[50,70]
    assert conservative_keep_box({**relation,'action':'remove_together'},target) is None
    assert conservative_keep_box({**relation,'point':[500,500]},target) is None


def test_box_fallback_rejects_invalid_coordinates():
    target=np.zeros((100,100),bool)
    assert conservative_keep_box(dict(action='keep',point=[1200,500],bbox=[400,300,1500,700]),target) is None


def test_v15_short_public_instruction_and_contact_contract():
    plan=dict(decision='accept',target='the left foreground subject',target_point=[500,500],
        support_check=dict(target_supports_other_living=False,other_living_still_supported=True),
        relations=[dict(action='keep',description='adjacent subject',reason='independent contact',
            point=[750,500],bbox=[700,200,900,800],segmentation_query='subject')],
        reconstruction='Continue the retained subject and background at their original depths.')
    parsed=parse_relation_response(json.dumps(plan),'relations-v15')
    assert validate_relation_plan(parsed,'relations-v15')=='accepted'
    assert parsed['instruction']=='Remove the left foreground subject.'
    assert 'excluded holes' in PROMPT_V15 and 'as keep' in PROMPT_V15
    assert len(PROMPT_V15.split())<350


def test_v16_uses_one_support_decision_not_two_error_prone_booleans():
    plan=dict(decision='accept',target='the left subject',target_point=[500,500],
        support_check='coherent',relations=[],reconstruction='Continue the wall.')
    parsed=parse_relation_response(json.dumps(plan),'relations-v16')
    assert validate_relation_plan(parsed,'relations-v16')=='accepted'
    parsed['support_check']='conflict'
    assert validate_relation_plan(parsed,'relations-v16')=='defer_support_conflict'


@pytest.mark.parametrize('rotation',range(4))
def test_seam_preserves_target_guard_and_write_envelope(rotation):
    target=np.zeros((96,96),bool);target[32:64,32:64]=True
    target[42:54,42:46]=False
    guard=np.zeros_like(target);guard[42:54,42:46]=True;guard[:,68:72]=True
    alpha=np.zeros((96,96),np.uint8);alpha[24:72,24:72]=255;alpha[guard]=0
    source=np.full((96,96,3),80,np.uint8);raw=source.copy()
    source[target]=[160,20,80];raw[guard]=[0,240,0];raw[24:72,24:72,0]+=5
    source,raw,target,guard,alpha=[np.rot90(x,rotation).copy() for x in [source,raw,target,guard,alpha]]
    result=np.asarray(boundary_seam_alpha(Image.fromarray(source),Image.fromarray(raw),target,Image.fromarray(alpha),guard))
    assert np.all(result[target]==255)
    assert np.all(result[guard]==0) and np.all(result[alpha==0]==0)
    assert np.all(result<=alpha)


def test_seam_cannot_hide_an_unwritable_target():
    source=Image.new('RGB',(32,32));target=np.zeros((32,32),bool);target[10:20,10:20]=True
    with pytest.raises(ValueError,match='exclude'):
        boundary_seam_alpha(source,source,target,Image.new('L',(32,32)))
