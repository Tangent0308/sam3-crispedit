import numpy as np
import pytest
from utils.region_denoise import editable_token_weights


def inputs():
    mask=np.zeros((128,128),bool);mask[32:64,32:64]=True
    protected=np.zeros_like(mask);protected[:,80]=True
    return mask,protected


def test_thin_neighbor_tokens_are_not_lost_to_majority_pooling():
    mask,protected=inputs()
    old=editable_token_weights(mask,(8,8),'remove',protected)
    new=editable_token_weights(mask,(8,8),'remove',protected,'guard-any-v1')
    assert old[3,5]==1 and new[3,5]==0
    assert new[3,3]==1


def test_mixed_tokens_trade_target_and_protected_coverage_explicitly():
    mask,protected=inputs();protected[:]=False
    mask[:,60:]=False;protected[32:64,60:64]=True
    any_guard=editable_token_weights(mask,(8,8),'remove',protected,'guard-any-v1')
    fractional=editable_token_weights(mask,(8,8),'remove',protected,'guard-fraction-v1')
    assert any_guard[3,3]==1
    assert fractional[3,3]==pytest.approx(.75)


@pytest.mark.parametrize('task',['add','remove','replace','attribute'])
@pytest.mark.parametrize('policy',['guard-any-v1','guard-fraction-v1'])
def test_no_protected_pixels_preserves_old_weights(task,policy):
    mask,_=inputs()
    old=editable_token_weights(mask,(8,8),task)
    assert np.array_equal(old,editable_token_weights(mask,(8,8),task,None,policy))


def test_target_wins_over_overlapping_annotation_without_mutation():
    mask,_=inputs();original=mask.copy()
    old=editable_token_weights(mask,(8,8),'remove')
    assert np.array_equal(old,editable_token_weights(mask,(8,8),'remove',mask,'guard-fraction-v1'))
    assert np.array_equal(mask,original)
