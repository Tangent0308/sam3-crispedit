import pytest
from sa1b_pipeline.group_audit import _enforce_multi_integrity, multi_audit_schema
from sa1b_pipeline.multi_transactional import member_passes
from sa1b_pipeline.multi_semantics import validate_multi_semantics


@pytest.mark.parametrize('field', ['ref_matches_mask', 'location_correct', 'addition_is_new'])
def test_acknowledged_wrong_target_site_or_repainting_cannot_pass(field):
    result = dict(decision='pass', host_integrity=True, natural=True,
                  operation_correct=True, target_done=True, non_target_preserved=True,
                  ref_matches_mask=True, location_correct=True, addition_is_new=True,
                  observed_change='a bag is visible', observed_location='left shoulder')
    result[field] = False
    checked = _enforce_multi_integrity(result)
    assert checked['decision'] == 'fail'
    assert not member_passes(checked)
    joint = _enforce_multi_integrity({'decision': 'pass', 'units': [result]}, joint=True)
    assert joint['decision'] == 'fail'


def test_new_decoding_contract_requires_location_and_novelty_evidence():
    single = multi_audit_schema()
    joint = multi_audit_schema(joint=True)['properties']['units']['items']
    for schema in (single, joint):
        assert {'observed_location', 'location_correct', 'addition_is_new'} <= set(schema['required'])
    assert list(single['properties'])[:3] == ['source_entity', 'observed_change', 'observed_location']


def test_invalid_possessive_site_is_rejected_before_rendering():
    bad = {'operation': 'add', 'scope': 'person', 'placement': {'site': 'her right shoulder'}}
    with pytest.raises(ValueError, match='host-relative'):
        validate_multi_semantics(bad, 'person')
    validate_multi_semantics({**bad, 'placement': {'site': 'right shoulder'}}, 'person')


def test_bind_independent_noun_without_replanning_or_mutating_raw():
    from sa1b_pipeline.group_planning import normalize_multi_grounding
    raw = {'targets': [{'scope': 'singular noun', 'grounding': {'entity': 'shirt'}}]}
    bound = normalize_multi_grounding(raw, [])
    assert bound['targets'][0]['scope'] == 'shirt'
    assert raw['targets'][0]['scope'] == 'singular noun'
    new = {'grounding_version': 3, 'targets': [{'grounding': {'entity': 'shirt'}}]}
    assert normalize_multi_grounding(new, [])['targets'][0]['scope'] == 'shirt'
    new['targets'][0]['scope'] = 'person'
    with pytest.raises(ValueError, match='overrides'):
        normalize_multi_grounding(new, [])
