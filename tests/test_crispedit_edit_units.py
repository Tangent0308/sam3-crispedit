import copy
import json

import pytest

from crispedit.mask.grounding import (parse_change_observation)
from crispedit.mask.checklist import grounding_checklist, parse_checklist_grounding


def observation():
    return {'edits': [
        {'change': 'The seated left person raises an arm holding a cup.', 'units': [
            {'source_ref': 'sleeved arm and hand', 'target_ref': 'sleeved arm and hand',
             'source_location': 'left seated person, arm beside torso',
             'target_location': 'left seated person, arm above head'},
            {'source_ref': 'white cup', 'target_ref': 'white cup',
             'source_location': 'held beside the torso', 'target_location': 'held above the head'}]},
        {'change': 'A red flower appears in the vase.', 'units': [
            {'source_ref': '', 'target_ref': 'red flower', 'source_location': '',
             'target_location': 'inside the vase on the right'}]}]}


def test_events_flatten_without_losing_instances_or_side_identity():
    parsed = parse_change_observation(json.dumps(observation()))
    assert len(parsed['changes']) == 3
    assert [u['edit_id'] for u in parsed['changes']] == [0, 0, 1]
    source = grounding_checklist(parsed, 'source')
    target = grounding_checklist(parsed, 'target')
    assert [u['change_id'] for u in source] == [0, 1]
    assert [u['change_id'] for u in target] == [0, 1, 2]
    assert 'beside torso' in source[0]['location']
    assert 'above head' in target[0]['location']
    assert source[1]['change'] == observation()['edits'][0]['change']
    boxes, unresolved = parse_checklist_grounding(json.dumps([
        {'change_id': 1, 'bbox_2d': [100, 200, 300, 400], 'ref': 'person'},
        {'change_id': 0, 'bbox_2d': None, 'reason': 'occluded'}]), parsed, 'source')
    assert boxes[0]['ref'] == 'white cup'
    assert boxes[0]['edit_id'] == 0 and boxes[0]['change_id'] == 1
    assert boxes[0]['change'] == source[1]['change']
    assert unresolved == [{'change_id': 0, 'reason': 'occluded'}]


@pytest.mark.parametrize('field,value', [('units', []), ('units', {}), ('change', '')])
def test_invalid_events_cannot_silently_drop_segmentation_units(field, value):
    payload = observation()
    payload['edits'][0][field] = value
    with pytest.raises(ValueError):
        parse_change_observation(json.dumps(payload))


@pytest.mark.parametrize('patch', [
    {'source_ref': [], 'target_ref': ''}, {'source_ref': '', 'target_ref': ''},
    {'source_location': ''}, {'layout': 'many'}, {'source_ref': None}])
def test_invalid_unit_requires_retry_instead_of_scope_invention(patch):
    payload = copy.deepcopy(observation())
    payload['edits'][0]['units'][0].update(patch)
    with pytest.raises(ValueError):
        parse_change_observation(json.dumps(payload))


def test_no_edits_and_mixed_schema_are_distinct():
    assert parse_change_observation('{"edits":[]}')['changes'] == []
    with pytest.raises(ValueError):
        parse_change_observation('{"edits":[],"changes":[]}')


def test_complete_event_array_is_only_a_wrapper_normalization():
    events = observation()['edits']
    assert parse_change_observation(json.dumps(events)) == parse_change_observation(json.dumps({'edits':events}))
    assert parse_change_observation('[]')['changes'] == []
    with pytest.raises(ValueError):
        parse_change_observation(json.dumps(events)[:-1])
    with pytest.raises(ValueError):
        parse_change_observation('[{"change":"remove", "units":[]}]')
    with pytest.raises(ValueError):
        parse_change_observation('[{"source_ref":"person"}]')


def test_default_two_pass_does_not_add_mllm_review_rounds(monkeypatch):
    from crispedit.mask.grounding_runner import parse_args
    monkeypatch.setattr('sys.argv', ['grounding', '--input-dir', '/input', '--output-dir', '/output'])
    args = parse_args()
    assert args.grounding_mode == 'two-pass'
    assert args.inference_backend == 'vllm'
    assert not hasattr(args, 'bbox_refinement')
    assert args.observation_max_new_tokens == 3072
    assert args.max_new_tokens == 1536
