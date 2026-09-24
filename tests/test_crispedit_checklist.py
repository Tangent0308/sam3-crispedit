import argparse
import json

import pytest
from PIL import Image

from crispedit.mask.checklist import grounding_checklist, parse_checklist_grounding, segmentation_ref
from crispedit.mask.checklist import observation_prompt
from crispedit.mask.grounding_runner import Qwen38Grounder, _result_row


OBS = {"changes": [{"change_id": 0, "source_ref": "black shirt", "target_ref": "red shirt"},
                    {"change_id": 1, "source_ref": "grey hoodie", "target_ref": "red hoodie"}]}










def test_refinement_identity_conflict_is_review_not_ok():
    payload = {'requests':[{'parse_ok':True,'refinement_identity_issues':[0]}],
               'boxes':{'source':[{'bbox_2d':[100,100,200,200]}],'target':[]}}
    assert _result_row(0,sample(),payload,{},'test',1)['qc_flag'] == 'GROUND_REVIEW'


def test_observation_uses_complete_edit_units_and_atomic_prop_refs():
    color = observation_prompt('color','recolor the chairs')
    assert 'recolor the chairs' in color
    assert 'One unit is one distinct object instance or edited part' in color
    motion = observation_prompt('motion','lower the phone')
    assert 'list a moved prop separately' in motion
    assert 'hand and sleeve' in motion
    assert len(color.split()) < 360


def record(identity):
    return {"change_id": identity, "status": "located", "boxes": [
        {"bbox_2d": [100,200,300,400], "region_mode": "object", "mask_density": "object"}]}


def test_id_and_source_appearance_are_owned_by_code():
    rows = [record(1), record(0)]
    rows[0]["ref"] = "bare arms"
    boxes, missing = parse_checklist_grounding(json.dumps(rows), OBS, "source")
    assert [b["ref"] for b in boxes] == ["grey hoodie", "black shirt"]
    assert not missing


def test_segmentation_noun_is_not_the_identity_attribute():
    observation = {"changes": [{"source_ref": "man in a black suit", "target_ref": "",
                                "sam_ref": "person", "region_description": "image-left"}]}
    boxes, _ = parse_checklist_grounding(json.dumps([record(0)]), observation, "source")
    assert boxes[0]['ref'] == 'person'
    assert boxes[0]['grounding_ref'] == 'man in a black suit'
    assert 'image-left' in boxes[0]['location'] and 'black suit' in boxes[0]['location']


def test_add_uses_target_subject_but_retains_target_identity():
    observation = {"changes": [{"source_ref": "", "target_ref": "green box",
                                "sam_ref": "box", "region_description": "girl's hands"}]}
    checklist = grounding_checklist(observation, 'target')
    assert checklist[0]['ref'] == 'green box' and checklist[0]['grounding_ref'] == 'green box'
    assert grounding_checklist(observation, 'source') == []


@pytest.mark.parametrize('original,expected', [
    ('man in a black suit','person'), ("man's left arm",'left arm'),
    ('man in black suit and glasses','person'),
    ('green robe','green robe'), ('dark wicker chairs with blue cushions','dark wicker chairs with blue cushions'),
    ('bird head with crest','bird head with crest'), ('head and neck of the large bird','head and neck of the large bird'),
    ("monkey's smiling face",'face'), ('closed eyes','eyes'), ('raised arm and hand','arm and hand'),
    ('man and horse','man and horse'), ('blue shirt','blue shirt'), ('closed book','closed book')])
def test_segmentation_normalization_preserves_scope(original, expected):
    assert segmentation_ref(original) == expected


def test_split_limb_boxes_do_not_ask_each_crop_to_ground_both_limbs():
    obs = {'changes':[{'source_ref':'woman arms','target_ref':'woman arms'}]}
    entry = record(0)
    entry['boxes'] *= 2
    boxes,_ = parse_checklist_grounding(json.dumps([entry]),obs,'source')
    assert [b['ref'] for b in boxes] == ['arm and hand','arm and hand']
    assert all(b['grounding_ref']=='woman arms' for b in boxes)


def test_evidence_uncertainty_is_not_silently_promoted_to_ok():
    payload = {'requests':[{'parse_ok':True,'evidence_issues':[{'decision':'uncertain'}]}],
               'boxes':{'source':[{'bbox_2d':[100,100,200,200]}],'target':[]}}
    row = _result_row(0,sample(),payload,{},'test',1)
    assert row['ground_parse_ok'] and row['qc_flag']=='GROUND_REVIEW'


def test_failed_coverage_review_is_not_silently_promoted_to_ok():
    payload = {'requests': [{'parse_ok': True}], 'coverage_review_failed': True,
               'boxes': {'source': [{'bbox_2d': [100,100,200,200]}], 'target': []}}
    row = _result_row(0, sample(), payload, {}, 'test', 1)
    assert row['ground_parse_ok'] and row['qc_flag'] == 'GROUND_REVIEW'


def test_group_layout_survives_explicit_object_mode_for_sam_matching():
    observation = {'changes': [{'source_ref': 'armchair', 'target_ref': 'armchair',
                               'region_layout': 'nearby_group'}]}
    boxes, _ = parse_checklist_grounding(json.dumps([record(0)]), observation, 'source')
    assert boxes[0]['region_mode'] == 'object'
    assert boxes[0]['region_layout'] == 'nearby_group'


def test_source_only_contract_cannot_claim_target_only_addition_is_covered():
    obs = {'changes':[{'source_ref':'arm','target_ref':'arm','change':'moved'},
                      {'source_ref':'','target_ref':'person','change':'added'}]}
    grounder = fake_grounder([json.dumps(obs),json.dumps([record(0)])])
    s = sample()
    s['type'] = 'motion change'
    payload = grounder.infer([s])[0]
    assert payload['canvas_issues'] == [{'change_id':1,'reason':'TARGET_ONLY_EDIT_IN_SOURCE_ONLY_TYPE'}]
    assert _result_row(0,s,payload,{},'test',1)['qc_flag']=='GROUND_REVIEW'


def test_model_cannot_replace_an_explicit_part_with_its_owner():
    obs = {'changes':[{'source_ref':'closed eyes','target_ref':'open eyes','sam_ref':'onion character'}]}
    assert grounding_checklist(obs,'source')[0]['ref'] == 'eyes'


def test_complete_wrappers_and_coordinate_arrays_keep_strict_identity():
    records = [record(0), record(1)]
    records[1].update(region_mode="object", mask_density="object", boxes=[[100,200,300,400]])
    boxes, unresolved = parse_checklist_grounding(json.dumps({"changes":records}), OBS, "source")
    assert len(boxes) == 2 and not unresolved
    with pytest.raises(ValueError):
        parse_checklist_grounding(json.dumps({"changes":[records[0]]}), OBS, "source")


def test_missing_optional_morphology_does_not_discard_valid_boxes():
    rows = [{"change_id":i,"status":"located","boxes":[[100,200,300,400]]} for i in (0,1)]
    boxes, unresolved = parse_checklist_grounding(json.dumps(rows), OBS, "source")
    assert len(boxes) == 2 and not unresolved
    assert "region_mode" not in boxes[0] and "mask_density" not in boxes[0]
    rows[0]["mask_density"] = "invalid"
    with pytest.raises(ValueError): parse_checklist_grounding(json.dumps(rows), OBS, "source")




@pytest.mark.parametrize("rows", [[record(0)], [record(0),record(0)], [record(0),record(2)],
                                  [record(True),record(1)]])
def test_missing_duplicate_unexpected_ids_rejected(rows):
    with pytest.raises(ValueError): parse_checklist_grounding(json.dumps(rows), OBS, "source")


def test_truncated_checklist_cannot_be_salvaged_as_success():
    with pytest.raises(ValueError): parse_checklist_grounding(json.dumps([record(0),record(1)])[:-3], OBS, "source")


def test_multiple_regions_keep_identity_and_unresolved_is_explicit():
    row = record(0)
    row["boxes"] *= 2
    boxes, unresolved = parse_checklist_grounding(json.dumps([row,
        {"change_id":1,"status":"not_visible","boxes":[],"reason":"occluded"}]), OBS,"source")
    assert len(boxes) == 2 and {b["change_id"] for b in boxes} == {0}
    assert unresolved == [{"change_id":1,"reason":"occluded"}]


def test_added_region_ids_are_not_renumbered_when_filtering_source():
    observation = {"changes":[{"change_id":0,"source_ref":"","target_ref":"lights"},
                               {"change_id":1,"source_ref":"shirt","target_ref":"shirt"}]}
    assert grounding_checklist(observation, "source")[0]["change_id"] == 1




def fake_grounder(responses):
    grounder = Qwen38Grounder.__new__(Qwen38Grounder)
    grounder.args = argparse.Namespace(grounding_mode="two-pass",background_observation_mode="foreground-audit",
        request_batch_size=4,parse_retries=1,observation_max_new_tokens=1024,max_new_tokens=512,bbox_refinement="off")
    grounder.prompt_version = "test"
    grounder.calls = []
    def generate(conversations, max_tokens=None):
        grounder.calls.append((conversations,max_tokens))
        grounder.last_generation_stats = [{"finish_reason":"stop","output_tokens":10} for _ in conversations]
        return [responses.pop(0) for _ in conversations]
    grounder._generate = generate
    return grounder


def sample():
    return {"type":"color","instruction":"fair skin wearing red jackets",
            "input_img":Image.new("RGB",(32,32)),"output_img":Image.new("RGB",(32,32))}


def test_no_visible_change_is_not_retried_into_an_invented_mask():
    grounder = fake_grounder(['{"changes":[]}'])
    payload = grounder.infer([sample()])[0]
    assert len(grounder.calls) == 1 and not payload['requests']
    assert payload['no_realized_changes']
    row = _result_row(0,sample(),payload,{},'test',1)
    assert row['ground_parse_ok'] and row['grounding_status'] == 'GROUND_FAIL'


def test_target_only_change_never_submits_empty_source_checklist():
    grounder = fake_grounder([json.dumps({'changes': [
        {'source_ref':'','target_ref':'herb','change':'added'}]})])
    s = sample()
    s['type'] = 'replace'
    payload = grounder.infer([s])[0]
    assert len(grounder.calls) == 1 and not payload['requests']
    assert payload['canvas_issues'] and not payload.get('no_realized_changes')
    row = _result_row(0,s,payload,{},'test',1)
    assert row['ground_parse_ok'] and row['grounding_status'] == 'GROUND_FAIL'


def test_length_retry_does_not_feed_truncated_answer_back_to_model():
    grounder = fake_grounder(['{"changes":', '{"changes":[]}'])
    generate = grounder._generate
    def truncated_once(conversations,max_tokens=None):
        output = generate(conversations,max_tokens)
        if len(grounder.calls) == 1:
            grounder.last_generation_stats[0]['finish_reason'] = 'length'
        return output
    grounder._generate = truncated_once
    payload = grounder.infer([sample()])[0]
    assert payload['observation']['parse_ok']
    retry = grounder.calls[1][0][0]
    assert len(retry) == 1 and retry[0]['role'] == 'user'
    assert 'COMPLETE concise JSON' in retry[0]['content'][-1]['text']


def test_instruction_guided_observation_uses_only_two_full_images():
    grounder = fake_grounder(['{"changes":[]}'])
    grounder.infer([sample()])
    conversation = grounder.calls[0][0][0]
    content = conversation[0]['content']
    assert sum(item['type'] == 'image' for item in content) == 2
    assert 'fair skin wearing red jackets' in content[-1]['text']


def test_observation_failure_stops_before_skin_grounding():
    grounder = fake_grounder(['{"changes":', '{"changes":'])
    payload = grounder.infer([sample()])[0]
    assert payload["observation_failed"] and not payload["requests"]
    assert len(grounder.calls) == 2
    assert [call[1] for call in grounder.calls] == [1024,2048]
    assert len(grounder.calls[1][0][0]) == 3  # Feedback, not identical greedy repetition.
    row = _result_row(0,sample(),payload,{},"test",1)
    assert not row["ground_parse_ok"] and row["qc_flag"] == "GROUND_FAIL"


def test_recovered_observation_then_single_image_id_grounding():
    observation = {"changes":[dict(OBS["changes"][0],change="black to red")]}
    grounder = fake_grounder(['{"changes":',json.dumps(observation),json.dumps([record(0)])])
    payload = grounder.infer([sample()])[0]
    assert payload["boxes"]["source"][0]["ref"] == "black shirt"
    conversation = grounder.calls[-1][0][0]
    assert sum(p["type"] == "image" for p in conversation[0]["content"]) == 1
    assert "fair skin" not in conversation[0]["content"][-1]["text"]


def test_unresolved_item_cannot_produce_ok_row():
    payload = {"requests":[{"parse_ok":True,"unresolved":[{"change_id":1}]}],
               "boxes":{"source":[{"bbox_2d":[0,0,100,100]}],"target":[]}}
    assert _result_row(0,sample(),payload,{},"test",1)["qc_flag"] == "GROUND_FAIL"






