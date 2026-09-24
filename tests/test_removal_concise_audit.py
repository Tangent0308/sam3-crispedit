import json
import pytest
from synthesis_pipeline.audit_removal_concise import admitted, parse_audit, PROMPT
from utils.removal_relations import concise_execution_prompt
from synthesis_pipeline.plan_removal_relations import PROMPT_V7, PROMPT_V6, validate_relation_plan


@pytest.mark.parametrize('quality,match,expected',[
    ('pass','pass',True),('pass','fail',False),('fail','pass',False),('fail','fail',False)])
def test_both_visual_quality_and_removal_required(quality,match,expected):
    parsed=parse_audit(json.dumps(dict(quality=quality,instruction_match=match,reason='Observed pixels.')))
    assert admitted(parsed)==expected


@pytest.mark.parametrize('raw',['{}','not json','<think>unfinished',
    '{"quality":"review","instruction_match":"pass","reason":"uncertain"}',
    '{"quality":"pass","instruction_match":"pass","reason":""}'])
def test_audit_fails_closed(raw):
    assert parse_audit(raw) is None
    assert not admitted(parse_audit(raw))


def test_audit_uses_final_verdict_after_thinking():
    raw = '<think>Possibly pass.</think>' + json.dumps(dict(
        quality='pass', instruction_match='fail', reason='The subject remains.'))
    assert parse_audit(raw)['instruction_match'] == 'fail'
    assert not admitted(parse_audit(raw))


def test_after_first_requires_separate_images(monkeypatch):
    from synthesis_pipeline.audit_removal_concise import main
    monkeypatch.setattr('sys.argv', ['audit', '--data-root', '/not/read',
        '--edited-dir', '/not/read', '--out-root', '/not/created',
        '--input-layout', 'paired', '--policy', 'pixels-first-v3'])
    with pytest.raises(ValueError, match='separate full images'):
        main()


def test_concise_prompts_have_bounded_length():
    assert len(PROMPT_V7.split()) < len(PROMPT_V6.split())*.55
    assert len(PROMPT.split()) < 180
    prompt=concise_execution_prompt('Remove the selected instance.',{'relations':[
        {'action':'remove_together','description':'attached item'},
        {'action':'keep','description':'independent neighbor'}]})
    assert 'attached item' in prompt and prompt.count('Remove ')==1
    assert len(prompt.split())<50


@pytest.mark.parametrize('policy',['relations-v5','relations-v6','relations-v7'])
def test_all_grounded_versions_validate_relation_bbox(policy):
    value=dict(decision='accept',target='object',instruction='Remove the object.',
        reconstruction='Fill the background.',target_point=[500,500],relations=[dict(
            description='attachment',action='remove_together',reason='exclusive',
            point=[500,500],segmentation_query='attachment',bbox=[900,900,1000,1000])])
    assert validate_relation_plan(value,policy)=='invalid_relation_bbox'
