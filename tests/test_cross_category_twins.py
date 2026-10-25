import copy
from pathlib import Path

import pytest
from PIL import Image

from sa1b_pipeline.blind_identifiability import probe_request, evaluate_probe, run_blind_probes
from sa1b_pipeline.group_planning import validate_twin_grounding, build_twin_group
from sa1b_pipeline.source_regions import blind_group_check


def group(tmp_path):
    path=tmp_path/'source.png';Image.new('RGB',(1000,1000)).save(path)
    return {'twin_id':'private-id','source_image':str(path),'instruction_noref':'Remove the selected region.',
            'members':[{'unit_id':'secret-A','bbox':[100,100,300,300],'scope':'lamp'},
                       {'unit_id':'secret-B','bbox':[600,600,800,800],'scope':'chair'}], 'group_checks':{}}


def response():
    return {'targets':[{'description':'lamp','bbox_1000':[100,100,300,300],'naturalness':.9,'preference_weight':.5},
                       {'description':'chair','bbox_1000':[600,600,800,800],'naturalness':.9,'preference_weight':.5}]}


def test_blind_probe_never_exposes_members_refs_masks_or_targets(tmp_path):
    g=group(tmp_path);g['members'][0].update(ref='private-ref',mask='private-mask',edited_image='private-target')
    images,prompt=probe_request(g)
    assert images==[g['source_image']]
    for leak in ['private-id','secret-A','secret-B','private-ref','private-mask','private-target','100,100,300,300']:
        assert leak not in prompt
    assert evaluate_probe(g,response())['pass']


@pytest.mark.parametrize('failure',['bias','unnatural','missing','ambiguous','invalid_score'])
def test_blind_probe_does_not_confuse_missing_evidence_with_unidentifiability(tmp_path,failure):
    g=group(tmp_path);raw=response()
    if failure=='bias':raw['targets'][0]['preference_weight']=.1
    if failure=='unnatural':raw['targets'][0]['naturalness']=.2
    if failure=='missing':raw['targets'].pop()
    if failure=='ambiguous':raw['targets'].append(copy.deepcopy(raw['targets'][0]))
    if failure=='invalid_score':raw['targets'][0]['preference_weight']=float('nan')
    assert not evaluate_probe(g,raw)['pass']


def test_probe_resume_uses_exact_request_and_never_repeats_completed_call(tmp_path):
    g=group(tmp_path)
    class Model:
        model_id='fake';batch_size=2;calls=0
        def generate_multimage(self,jobs):
            self.calls+=len(jobs);return [response() for _ in jobs]
    model=Model();run_blind_probes([g],model,tmp_path/'probe')
    run_blind_probes([g],model,tmp_path/'probe');assert model.calls==1
    g['instruction_noref']='Replace the selected region with a chair.'
    with pytest.raises(ValueError,match='inputs changed'):run_blind_probes([g],model,tmp_path/'probe')


def test_independent_member_scopes_and_mixed_granularity_are_allowed():
    plan={'grounding_version':3,'operation':'remove','members':[
        {'scope':noun,'grounding':{'entity':noun,'kind':'object','distinction':noun,'edit_feasible':True}}
        for noun in ['lamp','chair']]}
    validate_twin_grounding(plan)
    plan['members'][1]['grounding']['kind']='part'
    validate_twin_grounding(plan)


def test_candidate_quota_keeps_thirty_percent_same_and_four_operations(monkeypatch):
    from types import SimpleNamespace
    from collections import Counter
    from scripts import run_group_pipeline as runner
    def pool(candidates, *, operation, category_relation, **kwargs):
        result=[]
        for i in range(30):
            name=f'{operation}-{category_relation}-{i}'
            result.append({'candidate':{'source_id':name},'operation':operation,
                'objects':[{'object_id':'a','category':name},{'object_id':'b','category':name if category_relation=='same' else name+'-other'}],
                'category_relation':category_relation,
                'scope_level':['object','part_on_different_hosts','part_on_same_host'][i%3],
                'bucket':['lt0.5%','0.5-2%','gt2%'][(i//3)%3]})
        return result
    monkeypatch.setattr(runner,'select_twin_candidates',pool)
    monkeypatch.setattr(runner,'_row_area_bucket',lambda row:row['bucket'])
    args=SimpleNamespace(kind='twins',target=10,split='train',seed=8,candidate_pool_factor=1,
                         twin_three_member_fraction=0,twin_same_category_fraction=.3)
    result=runner._select(args,[],set())
    assert Counter(r['category_relation'] for r in result)=={'same':3,'cross':7}
    assert sorted(Counter(r['operation'] for r in result).values())==[2,2,3,3]


def test_native_qwen_bbox_alias_is_not_a_semantic_failure(tmp_path):
    raw=response()
    for target in raw['targets']:
        target['bbox_2d']=target.pop('bbox_1000')
    assert evaluate_probe(group(tmp_path),raw)['pass']
    assert 'bbox_1000' not in raw['targets'][0]


def test_cross_category_replace_cannot_be_only_a_finish():
    plan={'grounding_version':3,'operation':'replace','payload':'chrome finish','replacement_form':'chrome',
          'members':[{'scope':noun,'grounding':{'entity':noun,'kind':'part','distinction':noun,'edit_feasible':True}}
                     for noun in ['exhaust pipe','window panel']]}
    with pytest.raises(ValueError,match='only an attribute'):validate_twin_grounding(plan)
