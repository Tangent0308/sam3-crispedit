import json
from collections import Counter
import numpy as np
import pytest
from PIL import Image
from sa1b_pipeline.io import candidate_from_dict, read_jsonl
from sa1b_pipeline.operation_policy import assign_operations
from sa1b_pipeline.mllm import planning_prompt, selection_prompt
from sa1b_pipeline.select import select_and_plan


def test_assign_before_sharding_and_refill_deficit():
    ids=[f's{i}' for i in range(12)]
    assigned=assign_operations(ids)
    assert Counter(assigned.values())==dict(add=3,replace=3,remove=3,attribute=3)
    assert {k:assigned[k] for k in ids[1::3]}=={k:assigned[k] for k in ids if k in ids[1::3]}
    assert set(assign_operations(ids[:4],accepted_counts=dict(add=0,replace=4,remove=4,attribute=4)).values())=={'add'}
    with pytest.raises(ValueError):assign_operations(['same','same'])


def test_required_operation_is_in_both_stages_without_free_choice():
    for op in ['add','replace','remove','attribute']:
        assert f'HARD OPERATION ASSIGNMENT: {op}' in selection_prompt({'objects':[]},required_operation=op)
        p=planning_prompt({'objects':[]},[],{},required_operation=op)
        assert f"unit.operation MUST equal '{op}'" in p
        assert 'Choose exactly one suitable operation' not in p
        assert 'Prefer different operation types' not in p


def cohort(tmp_path,nested=False):
    source=tmp_path/'s.png';Image.new('RGB',(150,100),'white').save(source)
    objs=[]
    for i in range(2):
        mask=np.zeros((100,150),dtype='uint8')
        if nested:mask[10+i*10:80-i*10,10+i*10:80-i*10]=255
        else:mask[20:60,10+i*75:50+i*75]=255
        path=tmp_path/f'{i}.png';Image.fromarray(mask).save(path)
        objs.append(dict(object_id=str(i),category='cup',mask_path=str(path)))
    return candidate_from_dict(dict(source_id='s',image_path=str(source),objects=objs))


def run(tmp_path,ops,nested=False,assigned='remove'):
    c=cohort(tmp_path,nested);audit=tmp_path/'audit.jsonl'
    plan={'units':[dict(object_id=str(i),operation=op,ref_phrase='the ceramic cup',scope='cup',payload='blue' if op=='attribute' else '') for i,op in enumerate(ops)]}
    result=select_and_plan([c],None,operation_assignments={'s':assigned},audit_path=audit,
       selection_decisions={'s':{'accept':True,'selected_object_ids':['0','1']}},planning_decisions={'s':plan})
    return result,read_jsonl(audit)


def test_wrong_type_is_rejected_not_relabelled(tmp_path):
    result,audit=run(tmp_path,['attribute','remove'])
    assert result==[]
    assert any(x.get('error')=='assigned_operation_mismatch' for x in audit)
    assert any(x.get('error')=='fewer_than_min_quality_units' for x in audit)


def test_valid_type_preserves_two_targets(tmp_path):
    result,audit=run(tmp_path,['remove','remove'])
    assert len(result)==1
    assert {u.operation for u in result[0].units}=={'remove'}
    assert result[0].selector['required_operation']=='remove'


def test_nested_masks_rejected_before_generation(tmp_path):
    result,audit=run(tmp_path,['remove','remove'],nested=True)
    assert result==[]
    assert any(x.get('error')=='overlapping_edit_targets' for x in audit)


def test_operation_ineligible_masks_hidden_before_visual_selection(tmp_path):
    c=cohort(tmp_path)
    from sa1b_pipeline.schema import CandidateObject
    tiny=tmp_path/'tiny.png';a=np.zeros((100,150),dtype='uint8');a[5,5]=255;Image.fromarray(a).save(tiny)
    c.objects.append(CandidateObject(object_id='tiny',category='cup',mask_path=str(tiny)))
    class Selector:
        calls=0
        def generate(self,jobs):
            self.calls+=1
            if self.calls==1:
                assert '"tiny"' not in jobs[0][1]
                return [{'accept':True,'selected_object_ids':['0','1']}]
            return [{'units':[dict(object_id=str(i),operation='remove',ref_phrase='the ceramic cup',scope='cup') for i in range(2)]}]
    result=select_and_plan([c],Selector(),operation_assignments={'s':'remove'})
    assert len(result)==1
