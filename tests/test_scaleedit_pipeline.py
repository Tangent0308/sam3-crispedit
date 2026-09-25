import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import pytest

from scaleedit import policy, runner, quality
from scaleedit.mask.checklist import parse_checklist_grounding
from scaleedit.render import render_mask
from scaleedit.mask import pipeline as core


def row(task='object_replacement'):
    buf = io.BytesIO()
    Image.new('RGB', (100, 80), 'white').save(buf, format='PNG')
    return dict(sample_id='s', final_task=task, final_instruction='Replace the left cup',
                original_instruction='WRONG', source_image=buf.getvalue(), edited_image=buf.getvalue())


def units():
    return policy.parse_edit_units(json.dumps({'edits':[{'change':'Replace left cup; add spoon', 'units':[
        {'source_ref':'cup','target_ref':'bowl','source_location':'left','target_location':'left','geometry':'object'},
        {'source_ref':'','target_ref':'spoon','source_location':'','target_location':'right','geometry':'object'}]}]}))


def test_native_discovery_and_fields(tmp_path):
    for name in ('part-0.parquet', 'expand-x.parquet', 'other.parquet'):
        (tmp_path/name).touch()
    assert len(runner.discover(tmp_path)) == 2
    assert runner.base_row(3, row())['final_instruction'] == 'Replace the left cup'
    assert runner.decode(row()['source_image']).size == (100,80)


def test_mixed_units_source_and_target():
    obs = units()
    assert [u['change_id'] for u in policy.side_context(obs, 'source')['changes']] == [0]
    assert [u['change_id'] for u in policy.side_context(obs, 'target')['changes']] == [1]
    assert policy.mask_kind('count_change', obs['changes'][1]) == 'add'
    assert policy.mask_kind('object_addition', obs['changes'][0]) == 'replace'


def test_checklist_ids_and_semantics():
    obs = policy.side_context(units(), 'source')
    boxes, missing = parse_checklist_grounding('[{"change_id":0,"bbox_2d":[1,2,400,500]}]', obs, 'source')
    assert boxes[0]['ref'] == 'cup' and not missing
    with pytest.raises(ValueError):
        parse_checklist_grounding('[]', obs, 'source')
    with pytest.raises(ValueError):
        parse_checklist_grounding('[{"change_id":1,"bbox_2d":[1,2,400,500]}]', obs, 'source')


def test_scene_binary_and_reference():
    with pytest.raises(ValueError):
        policy.parse_scene('{"verdict":"PASS","target":"cup","reference":"NONE","reason":"clear"}')
    with pytest.raises(ValueError):
        policy.parse_scene('{"verdict":"REVIEW","target":"cup","reference":"left","reason":"clear"}')


def test_quality_no_change_overrides_all_pass():
    payload = {key:dict(status='PASS', evidence='visible') for key in quality.QUALITY_DIMENSIONS}
    payload.update(edit_observation=dict(source_state='red cup', target_state='red cup',
                                        visible_change='NONE', instruction_match='PASS'), reason_codes=[], summary='same')
    result = quality.normalize_quality_assessment(payload)
    assert result['keep'] is False


def test_global_no_model_call():
    args = SimpleNamespace(model_path='fake', stage='quality')
    result = runner.infer_filter(None, [(0,row('style_transfer'))], args)
    assert result[0]['verdict'] == 'DROP'
    assert result[0]['reason'] == 'excluded_edit_family'


def test_upstream_identity_and_instruction():
    records = [(0,row())]
    good = dict(runner.base_row(0,row()), keep=True, verdict='PASS')
    assert runner.passed_indices(records, {0:good}, 'test') == records
    for change in ({'sample_id':'other'}, {'final_instruction':'WRONG'}, {'keep':False}):
        with pytest.raises(ValueError):
            runner.passed_indices(records, {0:{**good,**change}}, 'test')
    with pytest.raises(ValueError):
        runner.passed_indices(records, {}, 'test')


def test_atomic_signature(tmp_path):
    path = tmp_path/'test.parquet'
    runner.write_table(path, [], runner.FILTER_SCHEMA, 'abc')
    assert runner.reusable(path, 'abc')
    with pytest.raises(ValueError):
        runner.reusable(path, 'changed')
    assert not path.with_suffix('.parquet.incomplete').exists()


def test_text_geometry_is_not_task_label(monkeypatch):
    obs = units()
    obs['changes'] = obs['changes'][:1]
    called = []
    def fake(processor, sample, ground, version):
        called.append(sample['type'])
        mask = np.zeros((80,100), dtype=np.uint8)
        mask[10:20,10:20] = 1
        rle = core.encode_rle(mask)
        return dict(mask=mask, qc_flags=['OK'], instances=[dict(mask_source='pvs',ref='cup',
                         area=int(mask.sum()),rle_size=rle['size'],rle_counts=rle['counts'])])
    monkeypatch.setattr(core, 'annotate_grounded_sample', fake)
    box = dict(change_id=0, ref='cup', bbox_2d=[100,100,300,300],
               polygon_2d=[[100,100],[300,200],[300,300],[100,200]])
    payload = dict(observation=dict(parsed=obs), boxes=dict(source=[box],target=[]))
    ground = dict(ground_json=json.dumps(payload), qc_flag='OK', error='')
    out = render_mask(None, 0, row('object_surface_text_editing'), ground)
    assert called == ['motion'] and out['mask_sum'] == 100
    obs['changes'][0]['geometry'] = 'text'
    payload['observation']['parsed'] = obs
    ground['ground_json'] = json.dumps(payload)
    out = render_mask(None, 0, row(), ground)
    assert called == ['motion']
    assert out['mask_source'] == 'text_polygon'
    assert out['mask_sum'] < 320
    image = np.asarray(Image.open(io.BytesIO(out['mask_png']))) > 0
    assert int(image.sum()) == out['mask_sum'] == out['instance_masks'][0]['area']


def test_ground_fail_is_not_valid_mask():
    out = render_mask(None, 0, row(), dict(ground_json='{}',qc_flag='GROUND_FAIL',error='failed'))
    assert out['qc_flag'] == 'GROUND_FAIL' and out['mask_sum'] == 0
    out = render_mask(None, 0, row(), dict(ground_json='{"observation":{"parsed":null}}',
                                          qc_flag='GROUND_FAIL',error='parse_failed'))
    assert out['qc_flag'] == 'GROUND_FAIL' and out['mask_sum'] == 0


def test_parse_missing_geometry_rejected():
    with pytest.raises(ValueError):
        policy.parse_edit_units('{"edits":[{"change":"a","units":[{"source_ref":"cup","target_ref":"bowl","source_location":"left","target_location":"left"}]}]}')


def test_grounding_batch_keeps_identity_and_canvas():
    from scaleedit.inference import Qwen38FilterEngine
    observation = {'edits':[{'change':'Replace cup and add spoon', 'units':[
        dict(source_ref='cup',target_ref='bowl',source_location='left',target_location='left',geometry='object'),
        dict(source_ref='',target_ref='spoon',source_location='',target_location='right',geometry='object')]}]}
    class Engine:
        _parse_with_retry = Qwen38FilterEngine._parse_with_retry
        args = SimpleNamespace(parse_retries=0)
        seen = []
        def generate(self, conversations, max_tokens):
            self.seen.extend(conversations)
            if len(self.seen) == 1:
                return [json.dumps(observation)]
            return ['[{"change_id":0,"bbox_2d":[10,20,300,500]}]',
                    '[{"change_id":1,"bbox_2d":[600,20,900,500]}]']
    engine = Engine()
    args = SimpleNamespace(model_path='fake',max_new_tokens=1024,batch_size=4)
    result = runner.infer_grounding(engine,[(5,row())],args)[0]
    assert result['qc_flag'] == 'OK' and result['row_idx'] == 5
    payload = json.loads(result['ground_json'])
    assert payload['boxes']['source'][0]['ref'] == 'cup'
    assert payload['boxes']['target'][0]['ref'] == 'spoon'
    image_counts = [sum(part['type']=='image' for part in conv[0]['content']) for conv in engine.seen]
    assert image_counts == [2,1,1]


def test_filter_reads_native_instruction_and_exposes_binary():
    from scaleedit.inference import Qwen38FilterEngine
    class Engine:
        _parse_with_retry = Qwen38FilterEngine._parse_with_retry
        args = SimpleNamespace(parse_retries=0)
        def generate(self, conversations, max_tokens):
            prompt = conversations[0][0]['content'][-1]['text']
            assert 'Replace the left cup' in prompt and 'WRONG' not in prompt
            return ['{"verdict":"PASS","target":"cup","reference":"left","reason":"selects one of two cups"}']
    args = SimpleNamespace(model_path='fake',stage='scene',max_new_tokens=256)
    result = runner.infer_filter(Engine(),[(0,row())],args)[0]
    assert result['keep'] and result['verdict'] == 'PASS'


def test_text_polygon_validation():
    obs = units()
    obs['changes']=obs['changes'][:1]
    obs['changes'][0]['geometry']='text'
    item=dict(change_id=0,bbox_2d=[100,100,300,300],polygon_2d=[[100,100],[300,200],[300,300],[100,200]])
    boxes,_=policy.parse_located_units(json.dumps([item]),obs,'source')
    assert boxes[0]['polygon_2d']==item['polygon_2d']
    for polygon in (None, [[100,100],[300,300],[300,100],[100,300]],
                    [[0,0],[300,0],[300,300],[0,300]], [[100,100]]*4):
        with pytest.raises(ValueError):
            policy.parse_located_units(json.dumps([{**item,'polygon_2d':polygon}]),obs,'source')
    octagon=[[110,100],[290,100],[300,110],[300,290],[290,300],[110,300],[100,290],[100,110]]
    boxes,_=policy.parse_located_units(json.dumps([{**item,'polygon_2d':octagon}]),obs,'source')
    assert boxes[0]['polygon_2d']==octagon


def test_parser_replay_preserves_raw_answers():
    from scripts.reparse_scaleedit_grounding import reparse
    observation={'edits':[{'change':'A to B','units':[dict(source_ref="'A'",target_ref="'B'",
                 source_location='left sign',target_location='left sign',geometry='text')]}]}
    polygon=[[110,100],[290,100],[300,110],[300,290],[290,300],[110,300],[100,290],[100,110]]
    raw=json.dumps([dict(change_id=0,bbox_2d=[100,100,300,300],polygon_2d=polygon)])
    payload=dict(observation=dict(attempts=[dict(raw_text=json.dumps(observation))]),
                 requests=[dict(grounding_image='source',attempts=[dict(raw_text=raw)])])
    result=reparse(dict(ground_json=json.dumps(payload),qc_flag='GROUND_FAIL',error='old parser required four vertices'))
    updated=json.loads(result['ground_json'])
    assert result['qc_flag']=='OK'
    assert updated['requests'][0]['attempts'][0]['raw_text']==raw
    assert updated['boxes']['source'][0]['polygon_2d']==polygon
    assert updated['parser_replay']['model_called'] is False
