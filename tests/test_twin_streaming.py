from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from sa1b_pipeline import twin_streaming as stream
from sa1b_pipeline.inference_services import RemoteBlind
from sa1b_pipeline.production_state import atomic_json


def test_resident_planner_preserves_per_request_schema_and_order():
    from sa1b_pipeline.inference_services import RemoteSelector
    calls=[]
    def request(payload):
        calls.append(payload)
        return {'ref':payload['prompt']}
    model=RemoteSelector(SimpleNamespace(model='27b',batch_size=2,devices=['0'],request=request))
    schemas=[{'type':'object','properties':{'addition':{'type':'string'}}},{'type':'object','properties':{}}]
    result=model.generate_multimage([(['src','a'],'first'),(['src','b'],'second')],schemas=schemas)
    assert result==[{'ref':'first'},{'ref':'second'}]
    assert {j['prompt']:j['schema'] for j in calls}==dict(zip(['first','second'],schemas))
    with pytest.raises(ValueError,match='schema count'):
        model.generate_multimage([(['src'],'x')],schemas=[])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    image = tmp_path/'source.png'
    Image.new('RGB', (20, 20), 'gray').save(image)
    plan = {'twin_id':'twin-a','operation':'remove','source_image':str(image),
            'group_checks':{},'members':[{'mask':str(image), 'unit':{}}]}
    calls = []
    def blind(groups, model, directory):
        calls.append('blind')
        for g in groups:
            g['group_checks']['blind_disambiguation'] = {'pass':True}
    def generate(item):
        calls.append('generate')
        g = deepcopy(item['group']); d = Path(item['directory']); d.mkdir(parents=True,exist_ok=True)
        out = d/'edited.png'; Image.new('RGB',(20,20),'white').save(out)
        g['members'][0]['edited_image'] = str(out)
        atomic_json(d/'group.json',g)
        return g
    def audit(groups, model, directory, **kwargs):
        calls.append('audit')
        groups[0]['group_checks']['group_pass'] = True
        return groups
    monkeypatch.setattr(stream,'run_blind_probes',blind)
    monkeypatch.setattr(stream,'audit_twin_groups_joint',audit)
    editors = SimpleNamespace(model='editor', request=generate)
    audits = SimpleNamespace(model='audit',batch_size=4,devices=['0'])
    return plan, editors, audits, calls


def test_resume_skips_models_and_checks_media(setup,tmp_path):
    plan,editors,audits,calls=setup
    original=deepcopy(plan)
    result=stream.process_twin(plan,tmp_path/'run',editors,audits)
    assert result['status']=='accepted' and calls==['blind','generate','audit']
    assert plan==original
    assert stream.process_twin(plan,tmp_path/'run',editors,audits)==result
    assert calls==['blind','generate','audit']
    Image.new('RGB',(20,20),'red').save(result['group']['members'][0]['edited_image'])
    with pytest.raises(ValueError,match='output changed'):
        stream.process_twin(plan,tmp_path/'run',editors,audits)


def test_failed_audit_resumes_saved_generation(setup,tmp_path,monkeypatch):
    plan,editors,audits,calls=setup
    original=stream.audit_twin_groups_joint
    def crash(*args, **kwargs): raise RuntimeError('service lost')
    monkeypatch.setattr(stream,'audit_twin_groups_joint',crash)
    with pytest.raises(RuntimeError,match='service lost'):
        stream.process_twin(plan,tmp_path/'run',editors,audits)
    monkeypatch.setattr(stream,'audit_twin_groups_joint',original)
    result=stream.process_twin(plan,tmp_path/'run',editors,audits)
    assert result['status']=='accepted' and calls.count('generate')==1


def test_blind_reject_never_generates(setup,tmp_path,monkeypatch):
    plan,editors,audits,calls=setup
    def reject(groups,model,directory):
        groups[0]['group_checks']['blind_disambiguation']={'pass':False}
    monkeypatch.setattr(stream,'run_blind_probes',reject)
    result=stream.process_twin(plan,tmp_path/'run',editors,audits)
    assert result['status']=='blind_rejected' and calls==[]


def test_changed_model_or_input_refuses_resume(setup,tmp_path):
    plan,editors,audits,calls=setup
    stream.process_twin(plan,tmp_path/'run',editors,audits)
    editors.model='other'
    with pytest.raises(ValueError,match='input changed'):
        stream.process_twin(plan,tmp_path/'run',editors,audits)


def test_shared_blind_preserves_full_resolution_and_detects_truncation():
    jobs=[]
    def request(payload):
        jobs.append(payload)
        return {'semantic_audit':{'decision':'ambiguous'},'semantic_audit_raw':'raw',
                'finish_reason':'length','input_tokens':9,'output_tokens':1000}
    model=RemoteBlind(SimpleNamespace(model='27b',batch_size=4,devices=['0'],request=request))
    results=model.generate_multimage([(['source.png'],'blind sentence')])
    assert jobs[0]['image_max_edge']==0
    assert jobs[0]['images']==['source.png'] and jobs[0]['prompt']=='blind sentence'
    assert results[0]['_generation_metadata']['finish_reason']=='length'


def test_overlap_gpu_assignment_rejected_before_start(tmp_path):
    from scripts.run_twins_streaming import main
    with pytest.raises(SystemExit):
        main(['--planned-manifest',str(tmp_path/'plans.jsonl'),'--output-root',str(tmp_path/'out'),
              '--audit-model','audit','--editor-model','editor','--site-model','sam3',
              '--mllm-python','python','--editor-python','python','--editor-devices','0,1','--audit-devices','1'])


def test_audit_and_blind_keep_distinct_resize_contracts(tmp_path):
    from sa1b_pipeline.audit_backend import audit_images
    path=tmp_path/'large.png';Image.new('RGB',(2048,1024)).save(path)
    assert audit_images([path])[0].size==(1536,768)
    assert audit_images([path],max_edge=0)[0].size==(2048,1024)


def test_omit_debug_views_preserves_exact_planning_pixels(tmp_path):
    import numpy as np
    from sa1b_pipeline.group_planning import group_views
    source=tmp_path/'source.png';Image.new('RGB',(100,80),'gray').save(source)
    mask=tmp_path/'mask.png';Image.new('L',(100,80),255).save(mask)
    candidate={'image_path':str(source)};objects=[{'object_id':'1','mask_path':str(mask)}]
    first,masks1,digest1=group_views(candidate,objects,tmp_path/'debug',locator_outline=True,locate_in_scene=True)
    second,masks2,digest2=group_views(candidate,objects,tmp_path/'fast',locator_outline=True,locate_in_scene=True,save_debug_views=False)
    assert digest1==digest2
    for a,b in zip(first+masks1,second+masks2):
        assert np.array_equal(np.asarray(Image.open(a)),np.asarray(Image.open(b)))
    assert not (Path(second[0]).parent/'marked_scene.png').exists()


def test_planner_replicas_fill_both_workers_without_losing_request_binding():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from sa1b_pipeline.inference_services import RemoteSelector
    barrier=threading.Barrier(4)
    def request(payload):
        barrier.wait(timeout=5)
        return {'prompt':payload['prompt'],'schema':payload['schema'],'images':payload['images']}
    model=RemoteSelector(SimpleNamespace(model='27b',batch_size=2,devices=['0','1'],request=request))
    jobs=[([f'image-{i}'],f'prompt-{i}') for i in range(4)]
    schemas=[{'const':i} for i in range(4)]
    result=model.generate_multimage(jobs,schemas=schemas)
    assert result==[{'prompt':p,'schema':s,'images':ims} for (ims,p),s in zip(jobs,schemas)]
