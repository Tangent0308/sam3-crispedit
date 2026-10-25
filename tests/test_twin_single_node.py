import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from sa1b_pipeline.twin_planning import normalize_compact_twin
from sa1b_pipeline.group_audit import twin_selected_comparison


def compact():
    return {'grounding_version':5,'members':[{'object_id':'1','entity':'lamp','ref_phrase':'the left lamp','kind':'object','host':'wall','mask_extent':'lamp body','excludes':'wall','distinction':'left of doorway','nearest_rival':'other lamp is right of doorway','ref_unique':True,'fit':'same wall mounting','edit_feasible':True,'site_visible':True}]}


def test_compact_binding_preserves_source_and_rejects_conflict():
    raw=compact();original=deepcopy(raw);normalized=normalize_compact_twin(raw)
    assert raw==original
    assert normalized['members'][0]['scope']==normalized['members'][0]['grounding']['entity']=='lamp'
    assert normalized['shared_fit'][0]['natural'] is True
    raw['members'][0]['scope']='chair'
    with pytest.raises(ValueError,match='overrides'):normalize_compact_twin(raw)


def test_compact_requires_actual_identification_evidence():
    raw=compact();raw['members'][0]['mask_extent']=''
    with pytest.raises(ValueError,match='incomplete'):normalize_compact_twin(raw)


def test_selector_audit_keeps_alignment_and_rejects_shape_mismatch(tmp_path):
    src=tmp_path/'src.png';tar=tmp_path/'tar.png';mask=tmp_path/'mask.png'
    Image.new('RGB',(100,80),'red').save(src);Image.new('RGB',(100,80),'blue').save(tar)
    m=Image.new('L',(100,80));ImageDraw.Draw(m).rectangle((20,20,70,60),fill=255);m.save(mask)
    group={'source_image':str(src),'members':[{'unit_id':'U1','mask':str(mask),'edited_image':str(tar)}]}
    out=twin_selected_comparison(group,tmp_path/'panel.jpg')
    im=Image.open(out)
    assert im.size==(1200,440)
    a,b=im.getpixel((300,230)),im.getpixel((900,230))
    assert a[0]>240 and a[2]<10 and b[2]>240 and b[0]<10
    Image.new('RGB',(99,80),'blue').save(tar)
    with pytest.raises(ValueError,match='shape mismatch'):twin_selected_comparison(group,tmp_path/'bad.jpg')


def test_single_node_replays_waves_and_keeps_resident_services(tmp_path,monkeypatch):
    from scripts import run_twins_single_node as entry
    from sa1b_pipeline.production_state import write_rows,atomic_json
    import sa1b_pipeline.source_isolation as isolation
    source=tmp_path/'source';root=tmp_path/'run'
    write_rows(source/'inputs/train_rank0.jsonl',[{'source_id':'a'},{'source_id':'b'}])
    atomic_json(source/'inputs/eval_reservation.json',{})
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(isolation,'assert_no_eval_leakage',lambda *a:None)
    monkeypatch.setattr(entry,'target_cells',lambda *a:{'remove:cross:2':2})
    monkeypatch.setattr(entry,'validate_twin_record',lambda *a,**kw:None)
    instances=[];calls=[]
    class Pool:
        def __init__(self,**kwargs):self.closed=0;instances.append(self)
        def close(self):self.closed+=1
    monkeypatch.setattr(entry,'ServicePool',Pool)
    for n,sid in enumerate(('a','b'),1):
        write_rows(root/'waves'/f'{n:05d}'/'inputs/selected.jsonl',[{'candidate':{'source_id':sid},'operation':'remove','category_relation':'cross','objects':[{'object_id':'1'},{'object_id':'2'}]}])
    def stream(argv,services):
        wave=Path(argv[argv.index('--output-root')+1]);calls.append((wave.name,services))
        sid='a' if wave.name=='00001' else 'b'
        write_rows(wave/'audited.jsonl',[{'twin_id':sid,'source_id':sid,'operation':'remove','category_relation':'cross','members':[{},{}],'group_checks':{'group_pass':True}}])
        atomic_json(wave/'summary.json',{})
    monkeypatch.setattr(entry,'run_stream',stream)
    argv=['--root',str(root),'--source-root',str(source),'--target','2','--mllm-python','python','--editor-python','python','--skip-model-cache']
    entry.main(argv)
    assert len(calls)==2 and calls[0][1]==calls[1][1]
    assert len(instances)==3 and all(x.closed==1 for x in instances)
    assert json.loads((root/'progress.json').read_text())['complete'] is True
    entry.main(argv)
    assert len(calls)==4  # Revalidates completed waves instead of trusting summary alone.
    assert len((root/'final/twins_train.jsonl').read_text().splitlines())==2


def test_generation_reference_drops_scene_locator_without_changing_labels():
    from sa1b_pipeline.editor import edit_prompt
    from sa1b_pipeline.instructions import build_unit
    raw={'feasible':True,'ref_phrase':'the white hat with a brown pom-pom','scope':'hat','operation':'attribute','payload':'','attributes':[{'property':'color','value':'blue'}]}
    unit=build_unit(raw,{'object_id':'1','category':'hat'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
    from dataclasses import asdict
    value=asdict(unit);original=deepcopy(value)
    value['generation_reference_frame']='crop-entity-v1'
    prompt=edit_prompt(value,target_hint='the center')
    assert prompt.startswith('Change the color of the hat to blue.')
    assert 'pom-pom' not in prompt
    assert value['instruction_ref']==original['instruction_ref']
    assert value['instruction_noref']==original['instruction_noref']


def test_local_generation_keeps_add_site():
    from sa1b_pipeline.editor import edit_prompt
    from sa1b_pipeline.instruction_protocol import make_spec, render
    s=make_spec('add','the red mailbox on the left','mailbox','a small star',placement={'relation':'on','site':'front panel','anchor_mode':'support'})
    u={'operation':'add','instruction_spec':s,'instruction_ref':render(s),'instruction_noref':render(s,False),'generation_reference_frame':'crop-entity-v1'}
    prompt=edit_prompt(u)
    assert prompt.startswith('Add a small star on the front panel of the mailbox.')
    assert 'on the left' not in prompt


def test_deployment_contains_fonts_and_refuses_overwrite(tmp_path):
    from scripts.prepare_twins_single_node import main
    import tarfile,hashlib
    main(['--root',str(tmp_path)])
    d=tmp_path/'deployment';bundle=d/'code.tar.gz'
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==(d/'code.sha256').read_text().strip()
    with tarfile.open(bundle) as f:
        names=f.getnames()
        assert 'sa1b_pipeline/assets/fonts/DejaVuSans.ttf' in names
        assert 'scripts/twins_single_node_entry.sh' in names
        assert 'scripts/groups_single_node_entry.sh' in names
        assert not any('__pycache__' in n or n.startswith('.git') for n in names)
    with pytest.raises(FileExistsError):main(['--root',str(tmp_path)])


def test_blind_requests_feed_editors_without_mixing_slow_audits(tmp_path):
    import os
    from scripts.group_production.inference_worker import next_batch
    audit=tmp_path/'a.json';blind=tmp_path/'z.json';audit.write_text('{}');blind.write_text('{"request_kind":"blind"}')
    os.utime(audit,ns=(1,1));os.utime(blind,ns=(2,2))
    assert next_batch([audit,blind],'audit',8)==[blind]
    assert next_batch([audit],'audit',8)==[audit]
    assert next_batch([blind,audit],'planner',8)==[audit,blind]


def test_planner_cannot_reject_without_identifying_members():
    import jsonschema
    from sa1b_pipeline.twin_planning import output_schema
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'feasible':False,'reason':'different categories'},output_schema())


def test_multi_entry_counts_retained_targets_and_replays_without_new_selection(tmp_path, monkeypatch):
    from scripts import run_twins_single_node as entry
    from sa1b_pipeline.production_state import write_rows, atomic_json
    import sa1b_pipeline.source_isolation as isolation
    source=tmp_path/'source';root=tmp_path/'multi'
    write_rows(source/'inputs/train_rank0.jsonl',[{'source_id':'a'}])
    atomic_json(source/'inputs/eval_reservation.json',{})
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(isolation,'assert_no_eval_leakage',lambda *a:None)
    monkeypatch.setattr(entry,'target_cells',lambda kind,n:{'2:attribute+remove':1})
    monkeypatch.setattr(entry,'validate_multi_record',lambda *a,**k:None)
    instances=[];calls=[]
    class Pool:
        def __init__(self,**kwargs):self.options=kwargs;self.closed=0;instances.append(self)
        def close(self):self.closed+=1
    monkeypatch.setattr(entry,'ServicePool',Pool)
    selection={'candidate':{'source_id':'a'},'operations':['attribute','remove','add'],
               'objects':[{'object_id':str(i)} for i in range(3)]}
    write_rows(root/'waves/00001/inputs/selected.jsonl',[selection])
    def wave(directory,args,reservation,pools):
        calls.append(args.kind)
        group={'multi_id':'a','source_id':'a','operations':['attribute','remove'],'n_targets':2,
               'units':[{'instruction_spec':{'operation':op}} for op in ['attribute','remove']],
               'joint_audit':{'pass':True}}
        write_rows(directory/'audited.jsonl',[group])
    monkeypatch.setattr(entry,'run_multi_wave',wave)
    argv=['--kind','multi','--root',str(root),'--source-root',str(source),'--target','1',
          '--mllm-python','python','--editor-python','python','--skip-model-cache']
    entry.main(argv);entry.main(argv)
    report=json.loads((root/'progress.json').read_text())
    assert report['complete'] and report['by_operation']=={'attribute':1,'remove':1}
    assert report['member_counts']=={'2':1} and calls==['multi','multi']
    assert len((root/'final/multi_train.jsonl').read_text().splitlines())==1
    assert all(p.closed==1 for p in instances)
    assert instances[0].options['role']=='editor'
    assert instances[2].options['planner_options']['thinking_token_budget']==512
    assert json.loads((root/'config.json').read_text())['audit_mode']=='component-final-v2'
