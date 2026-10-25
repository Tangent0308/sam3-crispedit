from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from sa1b_pipeline import per_mask_planning as pm
from sa1b_pipeline.twin_contract import ADAPTED, blind_probe_satisfied
from sa1b_pipeline.group_planning import build_twin_group
from sa1b_pipeline.group_schema import validate_twin_record
from sa1b_pipeline.source_regions import blind_group_check


def test_schema_count_checked_before_multimodal_inference():
    from sa1b_pipeline.mllm import VLLMSelector
    model=VLLMSelector.__new__(VLLMSelector)
    model._inherited_backend=False
    with pytest.raises(ValueError,match='schema count mismatch'):
        model.generate_multimage([(['missing.png'],'prompt')],schemas=[])


def test_single_image_entry_does_not_read_a_multimodal_schema_variable(tmp_path):
    from types import SimpleNamespace
    from sa1b_pipeline.mllm import VLLMSelector
    model=VLLMSelector.__new__(VLLMSelector)
    model.batch_size=4;model._inherited_backend=True;model.max_new_tokens=100
    model.backend=SimpleNamespace(chat_batch=lambda *a,**kw:['{"accept": true}'])
    path=tmp_path/'image.png';Image.new('RGB',(8,8)).save(path)
    assert model.generate([(str(path),'prompt')])==[{'accept':True}]


def data(tmp_path):
    p=tmp_path/'source.png';Image.new('RGB',(400,300),'gray').save(p)
    objects=[]
    for i,x in enumerate([25,280]):
        mask=np.zeros((300,400),dtype='uint8');mask[120:170,x:x+50]=255
        mp=tmp_path/f'{i}.png';Image.fromarray(mask).save(mp)
        objects.append(dict(object_id=str(i),mask_path=str(mp),category='chair',metadata={}))
    return dict(source_id='test',image_path=str(p)),objects


def edits():
    return [dict(ref='the left chair',edit={'property':'color','value':'blue'}),
            dict(ref='the right lamp',edit={'property':'color','value':'yellow'})]


def test_explicit_appearance_replacement_rejects_without_reassigning_operation():
    raws = [{'ref': 'the red chair beside the gate', 'edit': {'replacement': 'a blue chair'}},
            {'ref': 'the right chair', 'edit': {'replacement': 'a folding chair'}}]
    plan = pm.combine(raws, 'multi', {'U1': 'replace', 'U2': 'replace'},
                      [{'object_id': 'a'}, {'object_id': 'b'}])
    first, second = plan['targets']
    assert not first['feasible'] and 'replacement_only_changes_appearance' in first['reason']
    assert not first.get('_format_error')
    assert second['feasible'] and second['payload'] == 'a folding chair'
    assert plan['_per_mask_responses'] == raws
    from sa1b_pipeline.content_guard import appearance_only_replacement
    assert appearance_only_replacement('the teal headband worn by the runner',
                                       'a black fabric headband')
    assert appearance_only_replacement('the turquoise headband on the runner',
                                       'a black fabric headband')
    assert appearance_only_replacement('the reddish cross on the right-facing gable',
                                       'a plain white cross')
    assert appearance_only_replacement('the reddish cross on the right-facing gable',
                                       'a white star emblem') is None


def test_visible_letter_a_is_not_a_dangling_article_in_reference():
    raw = {'ref': 'the cream diagonal roof beam beneath the letter A',
           'edit': {'relation': 'along', 'site': 'its upper edge',
                    'addition': 'a thin row of warm-white LED lights'}}
    plan = pm.combine([raw], 'multi', {'U1': 'add'}, [{'object_id': 'beam'}])
    assert plan['targets'][0]['feasible']
    assert plan['targets'][0]['ref_phrase'] == raw['ref']
    assert plan['targets'][0]['payload'] == raw['edit']['addition']
    from sa1b_pipeline.content_guard import reference
    for incomplete in ['the beam beneath a', 'the beam beneath the',
                       'the beam beside the letter', 'the beam beneath the letter A...']:
        # "the letter" is grammatically complete; only reject the other
        # unfinished forms here, without inventing a semantic completeness gate.
        if incomplete.endswith('the letter'):
            assert reference(incomplete) == incomplete
        else:
            with pytest.raises(ValueError, match='unfinished phrase'):
                reference(incomplete)


@pytest.mark.parametrize('ref,payload', [
    ('the black chair', 'a white rocking chair'),
    ('the red pane', 'a frosted glass panel with a sliding vent'),
    ('the jeans worn by the man', 'a pair of linen trousers'),
    ('the blue drawer', 'a deep blue drawer'),
])
def test_replacement_guard_preserves_structural_or_uncertain_changes(ref, payload):
    from sa1b_pipeline.content_guard import appearance_only_replacement
    assert appearance_only_replacement(ref, payload) is None


def group(tmp_path):
    candidate,objects=data(tmp_path);views,masks,_=pm.views(candidate,objects,tmp_path/'views')
    from scripts.run_group_pipeline import _geometry
    plan=pm.combine(edits(),'twins',dict(U1='attribute',U2='attribute'),objects)
    g=build_twin_group(candidate,objects,plan,masks,split='train',scope_level='object',group_id='g',
                       geometry=_geometry(masks),assigned_operation='attribute',require_grounding=True)
    for m in g['members']:m.update(support_mask=m['mask'],blind_eligible=True)
    return g


def test_lossless_slots_reach_final_twin_member_and_nested_unit(tmp_path):
    from scripts.run_group_pipeline import _geometry
    c, objects = data(tmp_path)
    _, masks, _ = pm.views(c, objects, tmp_path/'views')
    raw = [{'ref': ref, 'edit': {'addition': 'a tray with a cup on top',
                                'relation': 'on', 'site': 'the upper shelf on the front'}}
           for ref in ['the left cabinet', 'the right cabinet']]
    plan = pm.combine(raw, 'twins', {'U1': 'add', 'U2': 'add'}, objects)
    g = build_twin_group(c, objects, plan, masks, split='train', scope_level='object',
                        group_id='slots', geometry=_geometry(masks), assigned_operation='add',
                        require_grounding=True)
    for m in g['members']:
        assert m['instruction_spec']['payload'] == 'a tray with a cup on top'
        assert m['unit']['instruction_spec'] == m['instruction_spec']
        assert m['instruction_noref'] == 'Add a tray with a cup on top on the selected region.'


def test_different_member_payloads_survive_build_validation_and_blind_protocol(tmp_path):
    g=group(tmp_path)
    assert g['twin_contract_version']==ADAPTED and g['instruction_noref'] is None and g['instruction_spec'] is None
    assert g['members'][0]['instruction_noref']=='Change the color of the selected region to blue.'
    assert g['members'][1]['instruction_noref']=='Change the color of the selected region to yellow.'
    assert g['members'][1]['unit']['instruction_spec']['attributes'][0]['value']=='yellow'
    validate_twin_record(g)
    check=blind_group_check(g);assert check['pass'] and not check['shared_noref_required'] and not check['shared_noref']
    g['members'][1]['instruction_spec']['operation']='remove'
    with pytest.raises(ValueError):validate_twin_record(g)


def test_independent_contract_does_not_disable_legacy_shared_gate(tmp_path):
    g=group(tmp_path);g['twin_contract_version']='blind-size-matched-2'
    assert not blind_probe_satisfied(g) and not blind_group_check(g)['pass']
    with pytest.raises(ValueError):validate_twin_record(g)


def test_one_mask_two_views_binding_no_extra_ids_or_images(tmp_path):
    c,o=data(tmp_path);pairs,masks,d=pm.views(c,o,tmp_path/'views')
    assert all(len(v)==2 for v in pairs) and pairs[0][0]==pairs[1][0] and pairs[0][1]!=pairs[1][1]
    assert np.array_equal(np.array(Image.open(pairs[0][0])),np.array(Image.open(c['image_path'])))
    meta=json.loads((Path(pairs[0][0]).parent/'views.json').read_text())['targets']
    assert [(x['unit_id'],x['object_id']) for x in meta]==[('U1','0'),('U2','1')]
    # PNG compression may differ; the selected pixels must be identical.
    for a,b in zip(masks,o):assert np.array_equal(np.asarray(Image.open(a)),np.asarray(Image.open(b['mask_path'])))


def test_context_trial_preserves_selected_pixels_holes_and_default_view(tmp_path):
    from PIL import ImageOps
    from scipy.ndimage import binary_dilation
    # Non-gray context, a hole, and a disconnected part expose target leakage.
    source = Image.new('RGB', (130, 90), (160, 70, 30))
    mask = np.zeros((90, 130), dtype=bool)
    mask[15:70, 20:85] = True
    mask[30:55, 40:65] = False
    mask[20:30, 105:115] = True
    box = (5, 5, 125, 85)
    tile = ImageOps.contain(source.crop(box), (900, 800), Image.Resampling.LANCZOS)
    selected = np.asarray(Image.fromarray(mask[5:85, 5:125].astype('uint8')).resize(
        tile.size, Image.Resampling.NEAREST)) > 0
    dark = np.asarray(pm.render_context_crop(source, mask, box))
    natural = np.asarray(pm.render_context_crop(source, mask, box, context_style='natural'))
    soft = np.asarray(pm.render_context_crop(source, mask, box, context_style='soft'))
    original = np.asarray(tile)
    ring = binary_dilation(selected, iterations=2) & ~selected
    assert np.array_equal(dark[selected], original[selected])
    assert np.array_equal(natural[selected], original[selected])
    assert np.array_equal(soft[selected], original[selected])
    assert np.array_equal(dark[ring], natural[ring])
    assert np.array_equal(soft[ring], natural[ring])
    assert np.all(natural[ring] == (0, 255, 100))
    context = ~selected & ~ring
    assert np.array_equal(natural[context], original[context])
    assert np.array_equal(soft[context], (original[context].astype('float32') * .5).astype('uint8'))
    assert np.any(dark[context] != natural[context])
    with pytest.raises(ValueError, match='context style'):
        pm.render_context_crop(source, mask, box, context_style='unknown')


def test_downsampled_scene_keeps_original_mask_coordinates_and_detail(tmp_path):
    source=tmp_path/'large.png';Image.new('RGB',(3000,2000),'white').save(source)
    mask=np.zeros((2000,3000),np.uint8);mask[500:600,1500:1600]=255
    mp=tmp_path/'mask.png';Image.fromarray(mask).save(mp)
    candidate={'image_path':str(source),'source_id':'large'}
    objects=[{'object_id':'x','mask_path':str(mp)}]
    images,masks,_=pm.views(candidate,objects,tmp_path/'views')
    assert Image.open(images[0][0]).size==(1536,1024)
    assert Image.open(images[0][1]).width <= 900 and Image.open(images[0][1]).height <= 800
    assert '"mask_box": [50.0, 25.0, 53.3, 30.0]' in pm.request_prompt('replace',images[0])
    assert np.array_equal(np.asarray(Image.open(masks[0])),mask)


def test_flattened_requests_resume_only_missing_mask_and_detect_prompt_change(tmp_path):
    class Planner:
        batch_size=1;calls=0
        def generate_multimage(self,jobs,schemas=None):
            assert len(jobs)==1 and len(jobs[0][0])==2
            self.calls+=1
            return [deepcopy(edits()[int('U2' in jobs[0][0][1])])]
    p=Planner();item=dict(gid='group',objects=[{'object_id':'0'},{'object_id':'1'}],selection={'operation':'attribute'},per_mask_views=[['src','U1'],['src','U2']])
    opts=dict(kind='twins',root=tmp_path)
    first=pm.plan_batch(p,[deepcopy(item)],**opts);assert p.calls==2 and first[0]['feasible']
    assert pm.plan_batch(p,[deepcopy(item)],**opts)==first and p.calls==2
    (tmp_path/'planning_attempts/group/U2.json').unlink()
    assert pm.plan_batch(p,[deepcopy(item)],**opts)==first and p.calls==3
    item['planning_source_hints']={'U1':'changed'}
    with pytest.raises(ValueError,match='checkpoint'):pm.plan_batch(p,[item],**opts)


def test_multi_rejection_and_format_failure_preserve_other_targets():
    raw=edits()+[{'reject':'Removing this support would leave the roof unsupported.'}]
    a=dict(U1='attribute',U2='attribute',U3='remove');o=[dict(object_id=str(i)) for i in range(3)]
    plan=pm.combine(raw,'multi',a,o);assert plan['feasible'] and [x['feasible'] for x in plan['targets']]==[True,True,False]
    raw[-1]={'reject':False};plan=pm.combine(raw,'multi',a,o)
    assert plan['feasible'] and plan['targets'][-1]['_format_error']
    raw=edits();raw[1]={'reject':'Target identity is ambiguous in the source.'}
    assert not pm.combine(raw,'twins',dict(U1='attribute',U2='attribute'),o[:2])['feasible']


def test_no_blind_call_and_audit_checks_each_own_edit(tmp_path):
    from sa1b_pipeline.blind_identifiability import run_blind_probes
    from sa1b_pipeline.group_audit import twin_audit_prompt,_apply_joint_twin_audit,JOINT_MEMBER_FIELDS
    g=group(tmp_path)
    class Never:
        batch_size=8
        def generate_multimage(self,*a,**k):raise AssertionError('no blind request for adapted twins')
    run_blind_probes([g],Never(),tmp_path/'blind')
    assert g['group_checks']['blind_disambiguation']['status']=='not_required' and blind_probe_satisfied(g)
    prompt=twin_audit_prompt(g);assert 'member_edits_valid' in prompt and 'shared_edit_valid' not in prompt
    for key in ['deterministic','cross_target_preservation','support_gap']:g['group_checks'][key]={'pass':True}
    audit=dict(members=[dict(unit_id=u['unit_id'],observed_entity='chair',ref_entity='chair',observed_change='color changed',checks={k:'pass' for k in JOINT_MEMBER_FIELDS},warnings=[],failure_reasons=[]) for u in g['members']],member_edits_valid='pass')
    _apply_joint_twin_audit(g,{'semantic_audit':audit});assert g['group_checks']['group_pass']
    audit['members'][1]['checks']['natural']='fail'
    _apply_joint_twin_audit(g,{'semantic_audit':audit});assert not g['group_checks']['group_pass']


def test_replacement_slot_names_new_content_and_site_possessive_normalizes():
    objects=[{'object_id':'a'},{'object_id':'b'}]
    raw=[{'ref':'the left chair','edit':{'replacement':'a folding stool'}},
         {'ref':'the right lamp','edit':{'replacement':'a pendant light'}}]
    plan=pm.combine(raw,'twins',{'U1':'replace','U2':'replace'},objects)
    assert plan['feasible'] and [m['payload'] for m in plan['members']]==['a folding stool','a pendant light']
    assert plan['_per_mask_responses']==raw
    add=[{'ref':'the left pot','edit':{'object':'a red ribbon','relation':'around','site':'its rim'}},
         {'ref':'the right pot','edit':{'object':'a blue ribbon','relation':'around','site':''}}]
    result=pm.combine(add,'twins',{'U1':'add','U2':'add'},objects)
    assert result['feasible'] and result['members'][0]['_minimal_instruction']=='Add a red ribbon around the rim of the left pot.'
    assert result['_per_mask_responses']==add
    raw[0]['edit']={'object':'a folding stool'}
    assert not pm.combine(raw,'twins',{'U1':'replace','U2':'replace'},objects)['feasible']


def test_geometry_is_bound_to_each_crop_and_cached_views_do_not_redraw(tmp_path, monkeypatch):
    c,o=data(tmp_path);pairs,masks,d=pm.views(c,o,tmp_path/'views')
    p1=pm.request_prompt('add',pairs[0]);p2=pm.request_prompt('add',pairs[1])
    assert '"mask_box": [6.2, 40.0, 18.8, 56.7]' in p1
    assert '"mask_box": [70.0, 40.0, 82.5, 56.7]' in p2
    assert 'Never copy coordinates into Ref' in p1
    def no_render(*args,**kwargs):raise AssertionError('cached crop rerendered')
    monkeypatch.setattr(pm.ImageOps,'contain',no_render)
    assert pm.views(c,o,tmp_path/'views')==(pairs,masks,d)
    metadata=Path(pairs[0][0]).parent/'views.json'
    raw=json.loads(metadata.read_text());raw['targets'][1]['unit_id']='U1';metadata.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='uniquely bound'):pm.request_prompt('add',pairs[0])


def test_new_addition_slot_preserves_payload_and_article_in_site():
    raw=[{'ref':'the tree beside the gate','edit':{'addition':'a bird','relation':'on','site':'a branch'}}]
    p=pm.combine(raw,'multi',{'U1':'add'},[{'object_id':'a'}])
    assert p['targets'][0]['feasible']
    assert p['targets'][0]['_minimal_instruction']=='Add a bird on a branch of the tree beside the gate.'
    assert pm.schema('add')['anyOf'][0]['properties']['edit']['required']==['addition','relation','site']


def test_remote_single_pixel_cannot_expand_crop_but_real_parts_survive():
    import numpy as np
    from sa1b_pipeline.per_mask_planning import clean_remote_speckles
    mask=np.zeros((300,600),bool);mask[100:130,50:60]=True;mask[100,550]=True
    cleaned,n=clean_remote_speckles(mask)
    assert n==1 and cleaned.sum()==300 and mask.sum()==301
    mask[101,550]=True;mask[102,550]=True
    assert clean_remote_speckles(mask)[1]==0  # More than two pixels is retained.
    small=np.zeros_like(mask);small[100:105,50:60]=True;small[100,550]=True
    assert clean_remote_speckles(small)[1]==0
    near=np.zeros_like(mask);near[100:130,50:60]=True;near[100,65]=True
    assert clean_remote_speckles(near)[1]==0


def test_declared_attached_anatomy_rejected_without_rejecting_accessories_or_depictions():
    from sa1b_pipeline.content_guard import incompatible_edit
    for ref,op in [('the nose and cheek of the man with sunglasses','remove'),
                   ('the forehead of the white-haired person','replace'),
                   ("the patch of skin on the back of the woman's hand",'replace'),
                   ("the area of bare skin on the man's cheek",'remove'),
                   ("the skin patch on the back of the woman's hand",'replace')]:
        assert incompatible_edit(ref,op)
        raw=[{'ref':ref,'edit':{} if op=='remove' else {'replacement':'a face mask'}}]
        row=pm.combine(raw,'multi',{'U1':op},[{'object_id':'1'}])['targets'][0]
        assert not row['feasible'] and not row.get('_format_error')
    assert incompatible_edit('the bare forearm and hand of the child in yellow','replace')
    assert incompatible_edit('the hands of the seated man','replace')
    assert incompatible_edit('the arms of the woman','remove')
    for ref,op in [('the nose ring worn by the woman','remove'),
                   ('the hat above the nose of the man','remove'),
                   ('the forehead of the statue of a man','replace'),
                   ('the prosthetic hand of the man','replace'),
                   ('the man in the foreground','remove'),
                   ('the nose of the aircraft','replace'),
                   ('the forehead of the person','attribute')]:
        assert incompatible_edit(ref,op) is None


def test_only_truncated_mask_retried_once_and_resume_reuses_both_attempts(tmp_path):
    class Planner:
        batch_size=8;model_id='test'
        def __init__(self):self.calls=[]
        def generate_multimage(self,jobs,*,schemas,token_limits=None):
            self.calls.append((len(jobs),token_limits))
            if token_limits is not None:
                return [edits()[0]]
            return [{'_generation_metadata':{'finish_reason':'length','output_tokens':1024}},edits()[1]]
    item=dict(gid='group',objects=[{'object_id':'0'},{'object_id':'1'}],
              selection={'operation':'attribute'},per_mask_views=[['src','U1'],['src','U2']])
    p=Planner();first=pm.plan_batch(p,[deepcopy(item)],kind='twins',root=tmp_path)
    assert p.calls==[(2,None),(1,[2048])]
    assert first[0]['feasible'] and first[0]['_planning_calls']==3
    assert pm.plan_batch(p,[deepcopy(item)],kind='twins',root=tmp_path)==first
    assert len(p.calls)==2


def test_excluded_view_never_modifies_selected_pixels():
    rng=np.random.default_rng(23)
    pixels=rng.integers(0,255,(180,240,3),dtype=np.uint8)
    selected=np.ones((180,240),bool);selected[30:160,45:200]=False
    result=np.asarray(pm.excluded_pixels_view(Image.fromarray(pixels),selected))
    assert np.array_equal(result[selected],pixels[selected])
    assert not np.array_equal(result[~selected],pixels[~selected])
    assert 'NOT the target' in pm.INTRO


def test_garment_extent_rejects_short_to_long_only_when_enabled():
    from sa1b_pipeline.content_guard import garment_extent
    # Labelled full-run cases: shorts/short skirts became knee-length shorts or legs cut at the old hem.
    assert garment_extent('the black skirt worn by the woman standing on the left', 'a pair of black tailored trousers')
    assert garment_extent('the blue plaid shorts worn by the cyclist in the red shirt', 'a pair of long dark denim jeans')
    assert garment_extent('the light blue skirt worn by the girl', 'a pair of black leggings')
    assert garment_extent('the denim shorts on the left', 'a long floral maxi dress')
    # Full run 4 X080: a knee-length skirt came back for "long", missed with one word allowed in between.
    assert garment_extent('the red skirt of the woman walking in the left foreground', 'a long black pleated skirt')
    # Same or shorter length, a long source garment, or the garment only as a locating landmark.
    assert garment_extent('the long dark skirt worn by the person', 'a pair of dark gray trousers') is None
    assert garment_extent('the light blue jeans of the rider', 'a pair of black leather riding pants') is None
    assert garment_extent('the blue shorts worn by the man', 'a pair of red swim shorts') is None
    assert garment_extent('the blue shorts worn by the man', 'a pair of cropped capri pants') is None
    assert garment_extent('the backpack of the man in shorts', 'a pair of trousers') is None
    raws = [{'ref': 'the blue shorts worn by the man', 'edit': {'replacement': 'a pair of long gray trousers'}},
            {'ref': 'the red cap on the boy', 'edit': {'replacement': 'a straw sun hat'}}]
    objects = [{'object_id': 'a'}, {'object_id': 'b'}]
    plan = pm.combine(raws, 'multi', {'U1': 'replace', 'U2': 'replace'}, objects)
    assert [t['feasible'] for t in plan['targets']] == [True, True]          # default contract unchanged
    plan = pm.combine(raws, 'multi', {'U1': 'replace', 'U2': 'replace'}, objects, options={'garment_extent': True})
    first, second = plan['targets']
    assert not first['feasible'] and first['reason'].startswith('garment_extent:') and second['feasible']
    assert plan['planning_options'] == {'garment_extent': True}
