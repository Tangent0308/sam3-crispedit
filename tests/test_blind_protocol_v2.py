import copy

import numpy as np
import pytest
from PIL import Image

from sa1b_pipeline.instruction_protocol import make_spec, render, validate_noref, mask_contract
from sa1b_pipeline.source_regions import localize_add_site, resolve_source_region, blind_group_check
from sa1b_pipeline.training import build_training_rows


def test_source_identity_removed_but_literal_payload_values_and_results_survive():
    spec = make_spec('replace', 'the left upper sleeve', 'sleeve', 'a flared bell sleeve', target_part='lower section')
    assert render(spec, False) == 'Replace the selected region with a flared bell sleeve.'
    assert render(spec) == 'Replace the lower section of the left upper sleeve with a flared bell sleeve.'
    validate_noref(spec, render(spec, False))
    for leaked in ('left region', 'sleeve', 'lower section of the selected region'):
        with pytest.raises(ValueError):
            validate_noref(spec, render(spec, False).replace('region', leaked, 1))
    text = make_spec('attribute', 'the upper left sign', 'sign', 'LEFT FIRST', attribute_kind='text')
    assert render(text, False) == 'Change the text on the selected region to "LEFT FIRST".'
    validate_noref(text, render(text, False))
    trim = make_spec('remove', 'the trousers', 'trousers', remove_mode='lowerlegs')
    assert render(trim, False) == 'Remove the selected region to make knee-length shorts.'


def test_no_ref_qualifiers_separate_scene_protections_and_edit_requirements():
    spec = make_spec('remove', 'the left planter', 'planter', qualifier='preserving the underlying surface')
    assert render(spec, False).endswith(', preserving the underlying surface.')
    spec['qualifier'] = 'preserving the plant growing in it and the pots on either side'
    assert render(spec, False) == 'Remove the selected region.'
    assert 'preserving the plant' in render(spec)
    validate_noref(spec, render(spec, False))
    spec['qualifier'] = 'near the left window'
    with pytest.raises(ValueError, match='qualifier'):
        validate_noref(spec, render(spec, False))
    spec['qualifier'] = 'preserving transparency'
    assert render(spec, False).endswith(', preserving transparency.')
    with pytest.raises(ValueError, match='qualifier'):
        validate_noref(spec, render(spec, False))


def test_token_attaches_to_actual_part_not_host_or_payload_literal():
    spec = make_spec('add', 'the left suitcase', 'suitcase', 'a label reading the selected region',
                     placement={'relation':'attached to','site':'handle'})
    token = '<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>'
    row = dict(operation='add', instruction_spec=spec, instruction=render(spec),
               instruction_noref=render(spec, False), source_image='s', edited_image='t', blind_eligible=True)
    rows = build_training_rows(row, token)
    assert rows[1]['prompt'].endswith('the handle of the left suitcase '+token+'.')
    assert rows[2]['prompt'] == 'Add a label reading the selected region attached to the selected region '+token+'.'
    row['blind_eligible'] = False
    with pytest.raises(ValueError, match='unresolved'):
        build_training_rows(row, token)


def _case(tmp_path):
    source = np.full((160,200,3),100,dtype=np.uint8)
    target = source.copy(); target[40:55,70:100] = 220
    host = np.zeros((160,200),dtype=bool);host[20:140,20:180] = True
    paths = {}
    for name, array in [('source',source),('target',target),('host',host.astype(np.uint8)*255)]:
        path = tmp_path/(name+'.png');Image.fromarray(array).save(path);paths[name]=str(path)
    spec = make_spec('add','the left cart','cart','a label',placement={'relation':'on','site':'front panel'})
    member = dict(unit_id='U1', mask=paths['host'], bbox=[20,20,180,140],point=[99,79],
                  area_fraction=float(host.mean()),instruction_spec=spec,instruction=render(spec),
                  instruction_noref=render(spec,False),unit={})
    return source,target,host,paths,member


def test_add_site_is_host_patch_not_output_silhouette_and_preserves_history(tmp_path):
    a,b,host,paths,member = _case(tmp_path)
    original = {k:open(v,'rb').read() for k,v in paths.items()}
    assert resolve_source_region(member,paths['source'],paths['target'],tmp_path/'resolved')
    site = np.asarray(Image.open(member['mask']))>0
    assert (site & ~host).sum()==0
    assert site.sum() < host.sum()
    assert site.sum() > np.any(a!=b,axis=2).sum()
    assert member['mask_v1']==paths['host'] and member['bbox_v1']==[20,20,180,140]
    ys,xs=np.where(site)
    assert member['bbox']==[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    assert 50 <= member['bbox'][0] <= 55 and 110 <= member['bbox'][2] <= 120
    x,y=member['point'];assert site[y,x]
    assert member['mask_semantics']['source_role']=='anchor'
    assert member['source_region_resolution']['dilation_radius']==16
    assert all(open(paths[k],'rb').read()==v for k,v in original.items())


def test_add_missing_contact_fails_closed_without_fabricated_site(tmp_path):
    a,b,host,paths,member = _case(tmp_path)
    b=a.copy();b[145:155,185:195]=255
    Image.fromarray(b).save(paths['target'])
    assert not resolve_source_region(member,paths['source'],paths['target'],tmp_path/'resolved')
    assert member['mask']==paths['host']
    assert member['blind_ineligible_reason']=='part_mask_missing'
    assert not member['blind_eligible']


def test_contents_mask_union_and_part_selector(tmp_path):
    a,b,host,paths,member = _case(tmp_path)
    part=np.zeros(host.shape,dtype=np.uint8);part[10:30,40:60]=255
    path=tmp_path/'contents.png';Image.fromarray(part).save(path)
    spec=make_spec('remove','the basket','basket',remove_mode='contents')
    member.update(instruction_spec=spec,instruction_noref=render(spec,False),unit={'execution_mask_path':str(path),
                  'mask_path':paths['host'],'bbox':member['bbox'],'point':member['point']})
    assert resolve_source_region(member,paths['source'],paths['target'],tmp_path/'resolved')
    mask=np.asarray(Image.open(member['mask']))>0
    assert np.array_equal(mask,host|(part>0))
    assert member['mask_semantics']['source_role']=='target'


def test_multi_allows_duplicate_sentences_twins_still_require_scope_and_sentence():
    spec=make_spec('remove','the left bag','bag')
    m=dict(instruction_spec=spec,instruction_noref=render(spec,False),area_fraction=.1,blind_eligible=True)
    group={'units':[copy.deepcopy(m),copy.deepcopy(m)]}
    assert blind_group_check(group)['pass']
    group={'twin_id':'tw','members':[copy.deepcopy(m),copy.deepcopy(m)]}
    assert blind_group_check(group)['pass']
    group['members'][1]['instruction_spec']['scope']='hat'
    assert not blind_group_check(group)['pass']
    group['members'][1]['instruction_spec']['scope']='bag'
    group['members'][1]['instruction_noref']='Remove the selected bag.'
    assert not blind_group_check(group)['pass']


def test_site_noise_filter_does_not_expand_to_entire_host(tmp_path):
    a,b,host,paths,member = _case(tmp_path)
    # Weak reconstruction noise throughout the host must not change the site.
    b[host] = np.clip(b[host].astype(int)+15,0,255).astype(np.uint8)
    site,info=localize_add_site(a,b,host)
    assert site is not None and site.sum()<host.sum()*.4
    assert info['dilation_radius']==16


def test_semantic_site_constraint_clips_occluded_host_and_fails_closed(tmp_path):
    a,b,host,paths,member=_case(tmp_path)
    seat=np.zeros_like(host);seat[48:70,60:110]=True
    def resolver(source,host,spec,difference):
        return seat & difference, {'method':'test_verified_site'}
    assert resolve_source_region(member,paths['source'],paths['target'],tmp_path/'resolved',site_resolver=resolver)
    assert np.array_equal(np.asarray(Image.open(member['mask']))>0,seat)
    assert member['source_region_resolution']['method']=='sam3_site_constrained_difference'
    other=copy.deepcopy(member)
    def unresolved(*args):return None, {'reason':'no_site_found'}
    assert not resolve_source_region(other,paths['source'],paths['target'],tmp_path/'failed',site_resolver=unresolved)
    assert other['blind_ineligible_reason']=='part_mask_missing'
