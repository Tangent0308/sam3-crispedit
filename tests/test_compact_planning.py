"""Candidate schema conversion must preserve scope and not invent repairs."""
from sa1b_pipeline.planning_candidates import expand_compact
from sa1b_pipeline.per_mask import validate_single_plan
from sa1b_pipeline.instructions import build_unit
import pytest


def row(**kw):
 r=dict(ref_core='the orange petal',distinguish='immediately right of the tallest upright petal on the foreground flower',rival='the tallest upright petal',scope='petal',kind='part',source_color='orange',op='attribute',change={'property':'color','value':'yellow'})
 r.update(kw);return r


def test_discriminator_is_in_both_structured_reference_and_rendered_ref():
 raw=expand_compact(row(),'7')
 u=build_unit(raw,{'object_id':'7','category':'petal'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert row()['distinguish'] in u.instruction_ref
 assert 'immediately' not in u.instruction_noref
 assert u.instruction_noref=='Change the color of the selected petal to yellow.'


def test_anatomy_removal_is_not_made_acceptable_by_conversion():
 raw=expand_compact(row(kind='anatomy',scope='forearm',op='remove',change={}),'7')
 with pytest.raises(ValueError,match='unnatural_removal'):
  validate_single_plan(raw,'7',['remove'])


def test_color_only_replacement_without_design_cannot_pass():
 raw=expand_compact(row(op='replace',change={'replacement':'a yellow petal'}),'7')
 with pytest.raises(ValueError,match='visible_design_change'):
  validate_single_plan(raw,'7',['replace'])


def test_no_edit_and_missing_reference_remain_unsuccessful():
 assert expand_compact(row(op='no_edit',change={'reason':'cannot distinguish this instance'}),'7')['feasible'] is False
 assert expand_compact(row(distinguish=None),'7').get('_parse_error')


def test_printed_motif_remove_does_not_invent_substrate_material():
 raw=expand_compact(row(kind='printed_motif',scope='printed petal motif',op='remove',change={}),'7')
 assert raw['qualifier']=='preserving the underlying surface'
 assert raw['payload']=='' and raw['target_part']==''


def test_complete_reference_is_preserved_without_duplicate_distinction():
 reference='the leaf immediately right of the second shirt button'
 raw=expand_compact(row(full_ref=reference,ref_core=None,distinguish=None),'7')
 validate_single_plan(raw,'7',['attribute'])
 unit=build_unit(raw,{'object_id':'7','category':'leaf'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert unit.instruction_ref==f'Change the color of {reference} to yellow.'
 assert 'second' not in unit.instruction_noref


@pytest.mark.parametrize('site,expected',[
 ('forearm',''),('the forearm',''),('center of the forearm','center'),
 ('skin of the forearm','skin'),('wrist','wrist'),
 ('center of the wrist','center of the wrist'),
])
def test_add_site_literal_host_dedup_preserves_distinct_parts(site,expected):
 original=row(ref_core='the bare forearm',scope='forearm',kind='anatomy',
  op='add',change={'item':'a small tattoo','relation':'on','site':site})
 raw=expand_compact(original,'7')
 assert raw['placement']['site']==expected
 assert raw['compact_output']['change']['site']==site
 assert raw['ref_phrase'].startswith('the bare forearm ')


def test_replace_literal_head_dedup_does_not_rewrite_other_designs():
 for design,expected in [
  ('a window with two panes','a window with two panes'),
  ('two panes','a window with two panes'),
  ('an arched window with two panes','a window with an arched window with two panes'),
 ]:
  raw=expand_compact(row(op='replace',change={'replacement':'a window','design':design}),'7')
  assert raw['payload']==expected
  assert raw['compact_output']['change']['design']==design


def test_add_host_pronoun_is_grammatical_cleanup_in_ref_and_noref():
 original=row(op='add',change={'item':'a small water droplet','relation':'on','site':'its upper surface'})
 raw=expand_compact(original,'7')
 unit=build_unit(raw,{'object_id':'7','category':'petal'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert raw['compact_output']['change']['site']=='its upper surface'
 assert raw['placement']['site']=='upper surface'
 assert unit.instruction_ref.startswith('Add a small water droplet on the upper surface of the orange petal ')
 assert unit.instruction_noref=='Add a small water droplet on the upper surface of the selected petal.'
 # Do not alter references to another object or a meaningful subpart.
 original['change']['site']='surface beside its stem'
 assert expand_compact(original,'7')['placement']['site']=='surface beside its stem'


def test_unique_target_needs_no_forced_extra_locator():
 import jsonschema
 from sa1b_pipeline.planning_schema import typed_compact_schema
 original=row(ref_core='the blue shoebox',distinguish='',scope='shoebox')
 jsonschema.validate(original,typed_compact_schema())
 raw=expand_compact(original,'7')
 validate_single_plan(raw,'7',['attribute'])
 unit=build_unit(raw,{'object_id':'7','category':'shoebox'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert unit.instruction_ref=='Change the color of the blue shoebox to yellow.'
 assert unit.instruction_noref=='Change the color of the selected shoebox to yellow.'
 assert raw['compact_output']==original


def test_adjacent_add_cannot_masquerade_as_direct_contact():
 import jsonschema
 from sa1b_pipeline.planning_schema import typed_compact_schema
 original=row(ref_core='the chair',distinguish='',scope='chair',kind='object',op='add',
              change={'item':'a small basket','relation':'beside','site':''})
 jsonschema.validate(original,typed_compact_schema())
 raw=expand_compact(original,'7')
 assert raw['placement']['anchor_mode']=='reference'
 with pytest.raises(ValueError,match='reference add requires'):
  build_unit(raw,{'object_id':'7','category':'chair'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 obj={'object_id':'7','category':'chair','metadata':{'execution_mask_path':'/resolved.png'}}
 unit=build_unit(raw,obj,'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert unit.instruction_noref=='Add a small basket beside the selected chair.'
 assert unit.execution_mask_path=='/resolved.png'


def test_landmark_in_reference_does_not_turn_direct_add_into_reference_add():
 raw=expand_compact(row(ref_core='the bag',distinguish='beside the chair',scope='bag',
      kind='object',op='add',change={'item':'a luggage tag','relation':'attached to','site':'handle'}),'7')
 unit=build_unit(raw,{'object_id':'7','category':'bag'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert unit.instruction_spec['placement']['anchor_mode']=='support'
 assert unit.instruction_ref=='Add a luggage tag attached to the handle of the bag beside the chair.'
 assert unit.instruction_noref=='Add a luggage tag attached to the handle of the selected bag.'


def test_single_phrase_replacement_keeps_public_result_and_internal_evidence_separate():
 original=row(full_ref='the lowest shirt button',scope='button',op='replace',
              change={'replacement':'a square four-hole button',
                      'form_change':'round two-hole to square four-hole'})
 raw=expand_compact(original,'7')
 validate_single_plan(raw,'7',['replace'])
 unit=build_unit(raw,{'object_id':'7','category':'button'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert unit.instruction_ref=='Replace the lowest shirt button with a square four-hole button.'
 assert unit.instruction_noref=='Replace the selected button with a square four-hole button.'
 assert raw['compact_output']==original
 assert 'round two-hole' not in unit.instruction_ref


def test_empty_form_evidence_does_not_certify_a_replacement():
 raw=expand_compact(row(full_ref='the shirt button',op='replace',
     change={'replacement':'a blue button','form_change':''}),'7')
 with pytest.raises(ValueError,match='visible_design_change'):
  validate_single_plan(raw,'7',['replace'])


def test_single_phrase_does_not_drop_essential_landmark_or_literal_text():
 raw=expand_compact(row(full_ref='the label on the rightmost bottle in the middle row',
     scope='label',op='attribute',change={'property':'text','value':'A, B (2026)'}),'7')
 unit=build_unit(raw,{'object_id':'7','category':'label'},'/mask.png',{'bbox':[1,2,3,4],'point':[2,3]},[])
 assert raw['ref_phrase'] in unit.instruction_ref
 assert 'middle row' not in unit.instruction_noref
 assert 'A, B (2026)' in unit.instruction_ref and 'A, B (2026)' in unit.instruction_noref
