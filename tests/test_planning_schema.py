import jsonschema
import pytest
from sa1b_pipeline.planning_schema import typed_compact_schema


def test_optional_skip_reason_cap_does_not_truncate_reference_or_payload():
    schema = typed_compact_schema(['remove', 'attribute'], single_phrase=True, skip_reason_max_chars=160)
    attribute = next(branch for branch in schema['anyOf'] if branch['properties']['op']['const'] == 'attribute')
    props = attribute['properties']
    assert props['skipped']['properties']['remove']['maxLength'] == 160
    assert 'maxLength' not in props['full_ref']
    assert 'maxLength' not in props['change']['properties']['value']
    value = {'kind': 'surface', 'full_ref': 'the grass infield', 'rival': None, 'scope': 'grass',
             'source_color': 'green', 'op': 'attribute', 'skipped': {'remove': 'An integral infield surface.'},
             'change': {'property': 'color', 'value': 'golden brown'}}
    jsonschema.validate(value, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**value, 'skipped': {'remove': 'x'*161}}, schema)


def proposal(kind, op, change):
    return dict(kind=kind,ref_core='the logo',distinguish='left of the handle',
                rival=None,scope='logo',source_color=None,op=op,change=change)


@pytest.mark.parametrize('contrast_first', [False, True])
def test_typed_schema_allows_pattern_edit_and_blocks_printed_replacement(contrast_first):
    schema=typed_compact_schema(contrast_first=contrast_first)
    jsonschema.validate(proposal('printed_motif','attribute',{'property':'pattern','value':'a star'}),schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(proposal('printed_motif','replace',{'replacement':'a star','design':'five points'}),schema)


@pytest.mark.parametrize('contrast_first', [False, True])
def test_typed_schema_does_not_allow_anatomy_removal_or_wrong_change_fields(contrast_first):
    schema=typed_compact_schema(contrast_first=contrast_first)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(proposal('anatomy','remove',{}),schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(proposal('part','attribute',{'replacement':'a cap','design':'flat'}),schema)
    jsonschema.validate(proposal('anatomy','add',{'item':'a small tattoo','relation':'on','site':''}),schema)


def test_choice_schema_requires_exact_skipped_prefix_and_does_not_force_preferred_type():
    schema=typed_compact_schema(['remove','add','attribute'])
    add=proposal('anatomy','add',{'item':'a small tattoo','relation':'on','site':''})
    add['skipped']={'remove':'Removing attached skin is not a natural edit.'}
    jsonschema.validate(add,schema)
    for wrong in [{},{'add':'irrelevant'},{'remove':'reason','attribute':'not skipped'}]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**add,'skipped':wrong},schema)
    remove=proposal('object','remove',{})
    jsonschema.validate({**remove,'skipped':{}},schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**proposal('object','replace',{'replacement':'a box','design':'square'}),'skipped':{}},schema)


def test_choice_schema_no_edit_accounts_for_all_allowed_types():
    schema=typed_compact_schema(['add','remove'])
    no_edit=proposal('anatomy','no_edit',{'reason':'No natural instruction for this ambiguous target.'})
    no_edit['skipped']={'add':'Reference remains ambiguous.','remove':'Attached anatomy.'}
    jsonschema.validate(no_edit,schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**no_edit,'skipped':{'remove':'Attached anatomy.'}},schema)


@pytest.mark.parametrize('contrast_first', [False, True])
def test_single_phrase_schema_uses_complete_reference_and_public_replacement(contrast_first):
 schema=typed_compact_schema(['replace','attribute'],single_phrase=True,contrast_first=contrast_first)
 p=proposal('part','replace',{'replacement':'a square four-hole button','form_change':'round to square'})
 del p['ref_core'],p['distinguish']
 p.update(full_ref='the lowest shirt button',scope='button',skipped={})
 jsonschema.validate(p,schema)
 # No second location slot or legacy design slot is available to concatenate.
 for wrong in [{**p,'distinguish':'above the belt'},
               {**p,'change':{'replacement':'a button','design':'square'}},
               {**p,'full_ref':''},
               {**p,'change':{'replacement':'a square button','form_change':''}}]:
  with pytest.raises(jsonschema.ValidationError):
   jsonschema.validate(wrong,schema)
 with pytest.raises(jsonschema.ValidationError):
  jsonschema.validate({**p,'kind':'anatomy'},schema)
