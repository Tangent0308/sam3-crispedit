from copy import deepcopy
import pytest
from sa1b_pipeline.multi_semantics import operation_exclusion, validate_multi_semantics
from sa1b_pipeline.group_selection import multi_operation_allowed
from sa1b_pipeline.group_planning import multi_prompt
from sa1b_pipeline.group_audit import _enforce_multi_integrity


def test_operation_assignment_rejects_bad_pairs_without_dropping_editable_masks():
    for category, blocked, allowed in [('jeans', 'remove', 'attribute'),
                                      ('head', 'replace', 'add'),
                                      ('person,woman', 'attribute', 'add'),
                                      ('sleeve', 'remove', 'attribute')]:
        obj = {'category': category, 'bbox': [0, 0, 50, 50]}
        assert not multi_operation_allowed(obj, blocked, (100, 100))
        assert multi_operation_allowed(obj, allowed, (100, 100))


def test_real_parts_and_removable_accessories_are_not_anatomy():
    for category, operation in [('chair leg', 'attribute'), ('mannequin head', 'replace'),
                                ('armchair', 'replace'), ('glove', 'remove'),
                                ('hat', 'remove'), ('whole person', 'remove')]:
        assert not operation_exclusion(category, operation)
    with pytest.raises(ValueError, match='source extent'):
        validate_multi_semantics({'scope':'bare arm','operation':'attribute'})


def test_multi_prompt_has_per_id_fallible_hints_and_fixed_operations():
    text = multi_prompt({'5':'attribute','7':'add'},
                        [{'object_id':'5','category':'sleeve'}, {'object_id':'7','category':'shirt'}], ['U1','U2'])
    assert 'unit_id=U1, operation=attribute, source_label_hint="sleeve"' in text
    assert 'unit_id=U2, operation=add, source_label_hint="shirt"' in text
    assert 'fallible' in text and 'the selected region' in text


def test_audit_cannot_accept_its_own_body_destruction_evidence():
    original = {'decision':'pass','natural':True,'host_integrity':True,
                'evidence':"The boy's lower body is cut off at the waist and blends with the background."}
    fixed = _enforce_multi_integrity(original)
    assert fixed['decision'] == 'fail'
    assert 'acknowledged_serious_defect' in fixed['failure_reasons']
    assert original['decision'] == 'pass'
    joint = {'decision':'pass','natural':True,'units':[{'host_integrity':True,'issue':'No artifacts'}],
             'evidence': original['evidence']}
    assert _enforce_multi_integrity(joint,joint=True)['decision']=='fail'


def test_audit_preserves_realistic_glove_removal_and_minor_variation():
    original={'decision':'pass','natural':True,'host_integrity':True,
              'evidence':'The glove was removed. The hand is intact, not erased. Minor texture variation.'}
    assert _enforce_multi_integrity(original)['decision']=='pass'
    missing=deepcopy(original);missing.pop('host_integrity')
    assert _enforce_multi_integrity(missing)['decision']=='fail'
    assert _enforce_multi_integrity({'decision':'pass','units':[]},joint=True)['decision']=='fail'


def test_negated_unrelated_clause_does_not_hide_destroyed_body():
    audit={'decision':'pass','natural':True,'host_integrity':True,
           'evidence':'No floating objects; the lower body is cut off at the waist.'}
    assert _enforce_multi_integrity(audit)['decision']=='fail'


def test_plural_anatomy_and_clothing_scope_mismatch():
    for noun in ('legs', 'hands', 'feet', 'bodies'):
        assert operation_exclusion(noun, 'replace')
    assert not operation_exclusion('chair legs', 'replace')
    with pytest.raises(ValueError, match='garment mask'):
        validate_multi_semantics({'scope': 'legs', 'operation': 'add'}, 'jeans,trousers')
    validate_multi_semantics({'scope': 'trousers', 'operation': 'add'}, 'jeans,trousers')


def test_multi_decoder_schema_rejects_nested_target_layout():
    import jsonschema
    from sa1b_pipeline.multi_plan_schema import multi_plan_schema
    schema = multi_plan_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate({'grounding_version': 3, 'feasible': False, 'targets': [], 'generation_order': [], 'evidence': 'incompatible'}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'grounding_version': 3, 'feasible': True, 'targets': [{'targets': []}], 'generation_order': [], 'evidence': ''}, schema)


def test_nonhuman_arm_and_impossible_garment_site_regressions():
    assert not operation_exclusion('awning arm', 'remove')
    assert operation_exclusion('human arm', 'remove')
    with pytest.raises(ValueError, match='outside the lower garment'):
        validate_multi_semantics({'operation':'add', 'scope':'trousers',
                                  'placement':{'site':'the breast pocket'}})
    validate_multi_semantics({'operation':'add', 'scope':'trousers',
                             'placement':{'site':'the side pocket'}})
    validate_multi_semantics({'operation':'add', 'scope':'jacket',
                             'placement':{'site':'the breast pocket'}})


def test_multi_audit_schema_requires_integrity_and_step_evidence():
    import jsonschema
    from sa1b_pipeline.group_audit import multi_audit_schema
    for joint in (False, True):
        jsonschema.Draft202012Validator.check_schema(multi_audit_schema(joint=joint))
    minimal = {'decision':'pass', 'failure_reasons':[], 'evidence':'Target changed.',
               'operation_correct':True, 'target_done':True, 'non_target_preserved':True, 'natural':True}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(minimal, multi_audit_schema())
    jsonschema.validate({**minimal, 'host_integrity':True, 'source_state':'white chair', 'result_state':'blue chair', 'observed_change':'Color changed.', 'source_entity':'chair', 'observed_location':'chair surface', 'ref_matches_mask':True, 'location_correct':True, 'addition_is_new':True}, multi_audit_schema(contrast=True))


def test_grounding_label_modifiers_do_not_cause_false_rejection():
    from sa1b_pipeline.multi_semantics import same_scope_identity
    for a,b in [("white t-shirt","t-shirt"),("vertical post","post"),
                ("plaid shirt sleeve","sleeve"),("chair","orange plastic chair")]:
        assert same_scope_identity(a,b)
    for a,b in [("rear bumper of the car","car"),("person","shirt"),
                ("chair backrest","chair"),("left sleeve","right sleeve"),
                ("lower body","body"),("", "shirt")]:
        assert not same_scope_identity(a,b)


def test_opacity_decoder_blocks_malformed_values_before_generation():
    import jsonschema
    from sa1b_pipeline.multi_plan_schema import multi_plan_schema
    schema=multi_plan_schema()["properties"]["targets"]["items"]["properties"]["attributes"]["items"]
    jsonschema.validate({"property":"opacity","value":"opaque"},schema)
    jsonschema.validate({"property":"finish","value":"frosted"},schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"property":"opacity","value":"frosted"},schema)
