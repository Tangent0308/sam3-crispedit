from copy import deepcopy
import jsonschema
import pytest
from sa1b_pipeline import content_planning as c
from sa1b_pipeline.instruction_protocol import render


@pytest.mark.parametrize('value', [
    'change the red side panel to blue',
    'Please recolor its front panel blue',
    'make it glossy',
    'replace the top with wood',
])
def test_attribute_command_isolated_without_changing_neighbor_or_input(value):
    raw = {'targets': [
        {'unit_id': 'U1', 'ref': 'the left package',
         'edit': {'property': 'color', 'value': value}},
        {'unit_id': 'U2', 'ref': 'the right package',
         'edit': {'property': 'color', 'value': 'blue'}},
    ]}
    original = deepcopy(raw)
    result = c.decode(raw, 'multi', {'U1': 'attribute', 'U2': 'attribute'})
    failed, kept = result['targets']
    assert not failed['feasible'] and failed['_format_error']
    assert 'attribute_value_is_instruction' in failed['reason']
    assert kept['feasible']
    assert kept['_minimal_instruction'] == 'Change the color of the right package to blue.'
    assert raw == original


@pytest.mark.parametrize('property_name,value', [
    ('text', 'Change the world'),
    ('text', 'Make it happen'),
    ('text', '欢迎'),
    ('color', 'warm orange'),
    ('pattern', 'a blue and white floral pattern'),
    ('texture', 'smooth with fine grain'),
])
def test_attribute_guard_preserves_values_and_literal_text(property_name, value):
    from sa1b_pipeline.content_guard import attribute_value
    assert attribute_value(property_name, value) == value


def test_nested_site_never_becomes_blind_payload():
    # Regression: the old render -> greedy parse path silently absorbed most
    # of this site into payload, leaking the source person's identity.
    raw = {'targets': [{'unit_id': 'U1',
                        'ref': 'the seated person behind the menu board',
                        'edit': {'object': 'a small black shoulder bag with a thin strap',
                                 'relation': 'on',
                                 'site': "the seated person's dark-clothed torso at the cafe"}}]}
    t = c.decode(raw, 'multi', {'U1': 'add'})['targets'][0]
    assert t['feasible']
    assert t['payload'] == raw['targets'][0]['edit']['object']
    assert t['placement']['site'] == 'dark-clothed torso at the cafe'
    from sa1b_pipeline.instruction_protocol import make_spec
    spec = make_spec('add', t['ref_phrase'], t['entity'], t['payload'], placement=t['placement'])
    assert render(spec, False) == 'Add a small black shoulder bag with a thin strap on the selected region.'
    # Passing this test is structural integrity, not a claim that this site or
    # reference is visually correct.


def test_literal_payload_and_relation_survive_nested_site_for_shared_twins():
    raw = {'members': [{'unit_id': 'U1', 'ref': 'the left cabinet'},
                       {'unit_id': 'U2', 'ref': 'the right cabinet'}],
           'edit': {'object': 'a tray with a red bowl on top', 'relation': 'resting on',
                    'site': 'the shelf above the open drawer'}}
    result = c.decode(raw, 'twins', {'U1': 'add', 'U2': 'add'})
    assert result['feasible']
    assert result['payload'] == raw['edit']['object']
    assert all(t['payload'] == raw['edit']['object'] and
               t['placement']['relation'] == 'resting on' and
               t['placement']['site'] == raw['edit']['site'] for t in result['members'])


def test_repeated_site_host_is_removed_only_when_literal():
    from sa1b_pipeline.content_guard import site
    ref = 'the blue band on the animal-print onesie'
    assert site('the blue band', ref=ref) == ''
    assert site('the upper edge of the blue band', ref=ref) == 'the upper edge'
    assert site("the blue band's front face", ref=ref) == 'front face'
    assert site('a branch', ref='the tree beside the gate') == 'a branch'
    assert site("the woman's shoulder", ref='the handbag') == "the woman's shoulder"


def test_structured_binding_cannot_swap_operation_or_target():
    from sa1b_pipeline.minimal_planning import decode
    from sa1b_pipeline.instruction_protocol import make_spec
    raw = {'targets': [{'unit_id': 'U1', 'ref': 'the left chair',
                        'instruction': 'Remove the left chair.'}]}
    spec = make_spec('remove', 'the right chair', 'chair')
    out = decode(raw, 'multi', {'U1': 'remove'}, content_specs={'U1': spec})
    assert not out['targets'][0]['feasible']
    assert 'different target' in out['targets'][0]['reason']


def test_structured_add_still_requires_an_explicit_item_count():
    raw = {'targets': [{'unit_id': 'U1', 'ref': 'the chair',
                        'edit': {'object': 'red cushion', 'relation': 'on', 'site': 'seat'}}]}
    t = c.decode(raw, 'multi', {'U1': 'add'})['targets'][0]
    assert not t['feasible'] and 'explicit new content/count' in t['reason']


@pytest.mark.parametrize('operation,edit,expected', [
    ('remove', {}, 'Remove the selected region.'),
    ('replace', {'object': 'a folding stool'}, 'Replace the selected region with a folding stool.'),
    ('attribute', {'property': 'color', 'value': 'blue'}, 'Change the color of the selected region to blue.'),
    ('attribute', {'property': 'text', 'value': 'Open'}, 'Change the text on the selected region to "Open".'),
    ('add', {'object': 'a cushion', 'relation': 'on', 'site': 'the seat'}, 'Add a cushion on the selected region.'),
])
def test_content_roundtrip_and_shared_blind(operation, edit, expected):
    from sa1b_pipeline.minimal_planning import parse_instruction
    assignments = {'U1': operation, 'U2': operation}
    raw = dict(members=[dict(unit_id='U1', ref='the left chair'), dict(unit_id='U2', ref='the right stool')], edit=edit)
    jsonschema.validate(raw, c.schema('twins', assignments))
    result = c.decode(raw, 'twins', assignments)
    assert result['feasible'] and result['_content_response'] == raw
    for member in result['members']:
        spec = parse_instruction(member['ref_phrase'], member['_minimal_instruction'], operation)
        assert render(spec, False) == expected
    normalized = c.normalize(result, 'twins', assignments, [{'object_id': 'a'}, {'object_id': 'b'}])
    assert normalized['planning_contract'] == c.VERSION


def test_multi_equal_blind_sentences_are_allowed():
    a = {'U1': 'remove', 'U2': 'remove'}
    raw = {'targets': [{'unit_id': 'U1', 'ref': 'the left chair', 'edit': {}},
                       {'unit_id': 'U2', 'ref': 'the right lamp', 'edit': {}}]}
    out = c.decode(raw, 'multi', a)
    assert all(t['feasible'] for t in out['targets'])
    assert out['_instruction_rendered_from_slots'] is True


def test_safe_normalization_keeps_refs_and_shared_blind_without_mutating_raw():
    a = {'U1': 'add', 'U2': 'add'}
    raw = {'members': [{'unit_id': 'U1', 'ref': 'the  left chair '},
                       {'unit_id': 'U2', 'ref': 'the right stool'}],
           'edit': {'object': 'a cushion', 'relation': 'on', 'site': 'the seat of'}}
    before = deepcopy(raw)
    out = c.decode(raw, 'twins', a)
    assert out['feasible'] and raw == before and out['_content_response'] == before
    assert out['members'][0]['ref_phrase'] == 'the left chair'
    assert out['members'][0]['_minimal_instruction'] == 'Add a cushion on the seat of the left chair.'
    assert out['members'][1]['_minimal_instruction'] == 'Add a cushion on the seat of the right stool.'


@pytest.mark.parametrize('ref', ['the poster to the left of', 'the window right of a', '...', 'the first such un'])
def test_incomplete_ref_is_not_repaired_by_guessing(ref):
    raw = {'members': [{'unit_id': 'U1', 'ref': ref}, {'unit_id': 'U2', 'ref': 'the other chair'}], 'edit': {}}
    out = c.decode(raw, 'twins', {'U1': 'remove', 'U2': 'remove'})
    assert out['_minimal_error'] and not out['feasible']
    assert 'unfinished' in out['reason']


def test_valid_quoted_ref_literal_text_and_dependent_noun_are_not_rejected():
    raw = {'targets': [{'unit_id': 'U1', 'ref': 'the sign reading "on"',
                        'edit': {'property': 'text', 'value': 'Come on'}},
                       {'unit_id': 'U2', 'ref': 'the chair with the cushion on top', 'edit': {}}]}
    out = c.decode(raw, 'multi', {'U1': 'attribute', 'U2': 'remove'})
    assert all(t['feasible'] for t in out['targets'])
    assert out['targets'][0]['attributes'][0]['value'] == 'Come on'
    from sa1b_pipeline.content_guard import reference
    assert reference('the window below') == 'the window below'
    assert reference('the chair the cat sits on') == 'the chair the cat sits on'


def test_uninformative_refusal_is_format_error_and_does_not_erase_multi_neighbor():
    raw = {'targets': [{'unit_id': 'U1', 'reject': 'false'},
                       {'unit_id': 'U2', 'ref': 'the chair', 'edit': {}}]}
    out = c.decode(raw, 'multi', {'U1': 'remove', 'U2': 'remove'})
    assert out['targets'][0]['_format_error'] and out['targets'][1]['feasible']
    out = c.decode({'reject': 'false'}, 'twins', {'U1': 'remove', 'U2': 'remove'})
    assert 'concrete' in out['_minimal_error'] and not out['feasible']


def test_worn_by_ref_checks_garment_not_wearer():
    from sa1b_pipeline.minimal_planning import validate_edit, parse_instruction
    ref = 'the straw hat worn by the woman in the beige dress'
    spec = parse_instruction(ref, f'Change the color of {ref} to red.', 'attribute')
    validate_edit({**spec, 'ref_phrase': ref})


def test_unquoted_language_leak_and_duplicate_twin_refs_are_rejected():
    from sa1b_pipeline.content_guard import reference
    assert reference('the sign reading "欢迎"') == 'the sign reading "欢迎"'
    assert reference('the café sign,') == 'the café sign'
    with pytest.raises(ValueError, match='non-English'):
        reference('the left w椅')
    with pytest.raises(ValueError, match='bracketed'):
        reference('the left chair (beside the table')
    raw = {'members': [{'unit_id': uid, 'ref': 'the left chair'} for uid in ['U1', 'U2']], 'edit': {}}
    out = c.decode(raw, 'twins', {'U1': 'remove', 'U2': 'remove'})
    assert 'identical refs' in out['_minimal_error'] and not out['feasible']


def test_invalid_refs_and_truncation_still_fail_closed():
    a = {'U1': 'remove', 'U2': 'remove'}
    raw = {'members': [{'unit_id': 'U1', 'ref': 'the U1 chair'}, {'unit_id': 'U2', 'ref': 'the lamp'}], 'edit': {}}
    assert not c.decode(raw, 'twins', a)['feasible']
    raw['_generation_metadata'] = {'finish_reason': 'length'}
    assert c.decode(raw, 'twins', a).get('_minimal_error')


def test_content_retry_preserves_good_targets_and_resumes(tmp_path):
    from sa1b_pipeline.lean_retry import plan_batch
    a = {'U1': 'remove', 'U2': 'remove'}
    first = {'targets': [{'unit_id': 'U1', 'ref': 'the left chair', 'edit': {}}, {'unit_id': 'U2', 'reject': 'would sever the arm'}]}
    second = {'targets': [{'unit_id': 'U2', 'ref': 'the right sleeve', 'edit': {'property': 'color', 'value': 'blue'}}]}
    class Planner:
        calls = 0
        def generate_multimage(self, jobs, schemas=None):
            self.calls += 1
            assert 'Supply edit CONTENT' in jobs[0][1]
            value = deepcopy(first if self.calls == 1 else second)
            jsonschema.validate(value, schemas[0])
            return [value]
    planner = Planner()
    item = dict(gid='content', views=[], objects=[{'object_id': 'a'}, {'object_id': 'b'}],
                selection={'operations': ['remove', 'remove']}, prompt=c.prompt('multi', a))
    opts = dict(kind='multi', root=tmp_path, retry=1, minimal=True, content=True,
                accepted_counts={'add': 20, 'replace': 20, 'attribute': 0})
    out = plan_batch(planner, [deepcopy(item)], **opts)[0]
    assert out['targets'][0]['_minimal_instruction'] == 'Remove the left chair.'
    assert out['targets'][1]['operation'] == 'attribute'
    assert out['planning_contract'] == c.VERSION
    assert plan_batch(planner, [deepcopy(item)], **opts)[0] == out and planner.calls == 2


def test_multi_invalid_value_keeps_neighbors_without_operation_retry():
    a = {'U1': 'attribute', 'U2': 'remove', 'U3': 'add'}
    raw = {'targets': [
        {'unit_id': 'U1', 'ref': 'the tree shadow', 'edit': {'property': 'opacity', 'value': 'lower'}},
        {'unit_id': 'U2', 'ref': 'the upper window', 'edit': {}},
        {'unit_id': 'U3', 'ref': 'the left window', 'edit': {'object': 'a flower box', 'relation': 'attached to', 'site': 'sill'}},
    ]}
    result = c.decode(raw, 'multi', a)
    assert '_minimal_error' not in result
    bad, *good = result['targets']
    assert not bad['feasible'] and bad['_format_error']
    assert 'opacity value' in bad['reason']
    assert all(t['feasible'] for t in good)
    assert result['_content_response'] == raw
    c.validate_response(result, 'multi', a)


def test_category_hints_bind_display_order_not_source_id_and_ignore_prose():
    objects = [{'object_id': 188, 'category': ' shirt ', 'description': 'hair'},
               {'object_id': 81, 'category': 'hat', 'description': 'wrong host'},
               {'object_id': 0, 'category': None}, {'object_id': 50, 'category': 'object'}]
    hints = c.category_hints(objects)
    assert hints == {'U1': 'shirt', 'U2': 'hat'}
    assignments = {f'U{i}': 'remove' for i in range(1, 5)}
    base = c.prompt('multi', assignments, view_style='clean-contours')
    assert base == c.prompt('multi', assignments, view_style='clean-contours', source_hints={})
    prompt = c.prompt('multi', assignments, view_style='clean-contours', source_hints=hints)
    assert prompt.startswith(base) and 'hair' not in prompt and 'wrong host' not in prompt
    assert 'may be wrong' in prompt and 'actual mask extent' in prompt
    with pytest.raises(ValueError, match='duplicate'):
        c.category_hints([{'object_id': 1}, {'object_id': '1'}])
    with pytest.raises(ValueError, match='unassigned'):
        c.prompt('multi', assignments, source_hints={'U5': 'shirt'})


def test_category_hint_retry_binding_and_checkpoint_reject_changed_hints(tmp_path):
    from sa1b_pipeline.lean_retry import plan_batch
    a = {'U1': 'remove', 'U2': 'remove'}
    hints = c.category_hints([{'object_id': 50, 'category': 'chair'}, {'object_id': 4, 'category': 'sleeve'}])
    first = {'targets': [{'unit_id': 'U1', 'ref': 'the left chair', 'edit': {}}, {'unit_id': 'U2', 'reject': 'would sever an arm'}]}
    second = {'targets': [{'unit_id': 'U2', 'ref': 'the right sleeve', 'edit': {'property': 'color', 'value': 'blue'}}]}
    class Planner:
        calls = 0
        def generate_multimage(self, jobs, schemas=None):
            self.calls += 1
            assert '"U1": "chair", "U2": "sleeve"' in jobs[0][1]
            return [deepcopy(first if self.calls == 1 else second)]
    item = dict(gid='hints', views=['original.png', 'contours.png'],
                objects=[{'object_id': 50}, {'object_id': 4}], selection={'operations': ['remove', 'remove']},
                planning_view_style='clean-contours', planning_source_hints=hints,
                prompt=c.prompt('multi', a, view_style='clean-contours', source_hints=hints))
    planner = Planner()
    opts = dict(kind='multi', root=tmp_path, retry=1, minimal=True, content=True,
                accepted_counts={'add': 20, 'replace': 20, 'attribute': 0})
    out = plan_batch(planner, [deepcopy(item)], **opts)[0]
    assert out['targets'][0]['_minimal_instruction'] == 'Remove the left chair.'
    assert out['targets'][1]['operation'] == 'attribute' and planner.calls == 2
    assert plan_batch(planner, [deepcopy(item)], **opts)[0] == out and planner.calls == 2
    item['prompt'] = c.prompt('multi', a, view_style='clean-contours', source_hints={'U1': 'lamp'})
    with pytest.raises(ValueError, match='checkpoint'):
        plan_batch(planner, [item], **opts)
