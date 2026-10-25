import pytest

from sa1b_pipeline.planning_quality import appearance_only_replacement


def test_color_intensity_is_not_a_new_tail_fin_structure():
    assert appearance_only_replacement('patterned vertical tail fin of the airplane', 'a solid deep-blue tail fin')
    assert appearance_only_replacement('wooden chair', 'a pale green chair')
    assert not appearance_only_replacement('shallow drawer', 'a deep drawer')
from sa1b_pipeline.multi_semantics import validate_multi_semantics
from sa1b_pipeline.group_planning import multi_prompt


@pytest.mark.parametrize('scope,payload', [
    ('windscreen', 'a blue foam windscreen'), ('jeans', 'a pair of beige trousers'),
    ('chair', 'a wooden chair'), ('window', 'a frosted window'),
])
def test_explicit_appearance_rewrite(scope, payload):
    assert appearance_only_replacement(scope, payload)
    with pytest.raises(ValueError, match='appearance'):
        validate_multi_semantics({'operation': 'replace', 'scope': scope, 'payload': payload})


@pytest.mark.parametrize('scope,payload', [
    ('chair', 'a slatted wooden chair'), ('trousers', 'a pair of knee-length shorts'),
    ('ceiling', 'a coffered ceiling'), ('hat', 'a straw fedora'),
])
def test_real_design_change_is_not_lexically_rejected(scope, payload):
    assert not appearance_only_replacement(scope, payload)


def test_source_guess_cannot_prime_multimodal_planner():
    prompt = multi_prompt({'1': 'add'}, [{'object_id': '1', 'category': 'UNTRUSTED_LABEL'}], ['U1'])
    assert 'UNTRUSTED_LABEL' not in prompt and 'operation=add' in prompt


def test_admitted_scope_conflict_cannot_be_approved():
    from sa1b_pipeline.group_audit import explicit_scope_contradiction
    assert explicit_scope_contradiction('Purple collar and cuff', 'Purple collar on right')
    assert explicit_scope_contradiction('the dial and lever of the machine', 'the dial')
    assert not explicit_scope_contradiction('black and white collar', 'collar')
    assert not explicit_scope_contradiction('collar and cuffs', 'collar and cuffs')
    assert not explicit_scope_contradiction('chair and stool', 'seating furniture')


def test_actual_operation_overrides_fluent_success_and_fails_closed():
    from sa1b_pipeline.operation_evidence import normalize_audit, CHECKS, audit_prompt
    raw = {**dict.fromkeys(CHECKS, True), 'observed_operation': 'attribute',
           'source_state': 'blue trousers', 'result_state': 'beige trousers',
           'changed_structure': 'same shape, new color', 'observed_location': 'legs', 'issue': 'none'}
    assert normalize_audit(raw, 'replace')['decision'] == 'fail'
    assert normalize_audit(raw, 'attribute')['decision'] == 'pass'
    assert normalize_audit({}, 'attribute')['decision'] == 'fail'
    assert normalize_audit({**raw, 'decision': 'fail', 'failure_reasons': ['truncated_audit']}, 'attribute')['decision'] == 'fail'
    assert 'Replace ' not in audit_prompt('Replace the chair with a stool.', 'replace')
    with pytest.raises(ValueError):
        audit_prompt('Add a hat on the head.', 'add')


def test_twin_audit_hides_assigned_label_and_checks_observed_type():
    from sa1b_pipeline.operation_evidence import twin_prompt, enforce_twin_observation
    group = {'operation': 'replace', 'members': [
        {'unit_id': 'U1', 'instruction': 'Replace the chair with a stool.'}]}
    prompt = twin_prompt('Photograph comparison.\nTarget contract:\noperation=replace; Replace the chair', group)
    assert 'operation=replace' not in prompt and 'Replace the chair' not in prompt
    assert 'Source referent: the chair' in prompt and 'a stool.' in prompt
    result = {'semantic_audit': {'members': [
        {'unit_id': 'U1', 'observed_operation': 'attribute', 'checks': {'operation_correct': 'pass'}}]}}
    enforce_twin_observation(result, 'replace')
    assert result['semantic_audit']['members'][0]['checks']['operation_correct'] == 'fail'


def test_replacement_requirement_uses_fields_not_ambiguous_with_separator():
    from sa1b_pipeline.operation_evidence import audit_prompt
    instruction = 'Replace the bean with red flakes with a white bean with speckles.'
    spec = {'operation': 'replace', 'ref': 'the bean with red flakes', 'payload': 'a white bean with speckles'}
    prompt = audit_prompt(instruction, 'replace', spec)
    assert 'Source referent: the bean with red flakes.' in prompt
    assert 'Required final content at that location: a white bean with speckles.' in prompt
    with pytest.raises(ValueError):
        audit_prompt(instruction, 'replace')


def test_v1_engine_shutdown_reaches_mp_client_without_gc():
    from types import SimpleNamespace
    from sa1b_pipeline.mllm_runtime import close_engine
    calls = []
    core = SimpleNamespace(shutdown=lambda **kw: calls.append(kw))
    close_engine(SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core)))
    assert calls == [{'timeout': 10}]
    calls.clear()
    close_engine(SimpleNamespace(llm_engine=SimpleNamespace(shutdown=lambda: calls.append('legacy'), engine_core=core)))
    assert calls == ['legacy']
    close_engine(None)
