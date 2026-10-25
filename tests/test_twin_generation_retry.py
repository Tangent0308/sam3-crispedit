import numpy as np
from PIL import Image

from sa1b_pipeline.twin_streaming import retry_plan


def _group(**checks):
    members = [{'unit_id': 'U1', 'semantic_audit': {'decision': 'pass', 'failure_reasons': []}, 'deterministic': {}},
               {'unit_id': 'U2', 'semantic_audit': {'decision': 'fail', 'failure_reasons': ['natural_failed']},
                'deterministic': {}}]
    base = {'group_pass': False, 'blind_protocol': {'pass': True}, 'support_gap': {'pass': True},
            'cross_target_preservation': {'pass': True, 'pairs': []}}
    return {'members': members, 'group_checks': {**base, **checks}}


def test_retry_plan_redoes_only_failed_members_and_never_text_or_contract_failures():
    assert retry_plan(_group()) == {'regenerate': ['U2'], 'keep': ['U1']}
    assert retry_plan(_group(group_pass=True)) is None
    assert retry_plan(_group(blind_protocol={'pass': False})) is None        # size/blind contract: plan property
    assert retry_plan(_group(support_gap={'pass': False})) is None
    text = _group()
    text['members'][1]['semantic_audit']['failure_reasons'] = ['natural_failed', 'reference_not_unique']
    assert retry_plan(text) is None                                            # a new image cannot fix the Ref
    spill = _group(cross_target_preservation={'pass': False, 'pairs': [
        {'member': 'U1', 'other': 'U2', 'changed_fraction_in_other_source_mask': 0.2}]})
    spill['members'][1]['semantic_audit'] = {'decision': 'pass', 'failure_reasons': []}
    assert retry_plan(spill) == {'regenerate': ['U1'], 'keep': ['U2']}
    clean = _group(); clean['members'][1]['semantic_audit'] = {'decision': 'pass', 'failure_reasons': []}
    assert retry_plan(clean) is None                                           # nothing member-specific failed


def test_generate_twin_group_keeps_marked_members_and_uses_member_seed(tmp_path, monkeypatch):
    from sa1b_pipeline import group_generation as gg
    calls = []

    class Editor:
        config = {'name': 'fake'}
        last_generation_context = {}

        def generate(self, source, unit, seed=0):
            calls.append((unit['unit_id'], seed))
            return Image.new('RGB', (64, 48), (200, 10, 10))

    monkeypatch.setattr(gg, 'twin_support_preflight', lambda group, out, editor: {
        'pass': True, 'pairs': [{'members': ['U1', 'U2'], 'pass': True}]})
    monkeypatch.setattr(gg, 'resolve_source_region', lambda *a, **k: None)
    monkeypatch.setattr(gg, '_row', lambda group, member, source, out, support, **k: {
        'edited_image': out, 'generation_context': k.get('generation_context')})
    monkeypatch.setattr(gg, 'deterministic_check', lambda row, **k: {'decision': 'pending_semantic_audit'})
    monkeypatch.setattr(gg, '_unit_dict', lambda member: {'unit_id': member['unit_id']})
    monkeypatch.setattr(gg, '_check_cross_preservation', lambda *a: 0.0)
    monkeypatch.setattr(gg, 'blind_group_check', lambda group: {'pass': True})
    source = tmp_path / 'source.png'
    Image.new('RGB', (64, 48), 'gray').save(source)
    first = tmp_path / 'first.png'
    Image.new('RGB', (64, 48), 'blue').save(first)
    group = {'source_image': str(source), 'members': [
        {'unit_id': 'U1', 'retry_keep': True, 'edited_image': str(first), 'generation_context': {'kept': 1},
         'support_mask': 's1', 'mask': 'm1'},
        {'unit_id': 'U2', 'retry_seed': 1, 'support_mask': 's2', 'mask': 'm2'}]}
    out = gg.generate_twin_group(group, tmp_path / 'retry', Editor(), seed=0)
    assert calls == [('U2', 1)]                                    # only the failed member is rendered, with seed 1
    kept, redone = out['members']
    assert kept['edited_image'] == str(first) and kept['generation_context'] == {'kept': 1} and kept['seed'] == 0
    assert redone['seed'] == 1 and redone['edited_image'].endswith('member_U2/edited.png')
