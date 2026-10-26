from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from test_multi_transactional import plan, Editor
from test_multi_component_audit import ComponentAuditor
from sa1b_pipeline.single_parallel import generate_independent_singles, singles_of, single_proof
from sa1b_pipeline.multi_parallel import conflict_components
from sa1b_pipeline.group_generation import multi_support_preflight


@pytest.fixture
def overlapping(monkeypatch):
    """U1 and U2 write overlapping supports: a multi group would edit U2 after U1 and export neither alone."""
    import sa1b_pipeline.single_parallel as mod
    real = mod.multi_support_preflight

    def preflight(g, root):
        result = real(g, root)
        size = Image.open(g['source_image']).size
        for u in g['units']:
            support = np.asarray(Image.open(u['mask']).convert('L')) > 0
            if u['unit_id'] in ('U1', 'U2'):
                support = support.copy(); support[0:8, 20:60] = True   # shared strip above both squares (no selector)
            path = root/f"support_{u['unit_id']}.png"
            Image.fromarray(support.astype(np.uint8) * 255).save(path)
            u['unit']['execution_mask_path'] = str(path)
        return result
    monkeypatch.setattr(mod, 'multi_support_preflight', preflight)


def test_every_target_edited_from_original_and_exported_alone(plan, tmp_path, overlapping):
    g = deepcopy(plan); multi_support_preflight(g, tmp_path/'probe')
    import sa1b_pipeline.single_parallel as mod
    mod.multi_support_preflight(g, tmp_path/'probe2')
    assert conflict_components(g)[0][0] == ['U1', 'U2']   # chained in the multi path
    editors = []

    def factory():
        e = Editor(); editors.append(e); return e
    auditor = ComponentAuditor(fail=['U3'])
    group = generate_independent_singles(plan, tmp_path/'run', factory, auditor)
    assert all(np.all(a == 100) for e in editors for a in e.inputs)   # every edit starts at the source
    assert sum(e.calls for e in editors) == 4
    assert [e['unit_id'] for e in group['single_parallel']['exported']] == ['U1', 'U2', 'U4']
    assert group['retention']['dropped_unit_ids'] == ['U3']
    assert group['single_parallel']['components'] == [['U1'], ['U2'], ['U3'], ['U4']]
    singles = singles_of(group)
    assert [s['unit_id'] for s in singles] == ['U1', 'U2', 'U4']
    source = np.asarray(Image.open(plan['source_image']))
    for s in singles:
        out = np.asarray(Image.open(s['edited_image']))
        changed = np.any(out != source, axis=2)
        mask = np.asarray(Image.open(s['mask']).convert('L')) > 0
        assert changed.any() and not (changed & ~mask).any()   # only its own target changed
        assert s['provenance']['export'] == 'independent-singles-1' and s['provenance']['single_proof']['pass']
    # U1 and U2 overlap: they cannot share one provisional audit image and are audited alone.
    jobs = [j['job_id'] for j in auditor.calls]
    assert any(j.endswith('::component::U1') for j in jobs) and any(j.endswith('::component::U2') for j in jobs)


def test_resume_makes_no_new_calls(plan, tmp_path, overlapping):
    e = Editor(); a = ComponentAuditor(); root = tmp_path/'run'
    group = generate_independent_singles(plan, root, lambda: e, a)
    before = (e.calls, len(a.calls))
    assert generate_independent_singles(plan, root, lambda: e, a) == group
    assert (e.calls, len(a.calls)) == before
    Image.new('RGB', (128, 128), 'black').save(group['single_parallel']['exported'][0]['image'])
    with pytest.raises(ValueError, match='checkpoint image changed'):
        generate_independent_singles(plan, root, lambda: e, a)


def test_lone_valid_target_is_audited(plan, tmp_path, overlapping, monkeypatch):
    import sa1b_pipeline.single_parallel as mod
    real = mod.run_committed_step

    def only_u4(*args, **kwargs):
        result = real(*args, **kwargs)
        if result['member']['unit_id'] != 'U4':
            result['accepted'] = False; result['reasons'] = ['test_deterministic_failure']
        return result
    monkeypatch.setattr(mod, 'run_committed_step', only_u4)
    a = ComponentAuditor()   # no keep_singles needed: every generated target is audited
    group = generate_independent_singles(plan, tmp_path/'run', Editor, a)
    assert [e['unit_id'] for e in group['single_parallel']['exported']] == ['U4'] and len(a.calls) == 1


@pytest.fixture
def disjoint(monkeypatch):
    """Execution masks equal to the source masks: the four targets' supports do not touch."""
    import sa1b_pipeline.single_parallel as mod
    real = mod.multi_support_preflight

    def preflight(g, root):
        result = real(g, root)
        for u in g['units']:
            u['unit']['execution_mask_path'] = u['mask']
        return result
    monkeypatch.setattr(mod, 'multi_support_preflight', preflight)


def test_disjoint_targets_share_one_audit_per_pair(plan, tmp_path, disjoint):
    a = ComponentAuditor()
    group = generate_independent_singles(plan, tmp_path/'run', Editor, a)
    assert len(group['single_parallel']['exported']) == 4
    assert len(a.calls) == 2 and group['single_parallel']['semantic_calls'] == 2
    assert sorted(j['job_id'].split('::')[-1] for j in a.calls) == ['U1+U2', 'U3+U4']


def test_proof_rejects_pixels_outside_support_or_unaudited(plan, tmp_path):
    source = plan['source_image']
    out = np.asarray(Image.open(source)).copy(); out[20:30, 20:30] = (0, 240, 0)
    path = tmp_path/'out.png'; Image.fromarray(out).save(path)
    support = np.zeros((128, 128), np.uint8); support[15:35, 15:35] = 255
    spath = tmp_path/'support.png'; Image.fromarray(support).save(spath)
    from sa1b_pipeline.production_state import file_hash
    member = {'step_output_image': str(path), 'support_mask': str(spath), 'component_audit_sha256': file_hash(path)}
    assert single_proof(source, member)['pass']
    assert single_proof(source, {**member, 'component_audit_sha256': 'x'})['reason'] == 'output_not_audited'
    out[100, 100] = 0; Image.fromarray(out).save(path)
    member['component_audit_sha256'] = file_hash(path)
    assert single_proof(source, member)['reason'] == 'changed_outside_support'
