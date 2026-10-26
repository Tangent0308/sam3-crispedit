from test_multi_transactional import plan, Editor
from test_multi_component_audit import ComponentAuditor
from test_single_parallel import disjoint  # noqa: F401 (fixture)
from test_audit_instruction_revision import proposal
from sa1b_pipeline.single_parallel import generate_independent_singles, singles_of


class RevisingAuditor(ComponentAuditor):
    """Proposes a verified Ref repair for U1, which the audit applies to the member in place."""

    def call_batch(self, jobs):
        results = super().call_batch(jobs)
        for result in results:
            for row in result['semantic_audit']['units']:
                if row['unit_id'] == 'U1':
                    row['instruction_revision'] = proposal('the cup in the upper-left corner')
        return results


def test_pack_audit_shared_when_it_revises_an_instruction(plan, tmp_path, disjoint):
    # The shared pack audit of the multi path (multi_component_audit.complete_components_together at b4c9991)
    # audits each component of a pack again; after a Ref repair the second call sees a changed input and the
    # group fails ('component audit input changed', 7 of 127 groups in single-target validation wave 1).
    # Single-target generation audits every pack once (single_parallel.audit_targets).
    a = RevisingAuditor()
    group = generate_independent_singles(plan, tmp_path/'run', Editor, a)
    singles = {s['unit_id']: s for s in singles_of(group)}
    assert set(singles) == {'U1', 'U2', 'U3', 'U4'} and len(a.calls) == 2
    assert singles['U1']['instruction'] == 'Remove the cup in the upper-left corner.'
