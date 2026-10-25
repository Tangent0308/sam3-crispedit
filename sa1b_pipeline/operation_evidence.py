"""Single-call observed-operation audit for replace/attribute only.

Add/remove keep their calibrated operation-specific audit. Hiding the assigned
label helps detect repainting mislabeled as replacement; it regresses accessory
occlusion judgments and must not be used as a universal operation classifier.
"""
import re

VERSION = 'observed-operation-1.3-count-detail'
CHECKS = ('ref_matches_mask', 'location_correct', 'target_done',
          'non_target_preserved', 'natural', 'host_integrity')
PROMPT = "Compare photographs, independently of the intended edit category. Image 1 SOURCE, image 2 RESULT, image 3 aligned SOURCE/RESULT/BINARY MASK. White in the last column selects the source region. Gray/black selector display is not damage. First record actual before/after form. Classify observed_operation: add = distinct new item on a retained host (natural occlusion is allowed); remove = selected item gone with coherent revealed content; replace = old entity replaced with a physically different shape/construction at its location; attribute = same entity with changed color/material/texture/text; none = no meaningful change; mixed = unrelated changes too; uncertain = cannot see the core change. Never use a requested payload's name as evidence that it exists. Changed color/material alone is attribute, even if described as another object. A panel/ceiling's different construction can be replace.\nThen check the desired final content below, original referent, exact selected extent, location, support/contact and preservation. Contact must be at the specified site: a strap on top of a hat does not meet a chin attachment. Scope means the actual mask extent, not a nearby item. Substantial selected surface remaining unchanged fails target_done. Fail severe defects, but allow mild texture differences, natural local occlusion/shading and unconventional plausible taste. Do not invent unrequested constraints. Return brief visual facts and all checks in supplied JSON.\nDesired final content: "


from .edit_quality import AUDIT_RULES

def audit_schema():
    fields = {k: {'type': 'string'} for k in
              ('source_state', 'result_state', 'changed_structure', 'observed_location', 'issue')}
    fields['source_count'] = {'type':['integer','null'],'minimum':0}
    fields['result_count'] = {'type':['integer','null'],'minimum':0}
    fields['serious_defects'] = {'type':'array','items':{'type':'string'}}
    fields['warnings'] = {'type':'array','items':{'type':'string'}}
    fields['observed_operation'] = {'enum': ['add', 'remove', 'replace', 'attribute', 'none', 'mixed', 'uncertain']}
    fields.update({k: {'type': 'boolean'} for k in CHECKS})
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def audit_prompt(instruction, operation, spec=None):
    if operation == 'replace':
        if spec:
            if spec.get('operation') != operation or not spec.get('ref') or not spec.get('payload'):
                raise ValueError('incomplete structured replacement requirement')
            source, payload = spec['ref'], spec['payload'].rstrip('.') + '.'
        elif not instruction.startswith('Replace ') or instruction.count(' with ') != 1:
            raise ValueError('unsupported replacement instruction rendering')
        else:
            source, payload = instruction[8:].split(' with ', 1)
        required = 'Source referent: ' + source + '. Required final content at that location: ' + payload
    elif operation == 'attribute':
        required = 'Required property: ' + re.sub(r'^(Change|Make|Tint) ', '', instruction)
    else:
        raise ValueError('observed-operation audit is calibrated only for replace/attribute')
    return PROMPT + required + '\n' + AUDIT_RULES + '\nRecord source_count and result_count: count the actual selected physical instances before/after (e.g. rods, sticks, panels). Use null only when not countable/visible. For attribute these counts must agree; explicitly trace each separate silhouette instead of assuming a pair stays a pair. For tint, an opaque painted result is not an acceptable transparent tint; mild reflection changes are acceptable.' 


def normalize_audit(audit, operation):
    """Fail closed on missing evidence; retain the exact observed facts."""
    checks = {k: audit.get(k) is True for k in CHECKS}
    operation_correct = audit.get('observed_operation') == operation
    reasons = list(audit.get('failure_reasons') or []) + list(audit.get('serious_defects') or [])
    reasons += [k + '_failed' for k, passed in checks.items() if not passed]
    if operation == 'attribute' and all(isinstance(audit.get(k), int) and not isinstance(audit.get(k), bool)
                                        for k in ('source_count', 'result_count')):
        if audit['source_count'] != audit['result_count']:
            reasons.append('attribute_instance_count_changed')
            checks['non_target_preserved'] = False
    if not operation_correct:
        reasons.append('operation_type_mismatch')
    if not all(isinstance(audit.get(k), str) and audit[k].strip()
               for k in ('source_state', 'result_state', 'changed_structure', 'observed_location', 'issue')):
        reasons.append('missing_visual_evidence')
    if audit.get('_parse_error') or audit.get('decision') == 'fail':
        reasons.append('invalid_or_failed_audit')
    return {**audit, **checks, 'operation_correct': operation_correct, 'addition_is_new': True,
            'source_entity': audit.get('source_state', ''),
            'observed_change': audit.get('changed_structure', ''), 'evidence': audit.get('issue', ''),
            'failure_reasons': list(dict.fromkeys(reasons)), 'decision': 'fail' if reasons else 'pass',
            'audit_profile': VERSION}


def twin_prompt(prompt, group):
    neutral = []
    for member in group['members']:
        required = audit_prompt(member['instruction'], group['operation'], member.get('instruction_spec')).split('Desired final content: ', 1)[1]
        neutral.append(member['unit_id'] + ': ' + required)
    return (prompt.split('\nTarget contract:\n')[0] +
            '\nIndependently classify each member observed_operation as add, remove, replace, attribute, none, mixed or uncertain. Color/material alone is ATTRIBUTE, not replace. Replacement requires a new physical form/construction, not just the requested object name. Return observed_operation for every member. The assigned category is intentionally hidden; set operation_correct to pass when the final-content requirement is met; a separate program compares the observed category.\nTarget contract:\n' + '\n'.join(neutral))


def enforce_twin_observation(result, operation):
    for row in (result.get('semantic_audit') or {}).get('members', []):
        if isinstance(row, dict) and row.get('observed_operation') != operation:
            row.setdefault('checks', {})['operation_correct'] = 'fail'
            row.setdefault('failure_reasons', []).append('observed_operation_mismatch')
