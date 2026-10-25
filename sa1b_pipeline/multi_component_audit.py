"""Audit the exact final output of each independent write component once.

Dependent chains may retain a verified contiguous prefix. Reuse requires exact
audit-view equality; changed context permits at most one prefix audit.
"""
from copy import deepcopy
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .fonts import load_font
from .geometry import load_mask
from .production_state import atomic_json, digest, file_hash
from .multi_transactional import _verify_hashes, _evidence_hashes

from .audit_instruction_revision import VERSION as REVISION_VERSION, revise, snapshot

POLICY = 'component-final-v2'
CHECKS = ('ref_matches_mask', 'location_correct', 'operation_correct', 'target_done',
          'non_target_preserved', 'natural', 'host_integrity')

PROMPT_VERSION = 'concise-three-full-images-3-site-ref'
PROMPT = """Review each U using three aligned images: 1 clean source, 2 source with colored mask contours and U labels, 3 final edited image. Contours select the source object/region; for add they select the attachment site. Judge final appearance from image 3, not the annotation.
Check image quality against the ORIGINAL request: the intended edit is visible at the correct region, looks natural, and preserves unrelated content. Reject wrong/missing edits, wrong attachment, obvious remnants, broken structure or collateral damage. Minor texture/boundary differences and source blur alone are not failures. All listed edits are intentional; dependent edits must survive together.
Check instruction quality: the Ref accurately names the target, with concise standard wording and a natural request. For add, Ref names the host; the source mask selects its direct contact/site and need not cover the whole host. Test uniqueness on clean image 1 ONLY: if another visible instance satisfies the complete Ref, repair the wording or reject it. Seeing its mask/U label is not evidence the WORDING is unique. Classify the actual operation independently: appearance-only change is attribute, not object replacement.
If the image fulfills the original goal and only wording/Ref needs improvement, return a concise corrected ref and optimized_instruction in this response; otherwise leave both empty. Keep the SAME source entity, mask extent, attachment site, operation, payload and attribute values; keep the request's operation-specific sentence format. Never rewrite the goal to excuse a bad image. Assess instruction_ok and reference_unique against the corrected wording when supplied. The program validates and renders Ref/no-ref; no-ref uses "the selected region" and preserves the edit content.
Return one entry per U: image_ok, instruction_ok, reference_unique, observed_operation, ref, optimized_instruction, and a brief visible reason (including the distinguishing cue when correcting). A valid text correction does not override image failure. No additional explanations.
"""


def audit_schema(ids, dependencies=()):
    text = {'type': 'string'}
    fields = {'unit_id': {'type': 'string', 'enum': ids},
              'image_ok': {'type': 'boolean'}, 'instruction_ok': {'type': 'boolean'},
              'reference_unique': {'type': 'boolean'},
              'observed_operation': {'enum': ['add', 'remove', 'replace', 'attribute', 'none', 'mixed', 'unclear']},
              'ref': text, 'optimized_instruction': text, 'reason': text}
    unit = dict(type='object', properties=fields, required=list(fields), additionalProperties=False)
    properties={'units': dict(type='array', items=unit, minItems=len(ids), maxItems=len(ids))}
    if dependencies:
        fields={'before':{'type':'string','enum':ids},'after':{'type':'string','enum':ids},
                'exposed_appearance':text,'adjacent_appearance':text,'consistent':{'type':'boolean'},'issue':text}
        properties['interfaces']=dict(type='array',items=dict(type='object',properties=fields,
            required=list(fields),additionalProperties=False),minItems=len(dependencies),maxItems=len(dependencies))
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def _box(mask, margin):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('empty component audit selector')
    pad = max(20, round(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)*margin))
    return (max(0, xs.min()-pad), max(0, ys.min()-pad),
            min(mask.shape[1], xs.max()+1+pad), min(mask.shape[0], ys.max()+1+pad))


def comparison_panel(group, members, final, path):
    """Exact aligned source/mask/final with context and detail; no raw model image."""
    source = Image.open(group['source_image']).convert('RGB')
    result = Image.open(final).convert('RGB')
    if result.size != source.size:
        raise ValueError('component result dimensions changed')
    panel = Image.new('RGB', (1536, 2*420*len(members)), (238, 240, 244))
    draw = ImageDraw.Draw(panel); font = load_font(21, bold=True)
    for i, member in enumerate(members):
        mask = load_mask({'mask_path': member['mask']}, source.size)
        support = load_mask({'mask_path': member['support_mask']}, source.size)
        changed = np.max(np.abs(np.asarray(source, dtype=np.int16)-np.asarray(result, dtype=np.int16)), axis=2)>12
        # Show contact context even when the addition failed to appear.
        focus = (changed & support) if member['operation']=='add' and (changed & support).any() else mask
        boxes = (_box(mask | (changed & support), .5), _box(focus, .18))
        binary = Image.fromarray(mask.astype('uint8')*255).convert('RGB')
        for row, box in enumerate(boxes):
            y = (i*2+row)*420
            for col, (im, label) in enumerate(((source, 'SOURCE'), (binary, 'MASK'), (result, 'FINAL RESULT'))):
                draw.text((col*512+10, y+8), f"{member['unit_id']} {label} / {'context' if row==0 else 'detail'}", font=font, fill='black')
                tile = ImageOps.contain(im.crop(box), (504, 372), Image.Resampling.NEAREST if col==1 else Image.Resampling.LANCZOS)
                panel.paste(tile, (col*512+(512-tile.width)//2, y+44+(372-tile.height)//2))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);panel.save(path, quality=95, subsampling=0)
    return str(path)


def expand_compact_result(raw, members):
    """Adapt the concise model response to the existing audited export contract."""
    result = deepcopy(raw)
    by_id = {m['unit_id']: m for m in members}
    for row in result.get('units', []) if isinstance(result.get('units'), list) else []:
        if not isinstance(row, dict) or 'image_ok' not in row:
            continue  # Historical evidence remains readable.
        member = by_id.get(row.get('unit_id'), {})
        if 'scope_status' in row:
            # The new audit reports observed scope/operation, rather than
            # agreeing with the assigned category. Older ledgers remain readable.
            row['scope_ok'] = row.get('scope_status') == 'matched'
            row['operation_ok'] = row.get('observed_operation') == member.get('operation')
            issue, proposed = row.get('ref_issue'), row.get('final_ref')
            if not isinstance(issue, str) or not isinstance(proposed, str):
                row['image_ok'] = False
            elif not issue.strip():
                # No demonstrated defect: retain the original wording exactly.
                # A gratuitous shortening cannot remove valid location cues.
                if proposed:
                    row['unrequested_ref_rewrite_ignored'] = proposed
                row['final_ref'] = (member.get('instruction_spec') or {}).get('ref', '')
            elif not proposed.strip():
                row['reference_unique'] = False
            row['reported_operation'] = row.get('observed_operation')
        if 'final_ref' in row:
            # A single authoritative final source phrase removes the ambiguity
            # between judging the old draft and judging its proposed repair.
            ref = row.get('final_ref')
            spec = member.get('instruction_spec') or {}
            site = (spec.get('placement') or {}).get('site')
            if spec.get('operation') == 'add' and site and isinstance(ref, str):
                from .instruction_protocol import noun, clean
                ref = clean(ref)
                prefix = noun(site) + ' of '
                # The model sometimes returns the public site+host phrase.
                # Strip only an EXACT echo of the immutable site slot; the
                # renderer restores it once, preserving the intended referent.
                if ref.casefold().startswith(prefix.casefold()) and ref[len(prefix):].startswith('the '):
                    row['site_echo_normalization'] = {'model_ref':ref,'host_ref':ref[len(prefix):]}
                    ref = ref[len(prefix):]
            row['ref'] = ref
            row['instruction_ok'] = (isinstance(ref,str) and ref.startswith('the ')
                                     and row.get('scope_ok') is True)
        if 'source_state' in row and 'result_state' in row and 'image_ok' in row:
            before,after=row.get('source_state'),row.get('result_state')
            if not all(isinstance(s,str) and s.strip() for s in (before,after)):
                row['image_ok']=False
            row['reason']='SOURCE: '+str(before)+'; RESULT: '+str(after)
        if 'operation_ok' in row:
            # New schema verifies the requested operation directly and returns
            # only a reference slot. Canonical sentence rendering stays in code.
            row['observed_operation'] = member.get('operation') if row['operation_ok'] is True else 'unclear'
            if type(row['operation_ok']) is not bool:
                row['image_ok'] = False
            if 'scope_ok' in row and row['scope_ok'] is not True:
                row['image_ok'] = False
            ref = row.get('ref')
            row['optimized_instruction'] = ''
            if isinstance(ref, str) and ref:
                from .instruction_protocol import render
                try:
                    spec = deepcopy(member.get('instruction_spec') or {})
                    spec['ref'] = ref
                    row['optimized_instruction'] = render(spec)
                except (ValueError, KeyError, TypeError):
                    row['instruction_ok'] = False
        reason = row.get('reason', '')
        valid = (all(type(row.get(k)) is bool for k in ('image_ok', 'instruction_ok', 'reference_unique'))
                 and all(isinstance(row.get(k), str) for k in ('ref','optimized_instruction','reason'))
                 and bool(reason.strip()))
        image_ok = (valid and row.get('image_ok') is True
                    and row.get('observed_operation') == member.get('operation'))
        instruction_ok = valid and row.get('instruction_ok') is True and row.get('reference_unique') is True
        for k in CHECKS:
            row[k] = instruction_ok if k == 'ref_matches_mask' else image_ok
        row.update(reference_count='one' if instruction_ok else 'unclear',
                   source_state=row.get('source_state') or 'See compact audit reason', result_state=row.get('result_state') or reason or 'Missing audit reason',
                   closest_alternative='', serious_defects=[], warnings=[], issue=reason)
        ref, instruction = row.get('ref'), row.get('optimized_instruction')
        # Some models echo the unchanged public anchor while leaving the optional
        # rewrite empty. This is a no-op, not a failed correction. Match exactly;
        # a genuinely different Ref still requires a complete validated rewrite.
        original_spec = member.get('instruction_spec') or {}
        original_ref = original_spec.get('ref')
        unchanged_refs = {original_ref} if isinstance(original_ref, str) else set()
        site = (original_spec.get('placement') or {}).get('site')
        if original_spec.get('operation') == 'add' and site and original_ref:
            from .instruction_protocol import noun
            unchanged_refs.add(noun(site) + ' of ' + original_ref)
        if not instruction and ref in unchanged_refs:
            row['reference_echo_ignored'] = True
            ref = ''
        if ref == original_ref:
            row['instruction_revision'] = dict(mode='keep',ref='',reason='',same_target=False,unique=False,evidence='')
        elif ref or instruction:
            # Exact canonical rendering enforces unchanged payload, operation and site.
            from .instruction_protocol import render
            spec = deepcopy(member.get('instruction_spec') or {})
            spec['ref'] = ref
            try:
                if render(spec) != instruction:
                    raise ValueError('changed edit goal or noncanonical correction')
            except (ValueError, KeyError, TypeError):
                row['serious_defects'].append('invalid_optimized_instruction')
            if instruction_ok:
                row['instruction_revision'] = dict(mode='ref', ref=ref, reason=row.get('ref_issue') or reason,
                    same_target=True, unique=True, evidence=reason)
            else:
                # An unverified Ref is a text rejection, not evidence of image
                # damage. Preserve the proposal without applying or promoting it.
                row['unverified_ref_proposal'] = ref
                row['instruction_revision'] = dict(mode='keep',ref='',reason='',same_target=False,unique=False,evidence='')
        else:
            row['instruction_revision'] = dict(mode='keep',ref='',reason='',same_target=False,unique=False,evidence='')
        if not valid:
            row['serious_defects'].append('invalid_compact_audit_response')
    return result


def normalize_result(raw, members, dependencies=()):
    raw = expand_compact_result(raw, members)
    rows = raw.get('units')
    ids = [m['unit_id'] for m in members]
    valid = (isinstance(rows, list) and len(rows)==len(ids) and all(isinstance(x, dict) for x in rows)
             and sorted(x.get('unit_id', '') for x in rows)==sorted(ids))
    by_id = {x['unit_id']: x for x in rows} if valid else {}
    by_member = {m['unit_id']:m for m in members}
    audits = {}
    for uid in ids:
        row = by_id.get(uid, {})
        reasons = [k+'_failed' for k in CHECKS if row.get(k) is not True]
        grounding = by_member[uid].get('grounding_evidence') or {}
        from .grounding_evidence import hard_failures
        grounding_failures = hard_failures(grounding)
        reasons.extend(grounding_failures)
        if grounding:
            row['grounding_evidence'] = deepcopy(grounding)
        if raw.get('_parse_error') or raw.get('decision') == 'fail' or raw.get('failure_reasons'):
            reasons.append('invalid_or_truncated_component_response')
        reasons.extend(row.get('serious_defects') or [])
        if row.get('reference_count') != 'one': reasons.append('reference_not_unique')
        expected = by_member[uid].get('operation')
        if expected in ('replace', 'attribute') and row.get('observed_operation') != expected:
            reasons.append('observed_operation_mismatch')
        if not valid: reasons.append('missing_duplicate_or_wrong_unit_evidence')
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('source_state','result_state')):
            reasons.append('missing_visual_contrast')
        image_ok = (valid and not grounding_failures and not raw.get('_parse_error') and raw.get('decision') != 'fail'
                    and not raw.get('failure_reasons') and not row.get('serious_defects')
                    and all(row.get(k) is True for k in CHECKS if k != 'ref_matches_mask')
                    and row.get('observed_operation') == expected)
        error = revise(by_member[uid], row, image_ok=image_ok)
        if error: reasons.append(error)
        audits[uid] = {**row, 'decision': 'fail' if reasons else 'pass', 'failure_reasons': reasons,
                       'evidence': row.get('issue', ''), 'audit_mode': POLICY}
    if dependencies:
        rows=raw.get('interfaces',[])
        expected={(e['before'],e['after']) for e in dependencies}
        actual=[(r.get('before'),r.get('after')) for r in rows if isinstance(r,dict)] if isinstance(rows,list) else []
        valid=len(actual)==len(expected) and set(actual)==expected
        for e in dependencies:
            row=next((r for r in rows if isinstance(r,dict) and (r.get('before'),r.get('after'))==(e['before'],e['after'])),{}) if valid else {}
            if (not valid or row.get('consistent') is not True or
                    not all(str(row.get(k,'')).strip() for k in ('exposed_appearance','adjacent_appearance'))):
                audit=audits[e['after']]
                audit['decision']='fail';audit['natural']=False
                audit['failure_reasons'].append('dependent_interface_inconsistent_or_missing')
            audits[e['after']].setdefault('interface_evidence',[]).append(row)
    return audits


def interface_panel(group, members, final, dependencies, path):
    by_id={m['unit_id']:m for m in members}
    result=Image.open(final).convert('RGB')
    panel=Image.new('RGB',(1536,520*len(dependencies)),(238,240,244))
    draw=ImageDraw.Draw(panel);font=load_font(21,bold=True)
    for i,e in enumerate(dependencies):
        target=by_id[e['after']]
        mask=load_mask({'mask_path':target.get('mask_v1') or target['mask']},result.size)
        box=_box(mask,.65)
        prior=Image.open(by_id[e['before']]['step_output_image']).convert('RGB')
        for col,(image,label) in enumerate(((prior,'AFTER PREDECESSOR'),(result,'COMPONENT FINAL'))):
            draw.text((col*768+10,i*520+8),e['before']+' -> '+e['after']+' / '+label,font=font,fill='black')
            tile=ImageOps.contain(image.crop(box),(752,470),Image.Resampling.LANCZOS)
            panel.paste(tile,(col*768+(768-tile.width)//2,i*520+44+(470-tile.height)//2))
    panel.save(path,quality=96,subsampling=0)
    return str(path)


def audit_component(group, members, final, auditor, directory, *, aligned=False):
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
    path = directory/'audit.json'
    from . import aligned_audit
    from .grounding_evidence import VERSION as GROUNDING_VERSION
    contract = dict(policy=POLICY, prompt_version=aligned_audit.VERSION if aligned else PROMPT_VERSION, instruction_revision_version=REVISION_VERSION, interface_audit_version=2, dependency_digest=digest(group.get('dependency_schedule', {})), unit_ids=[m['unit_id'] for m in members], final_sha256=file_hash(final),
                    source_sha256=file_hash(group['source_image']),
                    grounding_policy=GROUNDING_VERSION if getattr(auditor, 'grounding', None) else None,
                    target_contract=digest([{k:m.get(k) for k in ('unit_id','instruction','instruction_noref','mask','source_annotation_hint')} for m in members]))
    verifier = getattr(auditor, 'verifier', None)
    if verifier is not None:
        from .edit_verifier import contract as verifier_contract
        contract['verifier'] = verifier_contract(verifier)
    ref_reconcile = bool(getattr(auditor, 'ref_reconcile', False)) and aligned
    if ref_reconcile:
        from .audit_reconcile import VERSION as RECONCILE_VERSION
        contract['ref_reconcile'] = RECONCILE_VERSION
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['contract'] != contract: raise ValueError('component audit input changed')
        payload={k:v for k,v in saved.items() if k!='sha256'}
        if saved['sha256'] != digest(payload): raise ValueError('component audit ledger changed')
        _verify_hashes(saved['image_hashes'])
        return saved
    if getattr(auditor, 'grounding', None):
        for member in members:
            member['grounding_evidence'] = auditor.grounding.inspect(group['source_image'], member, final)
    from .full_contour_views import contour_pair
    source = Image.open(group['source_image']).convert('RGB')
    masks = [load_mask({'mask_path': m['mask']}, source.size) for m in members]
    _, marked, labels = contour_pair(source, masks, unit_ids=[m['unit_id'] for m in members])
    panel = str(directory/'source_masks.png'); marked.save(panel)
    atomic_json(directory/'mask_labels.json', labels)
    job_id = group['multi_id']+'::component::'+'+'.join(m['unit_id'] for m in members)
    targets=aligned_audit.requests(members, group['source_image'], [final]*len(members)) if aligned else '\n'.join(f"{m['unit_id']}: {m['instruction']}" for m in members)
    ids = {m['unit_id'] for m in members}
    dependencies = [e for e in group.get('dependency_schedule', {}).get('evidence', [])
                    if e.get('before') in ids and e.get('after') in ids]
    dependencies = list({(e['before'], e['after']):e for e in dependencies}.values())
    relations = ('\nDEPENDENT EDITS: ' + json.dumps(dependencies, ensure_ascii=False) +
                 '\nCheck that earlier edits remain and revealed surfaces agree with their surroundings; '
                 'an old-color patch or obvious residual shadow is a defect. Return interface observations and consistent.') if dependencies else ''
    images = aligned_audit.views(group['source_image'],members,[final]*len(members),directory) if aligned else [group['source_image'],panel,final]
    prompt = aligned_audit.PROMPT if aligned else PROMPT
    schema = aligned_audit.schema if aligned else audit_schema
    # The optional verifier jobs go out in the same batch as the main audit (one round trip);
    # a verdict can only fail a member the main audit passed.
    verify_jobs=[]
    if verifier is not None:
        from .edit_verifier import member_jobs
        verify_jobs=member_jobs(members,group['source_image'],[final]*len(members),directory,verifier,prefix=job_id)
    start=time.monotonic()
    results=auditor.call_batch([dict(job_id=job_id, images=images,
                                   prompt=prompt+'\nRequests:\n'+targets+relations,
                                   max_tokens=256+300*len(members)+140*len(dependencies), image_max_edge=1800 if aligned else 2304,
                                   **({'enable_thinking':True,'thinking_token_budget':256} if aligned else {}),
                                   output_schema=schema([m['unit_id'] for m in members],dependencies))]+verify_jobs)
    if len(results)!=1+len(verify_jobs) or results[0].get('job_id')!=job_id: raise RuntimeError('component audit job mismatch')
    raw=results[0].get('semantic_audit', {})
    if results[0].get('finish_reason') == 'length':
        raw={**raw, '_parse_error':'truncated response'}
    reconcile_records,reconcile_calls=[],0
    if ref_reconcile:
        from .audit_reconcile import reconcile
        raw,reconcile_records,reconcile_calls=reconcile(raw,members,group,auditor,prefix=job_id)
    audits=normalize_result(raw, members, dependencies)
    verifier_calls=len(verify_jobs)
    if verifier is not None:
        from .edit_verifier import record
        record(members,audits,results[1:])
        images=images+[p for j in verify_jobs for p in j['images']]
    images={**{p:h for m in members for p,h in _evidence_hashes(m).items()}, **{p:file_hash(p) for p in images}}
    images.update({final:file_hash(final), panel:file_hash(panel), group['source_image']:file_hash(group['source_image'])})
    saved=dict(contract=contract, raw=raw, audits=audits, instruction_updates={m['unit_id']:snapshot(m) for m in members}, accepted=all(a['decision']=='pass' for a in audits.values()),
               semantic_calls=1, seconds=time.monotonic()-start, image_hashes=images, panel=panel,
               inference={k:results[0].get(k) for k in ('input_tokens','output_tokens','finish_reason')})
    if verifier is not None:
        saved['verifier_calls']=verifier_calls
    if ref_reconcile:
        saved.update(ref_reconcile=reconcile_records,ref_reconcile_calls=reconcile_calls)
    if aligned:
        saved['aligned_crop_boxes']=json.loads((directory/'result_crop_boxes.json').read_text())
        saved['audited_final']=final
    saved['sha256']=digest(saved);atomic_json(path, saved)
    return saved


def generation_prefix(decisions, group=None):
    """Recover the committed chain, skipping failed writes only with proof.

    run_component does not advance its input on a deterministic rejection.
    Later accepted steps can therefore already be based on the last valid
    state; discarding all of them as a suffix loses valid work unnecessarily.
    Semantic rejection still uses the bounded prefix recovery below.
    """
    prefix=[]
    failed=set()
    expected=None
    dependencies=(group or {}).get('edit_dependencies',[])
    for decision in decisions:
        m=decision['member'];uid=m['unit_id']
        incoming=m.get('step_input_image')
        if expected is None:
            expected=incoming
        if not decision['accepted']:
            failed.add(uid)
            continue
        if failed:
            if any(a in failed and b==uid for a,b in dependencies):
                break
            # Do not accept path equality without verifying the saved pixels.
            if not incoming or not expected or file_hash(incoming)!=file_hash(expected):
                break
        prefix.append(decision)
        expected=m.get('step_output_image')
    return prefix


def _prefix_pixels_unchanged(report, members, image):
    """Reuse only if every displayed result crop is exactly unchanged."""
    boxes=report.get('aligned_crop_boxes') or []
    by_id={r['unit_id']:r for r in boxes}
    if not members or any(m['unit_id'] not in by_id for m in members):return False
    original=report.get('audited_final')
    if not original:return False
    _verify_hashes(report['image_hashes'])
    with Image.open(original) as old, Image.open(image) as new:
        if old.size != new.size:return False
        for member in members:
            row=by_id[member['unit_id']]
            for key in ('box','detail_box'):
                box=row[key]
                if not np.array_equal(np.asarray(old.crop(box)),np.asarray(new.crop(box))):return False
    return True


def complete_component(group, decisions, current, auditor, directory, *, joint_report=None, joint_image=None):
    """Keep a verified prefix; at most one recovery audit, never regenerate."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    generated=generation_prefix(decisions,group)
    members=[deepcopy(d['member']) for d in generated]
    generated_image=members[-1]['step_output_image'] if members else group['source_image']
    if joint_report is not None and members:
        report=deepcopy(joint_report)
        report['audits']={m['unit_id']:report['audits'][m['unit_id']] for m in members}
        report['accepted']=all(a['decision']=='pass' for a in report['audits'].values())
        report['semantic_calls']=0
        report.pop('sha256',None);report['sha256']=digest(report)
        atomic_json(directory/'audit.json',report)
    else:
        report=audit_component(group,members,generated_image,auditor,directory) if members else None
    initial_report=report
    calls=report['semantic_calls'] if report else 0
    count=0
    for member in members:
        if report['audits'][member['unit_id']]['decision']!='pass':break
        count+=1
    selected=members[:count]
    selected_image=selected[-1]['step_output_image'] if selected else group['source_image']
    recovery=None;report_path=directory/'audit.json'
    if selected and count<len(members):
        for member in selected:
            member.update(deepcopy(report.get('instruction_updates',{}).get(member['unit_id'],{})))
        if _prefix_pixels_unchanged(report,selected,selected_image):
            recovery={'mode':'identical_audit_crops','kept':[m['unit_id'] for m in selected],
                      'prior_audit_sha256':report['sha256'],'extra_semantic_calls':0}
        else:
            # Changed context is not covered by the old audit. One bounded
            # prefix audit replaces its evidence; failure does not recurse.
            report_path=directory/'prefix_once/audit.json'
            report=audit_component(group,selected,selected_image,auditor,report_path.parent,aligned=True)
            calls+=report['semantic_calls']
            recovery={'mode':'one_prefix_audit','kept':[m['unit_id'] for m in selected] if report['accepted'] else [],
                      'prior_audit_sha256':initial_report['sha256'],'extra_semantic_calls':report['semantic_calls']}
            joint_image=None
            if not report['accepted']:selected=[];selected_image=group['source_image']
        atomic_json(directory/'prefix_recovery.json',recovery)
    kept_ids={m['unit_id'] for m in selected}
    selected_by_id={m['unit_id']:m for m in selected}
    final_decisions=[];kept=[]
    for d in decisions:
        uid=d['member']['unit_id'];ok=uid in kept_ids
        m=deepcopy(selected_by_id.get(uid,d['member']))
        evidence=(report if ok or (recovery and recovery['mode']=='one_prefix_audit'
                                   and uid in (report or {}).get('audits',{}))
                  else initial_report)
        if evidence:
            m.update(deepcopy(evidence.get('instruction_updates',{}).get(uid,{})))
        audit=(evidence or {}).get('audits',{}).get(uid,dict(decision='fail',failure_reasons=d.get('reasons') or ['component_generation_check_failed']))
        m.pop('semantic_audit_pending',None);m['semantic_audit']=audit
        audited_image=selected_image if ok else generated_image
        m['component_audit_image']=audited_image;m['component_audit_sha256']=file_hash(audited_image)
        if joint_image and ok:
            m['joint_audit_image']=joint_image;m['joint_audit_sha256']=file_hash(joint_image)
        if recovery:m['prefix_recovery']=recovery
        reasons=[] if ok else (audit.get('failure_reasons') or ['dependent_component_rejected'])
        payload={**d,'member':m,'accepted':ok,'reasons':reasons,'defer_semantic':False,
                 'audit_mode':POLICY,'component_audit_path':str(report_path if ok else directory/'audit.json') if evidence else None}
        payload.pop('ledger_sha256',None);payload['ledger_sha256']=digest(payload)
        checkpoint=directory/(uid+'_decision.json');atomic_json(checkpoint,payload)
        final_decisions.append(dict(unit_id=uid,operation=m['operation'],accepted=ok,reasons=reasons,checkpoint=str(checkpoint)))
        if ok:kept.append(m)
    return dict(current=selected_image,kept=kept,decisions=final_decisions,audit_atomic=True,
                audit=report,semantic_calls=calls,prefix_recovery=recovery)


def complete_components_together(group, results, auditor, directory):
    """Pack up to two independent targets per audit; keep dependencies atomic.

    Only the exact audited pixels can survive fusion. Connected dependencies
    remain atomic. An unviable (<2 target) group incurs no semantic request.
    """
    from .multi_parallel import fuse_components
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    eligible=[]
    for index,r in enumerate(results):
        prefix=generation_prefix(r['provisional'],group)
        if prefix:
            eligible.append({**r,'provisional':prefix,'current':prefix[-1]['member']['step_output_image'],
                             '_component_index':index,'kept':[d['member'] for d in prefix]})
    members=[d['member'] for r in eligible for d in r['provisional']]
    # With opt-in single salvage, a lone valid target is still audited so it can ship as a single.
    if len(members)<2 and not (len(members)==1 and getattr(auditor,'keep_singles',False)):
        for r in results:
            for d in r['decisions']:
                if d['accepted']:
                    d['accepted']=False;d['reasons']=['fewer_than_two_provisionally_valid_targets']
            r.update(kept=[],current=group['source_image'],audit_atomic=True,semantic_calls=0)
        return results,0
    # Large unrelated tables caused observation/request confusion in 27B.
    # Pair independent writes, but never split a connected dependency chain.
    packs=[];pending=[];count=0
    remaining=list(eligible)
    # A broad edited surface can surround several disjoint foreground writes.
    # Show those together so their exposed/background colors are actually seen
    # in the existing audit. This changes audit packing, never edit eligibility,
    # serialization or the minimum target count.
    size=Image.open(group['source_image']).size
    geometry={}
    for r in remaining:
        if len(r['provisional'])!=1:continue
        member=r['provisional'][0]['member']
        mask=load_mask({'mask_path':member.get('mask_v1') or member['mask']},size)
        ys,xs=np.where(mask)
        if len(xs):
            geometry[r['_component_index']]=(int(mask.sum()),(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())))
    for anchor in sorted(remaining,key=lambda r:geometry.get(r['_component_index'],(0,()))[0],reverse=True):
        if anchor not in remaining or anchor['_component_index'] not in geometry:continue
        area,box=geometry[anchor['_component_index']]
        if area<.1*size[0]*size[1]:continue
        related=[]
        for other in remaining:
            if other is anchor or other['_component_index'] not in geometry:continue
            other_area,b=geometry[other['_component_index']]
            if other_area*4<area and box[0]<=(b[0]+b[2])/2<=box[2] and box[1]<=(b[1]+b[3])/2<=box[3]:
                related.append(other)
        if len(related)>=2:
            pack=[anchor,*related[:2]];packs.append(pack)
            used={r['_component_index'] for r in pack}
            remaining=[r for r in remaining if r['_component_index'] not in used]
    for r in sorted(remaining,key=lambda r:len(r['provisional']),reverse=True):
        n=len(r['provisional'])
        if pending and count+n>2:packs.append(pending);pending=[];count=0
        pending.append(r);count+=n
        if count>=2:packs.append(pending);pending=[];count=0
    if pending:packs.append(pending)
    from concurrent.futures import ThreadPoolExecutor
    def review(item):
        index,pack=item;out=directory/f'pack_{index:02d}'
        final,_=fuse_components(group['source_image'],pack,out/'provisional')
        ms=[d['member'] for r in pack for d in r['provisional']]
        report=audit_component(group,ms,final,auditor,out/'joint',aligned=True)
        return [(r['_component_index'],report,final) for r in pack]
    with ThreadPoolExecutor(max_workers=len(packs)) as executor:
        reports={key:(report,final) for rows in executor.map(review,enumerate(packs)) for key,report,final in rows}
    output=[]
    for i,r in enumerate(results):
        report,final=reports.get(i,(None,None))
        output.append(complete_component(group,r['provisional'],r['current'],auditor,directory/f'component_{i:02d}',
                                        joint_report=report,joint_image=final))
    return output,len(packs)


def verify_component_fusion(group):
    """Prove retained pixels are exactly the audited component pixels."""
    source=np.asarray(Image.open(group['source_image']).convert('RGB'))
    final=np.asarray(Image.open(group['joint_target']).convert('RGB'))
    if final.shape!=source.shape: raise ValueError('fused dimensions changed')
    owned=np.zeros(source.shape[:2],bool);found=[]
    units={u['unit_id']:u for u in group['units']}
    for c in group['parallel']['fusion']['components']:
        _verify_hashes({c['image']:c['image_sha256'],c['support']:c['support_sha256']})
        support=np.asarray(Image.open(c['support']).convert('L'))>0
        image=np.asarray(Image.open(c['image']).convert('RGB'))
        if (owned & support).any() or not np.array_equal(final[support],image[support]):
            raise ValueError('fusion differs from audited pixels')
        for uid in c['kept']:
            if uid not in units or units[uid].get('component_audit_sha256')!=c['image_sha256']:
                raise ValueError('retained component was not audited at final state')
            member=units[uid]
            if member.get('joint_audit_image'):
                _verify_hashes({member['joint_audit_image']:member['joint_audit_sha256']})
                audited=np.asarray(Image.open(member['joint_audit_image']).convert('RGB'))
                if not np.array_equal(audited[support],image[support]):
                    raise ValueError('retained pixels differ from joint audit image')
        found.extend(c['kept']);owned|=support
    if sorted(found)!=sorted(units) or not np.array_equal(final[~owned],source[~owned]):
        raise ValueError('fusion ownership or outside pixels changed')
    return dict(**{'pass':True}, exact_audited_pixels=True, outside_exact=True,
                retained_targets=len(found), group_mllm_calls=0)
