"""Focused post-audit defect check for one edit (opt-in; rejects only, never approves).

The main audit writes observations, Ref checks and its verdict in one long call;
it has passed edits whose own observations describe the failure ("the hut's porch
is filled in with a solid wall" for a removal). This verifier asks a short,
operation-specific list of yes/no questions about clearly visible defects, with
two images: the full AFTER photo and a BEFORE/AFTER zoom of the edited region
(the selected region outlined on BEFORE only). It runs on the resident
audit/planner model. A ``true`` answer to a hard question rejects the edit; soft
questions are recorded only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VERSION = 'edit-verifier-3-hard-remnant'

HEADER = ('You check ONE photo edit for clear defects. Image 1 is the full edited photo (AFTER). '
          'Image 2 shows the same zoomed region twice: BEFORE with the selected region outlined in yellow, '
          'and AFTER without outline, at the same scale.\n'
          'Requested edit: "{instruction}"\n'
          'Compare BEFORE and AFTER and answer each question with true or false. Answer true only for a defect '
          'you can clearly see in AFTER; small edge, blur or texture differences are not defects.\n')

# (key, question, hard). {target}: what was selected; {new}: requested new content; {change}: attribute value.
# Five targeted questions (old parts remaining, body cut at a replacement, unrecognizable addition, glowing
# recolor, wider host damage) were tried on 772 labelled edits (2026-10-24, edit-verifier-4) and did not
# catch the misses they targeted; with reasoning they added false rejects. Not kept.
QUESTIONS = {
    'remove': [
        ('target_remains', 'The {target} is still clearly visible in AFTER at its place, whole or in part, or a similar '
                           'object was redrawn there.', True),
        ('left_unsupported', 'Look at what rested on, hung from, leaned on or was held by the {target} in BEFORE: in AFTER '
                             'such a thing is still there but now floats or hangs with nothing holding it.', True),
        ('host_damaged', 'A neighbouring object, or the object the {target} was on or attached to, is now cut off, '
                         'broken or missing a part.', True),
        ('body_unnatural', 'A person or body part revealed or touched by the edit looks deformed, incomplete, '
                           'unnatural or like a mannequin.', True),
        # Hard since v3: on 425 labelled dev edits it was true on bad/minor removals only.
        ('remnant', 'The revealed area shows an obvious leftover: an outline, shadow or reflection of the removed thing, '
                    'a smear, or a patch that clearly differs from its surroundings.', True),
    ],
    'add': [
        ('not_added', 'No new {new} is visible in AFTER at or near the outlined region.', True),
        ('existing_changed', 'There is no NEW {new}: instead, an object that already existed at this place in BEFORE '
                             'was redrawn, recolored or moved.', True),
        ('wrong_size', 'The new {new} is clearly the wrong size for what it is attached to or for the scene.', True),
        ('unsupported', 'The new {new} floats or hovers: it is not attached to, worn on or resting on anything.', True),
        ('implausible', 'The new {new} clearly does not belong in this scene (wrong season, place, context or wearer).',
         True),
        ('pasted', 'The new {new} looks pasted in: clearly sharper or blurrier than its surroundings, or lit or colored '
                   'unlike the rest of the photo.', False),
    ],
    'replace': [
        ('old_remains', 'The original {target} is still there in AFTER, unchanged or only recolored or retextured, '
                        'instead of being replaced by a different object.', True),
        ('drawn_on', 'The new {new} was drawn on, attached to or placed beside the original {target}, which is still '
                     'in place.', True),
        ('new_missing', 'No {new} is visible in AFTER where the {target} was.', True),
        ('wrong_size', 'The new {new} is clearly the wrong size for its place in the scene.', True),
        ('unsupported', 'The new {new} floats or hovers instead of standing, hanging or resting where it should.', True),
        ('host_damaged', 'Something around the replaced object is now cut off, broken or missing a part.', True),
        ('pasted', 'The new {new} looks pasted in: clearly sharper or blurrier than its surroundings, or lit or '
                   'colored unlike the rest of the photo.', False),
    ],
    'attribute': [
        ('not_changed', 'The {target} looks the same in AFTER as in BEFORE: the requested change ({change}) is not '
                        'visible.', True),
        ('incomplete', 'Inside the outlined region, part of the {target} keeps its old appearance (patches, rims or '
                       'missed parts).', True),
        ('spill', 'The change spills onto neighbouring things outside the {target}.', True),
        ('shape_changed', 'The {target} changed its shape, size or identity, not only the requested property.', True),
        ('half_object', 'The same object continues outside the outlined region with its old appearance, so the object '
                        'now looks half changed.', False),
        ('flat_fill', 'The new appearance looks like a flat paint fill that wiped out the texture, shading, lighting '
                      'or reflections the surface had.', False),
    ],
}

TOKEN_LIMIT = 120
DESCRIBE_LIMIT = 140  # extra tokens for the optional before/after description fields


def _slots(spec: dict, instruction: str) -> dict:
    spec = spec or {}
    attrs = spec.get('attributes') or []
    change = ', '.join(f"{a.get('property')} {a.get('value')}" for a in attrs if isinstance(a, dict)) or 'the requested change'
    # Questions put their own determiner before the payload ("No new {new}").
    payload = re.sub(r'^(?:a|an|the|some)\s+', '', str(spec.get('payload') or '').strip(), flags=re.I)
    return {'instruction': instruction, 'target': str(spec.get('scope') or spec.get('ref') or 'selected object').strip(),
            'new': payload or 'item', 'change': change}


def prompt(operation: str, spec: dict, instruction: str, *, describe: bool = False) -> str:
    slots = _slots(spec, instruction)
    lines = [HEADER.format(**slots)]
    if describe:
        lines.append('First write "before" and "after": what is at the outlined place in BEFORE and in AFTER, '
                     'within 20 words each. Then answer:')
    for key, question, _ in QUESTIONS[operation]:
        lines.append(f'- {key}: ' + question.format(**slots))
    lines.append('Return JSON only with these keys' + (' (two short strings, then booleans).' if describe
                                                      else ' and boolean values.'))
    return '\n'.join(lines)


def schema(operation: str, *, describe: bool = False) -> dict:
    keys = [k for k, _, _ in QUESTIONS[operation]]
    properties = {k: {'type': 'boolean'} for k in keys}
    if describe:
        properties = {'before': {'type': 'string', 'maxLength': 160}, 'after': {'type': 'string', 'maxLength': 160},
                      **properties}
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def token_limit(describe: bool = False, thinking: int = 0) -> int:
    return TOKEN_LIMIT + (DESCRIBE_LIMIT if describe else 0) + thinking


def judge(operation: str, response) -> dict:
    """{'pass', 'defects' (hard true), 'advisory' (soft true), 'answers'}; malformed answers fail."""
    if not isinstance(response, dict) or any(not isinstance(response.get(k), bool) for k, _, _ in QUESTIONS[operation]):
        return {'version': VERSION, 'pass': False, 'defects': ['invalid_verifier_response'], 'advisory': [],
                'answers': response if isinstance(response, dict) else {}}
    hard = [k for k, _, h in QUESTIONS[operation] if h and response[k]]
    soft = [k for k, _, h in QUESTIONS[operation] if not h and response[k]]
    verdict = {'version': VERSION, 'pass': not hard, 'defects': hard, 'advisory': soft,
               'answers': {k: response[k] for k, _, _ in QUESTIONS[operation]}}
    if isinstance(response.get('before'), str) and isinstance(response.get('after'), str):
        verdict['observed'] = {'before': response['before'], 'after': response['after']}
    return verdict


def _load_mask(path, size):
    mask = Image.open(path).convert('L')
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask) > 127


def zoom_box(source: np.ndarray, result: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    """Box over the selected region and the changes connected to it.

    Changes that do not touch the (slightly dilated) selection belong to other
    edits of a multi image and would only shrink the target in the zoom.
    """
    import cv2
    ys, xs = np.nonzero(mask)
    ext = max(xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())
    radius = int(np.clip(.15 * ext, 8, 64))
    centre = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    # Work in a window of +-3 extents: a component cut at its border spans more than the
    # 3-extent limit below either way, so the box equals the full-image computation.
    h, w = mask.shape
    reach = 3 * ext + radius + 2
    wx0, wy0 = int(max(0, centre[0] - reach)), int(max(0, centre[1] - reach))
    wx1, wy1 = int(min(w, centre[0] + reach + 1)), int(min(h, centre[1] + reach + 1))
    window = (slice(wy0, wy1), slice(wx0, wx1))
    changed = np.abs(source[window].astype(np.int16) - result[window].astype(np.int16)).max(axis=2) > 24
    changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    local = mask[window]
    near = cv2.dilate(local.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0
    count, labels = cv2.connectedComponents(changed, connectivity=8)
    touching = np.unique(labels[near & (changed > 0)])
    focus = local | np.isin(labels, touching[touching > 0])
    fy, fx = np.nonzero(focus)
    x0, y0, x1, y1 = fx.min() + wx0, fy.min() + wy0, fx.max() + 1 + wx0, fy.max() + 1 + wy0
    # A change component may run far along a large surface; never let it dwarf the target.
    limit = 3 * ext
    if x1 - x0 > limit:
        x0, x1 = int(max(x0, centre[0] - limit / 2)), int(min(x1, centre[0] + limit / 2))
    if y1 - y0 > limit:
        y0, y1 = int(max(y0, centre[1] - limit / 2)), int(min(y1, centre[1] + limit / 2))
    pad = max(48, int(.6 * max(x1 - x0, y1 - y0)))
    return (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(w, x1 + pad)), int(min(h, y1 + pad)))


def views(source_path: str, result_path: str, mask_path: str, directory: str | Path, *, max_edge: int = 1280,
          canvas: tuple[int, int] = (1536, 1024)) -> list[str]:
    """[full AFTER, BEFORE/AFTER zoom pair]; side by side or stacked, whichever shows the zoom larger."""
    import cv2
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.open(source_path).convert('RGB')
    result = Image.open(result_path).convert('RGB')
    if result.size != source.size:
        raise ValueError('verifier source/result size mismatch')
    mask = _load_mask(mask_path, source.size)
    if not mask.any():
        raise ValueError('empty verifier mask')
    box = zoom_box(np.asarray(source), np.asarray(result), mask)
    w, h = box[2] - box[0], box[3] - box[1]
    gap, label = 16, 34
    side = min((canvas[0] - gap) / 2 / w, (canvas[1] - label) / h)
    stacked = min(canvas[0] / w, (canvas[1] - gap - 2 * label) / 2 / h)
    horizontal = side >= stacked
    scale = min(max(side, stacked), 4.0)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    before = np.asarray(source.crop(box).resize(size, Image.LANCZOS)).copy()
    local = np.asarray(Image.fromarray(mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(before, contours, -1, (255, 220, 0), 2)
    after = result.crop(box).resize(size, Image.LANCZOS)
    if horizontal:
        pair = Image.new('RGB', (2 * size[0] + gap, size[1] + label), 'white')
        spots = [(0, label), (size[0] + gap, label)]
        texts = [(6, 8), (size[0] + gap + 6, 8)]
    else:
        pair = Image.new('RGB', (size[0], 2 * size[1] + 2 * label + gap), 'white')
        spots = [(0, label), (0, size[1] + gap + 2 * label)]
        texts = [(6, 8), (6, size[1] + gap + label + 8)]
    pair.paste(Image.fromarray(before), spots[0])
    pair.paste(after, spots[1])
    draw = ImageDraw.Draw(pair)
    draw.text(texts[0], 'BEFORE (selected region outlined)', fill=(0, 0, 0))
    draw.text(texts[1], 'AFTER', fill=(0, 0, 0))
    full = result.copy()
    full.thumbnail((max_edge, max_edge), Image.LANCZOS)
    paths = [str(directory / 'verifier_after.jpg'), str(directory / 'verifier_pair.jpg')]
    full.save(paths[0], quality=92)
    pair.save(paths[1], quality=92)
    (directory / 'verifier_box.json').write_text(json.dumps({'version': VERSION, 'box': list(box),
                                                             'layout': 'side' if horizontal else 'stacked'}))
    return paths


def job(job_id: str, source_path: str, result_path: str, member: dict, directory, *, describe: bool = False,
        thinking: int = 0) -> dict:
    """Audit-service job (VLLMAuditor.call_batch format)."""
    spec = member.get('instruction_spec') or {}
    operation = member.get('operation') or spec.get('operation')
    images = views(source_path, result_path, member.get('mask_v1') or member['mask'], directory)
    out = {'job_id': job_id, 'images': images,
           'prompt': prompt(operation, spec, member.get('instruction', ''), describe=describe),
           'output_schema': schema(operation, describe=describe), 'max_tokens': token_limit(describe, thinking),
           'enable_thinking': bool(thinking), 'image_max_edge': 1536}
    if thinking:
        out['thinking_token_budget'] = thinking
    return out


# Per-operation thinking for options['by_operation'] (run_dual_validation --audit-verifier split).
# On 939 labelled edits (2026-10-24), reasoning caught more visible bad removals (13 vs 9 of 17) and
# replacements (10 vs 7 of 23) at the same false rejects, and nothing extra for additions (4 vs 5 of 15);
# for colour edits it caught 2 of 7 bad while rejecting 3 of 152 good (0 and 0 without reasoning).
SPLIT = {'remove': 512, 'replace': 512, 'add': 0, 'attribute': 'off'}


def _settings(options: dict, operation: str) -> dict | None:
    """Job settings for one operation, or None when this operation is not verified."""
    thinking = (options.get('by_operation') or {}).get(operation, options.get('thinking') or 0)
    if thinking == 'off':
        return None
    return {'describe': bool(options.get('describe')), 'thinking': int(thinking)}


def contract(options: dict) -> dict:
    out = {'version': VERSION, 'describe': bool(options.get('describe')), 'thinking': int(options.get('thinking') or 0)}
    if options.get('by_operation'):
        out['by_operation'] = dict(sorted(options['by_operation'].items()))
    return out


def member_jobs(members: list[dict], source_path: str, results: list[str], directory, options: dict, *,
                prefix: str) -> list[dict]:
    """One verifier job per verified member; submitted together with the main audit job, so the two run
    in one batch. Members of an operation set to 'off' get no job."""
    directory = Path(directory)
    jobs = []
    for m, r in zip(members, results):
        settings = _settings(options, m.get('operation') or m['instruction_spec']['operation'])
        if settings is not None:
            jobs.append(job(f"{prefix}::verify::{m['unit_id']}", source_path, r, m, directory / f"verifier_{m['unit_id']}",
                            **settings))
    return jobs


def record(members: list[dict], audits: dict, outputs: list[dict]) -> None:
    """Attach verdicts; only a member the main audit passed can be failed by its verdict.

    ``audits`` maps unit_id to the normalized audit row and is updated in place. A verdict
    for a member the main audit already rejected is kept for inspection and changes nothing.
    ``outputs`` answer the jobs of member_jobs, in order; a member without a job (operation
    not verified) is marked skipped.
    """
    by_id = {}
    for output in outputs:
        uid = str(output.get('job_id', '')).rpartition('::verify::')[2]
        if '::verify::' not in str(output.get('job_id', '')) or uid in by_id:
            raise RuntimeError('verifier job mismatch')
        by_id[uid] = output
    if set(by_id) - {m['unit_id'] for m in members}:
        raise RuntimeError('verifier output count mismatch')
    for member in members:
        output = by_id.get(member['unit_id'])
        if output is None:
            audits[member['unit_id']]['verifier'] = {'version': VERSION, 'pass': True, 'skipped': True,
                                                     'defects': [], 'advisory': [], 'answers': {}}
            continue
        response = None if output.get('finish_reason') == 'length' else output.get('semantic_audit')
        verdict = judge(member.get('operation') or member['instruction_spec']['operation'], response)
        audit = audits[member['unit_id']]
        audit['verifier'] = verdict
        if audit.get('decision') == 'pass' and not verdict['pass']:
            audit['decision'] = 'fail'
            audit['failure_reasons'] = list(audit.get('failure_reasons') or []) + ['verifier_' + d for d in verdict['defects']]


def apply(members: list[dict], audits: dict, source_path: str, results: list[str], auditor, directory, options: dict,
          *, prefix: str) -> int:
    """Verify, after the fact, each member the audit passed (serial variant; the pipeline submits
    member_jobs with the main audit instead). Returns the number of verifier calls."""
    from .group_audit import _call_batched
    pending = [(m, r) for m, r in zip(members, results) if audits[m['unit_id']].get('decision') == 'pass']
    if not pending:
        return 0
    jobs = member_jobs([m for m, _ in pending], source_path, [r for _, r in pending], directory, options, prefix=prefix)
    record([m for m, _ in pending], audits, _call_batched(auditor, jobs))
    return len(jobs)
