"""Native ScaleEdit prompts and per-edit-unit canvas policy, without old masks."""
from __future__ import annotations

import json
import re
from PIL import Image

from scaleedit.download import LOCAL_TASKS
from scaleedit import quality, scene
from scaleedit.mask.checklist import observation_prompt, grounding_prompt, strict_json, parse_checklist_grounding
from scaleedit.mask.grounding import parse_change_observation

METHOD = 'scaleedit-local-edit'
TASK_KINDS = {
    'object_addition': 'add', 'object_removal': 'remove', 'object_replacement': 'replace',
    'color_change': 'color', 'material_change': 'color', 'action_editing': 'motion',
}
TEXT_TASKS = frozenset(task for task in LOCAL_TASKS if 'text_editing' in task)


def task_name(value):
    return re.sub(r'[\s-]+', '_', str(value or '').strip().lower())


def type_guidance(task):
    if task in TEXT_TASKS:
        return ('When the instruction edits written text, verify the exact old and new characters, '
                'their selected location, and the preservation of other text and the carrier. '
                'A plausible but incorrectly spelled replacement fails completion. The reviewed task '
                'label may be inaccurate: a line/pattern or object edit is not necessarily a text edit.')
    if task == 'material_change':
        return ('Identify the visible source and requested target material on the selected object or surface. '
                'Expected changes in texture, reflections and shading are allowed; a lighting-only drift '
                'without the requested material transition is not completion.')
    if task == 'size_change':
        return ('The selected object must change size relative to unchanged scene anchors. '
                'Whole-image zooming or camera reframing does not satisfy a local size change.')
    if task == 'count_change':
        return ('Verify the requested addition or removal of instances in the selected group. '
                'Do not mistake occlusion or a changed camera view for a count change.')
    if task in TASK_KINDS:
        return quality.TYPE_GUIDANCE[TASK_KINDS[task]]
    return ('Use the instruction to define the intended local changes, including every subgoal of '
            'a compound edit. Judge visible evidence; do not infer hidden reasoning outcomes. '
            'Changes necessary for the requested action or interaction are allowed.')


def quality_prompt(task, instruction):
    return (quality.QUALITY_PROMPT
            .replace('__EDIT_TYPE__', task).replace('__INSTRUCTION__', instruction)
            .replace('__TYPE_GUIDANCE__', type_guidance(task))
            + '\nThe reviewed task label is advisory. The instruction and actual image pair define the operation.\n')


def scene_prompt(task, instruction):
    return (scene.SCENE_PROMPT.replace('__EDIT_TYPE__', task).replace('__INSTRUCTION__', instruction)
            + '\nFor text, a unique word or title alone is not a difficult referential selection. '
            'Judge the actual instruction, not just the reviewed task label.\n')


def edit_units_prompt(task, instruction):
    prompt = observation_prompt(task, instruction)
    prompt = prompt.replace('"layout":"single or nearby_group"',
                            '"layout":"single or nearby_group","geometry":"object or text"')
    return prompt + '''
ScaleEdit task labels can be inaccurate: follow the actual instruction and image pair.
For each unit, geometry=text ONLY for edited written characters or a compact glyph/logo block.
Quote the exact changed substring in each text ref, and put its carrier in the location.
Exclude unchanged neighboring words/lines. Pattern lines, material and surface changes
are geometry=object; name the changed object/surface, not an abstract pattern.
An addition has an empty source_ref; removal has an empty target_ref. A changed existing
object/part has both refs. In compound/count edits, make this distinction for EACH unit.
Regular rows of distinct manufactured items or edited parts are separate units, not particles.
Do not merge their individual caps, buttons or panels into a single enclosing object group.
'''


def parse_edit_units(text):
    raw = strict_json(text)
    if not isinstance(raw, dict) or not isinstance(raw.get('edits'), list):
        raise ValueError('Expected an edits array')
    normalized = parse_change_observation(text)
    raw_units = [unit for edit in raw['edits'] for unit in edit['units']]
    if len(raw_units) != len(normalized['changes']):
        raise ValueError('Edit-unit identity mismatch')
    for i, (unit, original) in enumerate(zip(normalized['changes'], raw_units)):
        geometry = original.get('geometry')
        if geometry not in {'object', 'text'}:
            raise ValueError('Each unit requires geometry=object or text')
        unit.update(change_id=i, geometry=geometry,
                    image_side='source' if unit['source_ref'] else 'target')
    return normalized


def side_context(observation, side):
    # A changed existing object is source-only, even if its target appearance differs.
    return {'changes': [unit for unit in observation['changes'] if unit['image_side'] == side]}


def locate_prompt(observation, side):
    context = side_context(observation, side)
    text_ids = [u['change_id'] for u in context['changes'] if u['geometry'] == 'text']
    return grounding_prompt(context, side) + (f'''
Text change_ids: {text_ids}.
For EACH text unit, also return polygon_2d: four [x,y] corners in perimeter order,
normalized to [0,1000], tightly following the orientation of ONLY the changed characters.
Use the change description to identify the exact substring; exclude unchanged neighboring
words/lines and the carrier. The quadrilateral may be rotated/perspective-skewed; bbox_2d
must enclose it. Example field: "polygon_2d":[[100,200],[400,250],[390,300],[90,250]].
Non-text units need only bbox_2d. Not-visible text has bbox_2d=null and no polygon.
''' if text_ids else '')


def parse_located_units(text, observation, side):
    boxes, unresolved = parse_checklist_grounding(text, observation, side)
    raw = strict_json(text)
    raw = raw['changes'] if isinstance(raw, dict) else raw
    raw = {item['change_id']:item for item in raw}
    text_ids = {u['change_id'] for u in observation['changes'] if u['geometry']=='text'}
    for box in boxes:
        if box['change_id'] not in text_ids:
            continue
        if sum(b['change_id']==box['change_id'] for b in boxes) != 1:
            raise ValueError('Split separate text blocks into distinct edit units')
        polygon = raw[box['change_id']].get('polygon_2d')
        if not isinstance(polygon,list) or not 4<=len(polygon)<=8:
            raise ValueError('Each located text unit requires polygon_2d with 4-8 vertices')
        import math
        if any(not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float)
               or not math.isfinite(v) or not 0<=v<=1000 for v in p) for p in polygon):
            raise ValueError('Invalid polygon coordinates')
        cross = []
        n=len(polygon)
        for i in range(n):
            a,b,c = polygon[i],polygon[(i+1)%n],polygon[(i+2)%n]
            cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
        if not (all(v>0 for v in cross) or all(v<0 for v in cross)):
            raise ValueError('Text polygon must be convex, nonempty and in perimeter order')
        # Same-turn signs alone do not exclude self-intersecting star polygons.
        for i in range(n):
            a,b=polygon[i],polygon[(i+1)%n]
            for j in range(i+1,n):
                if j in {i,(i+1)%n} or (j+1)%n==i:
                    continue
                c,d=polygon[j],polygon[(j+1)%n]
                orient=lambda p,q,r:(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
                if orient(a,b,c)*orient(a,b,d)<=0 and orient(c,d,a)*orient(c,d,b)<=0:
                    raise ValueError('Text polygon edges must not intersect')
        x1,y1,x2,y2 = box['bbox_2d']
        if any(not x1-1<=x<=x2+1 or not y1-1<=y<=y2+1 for x,y in polygon):
            raise ValueError('Text polygon must be enclosed by bbox_2d')
        box['polygon_2d'] = polygon
    return boxes, unresolved


def conversation(images, prompt):
    content = []
    for label, image in images:
        content.extend([{'type': 'text', 'text': label}, {'type': 'image', 'image': image.convert('RGB')}])
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]


def mask_kind(task, unit):
    if unit['image_side'] == 'target':
        return 'add'
    if not unit['target_ref']:
        return 'remove'
    return TASK_KINDS.get(task, 'motion') if TASK_KINDS.get(task) != 'add' else 'replace'


def parse_quality(text):
    return quality.normalize_quality_assessment(quality.extract_json_object(text))


def parse_scene(text):
    return scene.parse_scene_response(text)
