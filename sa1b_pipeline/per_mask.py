"""One model request per source mask; deterministic instruction-protocol checks."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import re
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .geometry import sha256, load_mask, mask_geometry
from .instructions import build_unit
from .io import load_candidates, write_jsonl
from .schema import OPERATIONS, SelectionCase, validate_case
from .reference_diagnostics import reference_collisions
from .instruction_style import INSTRUCTION_STYLE

VERSION = 'per-mask-single-call-v2.4-concise2-directadd1-assigned1'

SINGLE_CALL_PROMPT = '''Design ONE natural photo-edit request for ONE supplied mask. Identify the selected instance and design its edit in this same response.

TWO INPUT IMAGES: Image 1 is the original full scene with a thin CYAN RECTANGLE locating the target bounding box; the rectangle is only a locator, NOT the mask. Image 2 is ONE enlarged context crop: selected pixels retain their ORIGINAL color/brightness; unselected context is dimmed, and a MAGENTA/WHITE contour is drawn just OUTSIDE the mask. Do not read guide colors as source color. The bright ORIGINAL pixels inside this contour are the only target, even if they are a green leaf, plain ground or an uninteresting fragment. Identify their actual color and texture before naming them; do not let an orange flower or another prominent neighbor distract you. For background masks, the selected region can surround excluded foreground objects. Use image 1 to locate this crop and compare same-category neighbors. Left/right means image-left/right.

BINDING AND REFERENCE (highest priority): Describe exactly the selected object, part, surface, or printed motif. A petal mask names one petal, a door-panel mask names that panel, a ground mask names that patch of ground; do not expand to flower, car, or a nearby object. A recognizable partly occluded object can be named as that object. Printed flowers/leaves are motifs on fabric, not live plants. Smallness, blur, imperfect borders and occlusion alone do NOT make a mask unusable.
The final ref_phrase is ONE minimal noun phrase identifying this instance in the UNMARKED full image. Start with the actual noun; if unique, stop. Otherwise add the shortest verified adjective, possessive, ordinal or landmark relation that excludes the closest similar instance. Examples only when true and sufficient: 'the middle cup', 'the leftmost window', 'the right man\'s shirt', 'the petal above the lowest leaf'. Do not append a separate location clause, repeat the noun, or routinely combine host, color and position. Retain another modifier only if a rival still matches. Generic 'upper-left petal' fails if two petals match. Use an ordinal only within an explicitly defined visible group and counting direction. Never guess counts, hidden function or anatomical left/right. Necessary distinctions must appear in ref_phrase, not just evidence. Do not refer to annotation colors, masks, crop coordinates or 'selected'. Refine an ambiguous draft internally; if still unresolved, report a reference limitation rather than blaming the source mask.


OPERATION DESIGN: Try the assigned preferred type first; do not invent an unreasonable request to satisfy a quota. Check actual target and ordinary scene use.
- add: preserve this host and add a conventional, useful/plausible item at a site belonging to it. New item may extend beyond the host silhouette. A bottle in a backpack pocket or a bracelet around a wrist is natural without requiring microscopic contact evidence. Do not place physical dew on a printed flower, glue arbitrary ornaments to tiny car parts, obstruct equipment, or use an unselected neighboring support. A new printed/embroidered motif must explicitly be described as printed/embroidered, not as a real object. Put item/count only in payload; placement contains location. If the mask already selects a wrist, placement.site is EMPTY (around the wrist, not around the wrist of the wrist).
- replace: replace only the selected identity with a visibly different SHAPE/DESIGN serving the same specific function and fitting existing contacts. A different style of bag, chair, handle or planter is reasonable. Changing only color/material/finish/tint is ATTRIBUTE, even if worded 'replace'. Name the visible design difference in payload; replacement_changes_form must be true. Do not replace a vent with a solid plate, a partial car panel with a whole car, or a petal with an unrelated decoration. Preserve unselected contents/flowers of containers and their support.
- remove: remove exactly the named mask target while leaving a natural intact scene. Do NOT remove a person's arm, hand, wrist, leg, face or other attached anatomy. Do NOT remove structural supports holding unselected objects, or cut out fabric when the intended request is erasing its printed motif. For a printed motif, name the motif in both ref_phrase and scope, preserving the underlying fabric. A bag, unwanted background object or small flower can be removed despite blur/occlusion. Set removal_is_natural=false for an anatomically/physically nonsensical removal; set removal_preserves_unselected=false if it affects unselected contents/supports. With multiple allowed types choose a reasonable alternative in this SAME response. With only remove allowed, return feasible=false for such a target.
- attribute: For human skin use subtle texture retouching rather than recoloring a single limb to a mismatched skin tone. Never tint over a helmet's viewing window with opaque paint. Choose a plausible visibly DIFFERENT color/material/finish/pattern/texture/text/tint/opacity on exactly the named surface. Keep unrequested geometry, markings and other components unchanged. Do not recolor a whole person to edit clothing. Do not request the already present color or transparency. Opacity values are exactly transparent, translucent, opaque (never numbers or 'clear'); tint changes color while retaining transparency. Text values are exact strings. A visible motif can change color while the fabric stays unchanged.

OUTPUT PROTOCOL 1.0: Code renders Ref/No-ref from shared structured slots. One operation, one concise sentence. Aim ref<=26 words and no-ref<=16, but retain necessary reference distinctions. scope is the specific short noun without article, source color or position. target_part is EMPTY when the mask already selects that part; do not request an unlocalized smaller subpart unless a trusted execution region is supplied. Never invent masks/paths. remove_mode=whole means the entirety of the named target, not its parent. remove/attribute payload is empty. attribute requires attributes=[{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}], others attributes=[]. tint/opacity stand alone. add uses placement={"relation":"on|in|around|to|...","site":"optional part name without article","anchor_mode":"support"}; do not repeat the target noun in site. A nonempty site requires a spatial relation other than to. Relative beside/above/below/near placement requires a trusted execution region and anchor_mode=reference. Non-add placement=null. qualifier is empty or a short preservation clause, never another edit.

Silently check binding, uniqueness of the FINAL reference, naturalness, type and scope before responding. No reasoning trace, no separate reviewer, no alternative plans. feasible=false only when you cannot produce a clear reference or any reasonable allowed edit; state the specific instruction limitation in evidence. Return JSON only. Keep observation/evidence fields brief.
{"selected_pixels":"exact selected identity/extent","unselected_neighbors":"nearest competing instances","target_kind":"object|part|surface|printed_motif|anatomy","removal_preserves_unselected":true,"removal_is_natural":true,"replacement_changes_form":false,"feasible":true,"object_id":"...","category":"target noun","operation":"...","ref_phrase":"the ...","scope":"...","source_color":null,"payload":"...","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","skipped_operations":[],"evidence":"one brief factual sentence"}
'''


SINGLE_CALL_PROMPT += "\n" + INSTRUCTION_STYLE + "\n"

def single_call_prompt(obj, priorities):
    context = {
        'object_id': obj.object_id,
        'preferred_operation': priorities[0],
        'operation_priority': priorities,
        'resolved_execution_region_available': bool(obj.metadata.get('execution_mask_path')),
    }
    if len(priorities) == 1:
        return SINGLE_CALL_PROMPT + '\nTask: ' + json.dumps(context, ensure_ascii=False) + '''
ASSIGNED TYPE: Only the single operation in operation_priority is allowed. Return one natural in-scope edit of that type, or feasible=false with one brief concrete limitation. Never change types or disguise an attribute edit as replace/add. An unsuitable assigned operation is not a bad-mask verdict. Keep the shortest unambiguous reference. Do not remove attached anatomy or invent unnatural accessories to fill a quota. There is no retry or alternative plan.'''
    return SINGLE_CALL_PROMPT + '\nTask: ' + json.dumps(context, ensure_ascii=False) + '''
The operation priority is assigned BEFORE this call to balance accepted types. preferred_operation is a FIRST CHOICE, not a prohibition on the other listed types. Try it first. If it genuinely cannot yield a natural in-scope edit, use the first suitable type in operation_priority and record each skipped preceding type as {"operation":"...","reason":"concrete semantic reason"}. This happens within this ONE response; do not invent a reason merely to choose an easier type. With a single allowed type do not switch types. For example, if removal of a filled planter body is unsafe, choose a plausible color change of that body or replace it with a compatible planter while preserving its flowers. Do NOT return feasible=false simply because removal was the first choice. Return one plan only, not alternatives.'''


def _outer_mask_band(mask, width):
    """Same square max-filter contour, using fast morphology when available."""
    arr = np.asarray(mask)
    try:
        import cv2
    except ImportError:
        expanded = np.asarray(mask.filter(ImageFilter.MaxFilter(width)))
    else:
        expanded = cv2.dilate(arr, np.ones((width, width), dtype=np.uint8),
                              borderType=cv2.BORDER_REPLICATE)
    return (expanded > 0) & (arr == 0)


def selected_pixel_evidence(source, mask, max_samples=8192):
    """Small deterministic appearance summary from unannotated selected pixels."""
    rgb = np.asarray(source.convert('RGB'))
    selected = np.asarray(mask) > 0
    if selected.shape != rgb.shape[:2] or max_samples < 1 or not selected.any():
        raise ValueError('nonempty aligned mask and positive sample bound required')
    pixels = rgb[selected]
    count = len(pixels)
    if count > max_samples:
        pixels = pixels[np.linspace(0, count - 1, max_samples, dtype=np.int64)]
    bins = pixels.astype(np.int32) // 32
    codes = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    counts = np.bincount(codes, minlength=512)
    top = np.argsort(-counts, kind='stable')[:3]
    palette = [{'rgb': np.rint(pixels[codes == code].mean(axis=0)).astype(int).tolist(),
                'sample_fraction': round(float(counts[code] / len(pixels)), 3)}
               for code in top if counts[code]]
    return {'selected_pixels': count, 'sampled_pixels': len(pixels),
            'median_rgb': np.rint(np.median(pixels, axis=0)).astype(int).tolist(),
            'frequent_rgb_bins': palette,
            'basis': 'raw source pixels within original mask; includes lighting, not semantic colour labels'}


def single_call_views(candidate, obj, directory, *, crop_padding_factor=.45,
                      crop_min_fraction=0., crop_min_pixels=16,
                      crop_max_edge=896, scene_max_edge=1280, source_cache=None):
    """Exactly two model inputs: scene locator and one visibly annotated crop."""
    source = (source_cache.get(candidate.image_path) if source_cache is not None
              else Image.open(candidate.image_path).convert('RGB'))
    mask = load_mask(obj.__dict__, source.size)
    geometry = mask_geometry(mask)
    settings = (crop_padding_factor, crop_min_fraction, crop_min_pixels, crop_max_edge, scene_max_edge)
    if min(crop_padding_factor, crop_min_fraction, crop_min_pixels) < 0 or min(crop_max_edge, scene_max_edge) < 1:
        raise ValueError('invalid planning view dimensions')
    settings_tag = b'' if settings == (.45, 0., 16, 896, 1280) else repr(settings).encode()
    digest = hashlib.sha256(VERSION.encode() + settings_tag + source.tobytes() + mask.tobytes()).hexdigest()
    root = Path(directory) / digest
    root.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = geometry['bbox']
    padding = round(max(crop_min_pixels, max(x1-x0, y1-y0)*crop_padding_factor,
                        max(source.size)*crop_min_fraction))
    crop_box = [max(0,x0-padding), max(0,y0-padding), min(source.width,x1+padding), min(source.height,y1+padding)]
    paths = [root/'scene_locator.png', root/'annotated_crop.png']
    mask_path = root/'source_mask.png'
    if not all(p.exists() for p in [*paths, mask_path]):
        binary = Image.fromarray(mask.astype('uint8')*255)
        binary.save(mask_path, compress_level=1)
        scene = source.copy()
        scene.thumbnail((scene_max_edge,scene_max_edge), Image.Resampling.LANCZOS)
        sx, sy = scene.width/source.width, scene.height/source.height
        ImageDraw.Draw(scene).rectangle([x0*sx,y0*sy,(x1-1)*sx,(y1-1)*sy], outline=(0,230,255), width=2)
        scene.save(paths[0], compress_level=1)
        crop = source.crop(crop_box)
        factor = crop_max_edge/max(crop.size)
        size = (max(1,round(crop.width*factor)), max(1,round(crop.height*factor)))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        cropped_mask = binary.crop(crop_box).resize(size, Image.Resampling.NEAREST)
        # Preserve EVERY selected pixel after resizing. Annotation is outside
        # the mask so that even one-pixel-wide targets keep their source color.
        original_crop = crop.copy()
        crop = Image.blend(crop, Image.new('RGB', size, 'black'), .40)
        crop.paste(original_crop, mask=cropped_mask)
        outer = _outer_mask_band(cropped_mask, 11)
        ring = _outer_mask_band(cropped_mask, 5)
        crop.paste('white', mask=Image.fromarray(outer.astype('uint8')*255))
        crop.paste((255,0,180), mask=Image.fromarray(ring.astype('uint8')*255))
        panel = Image.new('RGB', (size[0], size[1]+38), 'white')
        panel.paste(crop, (0,38))
        ImageDraw.Draw(panel).text((8,8), 'TARGET INSIDE MAGENTA CONTOUR', fill=(180,0,120), font=ImageFont.load_default(size=16))
        panel.save(paths[1], compress_level=1)
    return list(map(str,paths)), geometry, str(mask_path), digest, crop_box


def _explicit_human_part(raw):
    """Catch an explicit human body reference even when the model mislabels kind.

    Deliberately narrow: this is a textual contradiction check, not a visual
    anatomy classifier. Furniture/clock parts and motifs remain editable.
    """
    if raw.get('target_kind') == 'printed_motif':
        return False
    parts = r'(?:arm|forearm|hand|wrist|leg|foot|face|head|finger|toe)s?'
    scope = str(raw.get('scope', '')).lower().strip()
    if not re.fullmatch(r'(?:the |left |right |bare |exposed |upper |lower )*' + parts, scope):
        return False
    ref = str(raw.get('ref_phrase', '')).replace('\u2019', "'").lower()
    people = r'(?:person|man|woman|boy|girl|child|worker|wearer|adult|baby)'
    modifiers = r'(?:(?:left|right|bare|exposed|visible|upper|lower)\s+)*'
    # Permit visual qualifiers without accepting arbitrary nouns (clock,
    # chair, mannequin, etc.) between the body part and its human owner.
    person_modifiers = r'(?:(?:blurred|blurry|out-of-focus|background|foreground|young|elderly|older|adult)\s+)*'
    return bool(re.search(r'\b' + people + r"'s\s+" + modifiers + parts + r'\b', ref)
                or re.search(r'\b' + parts + r'\s+of\s+the\s+' + person_modifiers
                             + people + r"\b(?!'s\b)", ref))


def validate_single_plan(raw, object_id, priorities):
    if not isinstance(raw, dict) or raw.get('_parse_error'):
        raise ValueError('malformed_model_response')
    if raw.get('feasible') is not True:
        raise ValueError('no_suitable_instruction: ' + str(raw.get('reason') or raw.get('evidence') or 'unspecified'))
    if str(raw.get('object_id')) != object_id:
        raise ValueError('object_id_mismatch')
    for key in ('category', 'ref_phrase', 'scope'):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError('missing_source_description: ' + key)
    # Reject an unusable instruction, never the source mask. No model retry.
    if re.search(r'\b(?:selection|selected|(?:in|inside|within|shown in)\s+the\s+(?:mask|crop)|bounding\s+(?:box|rectangle)|'
                 r'(?:cyan|magenta|annotated|highlighted)\s+(?:box|rectangle|contour|outline)|'
                 r'(?:cyan|magenta|pink)[ -]+(?:marked|filled|outlined)\b|'
                 r'(?:shown|displayed|pictured)\s+(?:in|inside)\s+(?:the|an)\s+inset\b|'
                 r'(?:shown|displayed|pictured)\s+(?:in|inside)\s+(?:the|an)\s+'
                 r'(?:left|right|upper|lower|top|bottom)(?:[ -]+side)?[ -]+inset|'
                 r'outlined\s+in\s+(?:cyan|magenta))\b',
                 raw['ref_phrase'], re.I):
        raise ValueError('reference_depends_on_annotation')
    operation = raw.get('operation')
    if operation not in priorities:
        raise ValueError('operation_not_allowed')
    if operation == 'remove' and (raw.get('target_kind') == 'anatomy'
                                  or raw.get('removal_is_natural') is False
                                  or _explicit_human_part(raw)):
        raise ValueError('unnatural_removal_of_target')
    if operation == 'replace' and raw.get('replacement_changes_form') is False:
        raise ValueError('replacement_requires_visible_design_change')
    if operation == 'remove' and raw.get('removal_preserves_unselected') is False:
        raise ValueError('remove_would_change_or_unsupported_unselected_objects')


def plan_all_masks(candidates, model, *, work_dir, operation_policy='balanced', counts=None,
                   attempt_counts=None, output_path=None, allow_operation_fallback=False):
    """One request for EVERY decodable mask; no semantic audits/retries/geometry gates."""
    if operation_policy not in ('balanced', *OPERATIONS):
        raise ValueError('unknown operation policy')
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = counts if counts is not None else Counter()
    attempt_counts = attempt_counts if attempt_counts is not None else Counter()
    audit_file = work_dir / 'audit.jsonl'
    cases, seen = [], set()

    def record(row):
        with audit_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    for candidate in candidates:
        candidate.image_sha256 = sha256(candidate.image_path)
        with Image.open(candidate.image_path) as source_image:
            image_size = list(source_image.size)
        for index, obj in enumerate(candidate.objects, 1):
            key = (candidate.source_id, obj.object_id)
            if key in seen:
                raise ValueError(f'duplicate source/object key: {key}')
            seen.add(key)
            base = {'source_id': candidate.source_id, 'object_id': obj.object_id, 'planner_version': VERSION}
            try:
                paths, geometry, mask_path, digest, crop_box = single_call_views(candidate, obj, work_dir/'views')
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                record({**base, 'stage': 'terminal', 'status': 'rejected', 'reason': 'unreadable_or_empty_mask', 'detail': str(exc)})
                continue
            priorities = sorted(OPERATIONS, key=lambda op: (counts.get(op, 0), attempt_counts.get(op, 0), OPERATIONS.index(op))) if operation_policy == 'balanced' else [operation_policy]
            if not allow_operation_fallback:
                priorities = priorities[:1]
            prompt = single_call_prompt(obj, priorities)
            attempt_counts[priorities[0]] = attempt_counts.get(priorities[0], 0) + 1
            started = time.perf_counter()
            # Exactly one model invocation. Errors are recorded, never retried here.
            raw = model.generate_multimage([(paths, prompt)])[0]
            record({**base, 'stage': 'single_call_plan', 'images': paths, 'prompt': prompt,
                    'result': raw, 'seconds': time.perf_counter()-started, 'binding_hash': digest,
                    'preferred_operation': priorities[0], 'operation_priority': priorities})
            try:
                validate_single_plan(raw, obj.object_id, priorities)
                raw = {**raw, 'unit_index': index}
                unit = build_unit(raw, {**obj.__dict__, 'category': raw['category']}, mask_path, geometry, [])
                single_source = replace(candidate, objects=[replace(obj, mask_path=mask_path)],
                    provenance={**candidate.provenance, 'planning_mode': VERSION,
                                'selection_status': 'per_mask_planned', 'image_size': image_size})
                case = SelectionCase(
                    'sa1b-'+candidate.source_id+'-mask-'+hashlib.sha256(obj.object_id.encode()).hexdigest()[:12],
                    single_source, [unit], {'target_count': 1, 'target_features': [geometry]},
                    {'planning_mode': VERSION, 'model_plan': raw, 'model_id': getattr(model, 'model_id', 'unknown'),
                     'preferred_operation': priorities[0], 'operation_priority': priorities,
                     'binding_hash': digest, 'crop_box': crop_box, 'model_calls': 1,
                     'semantic_audit_performed': False, 'joint_multi_target': False,
                     'operation_fallback': unit.operation != priorities[0]})
                validate_case(case, 1, 1)
            except (ValueError, TypeError, KeyError) as exc:
                record({**base, 'stage': 'terminal', 'status': 'rejected', 'reason': str(exc)})
                continue
            cases.append(case)
            counts[unit.operation] = counts.get(unit.operation, 0) + 1
            if output_path:
                with Path(output_path).open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(case.to_dict(), ensure_ascii=False) + '\n')
            record({**base, 'stage': 'terminal', 'status': 'accepted', 'operation': unit.operation, 'case_id': case.case_id})
    # A review signal, never a source-mask rejection or another model call.
    collisions = reference_collisions(case.to_dict() for case in cases)
    write_jsonl(work_dir / 'reference_collisions.jsonl', collisions)
    return cases

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--operation-policy',choices=['balanced',*OPERATIONS],default='balanced')
    parser.add_argument('--allow-operation-fallback',action='store_true',
                        help='Historical planning experiment only; can produce a strongly skewed pool')
    parser.add_argument('--max-candidates',type=int,help='explicit source-level pilot bound; all masks of each included source are processed')
    parser.add_argument('--backend',choices=['hf','vllm'],default='hf')
    parser.add_argument('--model-id',default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')
    parser.add_argument('--tensor-parallel-size',type=int,default=1)
    parser.add_argument('--max-new-tokens',type=int,default=1000)
    args = parser.parse_args(argv)
    root = args.work_dir or args.output.with_suffix('.per_mask')
    if args.output.exists() or (root/'audit.jsonl').exists():
        raise ValueError('Use a new output/work directory; refusing stale identity decisions or duplicate appends')
    candidates = load_candidates(args.manifest)
    if args.max_candidates is not None:
        if args.max_candidates < 1: parser.error('--max-candidates must be positive')
        candidates = candidates[:args.max_candidates]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_jsonl(args.output,[])
    if args.backend == 'hf':
        from .hf_mllm import HFSelector
        model = HFSelector(model_id=args.model_id,max_new_tokens=args.max_new_tokens)
    else:
        from .mllm import VLLMSelector
        model = VLLMSelector(args.model_id,tensor_parallel_size=args.tensor_parallel_size,max_new_tokens=args.max_new_tokens)
    counts, attempts = Counter(), Counter()
    started = time.perf_counter()
    try:
        cases = plan_all_masks(candidates,model,work_dir=root,operation_policy=args.operation_policy,
                               counts=counts,attempt_counts=attempts,output_path=args.output,
                               allow_operation_fallback=args.allow_operation_fallback)
        summary = {'version':VERSION,'sources':len(candidates),'input_masks':sum(len(c.objects) for c in candidates),
            'operation_mode':'fallback' if args.allow_operation_fallback else 'assigned',
            'semantic_audit_performed':False,'max_calls_per_decodable_mask':1,'accepted_masks':len(cases),'accepted_operations':dict(counts),'design_attempts':dict(attempts),
            'seconds_excluding_model_load':time.perf_counter()-started,'independent_human_review':False,
            'target_images_generated':False,'joint_multi_target':False}
        (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:
        if hasattr(model,'close'): model.close()


if __name__ == '__main__':
    main()
