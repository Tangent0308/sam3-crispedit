"""Visually check a proposed action against its exact source mask before editing.

This is not output-image QA. It may shorten/correct a plan but cannot silently
expand the mask, change the task type, or authorize external dependencies.
"""
import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.generate_samtok_plan import (
    allowed_mask_policies, ambiguous_anatomical_side, mask_geometry_hint,
    parse_json_object, protected_dependency_conflicts, write_jsonl,
)
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
import utils.vlm_utils as vlm


TYPE_CHECKS = {
    'add': 'The mask is a HOST, not the footprint of the new item. The attachment can occupy a small visible part inside that host; it need not cover the whole mask. Check that the proposed attachment is a real visible surface or garment part. If the named part does not exist, revise to an actually visible nearby part of the SAME selected host, keeping the new item. Reject only if no coherent attachment exists. The new item must be absent and physically supported.',
    'remove': 'Every visible part of the named removal unit must be INSIDE. Trace any thin extension beyond the outline. If a separate item resting in/on or held by the target would be left behind, reject; do not claim it is covered and do not remove it by changing the wording. Do not confuse the supporting background with a dependent item.',
    'replace': 'The old object must be fully selected. Reject if held/carried objects or a required external contact would be left unsupported. Protecting such an item does not solve the interaction. The replacement must fit the existing physical support without inventing another holder or limb.',
    'attribute': 'The mask is the MAXIMUM allowed extent. A valid edit may change a smaller material surface inside it. NEVER broaden a specific material edit to recolor the whole silhouette merely because the mask includes other components. Keep a valid narrow instruction and use surface refinement with a query naming that exact material/part, not its whole owner. Conversely, a narrow selected strip must NOT be called the whole repeated group; name its precise level and part. For a mixed-material front face choose a visible paintable body panel and use surface refinement; reject only if no coherent surface can be isolated. Keep the proposed attribute value.',
}


def scope_prompt(row):
    feedback = row.get('_scope_feedback', '')
    return f'''Check source-mask/action agreement. No edited image exists yet.
Image 1: unmarked full photograph. Image 2: the same photograph cropped with context; the selected pixels are ONLY inside the black/white contour. The blue leader from the external TARGET INSIDE label stops at the selected boundary. Holes and everything outside the contour are NOT selected. Outline, leader and label are annotations, not objects or scene colors.
{row.get('mask_geometry_hint', '')}
The proposal below may have misread the outline. Do not trust its object coverage, body-part names or spatial claims. Ignore the original dataset's broad object label. First trace the actual boundary, then identify the selected physical unit, then compare the proposal. An adjacent item belongs to the selected object only if its visible attachment/contact supports that conclusion; proximity alone is not ownership. Occlusion is not a missing body part: judge all actually visible target pixels.
Type: {row['task_type']}
Proposed instruction: {row['editing_instruction']}
{TYPE_CHECKS[row['task_type']]}
Keep the proposal's valid FULL-image instance locator, correcting it only if visually wrong; do not drop it while shortening the command. Keep the exact part. Do not guess anatomical left/right, including garment sides; use a visible action or neighbor relation instead. Do not assume a specialized garment feature exists just because the person wears a shirt. Prefer a plainly visible surface on the SAME host when the proposed feature is doubtful. Do not infer a functional object category from shape alone; use a factual shape/material description if the purpose is uncertain. Do not describe default preservation. No coordinate, mask, image or annotation language in the instruction. Do not invent a new task or expand the allowed region to rescue a proposal. If the wording is already valid, preserve it verbatim. Use revise, not accept, for ANY change in wording or requested surface. If your reason says only a strip or segment is selected, those same scope words MUST appear in the corrected instruction; a reason does not repair a still-overbroad command.
Return JSON with:
- reason: one short visual observation of the actual boundary, visible contact or specific mismatch, BEFORE deciding.
- decision: accept, revise, or reject.
- refer_object: the exact existing unit/part with a unique full-image locator, 2-14 words. For a strip/segment name its position WITHIN the larger owner, not just the owner's position in the photo. The words selected/visible/marked do NOT identify which part; use an observable level or neighboring feature instead.
- editing_instruction: one direct 4-24-word command; unchanged if accept, corrected if revise, empty if reject.
- segmentation_target: 1-8 words for the EXACT existing surface being changed, not its whole owner or the new object; omit positional clauses.
- mask_refinement: one of {sorted(allowed_mask_policies(row['task_type']))}; surface means narrow down inside the original extent, never expand.
An accessory outside the outline is a reason to reject removal/replacement, not to silently omit it from the observation. State uncertainty as rejection, not invented completeness.
{feedback}'''


def apply_scope_result(row, result):
    """Fail closed on malformed or out-of-policy revisions; preserve provenance."""
    if not isinstance(result, dict) or result.get('decision') not in {'accept', 'revise', 'reject'}:
        return None, 'invalid_decision'
    if not isinstance(result.get('reason'), str) or not result['reason'].strip():
        return None, 'missing_visual_reason'
    if result['decision'] == 'reject':
        return None, 'rejected_scope'
    instruction = result.get('editing_instruction', '')
    reference = result.get('refer_object', '')
    phrase = result.get('segmentation_target', '')
    if not all(isinstance(x, str) and x.isascii() for x in [instruction, reference, phrase]):
        return None, 'invalid_text'
    if not 4 <= len(instruction.split()) <= 24 or not 2 <= len(reference.split()) <= 14 or not 1 <= len(phrase.split()) <= 8:
        return None, 'invalid_length'
    policy = result.get('mask_refinement')
    if policy not in allowed_mask_policies(row['task_type']):
        return None, 'invalid_mask_policy'
    if result['decision'] == 'accept' and instruction != row['editing_instruction']:
        return None, 'accept_changed_instruction'
    reason = result['reason'].lower()
    for term in ('strip', 'segment', 'riser'):
        if row['task_type'] == 'attribute' and re.search(r'\b'+term+r'\b', reason) and not all(re.search(r'\b'+term+r'\b', text.lower()) for text in (instruction, reference)):
            return None, 'scope_word_missing_'+term
    if re.search(r'\b(?:left|right)[ -]+(?:lapel|cuff|sleeve|pocket)\b', instruction, re.I):
        return None, 'unchecked_garment_side'
    if re.search(r'\b(?:visible|selected|marked|outlined|target) (?:strip|segment|section|part|area)\b', instruction, re.I):
        return None, 'vague_part_locator'
    from synthesis_pipeline.generate_samtok_plan import (
        TYPE_ACTION_PATTERNS, has_annotation_language, STRONG_LOCATOR_PATTERN,
        contains_distinctive_reference,
    )
    if not TYPE_ACTION_PATTERNS[row['task_type']].search(instruction) or has_annotation_language(instruction):
        return None, 'invalid_action'
    if not STRONG_LOCATOR_PATTERN.search(reference) or not contains_distinctive_reference(instruction, reference):
        return None, 'missing_unique_reference'
    if ambiguous_anatomical_side(result) or protected_dependency_conflicts(row, row['task_type']):
        return None, 'unresolved_dependency_or_laterality'
    updated = {**row, 'editing_instruction': instruction, 'new_instruction': instruction,
               'refer_object': [reference], 'segmentation_target': phrase,
               'mask_refinement': policy, 'scope_preflight': {
                   'version': 1, 'decision': result['decision'], 'reason': result['reason'],
                   'original_instruction': row['editing_instruction'],
                   'original_segmentation_target': row.get('segmentation_target'),
                   'verification': 'model_scope_checked_not_manually_verified'}}
    updated.pop('_scope_feedback', None)
    return updated, 'accepted'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--attempts', type=int, default=2, help='Retry malformed/self-contradictory outputs, never retry a visual rejection')
    p.add_argument('--ids', default='', help='Optional comma-separated case prefixes for a bounded regression')
    p.add_argument('--model-id', default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')
    a = p.parse_args()
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'sources').symlink_to((a.data_root/'sources').resolve())
    (a.out_root/'inputs').mkdir()
    rows = [json.loads(x) for x in (a.data_root/'annotations.jsonl').read_text().splitlines() if x.strip()]
    if a.ids:
        wanted = {int(x) for x in a.ids.split(',')}
        rows = [r for r in rows if int(r['image'].split('_')[0]) in wanted]
        if len(rows) != len(wanted):
            raise ValueError('Requested scope IDs are missing or duplicated')
    started = time.perf_counter()
    vlm.configure_backend('qwen38-vllm', model_id=a.model_id, device='cuda:0', dtype='bf16')
    backend = vlm.get_backend()
    load_seconds = time.perf_counter()-started
    responses, accepted = [], []
    inference = 0.
    try:
      pending = rows
      for attempt in range(1, a.attempts+1):
        again = []
        for offset in range(0, len(pending), a.batch_size):
            batch = pending[offset:offset+a.batch_size]
            messages = []
            for row in batch:
                source = Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
                mask = mask_array(source.size, row['mask'])
                crop = instruction_target_crop(source, mask, target_pointer=True)
                crop.save(a.out_root/'inputs'/row['image'])
                prompt = scope_prompt({**row, 'mask_geometry_hint': mask_geometry_hint(mask)})
                messages.append([{'role': 'user', 'content': [
                    {'type': 'image', 'image': source}, {'type': 'image', 'image': crop},
                    {'type': 'text', 'text': prompt}]}])
            t = time.perf_counter()
            outputs = backend.chat_batch(messages, max_new_tokens=512)
            inference += time.perf_counter()-t
            for row, raw, message in zip(batch, outputs, messages):
                result = parse_json_object(raw)
                updated, status = apply_scope_result(row, result)
                response = dict(image=row['image'], attempt=attempt, raw_response=raw, parsed=result,
                                status=status, prompt=message[0]['content'][-1]['text'])
                responses.append(response)
                if updated is not None:
                    accepted.append(updated)
                elif status != 'rejected_scope':
                    feedback = ('Your previous answer failed the output contract: '+status+
                                '. Correct this specific issue without dropping the unique scene locator. '
                                'If changing ANY wording, use revise. Respect the 1-8 word source query. '
                                'If a strip/segment/riser is the selected scope, name it in the instruction. '
                                'A visible strip is not a unique locator: identify its level or position within its owner, '
                                'as observed in the two images. '
                                'Previous answer: '+raw)
                    again.append({**row, '_scope_feedback':feedback})
                print(json.dumps(response, ensure_ascii=False), flush=True)
            write_jsonl(a.out_root/'responses.jsonl', responses)
        pending = again
        if not pending:
            break
    finally:
        vlm.shutdown_backend()
    write_jsonl(a.out_root/'annotations.jsonl', accepted)
    frozen = a.data_root/'input_annotations.jsonl'
    write_jsonl(a.out_root/'input_annotations.jsonl',
                [json.loads(x) for x in frozen.read_text().splitlines()] if frozen.exists() else rows)
    summary = dict(input_cases=len(rows), accepted=len(accepted), calls=len(responses),
                   load_seconds=load_seconds, inference_seconds=inference,
                   wall_seconds=time.perf_counter()-started)
    (a.out_root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
