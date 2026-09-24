"""Direct context editing with a single, late, soft pixel-space composition."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox


GROUNDED_EDIT_METHODS = {
    'context_grounded_v3', 'context_grounded_v4',
    'context_grounded_v3_qwen21',
    'context_grounded_v4_qwen21',
}


def removal_execution_evidence(row):
    """Use only accepted, frozen visual evidence; never derive category rules."""
    grounding = row.get('visual_grounding', {})
    revision = row.get('planning_revision', {})
    if (not isinstance(grounding, dict) or not isinstance(revision, dict)
            or row.get('task_type') != 'remove' or grounding.get('decision') != 'accept'
            or revision.get('decision') != 'accept'):
        return {}
    parts = grounding.get('selected_surfaces', [])
    outside = grounding.get('outside_description', '')
    target = grounding.get('target_description', '')
    if (not isinstance(parts, list) or not 1 <= len(parts) <= 5
            or not all(isinstance(part, str) and 1 <= len(part.split()) <= 24 for part in parts)
            or not isinstance(outside, str) or len(outside.split()) > 100
            or not isinstance(target, str) or not target.strip()):
        return {}
    return dict(target=target, visible_parts=[part.strip() for part in parts],
                excluded_context=outside.strip(), provenance='accepted_frozen_visual_grounding')


def qwen21_edit_prompt(action, task_type, policy='legacy', evidence=None):
    base = 'Edit the provided photo in place. ' + action + ' '
    if policy == 'legacy':
        return base + ('Preserve the full photographic scene and reconstruct any newly exposed '
                       'background naturally. Return the complete edited photograph.')
    if policy not in {'typed-v1','remove-shadow-v2','remove-evidence-v1','remove-parts-v1','remove-context-v1','relation-compact-v1','relation-spatial-v1','relation-located-v2','relation-action-v3'}:
        raise ValueError(f'Unknown Qwen-Image-2.1 prompt policy: {policy}')
    rules = {
        'add': ('Integrate the addition at the specified attachment or placement site. '
                'Match the surrounding perspective, focus, lighting and surface texture. '
                'Keep the host intact and make contact physically coherent. '),
        'remove': ('Remove the entire visible target, including its thin extremities. '
                   'Continue the surrounding background texture, edges, lighting and perspective '
                   'through its former footprint, without fragments or uniform painted patches. '),
        'replace': ('Substitute the new instance for the complete visible old target, '
                    'rather than placing a small depiction inside it. Match the existing '
                    'footprint, pose, depth, focus and physical support. '),
        'attribute': ('Apply only the requested property change to the named surface in place. '
                      'Retain its silhouette, fine detail, shading, folds and existing markings '
                      'unless that property is explicitly requested to change. A color change '
                      'must follow the existing texture and illumination. '),
    }
    if policy=='remove-shadow-v2' and task_type=='remove':
        rules['remove'] += ('Remove the target\'s own cast shadow and contact shadow where they would '
            'remain as isolated evidence of the missing object. Keep shadows belonging to other '
            'objects and the scene\'s ambient lighting. Continue each exposed surface across '
            'the old boundary instead of tracing the old silhouette. ')
    if policy in {'remove-evidence-v1','remove-parts-v1','remove-context-v1'} and task_type == 'remove' and evidence:
        if policy in {'remove-evidence-v1','remove-parts-v1'}:
            rules['remove'] += (
            'The visible parts of this same selected instance include: '
            + '; '.join(evidence['visible_parts']) + '. Remove them together, '
            'not the corresponding parts of neighboring instances. ')
        if policy in {'remove-evidence-v1','remove-context-v1'}:
            if evidence.get('excluded_context') and evidence['excluded_context'].lower() != 'none':
                rules['remove'] += ('Existing surrounding content, not removal targets: '
                    + evidence['excluded_context'].rstrip('.') + '. ')
            rules['remove'] += (
                'Reconstruct each directly exposed surface from its visible continuation. '
                'Do not substitute a farther background layer for a nearer supporting surface. ')
    return base + rules[task_type] + (
        'Keep the framing and unrelated instances unchanged. Return the complete edited photograph.')


def validate_refinement_execution(row, edit_method):
    """Do not let a provisional subpart plan reach a legacy stale-mask editor."""
    needs_refinement = row.get('mask_refinement') == 'surface' or (
        row.get('task_type') in {'remove','replace'} and row.get('edit_unit_status') == 'complete_part'
        and bool(row.get('structural_parts')))
    if needs_refinement or row.get('region_contract'):
        if edit_method not in GROUNDED_EDIT_METHODS:
            raise ValueError('Semantic-mask plans require context_grounded_v3; legacy crop masks may be stale')
        if row.get('region_contract',{}).get('status') not in {'original','segmented_candidate','visually_verified'}:
            raise ValueError('Run semantic region refinement before generating this provisional plan')


def edit_context_crop(
    pipe,
    source,
    row,
    mask,
    generator,
    steps=40,
    true_cfg_scale=4.0,
    guidance_scale=1.0,
    negative_prompt=" ",
    remove_context_window=False,
    diagnostics_dir=None,
    target_guide=False,
    attribute_mask_composition=False,
    preserve_attribute_texture=False,
    guarded_composition=False,
    protected_mask=None,
    replacement_context_window=False,
    connected_support=False,
    grounded_composition=False,
    regional_denoising=False,
    qwen21_prompt_policy='legacy',
    remove_composition_policy='legacy',
    latent_protection_policy='legacy',
    relation_geometry_policy='legacy',
    removal_conditioning='source',
):
    """Keep surrounding evidence during diffusion and protect distant pixels.

    Unlike exact-mask latent reinjection, the old target silhouette is not
    restored at each denoising step. The final editable rectangle includes a
    collar around the whole old footprint, admitting a new object silhouette.
    Add masks identify an anchor, so their full context window is writable.
    This is not a quality guarantee: collateral edits and seams still need audit.
    """
    from utils.qwen_pipeline_loader import is_qwen21_omni_pipeline, is_qwen21_pipeline
    qwen21 = is_qwen21_pipeline(pipe)
    qwen21_omni = is_qwen21_omni_pipeline(pipe)
    if removal_conditioning not in {'source','erase-neutral-v1','erase-prefill-v1','erase-neutral-v2'}:
        raise ValueError('Unknown removal conditioning')
    if removal_conditioning != 'source' and (not qwen21 or row['task_type'] != 'remove' or not grounded_composition):
        raise ValueError('Erased conditioning requires grounded Qwen21 removal')
    if relation_geometry_policy not in {'legacy', 'visible-v1'}:
        raise ValueError('Unknown relation geometry policy')
    if grounded_composition:
        contract = row.get('region_contract', {})
        execution = row.get('execution_region')
        if execution is not None:
            from synthesis_pipeline.audit_edit_pairs import mask_array
            if (row['task_type'] != 'remove' or execution.get('status') != 'resolved_auxiliary'
                    or execution.get('source_size') != list(source.size)):
                raise ValueError('Unresolved or incompatible auxiliary execution region')
            resolved = mask_array(source.size, execution['mask']).astype(bool)
            if np.any(mask.astype(bool) & ~resolved):
                raise ValueError('Auxiliary execution cannot shrink the source target')
            mask = resolved
        target_guide = target_guide or bool(row.get('editor_target_guide', False))
        if contract.get('status') not in {'original', 'segmented_candidate', 'visually_verified'}:
            raise ValueError('Grounded editing requires a resolved region contract')
        if contract.get('source_size') != list(source.size):
            raise ValueError('Region contract source dimensions differ')
        from synthesis_pipeline.audit_edit_pairs import mask_array
        if contract.get('protected_mask'):
            extra = mask_array(source.size, contract['protected_mask']).astype(bool)
            protected_mask = extra if protected_mask is None else (protected_mask | extra)
        bbox = padded_mask_bbox(mask,
            padding_fraction=float(row.get('editor_crop_padding', .75 if row['task_type']=='remove' else .5)),
            min_padding=int(row.get('editor_crop_min',128 if row['task_type']=='remove' else 64)))
    else:
        bbox = padded_mask_bbox(mask, padding_fraction=0.75, min_padding=128)
    crop = source.crop(bbox)
    image_input = crop
    condition_evidence = None
    if removal_conditioning != 'source':
        if regional_denoising or target_guide or row.get('editor_target_guide'):
            raise ValueError('Erased conditioning pilot uses one image without regional reinjection')
        from utils.remove_support import erased_removal_condition
        image_input,condition_evidence=erased_removal_condition(
            crop,mask[bbox[1]:bbox[3],bbox[0]:bbox[2]],
            protected_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] if protected_mask is not None else None,
            policy=removal_conditioning)
    prompt = row["editing_instruction"]
    action = (
        row["editing_instruction"]
        if row["task_type"] == "remove"
        else row.get("new_instruction", row["editing_instruction"])
    )
    if grounded_composition:
        local = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(bool)
        ys, xs = np.nonzero(local)
        target = row['region_contract'].get('segmentation_target') or row.get('masked_content', 'object')
        prompt = (
            action + ' '
            f"The existing {target} to edit is at x={xs.min()/crop.width:.0%} to {(xs.max()+1)/crop.width:.0%}, "
            f"y={ys.min()/crop.height:.0%} to {(ys.max()+1)/crop.height:.0%} in this photo "
            "(left/top are 0%). The rectangle locates the existing instance; it is not a shape to paint. "
            + " Keep the framing and every other existing instance in its original place. "
        )
        if row['task_type'] == 'attribute':
            prompt += (
                "Modify the existing surface in place. Preserve its size, silhouette and position. "
                "Keep photographic shading, material texture and fine details. Do not add a new copy "
                "or cover the object with a solid-color shape. "
            )
        elif row['task_type'] == 'remove':
            prompt += (
                "Remove this one complete object. Fill its former footprint by continuing the visible "
                "background materials, edges and perspective. Keep adjacent objects and surfaces intact. "
            )
        else:
            prompt += "Keep the new object's scale, support and contact consistent with this scene. "
        prompt += row.get('generation_context', '')
    elif guarded_composition:
        if target_guide:
            raise ValueError("Guarded editing uses clean photographic input only")
        local = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(bool)
        ys, xs = np.nonzero(local)
        prompt += (
            f" The intended target is centered approximately at {xs.mean()/crop.width:.0%} "
            f"from the left and {ys.mean()/crop.height:.0%} from the top of this photo. "
            "Edit only that instance. Keep the camera, framing and surrounding objects fixed."
        )
        if row['task_type'] == 'attribute':
            prompt += (
                " Retain its exact silhouette, pose, photographic texture, shading and fine "
                "surface detail. Apply the property change visibly, following existing lighting."
            )
        elif row['task_type'] == 'remove':
            prompt += (
                " Reconstruct the entire removed footprint with continuous background texture "
                "and lighting; leave no target fragments or ghost edges. Do not remove neighbors."
            )
        elif row['task_type'] in {'add', 'replace'}:
            prompt += (
                " Use the existing physical support and scene perspective, with natural contact "
                "and scale. Do not invent a new supporting hand or person."
            )
    if target_guide:
        # Official Qwen-Image-Edit supports multiple reference images. The first
        # stays clean; the second supplies non-semantic region geometry only.
        from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs

        local_mask = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        guide, _ = audit_two_image_inputs(
            crop, crop, local_mask, longest_side=max(crop.size), scope="full"
        )
        guide = guide.crop((0, 40, guide.width, guide.height))
        image_input = [crop, guide]
        prompt = (
            "Edit Image 1. Image 2 is the same photo with a black/white outline "
            "identifying the exact target; the outline is a guide, not an object. "
            + prompt
            + " Apply this action to that target only. Return one natural "
            "edited version of Image 1 without any guide outline or annotation."
        )
    if preserve_attribute_texture:
        prompt += (
            " Make the requested property change clearly visible while preserving the "
            "target's original geometry, photographic texture, local shading and fine "
            "surface detail. A color change must follow the existing lighting and folds, "
            "not become a flat painted shape. Match the original level of focus."
        )
        negative_prompt = (
            "sticker border, black and white annotation outline, flat painted fill, "
            "posterization, cartoon rendering, ghost contour, pasted patch"
        )
    denoise_kwargs={}
    token_weights=None
    if regional_denoising:
        if not grounded_composition or (target_guide and not qwen21):
            raise ValueError('Regional denoising requires grounded masks; guides require Qwen 2.1')
        from utils.region_denoise import region_callback_kwargs
        denoise_kwargs,token_weights=region_callback_kwargs(pipe,crop,
            mask[bbox[1]:bbox[3],bbox[0]:bbox[2]],row['task_type'],generator,
            protected_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] if protected_mask is not None else None,
            protection_policy=latent_protection_policy)
    relation_context_evidence=None
    if qwen21:
        from utils.removal_relations import is_grounded_relation_policy
        # Follow 2.1's official natural-language editing style.  Reusing the
        # 2511-only coordinate/"rectangle" protocol can trigger 2.1's native
        # transparent-layer mode instead of a complete edited photograph.
        if qwen21_prompt_policy=='relation-spatial-v1':
            if not is_grounded_relation_policy(row.get('relation_policy')) or not row.get('execution_region') or row['task_type']!='remove':
                raise ValueError('Spatial relation prompt requires resolved v4 removal relations')
            from utils.removal_relations import crop_relative_target_hint
            from synthesis_pipeline.audit_edit_pairs import mask_array
            action+=' '+crop_relative_target_hint(mask_array(source.size,row['mask']),bbox)
        prompt = qwen21_edit_prompt(action, row['task_type'], qwen21_prompt_policy,
            evidence=removal_execution_evidence(row))
        if row.get('execution_region'):
            if is_grounded_relation_policy(row.get('relation_policy')):
                from utils.removal_relations import compile_relation_context
                context,relation_context_evidence=compile_relation_context(row['relation_plan'],bbox,source.size)
                prompt += ' '+context
                if qwen21_prompt_policy=='relation-compact-v1':
                    from utils.removal_relations import compact_removal_prompt
                    prompt=compact_removal_prompt(action,row['relation_plan'])
                    relation_context_evidence['prompt_variant']='one_removal_clause_generic_preservation'
                if row.get('relation_policy') in {'relations-v7','relations-v8','relations-v9','relations-v10'} and qwen21_prompt_policy=='typed-v1':
                    from utils.removal_relations import concise_execution_prompt
                    prompt=concise_execution_prompt(row['editing_instruction'],row['relation_plan'])
                    relation_context_evidence['prompt_variant']='concise_relation_execution_v1'
            else:
                prompt += ' ' + row.get('relation_execution_context', '')
            if relation_geometry_policy == 'visible-v1':
                prompt += (' Keep each retained entity in its existing pose and location. '
                    'Use its already visible silhouette and parts as fixed anchors; '
                    'complete only the portions newly exposed by removal. '
                    'Do not duplicate or relocate visible parts, or add extra limbs. '
                    'Connect completed parts naturally to the retained entity and its support.')
        if qwen21_prompt_policy=='relation-compact-v1' and (
                not is_grounded_relation_policy(row.get('relation_policy')) or not row.get('execution_region') or row['task_type']!='remove'):
            raise ValueError('Compact relation prompt requires resolved v4 removal relations')
        if qwen21_prompt_policy in {'relation-located-v2','relation-action-v3'}:
            if not is_grounded_relation_policy(row.get('relation_policy')) or not row.get('execution_region') or row['task_type']!='remove':
                raise ValueError('Located removal prompt requires resolved removal relations')
            from utils.removal_relations import located_removal_prompt
            from synthesis_pipeline.audit_edit_pairs import mask_array
            prompt=located_removal_prompt(row['editing_instruction'],row['relation_plan'],
                mask_array(source.size,row['mask']),bbox,row['execution_region'])
            relation_context_evidence['prompt_variant']='concise_located_removal_v2'
            if qwen21_prompt_policy=='relation-action-v3':
                from utils.removal_relations import action_removal_prompt
                prompt=action_removal_prompt(row['relation_plan'],mask_array(source.size,row['mask']),bbox,row['execution_region'])
                relation_context_evidence['prompt_variant']='single_action_with_fill_context_v3'
    if removal_conditioning != 'source':
        prompt=('Complete the missing background in the retouched area. Continue the surrounding '
                'surfaces, lighting and perspective naturally. Keep the existing scene and remaining '
                'subjects unchanged. Return a complete opaque photograph.')
        if removal_conditioning == 'erase-neutral-v2':
            prompt=('Fill the gray area with the background behind the removed subject. '
                    'Continue the surrounding surfaces and lighting. Keep the rest of the photograph unchanged.')
    if qwen21 and target_guide:
        # Qwen21 reconstructs its prompt above; do not lose the guide binding
        # when replacing the legacy 2511 prompt. Image 1 stays entirely clean.
        prompt = ('Edit Image 1. Image 2 marks the target with a black/white boundary; '
                  'it is a location guide, not image content. ' + prompt +
                  ' Return Image 1 after the edit, without any annotation.')
    call_kwargs=dict(
        image=image_input,prompt=prompt,num_inference_steps=steps,
        generator=generator,**denoise_kwargs)
    if qwen21:
        # Official Qwen-Image-2.1 defaults: 40 steps, no CFG, KV cache on.
        # Do not pass the legacy guidance_scale argument, which is absent from
        # QwenImage21Pipeline.__call__.
        call_kwargs['true_cfg_scale']=true_cfg_scale
        call_kwargs.setdefault('use_kv_cache',True)
        if true_cfg_scale>1 and negative_prompt is not None:
            call_kwargs['negative_prompt']=negative_prompt
    else:
        call_kwargs.update(
            negative_prompt=negative_prompt,true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale)
    model_output = pipe(**call_kwargs).images[0]
    model_alpha = None
    if qwen21 and model_output.mode == 'RGBA':
        # Qwen-Image-2.1 natively predicts RGBA.  For a local edit it may
        # represent untouched pixels as transparent while their hidden RGB is
        # an arbitrary matte color (often purple).  Dropping alpha with
        # `convert("RGB")` leaks that matte into replacement boundaries.
        # Interpret transparency according to the official RGBA semantics and
        # composite the model layer over the clean condition crop first.
        layer = model_output.resize(crop.size, Image.Resampling.LANCZOS)
        model_alpha = layer.getchannel('A')
        edited = Image.alpha_composite(crop.convert('RGBA'), layer).convert('RGB')
    else:
        edited = model_output.convert('RGB').resize(
            crop.size, Image.Resampling.LANCZOS)
    result, alpha = compose_context_crop(
        source, edited, mask, row["task_type"], bbox, remove_context_window
    )
    if attribute_mask_composition:
        if row["task_type"] != "attribute":
            raise ValueError(
                "Exact late composition is only intended for same-geometry attributes"
            )
        result, alpha = compose_attribute_crop(source, edited, mask, bbox)
    if guarded_composition:
        result, alpha = compose_guarded_crop(
            source, edited, mask, row['task_type'], bbox, protected_mask,
            replacement_context_window=replacement_context_window,
            connected_support=connected_support,
        )
    if grounded_composition:
        result, alpha = compose_grounded_crop(source, edited, mask, row['task_type'], bbox, protected_mask,
            remove_composition_policy=remove_composition_policy)
    if diagnostics_dir is not None:
        import json
        from pathlib import Path

        directory = Path(diagnostics_dir)
        directory.mkdir(parents=True, exist_ok=True)
        crop.save(directory / "source_crop.png")
        if qwen21:
            # Preserve native RGBA before alpha composition. Otherwise a
            # transparent failed generation can masquerade as a model no-op.
            model_output.save(directory / 'native_model_output.png')
        if removal_conditioning != 'source':
            image_input.save(directory / 'condition_image.png')
        edited.save(directory / "raw_edited_crop.png")
        alpha.save(directory / "composition_alpha.png")
        if protected_mask is not None:
            Image.fromarray(protected_mask.astype(np.uint8) * 255).save(directory / "protected_instances.png")
        if target_guide:
            guide.save(directory / "target_guide.png")
        if token_weights is not None:
            Image.fromarray((token_weights*255).astype(np.uint8)).save(directory/'editable_tokens.png')
        if model_alpha is not None:
            model_alpha.save(directory/'qwen21_output_alpha.png')
        (directory / "generation_request.json").write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "task_type": row["task_type"],
                    "remove_composition_policy": remove_composition_policy,
                    "steps": steps,
                    "removal_conditioning":removal_conditioning,
                    "condition_evidence":condition_evidence,
                    "model_output_size": list(model_output.size),
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "crop_bbox": bbox,
                    "source_size": source.size,
                    "reference_images": 2 if target_guide else 1,
                    "attribute_mask_composition": attribute_mask_composition,
                    "preserve_attribute_texture": preserve_attribute_texture,
                    "guarded_composition": guarded_composition,
                    "replacement_context_window": replacement_context_window,
                    "connected_support": connected_support,
                    "grounded_composition": grounded_composition,
                    "regional_denoising":regional_denoising,
                    "latent_protection_policy":latent_protection_policy,
                    "relation_geometry_policy":relation_geometry_policy,
                    "relation_context_evidence":relation_context_evidence,
                    "grounded_prompt_version": 2 if grounded_composition else None,
                    "pipeline_family": "qwen21" if qwen21 else "qwen2511",
                    "qwen21_backend": "vllm-omni" if qwen21_omni else ("diffusers" if qwen21 else None),
                    "qwen21_official_no_cfg": bool(qwen21 and true_cfg_scale <= 1),
                    "qwen21_prompt_policy": qwen21_prompt_policy if qwen21 else None,
                    "removal_execution_evidence": removal_execution_evidence(row)
                        if qwen21 and qwen21_prompt_policy in {'remove-evidence-v1','remove-parts-v1','remove-context-v1'} else None,
                    "mask_semantics": "original target; composition_alpha is the separate writable support",
                    "execution_region":row.get('execution_region'),
                },
                indent=2,
            )
        )
    return result


def compose_grounded_crop(source, edited, mask, task_type, bbox, protected_mask=None,
                          replacement_mask=None, poisson_blend=False, remove_composition_policy='legacy'):
    """Explicit semantic edit unit, narrow repair collar and visible-neighbor guard.

The source mask must already include structural parts for removal. A replacement
can supply its separately segmented new silhouette, avoiding a crop-wide write.
"""
    import cv2
    if mask.shape != (source.height, source.width) or edited.size != source.crop(bbox).size:
        raise ValueError('Source, mask and edited crop must be aligned')
    x1,y1,x2,y2 = bbox
    inside = mask[y1:y2,x1:x2].astype(bool)
    if not inside.any():
        raise ValueError('Expected nonempty effective edit mask')
    if task_type == 'attribute':
        _, alpha_image = compose_attribute_crop(source, edited, mask, bbox)
        alpha = np.asarray(alpha_image).astype(np.float32)/255
    elif task_type == 'add':
        _, alpha_image = compose_context_crop(source, edited, mask, task_type, bbox)
        alpha = np.asarray(alpha_image).astype(np.float32)/255
    elif task_type in {'remove','replace'}:
        support = inside.copy()
        if remove_composition_policy not in {'legacy','adaptive-remove-v1','adaptive-remove-v2','adaptive-remove-v3','adaptive-remove-v4','adaptive-remove-v5'}:
            raise ValueError('Unknown removal composition policy')
        if task_type=='remove' and remove_composition_policy.startswith('adaptive-remove-'):
            from utils.remove_support import adaptive_remove_support
            support,_=adaptive_remove_support(source.crop(bbox),edited,inside,
                protected_mask[y1:y2,x1:x2] if protected_mask is not None else None,
                policy='adaptive-remove-v2' if remove_composition_policy in {'adaptive-remove-v3','adaptive-remove-v4','adaptive-remove-v5'} else remove_composition_policy)
        ys,xs = np.nonzero(inside)
        if task_type == 'replace':
            if replacement_mask is not None:
                if replacement_mask.shape != mask.shape:
                    raise ValueError('Replacement mask dimensions differ')
                support |= replacement_mask[y1:y2,x1:x2].astype(bool)
            else:
                # Bounded fallback until a new-instance segmentation is available.
                padx,pady = max(16,round(np.ptp(xs)*.25)),max(16,round(np.ptp(ys)*.25))
                support[max(0,ys.min()-pady):min(support.shape[0],ys.max()+pady+1),
                        max(0,xs.min()-padx):min(support.shape[1],xs.max()+padx+1)] = True
        collar = float(np.clip(min(np.ptp(xs)+1,np.ptp(ys)+1)*.025,3,10))
        distance = cv2.distanceTransform((~support).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        alpha = np.clip((collar+3-distance)/3,0,1)
        alpha[support] = 1
    else:
        raise ValueError(f'Unknown task type {task_type}')
    if protected_mask is not None:
        if protected_mask.shape != mask.shape:
            raise ValueError('Protected mask dimensions differ')
        protected = protected_mask[y1:y2,x1:x2].astype(bool) & ~inside
        # A narrow transition preserves the actual neighboring instance exactly.
        distance = cv2.distanceTransform((~protected).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        alpha *= np.where(inside,1,np.clip(distance/2,0,1))
    alpha_image = Image.fromarray(np.round(alpha*255).astype(np.uint8))
    result = source.copy()
    if task_type=='remove' and remove_composition_policy in {'adaptive-remove-v3','adaptive-remove-v4','adaptive-remove-v5'}:
        from utils.removal_harmonization import harmonize_removal
        edited,_=harmonize_removal(source.crop(bbox),edited,alpha_image)
        if remove_composition_policy=='adaptive-remove-v4':
            from utils.remove_support import boundary_seam_alpha
            alpha_image=boundary_seam_alpha(source.crop(bbox),edited,inside,alpha_image,
                protected_mask[y1:y2,x1:x2] if protected_mask is not None else None)
        elif remove_composition_policy=='adaptive-remove-v5':
            from utils.removal_harmonization import correct_boundary_band
            edited=correct_boundary_band(source.crop(bbox),edited,alpha_image)
    composed = Image.composite(edited,source.crop(bbox),alpha_image)
    if poisson_blend and task_type == 'remove':
        binary=(alpha>0).astype(np.uint8)*255
        x,y,w,h=cv2.boundingRect(binary)
        if x>1 and y>1 and x+w<edited.width-1 and y+h<edited.height-1:
            # OpenCV mutates its mask (including erosion); keep the final support
            # intact or thin removed parts are pasted back from the source.
            cloned=cv2.seamlessClone(np.asarray(edited).copy(),np.asarray(source.crop(bbox)).copy(),binary.copy(),
                                    (x+w//2,y+h//2),cv2.NORMAL_CLONE)
            composed=Image.composite(Image.fromarray(cloned),source.crop(bbox),Image.fromarray(binary))
    result.paste(composed,bbox[:2])
    return result,alpha_image


def protected_neighbors(data_root, row, size):
    """Union other annotated instances of this source; never infer missing masks.

    Source/target overlap is resolved later in favor of the selected target.
    A replanning manifest may omit incompatible siblings, so use its frozen
    input manifest for protection when available.
    """
    import json
    from pathlib import Path
    from synthesis_pipeline.audit_edit_pairs import mask_array

    root = Path(data_root)
    manifest = root / 'input_annotations.jsonl'
    if not manifest.exists():
        manifest = root / 'annotations.jsonl'
    protected = np.zeros((size[1], size[0]), dtype=bool)
    for line in manifest.read_text().splitlines():
        other = json.loads(line)
        if other['source_image'] == row['source_image'] and other['image'] != row['image']:
            protected |= mask_array(size, other['mask'])
    return protected


def compose_guarded_crop(source, edited, mask, task_type, bbox, protected_mask=None,
                         replacement_context_window=False, connected_support=False,
                         poisson_blend=False):
    """Type-aware write support, preserving known neighboring instances exactly.

    Remove/replace allow a collar around the *shape*, not its whole rectangle.
    The original target is never feathered back into a removal. Attribute uses
    exact original membership. This cannot fix semantic or diffusion failures.
    """
    import cv2

    if mask.shape != (source.height, source.width) or edited.size != source.crop(bbox).size:
        raise ValueError('Source, mask and edited crop must be aligned')
    inside = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(bool)
    if not inside.any():
        raise ValueError('Expected nonempty target')
    if connected_support and task_type in {'remove', 'replace'}:
        old_pixels = np.asarray(source.crop(bbox), dtype=np.float32)
        new_pixels = np.asarray(edited, dtype=np.float32)
        changed = np.abs(old_pixels - new_pixels).mean(axis=2) >= 255 * .08
        protect = np.zeros_like(inside)
        if protected_mask is not None:
            if protected_mask.shape != mask.shape:
                raise ValueError('Protected mask shape mismatch')
            protect = protected_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(bool) & ~inside
        changed &= ~protect
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        changed[protect] = 0
        _, components = cv2.connectedComponents(changed, connectivity=8)
        connected_ids = np.unique(components[inside & (changed > 0)])
        connected_ids = connected_ids[connected_ids != 0]
        support = inside | np.isin(components, connected_ids)
        distance = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = np.clip((9. - distance) / 5., 0, 1)
        alpha[support] = 1
    elif task_type == 'attribute':
        _, alpha_image = compose_attribute_crop(source, edited, mask, bbox)
        alpha = np.asarray(alpha_image).astype(np.float32) / 255
    elif task_type == 'add' or (task_type == 'replace' and replacement_context_window):
        _, alpha_image = compose_context_crop(source, edited, mask, 'add', bbox)
        alpha = np.asarray(alpha_image).astype(np.float32) / 255
    elif task_type in {'remove', 'replace'}:
        ys, xs = np.nonzero(inside)
        collar = max(16., min(64., min(np.ptp(xs)+1, np.ptp(ys)+1) * .15))
        feather = max(8., collar * .5)
        distance = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = np.clip((collar + feather - distance) / feather, 0, 1)
        alpha[inside] = 1
    else:
        raise ValueError(f'Unknown edit type: {task_type}')
    if protected_mask is not None:
        if protected_mask.shape != mask.shape:
            raise ValueError('Protected mask shape mismatch')
        protected = protected_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(bool) & ~inside
        if protected.any():
            distance = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            guard = np.clip(distance / 6., 0, 1)
            guard[inside] = 1
            alpha *= guard
    alpha_image = Image.fromarray(np.round(alpha * 255).astype(np.uint8), mode='L')
    result = source.copy()
    composed = Image.composite(edited, source.crop(bbox), alpha_image)
    if poisson_blend and task_type in {'remove', 'replace'}:
        clone_mask = (alpha > .01).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(clone_mask)
        # OpenCV zeros the mask border; do not restore old pixels at a real
        # photograph/crop boundary by sending edge-touching masks through it.
        if x > 1 and y > 1 and x + w < edited.width - 1 and y + h < edited.height - 1:
            cloned = cv2.seamlessClone(np.asarray(edited).copy(), np.asarray(source.crop(bbox)).copy(),
                clone_mask.copy(), (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
            # Restrict solver drift to writable pixels and retain protected instances exactly.
            composed = Image.composite(Image.fromarray(cloned), source.crop(bbox),
                                      Image.fromarray((alpha > 0).astype(np.uint8) * 255))
    result.paste(composed, bbox[:2])
    return result, alpha_image


def compose_attribute_crop(source, edited, mask, bbox):
    """Restrict a same-geometry property edit to original target pixels.

    This is a late pixel-space operation, not reference-latent reinjection.
    External guide strokes and unrelated instances are excluded. New object
    shapes must NOT use this policy: they would be clipped by the old shape.
    """
    crop = source.crop(bbox)
    if mask.shape != (source.height, source.width) or edited.size != crop.size:
        raise ValueError("Source, mask and edited crop must be aligned")
    inside = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype(bool)
    if not inside.any():
        raise ValueError("Attribute target mask must be nonempty")
    binary = Image.fromarray(inside.astype(np.uint8) * 255)
    soft = binary.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5))
    alpha = Image.fromarray(np.where(inside, np.asarray(soft), 0).astype(np.uint8))
    result = source.copy()
    result.paste(Image.composite(edited, crop, alpha), bbox[:2])
    return result, alpha


def compose_context_crop(
    source, edited, mask, task_type, bbox, remove_context_window=False
):
    """Feather only internal crop boundaries, never a true image boundary."""
    if mask.shape != (source.height, source.width):
        raise ValueError("Mask shape must match the original image")
    if task_type not in {"add", "remove", "replace", "attribute"}:
        raise ValueError(f"Unsupported task type: {task_type}")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height):
        raise ValueError("Crop box must lie within the original image")
    crop = source.crop(bbox)
    if edited.size != crop.size:
        raise ValueError("Edited crop must be aligned with the source crop")
    local_mask = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    alpha = Image.new("L", crop.size, 0)
    ys, xs = np.nonzero(local_mask)
    if not len(xs):
        raise ValueError("Crop must contain a nonempty target mask")
    pad = max(32, round(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.2))
    ImageDraw.Draw(alpha).rectangle(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(crop.width - 1, int(xs.max()) + pad),
            min(crop.height - 1, int(ys.max()) + pad),
        ),
        fill=255,
    )
    if task_type == "add" or (task_type == "remove" and remove_context_window):
        edge = min(24, min(crop.size) // 8)
        arr = np.ones((crop.height, crop.width), np.float32)
        for i in range(edge):
            v = (i + 1) / (edge + 1)
            if bbox[1] > 0:
                arr[i, :] *= v
            if bbox[3] < source.height:
                arr[-i - 1, :] *= v
            if bbox[0] > 0:
                arr[:, i] *= v
            if bbox[2] < source.width:
                arr[:, -i - 1] *= v
        alpha = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        alpha = alpha.filter(ImageFilter.GaussianBlur(10))
    result = source.copy()
    result.paste(Image.composite(edited, crop, alpha), bbox[:2])
    return result, alpha
