import numpy as np
from PIL import Image
from synthesis_pipeline.plan_dataset_regions import (
    addition_conflicts_with_excluded_contact,
    addition_is_microscopic, addition_places_loose_item_on_bare_body,
    addition_places_unfastened_item_on_fur, addition_surface_detail_is_unjudgeable,
    addition_is_oversized, addition_uses_underspecified_repeated_site,
    addition_host_is_broad_natural_background, addition_reason_has_novelty_check,
    addition_reason_admits_existing_item, addition_requires_unsupported_material_claim,
    addition_site_is_grounded, addition_site_is_sufficiently_local,
    addition_requires_pixels_outside_mask, addition_uses_ambiguous_body_side,
    addition_uses_unstable_narrow_support,
    broad_permanent_scene_edit, canonical_target,
    component_hint, dependency_from_grounding, edit_conflicts_with_reflection,
    enclosed_hole_hint,
    explicit_outside_dependency, grounding_marker_error, locator_coverage,
    locator_is_preserved, replacement_is_merely_surface_variant,
    partial_living_body_part_edit, partial_integrated_mechanical_assembly,
    replacement_has_incompatible_footprint,
    whole_living_target_has_visible_part_outside,
    selected_external_rigging_risk, unresolved_tiny_fragment, unsampled_held_target,
    unsampled_relational_surface, attribute_color_change_is_low_contrast,
    attribute_uses_implausible_living_tissue_color,
    attribute_uses_implausible_intrinsic_material_color,
    attribute_targets_unresolved_background,
    attribute_recolors_whole_pattern, instruction_uses_ambiguous_member_reference,
    target_promotes_unsampled_outside_noun, textured_or_weathered,
    validate, fixed_region, prompt_for, topology_hint,
)
from utils.context_edit import validate_refinement_execution


def test_short_valid_plan_keeps_mask_and_allows_unique_nonpositional_identity():
    target='purple floral lower garment'
    row={'task_type':'attribute','mask':{'counts':'opaque'},'editing_instruction':'',
         'visual_grounding':{'target_description':target}}
    value={'decision':'accept','reason':'A visibly patterned lower garment is selected.',
           'dependency_class':'none',
           'target_description':target,
           'editing_instruction':'Change the purple floral lower garment to yellow.'}
    result,status=validate(value,row,'scope')
    assert status=='accepted' and result['mask']==row['mask']
    assert result['mask_refinement']=='original'
    contract=fixed_region({**result,'sam_target_mask':{'counts':'stale_legacy_mask'}},(10,10),np.zeros((10,10),bool))
    assert contract['mask']==row['mask']
    assert contract['sam_target_mask']==row['mask']
    assert contract['region_contract']['source_sam_called'] is False
    validate_refinement_execution(contract,'context_grounded_v4')


def test_model_rejection_and_wrong_type_cannot_enter_editing():
    row={'task_type':'remove','editing_instruction':'',
         'visual_grounding':{'target_description':'left object'}}
    assert validate({'decision':'reject','reason':'A dependent item is outside.'},row,'scope')[1]=='model_rejected'
    value={'decision':'accept','reason':'Visible target.','dependency_class':'none',
           'target_description':'left object',
           'editing_instruction':'Add a ribbon to the object on the left.'}
    assert validate(value,row,'scope')[1]=='wrong_action_type'
    value['editing_instruction']='Remove the object in the mask.'
    assert validate(value,row,'scope')[1]=='annotation_language_in_instruction'


def test_prompt_explicitly_uses_existing_mask_and_visible_occluded_instances():
    prompt=prompt_for({'task_type':'replace','visual_grounding':{
        'target_description':'person beside the tree','outside_description':'bicycle'}},'scope')
    assert 'no later source segmentation or mask expansion' in prompt
    assert 'People and animals ARE valid' in prompt
    assert 'hidden parts behind occluders' in prompt


def test_real_face_mask_is_not_annotation_language():
    target='person wearing a face mask'
    row={'task_type':'remove','editing_instruction':'',
         'visual_grounding':{'target_description':target}}
    value={'decision':'accept','reason':'Visible person with face covering.',
           'dependency_class':'none',
           'target_description':'person wearing a face mask',
           'editing_instruction':'Remove the person wearing a face mask on the right.'}
    assert validate(value,row,'scope')[1]=='accepted'
    value['editing_instruction']='Remove the person in the masked region.'
    assert validate(value,row,'scope')[1]=='annotation_language_in_instruction'


def test_outer_region_is_not_confused_with_central_hole():
    mask=np.ones((20,20),bool);mask[3:17,3:17]=False
    assert 'photo center is EXCLUDED' in topology_hint(mask)
    assert topology_hint(~mask)==''


def test_shared_group_hint_is_not_injected_as_single_region_evidence():
    row={'task_type':'replace','reference_binding':{'status':'bound_shared_group',
         'label':'one of the baby elephant and big elephant'},'num_masks':2}
    assert 'baby elephant and big elephant' not in prompt_for(row,'ground')


def test_long_target_locator_does_not_reject_a_concise_valid_command():
    target='the vertical white metal fence post in the middle ground, located between a palm tree and a white car'
    value={'decision':'accept','reason':'A single selected fence post is uniquely located.',
           'dependency_class':'none',
           'target_description':target,'editing_instruction':'Change the color of '+target+' to red.'}
    row={'task_type':'attribute','editing_instruction':'',
         'visual_grounding':{'target_description':target}}
    assert validate(value,row,'scope')[1]=='accepted'


def test_grounding_is_frozen_before_instruction_generation():
    row={'task_type':'remove','editing_instruction':''}
    value={'decision':'accept','reason':'The selected person is left of the bicycle.',
           'marker_observations':['TARGET 1: person clothing'],
           'target_description':'person left of the bicycle',
           'selected_surfaces':['person clothing'],
           'outside_description':'the bicycle'}
    grounded,status=validate(value,row,'ground')
    assert status=='accepted' and grounded['editing_instruction']==''
    prompt=prompt_for(grounded,'scope')
    assert 'selected target: "person left of the bicycle"' in prompt
    changed={'decision':'accept','reason':'Another target is selected.',
             'dependency_class':'none',
             'target_description':'person right of the bicycle',
             'editing_instruction':'Remove the person right of the bicycle.'}
    assert validate(changed,grounded,'scope')[1]=='target_changed_from_visual_grounding'


def test_ground_prompt_does_not_design_an_edit():
    prompt=prompt_for({'task_type':'replace'},'ground')
    assert 'VISUAL GROUNDING ONLY' in prompt
    assert 'marker_observations' in prompt
    assert 'selected_surfaces' in prompt
    assert 'Task type:' not in prompt
    assert 'People and animals ARE valid' not in prompt


def test_component_hint_does_not_equate_fragments_with_instances():
    mask=np.zeros((20,20),bool)
    mask[2:8,2:8]=True
    mask[12:18,12:18]=True
    hint=component_hint(mask)
    assert '2 substantial or elongated visible mask fragment(s)' in hint
    assert 'not object counts' in hint


def test_component_hint_reports_objective_appearance_for_multiple_fragments():
    mask=np.zeros((20,30),dtype=bool)
    mask[2:10,2:10]=True
    mask[12:19,22:29]=True
    source=Image.new('RGB',(30,20),'white')
    hint=component_hint(mask,source)
    assert 'Objective fragment evidence' in hint
    assert 'fragment 1:' in hint
    assert 'median source RGB' in hint


def test_enclosed_excluded_hole_is_explicit_membership_evidence():
    mask=np.ones((30,30),bool)
    mask[10:20,12:18]=False
    hint=enclosed_hole_hint(mask)
    assert '1 meaningful enclosed EXCLUDED hole' in hint
    assert 'inside such a hole is outside the target' in hint


def test_grounding_target_is_canonicalized_for_natural_embedding():
    assert canonical_target('The selected wooden bench.')=='selected wooden bench'
    row={'task_type':'replace','editing_instruction':''}
    grounded,status=validate({'decision':'accept','reason':'Visible section.',
        'marker_observations':['TARGET 1: wooden bench surface'],
        'target_description':'The selected wooden bench.',
        'selected_surfaces':['wooden slats'],
        'outside_description':'foreground bench continuation'},row,'ground')
    assert status=='accepted'
    assert grounded['visual_grounding']['target_description']=='selected wooden bench'
    planned,status=validate({'decision':'accept','reason':'The section is coherent.',
        'dependency_class':'none',
        'target_description':'the selected wooden bench',
        'editing_instruction':'Replace the selected wooden bench with a stone fountain.'},grounded,'scope')
    assert status=='accepted'


def test_real_costume_mask_is_not_annotation_language():
    target='costumed performer wearing a large purple mask'
    row={'task_type':'add','editing_instruction':'',
         'visual_grounding':{'target_description':target}}
    value={'decision':'accept','reason':'Novelty check: no similar bell is visible. Visible performer.',
           'dependency_class':'none',
           'target_description':target,
           'editing_instruction':'Add a gold bell to the costumed performer wearing a large purple mask.'}
    assert validate(value,row,'scope')[1]=='accepted'


def test_dependency_contract_blocks_contradictory_accept_without_extra_model_call():
    target='woman in blue dress on the right'
    row={'task_type':'remove','editing_instruction':'',
         'visual_grounding':{'target_description':target}}
    value={'decision':'accept','reason':'A carried item remains outside.',
           'dependency_class':'held_attached','target_description':target,
           'editing_instruction':'Remove the woman in blue dress on the right.'}
    assert validate(value,row,'scope')[1]=='contradictory_dependency_accept'


def test_scope_dependency_decision_is_authoritative_over_brittle_text_inference():
    grounding={'target_description':'baseball batter in red jersey',
               'reason':'The boundary excludes the bat held in the hands.',
               'outside_description':'black bat'}
    assert dependency_from_grounding(grounding)=='held_attached'
    row={'task_type':'remove','editing_instruction':'','visual_grounding':grounding}
    value={'decision':'accept','reason':'Removal is feasible.',
           'dependency_class':'none','target_description':'baseball batter in red jersey',
           'editing_instruction':'Remove the baseball batter in red jersey.'}
    assert validate(value,row,'scope')[1]=='accepted'


def test_grounding_relation_policy_does_not_confuse_target_clothes_or_position():
    wearing={'target_description':'worker in a white uniform standing behind the counter',
             'reason':'The selected person is wearing a white uniform and cap.',
             'outside_description':'bananas on the counter, customer in foreground'}
    assert dependency_from_grounding(wearing) is None
    positional={'target_description':'girl sitting on the right',
                'reason':'The girl is sitting on the right and is fully selected.',
                'outside_description':'woman on the left, wooden table'}
    assert dependency_from_grounding(positional) is None
    standing_neighbor={'target_description':'person standing on the left',
                       'reason':'The selected person is fully enclosed.',
                       'outside_description':'person standing on the right side of the room'}
    assert dependency_from_grounding(standing_neighbor) is None
    other_person_holds_item={'target_description':'standing person in a dark hoodie',
                             'reason':'The neighbour and their equipment are excluded.',
                             'outside_description':'pink skateboard held by person in white shirt'}
    assert dependency_from_grounding(other_person_holds_item) is None
    lexical_collision={'target_description':'woman holding food and a bag',
                       'reason':'Her food and bag are selected.',
                       'outside_description':'man in hoodie, food poster'}
    assert dependency_from_grounding(lexical_collision) is None
    included_attachment={'target_description':'large white tent',
                         'reason':'The selection includes a smaller attached section.',
                         'outside_description':'surrounding lawn and trees'}
    assert dependency_from_grounding(included_attachment) is None
    repeated_clothing_words={'target_description':'child wearing denim overalls and a floral shirt',
                             'reason':'The selected child is fully visible.',
                             'outside_description':'child in a blue striped shirt, person in a red hat'}
    assert dependency_from_grounding(repeated_clothing_words) is None


def test_grounding_relation_policy_reads_only_explicit_excluded_dependency():
    pulled={'target_description':'person in dark blue jacket walking away',
            'reason':'The mask excludes the black suitcase being pulled by the person.',
            'outside_description':'black rolling suitcase'}
    assert dependency_from_grounding(pulled)=='held_attached'
    relation_locator={'target_description':'young boy in red jersey holding a bat',
                      'reason':'The hands are selected.',
                      'outside_description':'red baseball bat'}
    assert dependency_from_grounding(relation_locator) is None
    reverse_support={'target_description':'bald man seated at a table',
                     'reason':'The upper body is selected.',
                     'outside_description':'chair the man is sitting on, table'}
    assert dependency_from_grounding(reverse_support) is None


def test_attribute_requires_surface_and_preserves_visible_pattern():
    target='person in a grey puffy coat on the far left'
    row={'task_type':'attribute','editing_instruction':'','visual_grounding':{
        'target_description':target,
        'reason':'The person wears a patterned grey puffy coat.'}}
    whole_person={'decision':'accept','reason':'The coat is visible.',
                  'dependency_class':'none','target_description':target,
                  'editing_instruction':'Change the color of the person in a grey puffy coat on the far left to red.'}
    assert validate(whole_person,row,'scope')[1]=='attribute_missing_specific_surface'
    erased={**whole_person,
            'editing_instruction':'Change the patterned coat of the person in a grey puffy coat on the far left to solid red.'}
    assert validate(erased,row,'scope')[1]=='attribute_would_erase_visible_pattern'
    valid={**whole_person,
           'editing_instruction':'Change the coat of the person in a grey puffy coat on the far left to red.'}
    assert validate(valid,row,'scope')[1]=='accepted'


def test_attribute_cannot_invent_an_excluded_surface_absent_from_grounding():
    target='person in the green beanie and blue jeans walking away'
    row={'task_type':'attribute','editing_instruction':'','visual_grounding':{
        'target_description':target,
        'reason':'The selected pixels cover the upper body, legs, and green beanie.'}}
    backpack={'decision':'accept','reason':'The backpack is editable.',
              'dependency_class':'none','target_description':target,
              'editing_instruction':'Change the backpack color of the person in the green beanie and blue jeans walking away to red.'}
    assert validate(backpack,row,'scope')[1]=='attribute_surface_not_grounded_inside_mask'
    beanie={**backpack,
            'editing_instruction':'Change the green beanie of the person in the green beanie and blue jeans walking away to red.'}
    assert validate(beanie,row,'scope')[1]=='accepted'


def test_natural_locator_reordering_is_allowed_but_referent_loss_is_not():
    target='hand holding a mobile phone in the foreground'
    assert locator_coverage(
        target,
        'Change the screen of the mobile phone held by the hand in the foreground to red.',
    ) >= 0.8
    assert locator_coverage(target,'Change the other phone screen to red.') < 0.8
    assert locator_is_preserved(
        'woman with blonde hair and glasses wearing a light green turtleneck sweater',
        'Add a brooch to the green turtleneck worn by the woman on the left.',
    )
    assert not locator_is_preserved(target, 'Change the other phone screen to red.')
    row={'task_type':'attribute','editing_instruction':'','visual_grounding':{
        'target_description':target,
        'reason':'The selected pixels include the hand and mobile phone.'}}
    natural={'decision':'accept','reason':'The phone is selected.',
             'dependency_class':'none','target_description':target,
             'editing_instruction':'Change the screen of the mobile phone held by the hand in the foreground to red.'}
    assert validate(natural,row,'scope')[1]=='accepted'
    assert validate({**natural,'editing_instruction':'Change the other phone screen to red.'},row,'scope')[1]=='instruction_lost_distinctive_target_locator'


def test_attribute_owner_possessive_names_a_specific_surface():
    target='woman in black tank top and blue skirt'
    row={'task_type':'attribute','editing_instruction':'','visual_grounding':{
        'target_description':target,
        'reason':'The black tank top and blue skirt are selected.',
        'selected_surfaces':['black tank top','blue skirt']}}
    value={'decision':'accept','reason':'The skirt is visibly selected.',
           'dependency_class':'none','target_description':target,
           'editing_instruction':"Change the color of the woman's blue skirt to red."}
    assert validate(value,row,'scope')[1]=='accepted'


def test_grounding_accept_requires_point_checks_and_inside_surfaces():
    row={'task_type':'attribute','editing_instruction':''}
    base={'decision':'accept','reason':'The point is on a jacket.',
          'target_description':'black jacket on the left',
          'outside_description':'person behind it'}
    assert validate(base,row,'ground')[1]=='missing_marker_observations'
    with_markers={**base,'marker_observations':['TARGET 1: black jacket fabric']}
    assert validate(with_markers,row,'ground')[1]=='missing_selected_surfaces'
    complete={**with_markers,'selected_surfaces':['black jacket fabric']}
    assert validate(complete,row,'ground')[1]=='accepted'


def test_grounding_rejects_skipped_or_semantically_wrong_target_points():
    base={'decision':'accept','reason':'A cyclist is selected.',
          'target_description':'cyclist in a yellow jersey',
          'selected_surfaces':['yellow jersey','white helmet'],
          'outside_description':'red bus behind the cyclist'}
    skipped={**base,'marker_observations':[
        'TARGET 1 lies on the yellow jersey.',
        'TARGET 3 lies on the red bus stripe.',
    ]}
    assert grounding_marker_error(skipped)=='marker_observations_not_numbered_in_order'
    unrelated={**base,'marker_observations':[
        'TARGET 1 lies on the yellow jersey.',
        'TARGET 2 lies on the red bus stripe.',
    ]}
    assert grounding_marker_error(unrelated)=='marker_observation_incoherent_with_target'
    explicit={**base,'marker_observations':[
        'TARGET 1 lies on the bus window outside the main subject boundary.',
    ]}
    assert grounding_marker_error(explicit)=='marker_explicitly_outside_claimed_target'


def test_grounding_allows_repeating_own_marker_number_in_observation():
    value={
        'reason':'The umbrella canopy is selected.',
        'target_description':'red umbrella canopy',
        'selected_surfaces':['red fabric'],
        'outside_description':'metal pole below',
        'marker_observations':[
            'TARGET 1: Red fabric beside TARGET 1.',
            'TARGET 2: Red fabric above TARGET 2.',
        ],
    }
    assert grounding_marker_error(value) is None


def test_grounding_retries_duplicate_marker_claims_and_unsampled_held_object():
    duplicated={
        'target_description':'black-faced sheep in the center',
        'selected_surfaces':['white wool','black face'],
        'outside_description':'another sheep on the left',
        'marker_observations':[
            'TARGET 1: White wool on the sheep.',
            'TARGET 2: Dark furry ear of the sheep.',
            'TARGET 3: Dark furry ear of the sheep.',
        ],
    }
    assert grounding_marker_error(duplicated)=='duplicate_marker_observations'
    holding={**duplicated,
             'target_description':'child in a red jacket holding a zebra umbrella',
             'marker_observations':['TARGET 1: Red jacket on the child.']}
    assert unsampled_held_target(holding)
    holding['marker_observations'].append('TARGET 2: Zebra fabric on the umbrella canopy.')
    assert not unsampled_held_target(holding)
    contradictory={
        'target_description':'black-faced sheep in the center',
        'selected_surfaces':['white wool','black face'],
        'outside_description':'partial sheep on the left, snowy field',
        'marker_observations':[
            'TARGET 1: White wool on the black-faced sheep.',
            'TARGET 2: Brown body of the partial sheep on the left.',
        ],
    }
    assert grounding_marker_error(contradictory)=='marker_observation_matches_outside_item'
    inanimate={
        'target_description':'red and yellow stadium bleachers with a blue floor',
        'selected_surfaces':['red seats','yellow seats','blue floor'],
        'outside_description':'people standing in the aisles',
        'marker_observations':[
            'TARGET 1: A person in a white shirt standing in the bleachers.'
        ],
    }
    assert grounding_marker_error(inanimate)=='marker_observation_matches_outside_item'


def test_grounding_rejects_self_reported_foreign_selected_fragment():
    value={
        'reason': ('The white bus is excluded, with only a tiny erroneous fragment '
                   'of its rear corner included in the mask.'),
        'target_description':'red double-decker bus on the right',
        'selected_surfaces':['red body panels'],
        'outside_description':'white double-decker bus on the left',
        'marker_observations':[
            'TARGET 1: Red body panel of the right bus.',
            'TARGET 2: White rear panel of the left bus.',
        ],
    }
    assert grounding_marker_error(value)=='grounding_admits_foreign_selected_fragment'


def test_grounding_does_not_promote_unsampled_support_or_equipment():
    seated={
        'target_description':'man in a red tunic sitting on a patterned seat',
        'selected_surfaces':['red tunic','brown hat','patterned seat'],
        'marker_observations':['TARGET 1: Red tunic fabric.',
                               'TARGET 2: Brown hat on the man.'],
    }
    assert unsampled_relational_surface(seated)
    seated['selected_surfaces'].remove('patterned seat')
    assert not unsampled_relational_surface(seated)
    equipped={
        'target_description':'blue sprayer tank with a yellow lid and black hose',
        'selected_surfaces':['blue tank','yellow lid','black hose'],
        'marker_observations':['TARGET 1: Blue plastic tank surface.'],
    }
    # A component introduced by "with" may be selected even when a point was
    # sampled only on the main body; the boundary and hole evidence decide it.
    assert not unsampled_relational_surface(equipped)


def test_marker_comparison_ignores_shared_colors_and_allows_homogeneous_network():
    container={
        'target_description':'large yellow container in the foreground',
        'selected_surfaces':['yellow horizontal slats','white paper label'],
        'outside_description':'red and white tape to the right, crate to the left',
        'marker_observations':[
            'TARGET 1: Yellow slats to the left of a white paper label.'
        ],
    }
    assert grounding_marker_error(container) is None
    network={
        'target_description':'network of leafless branches and twigs',
        'selected_surfaces':['bare twigs','branch segments'],
        'outside_description':'road below',
        'marker_observations':[
            'TARGET 1: Thin bare twig.',
            'TARGET 2: Vertical branch segment.',
            'TARGET 3: Thin bare twig.',
        ],
    }
    assert grounding_marker_error(network) is None


def test_explicit_target_attached_outside_item_is_a_hard_remove_conflict():
    grounding={
        'target_description':'man in a black shirt near the carousel',
        'outside_description':"orange backpack strap on the selected man's shoulder",
    }
    assert explicit_outside_dependency(grounding)=='held_attached'
    row={'task_type':'remove','editing_instruction':'','visual_grounding':grounding}
    value={'decision':'accept','reason':'The man can be removed.',
           'dependency_class':'none','target_description':grounding['target_description'],
           'editing_instruction':'Remove the man in a black shirt near the carousel.'}
    assert validate(value,row,'scope')[1]=='policy_rejected_explicit_outside_dependency'


def test_explicit_supported_person_is_a_hard_replace_conflict():
    grounding={
        'target_description':'man lying on the green sofa',
        'outside_description':"baby resting across the man's torso",
    }
    assert explicit_outside_dependency(grounding)=='supported_by_target'
    row={'task_type':'replace','editing_instruction':'','visual_grounding':grounding}
    value={'decision':'accept','reason':'The baby is separate.',
           'dependency_class':'independent','target_description':grounding['target_description'],
           'editing_instruction':'Replace the man lying on the green sofa with a woman.'}
    assert validate(value,row,'scope')[1]=='policy_rejected_explicit_outside_dependency'


def test_hanging_outside_accessory_is_a_hard_replace_conflict():
    grounding={
        'target_description':'standing man in a black robe',
        'outside_description':"glasses hanging from the man's collar",
    }
    assert explicit_outside_dependency(grounding)=='held_attached'


def test_broad_permanent_structures_and_continuous_sections_are_rejected():
    assert broad_permanent_scene_edit(
        {'task_type':'remove'}, 'stone castle walls and towers behind the colonnade'
    )
    assert broad_permanent_scene_edit(
        {'task_type':'replace'}, 'section of dark metal railing behind the musician'
    )
    assert broad_permanent_scene_edit(
        {'task_type':'remove'}, 'modern skyscraper visible behind the traditional roof'
    )
    assert broad_permanent_scene_edit(
        {'task_type':'replace'}, 'weathered stone pillar on the temple entrance'
    )
    assert broad_permanent_scene_edit(
        {'task_type':'remove'},
        'vertical white marble panel with gold medallions between two fluted columns',
    )
    assert not broad_permanent_scene_edit(
        {'task_type':'remove'}, 'small white boat moored against the stone wall'
    )
    assert not broad_permanent_scene_edit(
        {'task_type':'remove'}, 'black freestanding lamp post on the right side of the street'
    )
    from pycocotools import mask as coco_mask
    encoded=coco_mask.encode(np.asfortranarray(np.ones((10,10),dtype=np.uint8)))
    encoded['counts']=encoded['counts'].decode('ascii')
    assert broad_permanent_scene_edit(
        {'task_type':'replace','mask':encoded,'visual_grounding':{
            'selected_surfaces':['weathered brick wall','tiled roof']}},
        'small gabled brick structure on the canal bank',
    )
    tree=np.zeros((10,10),dtype=np.uint8);tree[:4,:4]=1
    tree_encoded=coco_mask.encode(np.asfortranarray(tree))
    tree_encoded['counts']=tree_encoded['counts'].decode('ascii')
    assert broad_permanent_scene_edit(
        {'task_type':'replace','mask':tree_encoded},
        'large deciduous tree with yellowing leaves on the right',
    )


def test_addition_cannot_invent_a_garment_attachment_site():
    grounding={'target_description':'man in a grey cardigan',
               'reason':'The grey cardigan and blue shirt are selected.',
               'selected_surfaces':['grey knit cardigan','blue shirt']}
    assert not addition_site_is_grounded(
        "Add a red square to the breast pocket of the man's grey cardigan.", grounding
    )
    assert addition_site_is_grounded(
        "Add a red pin to the man's grey cardigan.", grounding
    )


def test_addition_body_site_must_be_visible_and_not_anatomically_ambiguous():
    rear_view={
        'target_description':'baseball player wearing a dark blue jersey with number 7',
        'reason':'The player is seen from behind.',
        'selected_surfaces':['dark blue jersey','grey baseball pants'],
    }
    assert not addition_site_is_grounded(
        "Add a white sticker to the player's upper chest.", rear_view
    )
    assert addition_uses_ambiguous_body_side(
        "Add a white sticker to the player's upper left chest."
    )
    assert addition_uses_ambiguous_body_side(
        "Add a gold brooch to the upper left side of the woman's patterned top."
    )
    assert not addition_uses_ambiguous_body_side(
        "Add a white sticker to the viewer's left side of the visible torso."
    )
    front_view={
        'target_description':'soldier facing the camera',
        'reason':'The front of the torso is visible.',
        'selected_surfaces':['camouflage helmet','camouflage uniform torso'],
    }
    assert addition_site_is_grounded(
        "Add a unit patch to the soldier's chest.", front_view
    )


def test_addition_must_be_visible_and_locally_placed_on_broad_host():
    assert addition_is_microscopic(
        'Add a small red pepper flake to the top slice of bread.'
    )
    assert not addition_is_microscopic(
        'Add a red label to the top slice of bread.'
    )
    building={
        'target_description':'large residential building on the left bank',
        'selected_surfaces':['white facade','blue glass panels'],
    }
    assert not addition_site_is_sufficiently_local(
        'Add a rectangular sign to the white facade of the building.', building
    )
    assert addition_site_is_sufficiently_local(
        'Add a rectangular sign above the main entrance of the building.', building
    )
    assert not addition_site_is_sufficiently_local(
        'Add a small square sign to the central glass facade.', building
    )
    assert addition_site_is_sufficiently_local(
        'Add a small square sign directly below the roof logo.', building
    )


def test_surface_detail_on_tiny_host_is_not_full_image_judgeable():
    from pycocotools import mask as coco_mask
    tiny=np.zeros((100,100),dtype=np.uint8);tiny[10:13,10:13]=1
    encoded=coco_mask.encode(np.asfortranarray(tiny))
    encoded['counts']=encoded['counts'].decode('ascii')
    row={'mask':encoded}
    assert addition_surface_detail_is_unjudgeable(
        row, 'Add a small blue sticker to the tiny traffic sign.'
    )
    assert not addition_surface_detail_is_unjudgeable(
        row, 'Add a clearly visible ribbon around the tiny traffic sign.'
    )
    small=np.zeros((100,100),dtype=np.uint8);small[10:16,10:16]=1
    encoded_small=coco_mask.encode(np.asfortranarray(small))
    encoded_small['counts']=encoded_small['counts'].decode('ascii')
    assert addition_surface_detail_is_unjudgeable(
        {'mask':encoded_small}, 'Add a small silver padlock to the metal gate.'
    )


def test_addition_stays_fine_grained_and_names_one_repeated_site():
    assert addition_is_oversized(
        'Add a large vertical red banner to the glass facade below the roof logo.'
    )
    assert not addition_is_oversized(
        'Add a compact red banner below the roof logo.'
    )
    forest={
        'target_description':'forest background of pine trees and branches',
        'reason':'Dense branches fill the selected background.',
        'selected_surfaces':['green pine needles','tree branches','patches of sky'],
    }
    assert addition_uses_underspecified_repeated_site(
        'Add green lichen to a pine branch in the upper-left forest background.',
        forest,
    )
    assert not addition_uses_underspecified_repeated_site(
        'Add green lichen to the branch immediately above the left antler.',
        forest,
    )
    assert addition_host_is_broad_natural_background(forest)
    assert not addition_host_is_broad_natural_background({
        'target_description':'single pine branch above the left antler',
        'selected_surfaces':['one thick branch'],
    })
    assert addition_reason_has_novelty_check(
        'Novelty check: no visually similar item is visible in the source.'
    )
    assert not addition_reason_has_novelty_check(
        'The proposed item is absent from the image.'
    )
    assert addition_reason_admits_existing_item(
        'Add a single chocolate chip to the center of the cookie.',
        'Novelty check: The source image contains chocolate chips on other cookies.',
    )
    assert not addition_reason_admits_existing_item(
        'Add a red star decal to the panel.',
        'Novelty check: The source image contains no star decals.',
    )


def test_addition_attachment_material_must_be_visually_grounded():
    wall={'target_description':'patterned wall panel',
          'selected_surfaces':['patterned white surface']}
    fridge={'target_description':'metal refrigerator door',
            'selected_surfaces':['steel door surface']}
    assert addition_requires_unsupported_material_claim(
        'Add a heart-shaped magnet to the wall panel.',wall
    )
    wall_with_outside_metal={**wall,
        'reason':'The panel is selected while metal cooking pots are outside.'}
    assert addition_requires_unsupported_material_claim(
        'Add a heart-shaped magnet to the wall panel.',wall_with_outside_metal
    )
    wall_with_metal_locator={
        'target_description':'patterned wall panel located to the left of the metal refrigerator',
        'selected_surfaces':['patterned white surface'],
    }
    assert addition_requires_unsupported_material_claim(
        'Add a heart-shaped magnet to the wall panel.',wall_with_metal_locator
    )
    assert not addition_requires_unsupported_material_claim(
        'Add a heart-shaped magnet to the refrigerator door.',fridge
    )


def test_addition_cannot_balance_a_loose_prop_on_bare_skin():
    grounding={'target_description':'shirtless man lying on the lower steps',
               'reason':'The man is lying prone with his bare back visible.',
               'selected_surfaces':['bare torso','dark swim trunks']}
    assert addition_places_loose_item_on_bare_body(
        'Place small white sunglasses on the back of the shirtless man.', grounding
    )
    assert not addition_places_loose_item_on_bare_body(
        'Add a silver bracelet to the wrist of the shirtless man.', grounding
    )
    assert addition_places_unfastened_item_on_fur(
        'Add a small yellow flower to the fur on the back of the polar bear.'
    )
    assert not addition_places_unfastened_item_on_fur(
        'Pin a small yellow flower to the selected costume fur.'
    )


def test_addition_requires_stable_free_support():
    assert addition_uses_unstable_narrow_support(
        'Place a small potted succulent on the top rail of the wooden chair.'
    )
    assert not addition_uses_unstable_narrow_support(
        'Attach a small brass nameplate to the front rail of the wooden chair.'
    )
    grounding={
        'target_description':'head and raised trunk of the elephant',
        'outside_description':'white hat resting on the tip of the selected elephant trunk',
    }
    assert addition_conflicts_with_excluded_contact(
        "Add a red flower to the tip of the elephant's trunk.", grounding
    )
    assert not addition_conflicts_with_excluded_contact(
        "Add a red ribbon around the elephant's neck.", grounding
    )
    occupied_top={
        'outside_description':(
            'The woman behind the stand and brochures resting on the top surface.'
        )
    }
    assert not addition_conflicts_with_excluded_contact(
        'Add a white sticker to the blue side panel of the political stand.',
        occupied_top,
    )


def test_addition_visible_body_must_fit_inside_mask():
    for instruction in [
        'Place a small traffic cone on the top surface of the container.',
        'Add a small bird perched on the thin branch.',
        'Attach a red flag to the metal pole.',
        'Add a small red flag to the stroller handle.',
        "Add a small red flag to the top of the park bench's backrest.",
        'Add a top hat to the head of the statue.',
        'Add a small white antenna to the center of the skyscraper roof.',
        'Place a small white coffee cup on the brochures on the stand.',
        'Place a small yellow hard hat on the top white bag of the pallet stack.',
        'Add a small wooden sign to the slatted picket fence.',
        "Add a small red flag to the upper center of the rusty gate's frame.",
    ]:
        assert addition_requires_pixels_outside_mask(instruction)
    for instruction in [
        'Add a red sticker to the front of the container.',
        'Attach a brass nameplate to the broad sign face.',
        'Add a silver watch to the selected wrist.',
    ]:
        assert not addition_requires_pixels_outside_mask(instruction)


def test_rigged_target_is_not_removed_from_external_equipment():
    grounding={
        'target_description':'second horse behind the foreground horse',
        'reason':'The selected horse is wearing tack.',
        'selected_surfaces':['dark coat','black harness and bridle'],
    }
    assert selected_external_rigging_risk(grounding)
    row={'task_type':'remove','editing_instruction':'','visual_grounding':grounding}
    value={'decision':'accept','reason':'The horse is selected.',
           'dependency_class':'none','target_description':grounding['target_description'],
           'editing_instruction':'Remove the second horse behind the foreground horse.'}
    assert validate(value,row,'scope')[1]=='policy_rejected_external_rigging_risk'


def test_ridden_transport_with_excluded_rider_or_attachment_is_not_replaced():
    grounding={
        'target_description':'beige bicycle frame and wheels ridden by the woman',
        'reason':'The bicycle is selected while the rider is excluded.',
        'selected_surfaces':['beige frame tubes','black tires','handlebars'],
        'outside_description':"the beige front basket and the woman's legs and shoes",
    }
    assert selected_external_rigging_risk(grounding)
    row={'task_type':'replace','editing_instruction':'','visual_grounding':grounding}
    value={'decision':'accept','reason':'The bicycle is selected.',
           'dependency_class':'none','target_description':grounding['target_description'],
           'editing_instruction':(
               'Replace the beige bicycle frame and wheels ridden by the woman '
               'with a red mountain bike.'
           )}
    assert validate(value,row,'scope')[1]=='policy_rejected_external_rigging_risk'


def test_remove_rejects_noninterchangeable_continuous_owner_part():
    target='red spotted cap of the mushroom structure'
    row={'task_type':'remove','editing_instruction':'','visual_grounding':{
        'target_description':target}}
    value={'decision':'accept','reason':'The cap is attached to the structure.',
           'dependency_class':'continuous_owner','target_description':target,
           'editing_instruction':'Remove the red spotted cap of the mushroom structure.'}
    assert validate(value,row,'scope')[1]=='contradictory_dependency_accept'


def test_remove_rejects_partial_living_body_and_integrated_rebar():
    assert partial_living_body_part_edit("person's hand gripping a blender")
    assert partial_living_body_part_edit('arm and torso of the person at the edge')
    assert partial_living_body_part_edit(
        'dark, shadowed legs and shoes of a group of people standing on the left'
    )
    assert not partial_living_body_part_edit('person partly cropped by the right image edge')
    target="person's hand gripping the side of a blender base"
    row={'task_type':'remove','editing_instruction':'','visual_grounding':{
        'target_description':target}}
    value={'decision':'accept','reason':'The hand is selected.',
           'dependency_class':'none','target_description':target,
           'editing_instruction':'Remove the hand gripping the side of the blender base.'}
    assert validate(value,row,'scope')[1]=='policy_rejected_partial_living_body_part'
    incomplete={
        'target_description':'person in a dark blue shirt and jeans',
        'outside_description':'another person to the left, and the head of the selected person',
    }
    assert whole_living_target_has_visible_part_outside(incomplete)
    rebar='rusted metal rebar cage above the concrete pillar'
    assert broad_permanent_scene_edit({'task_type':'remove'},rebar)


def test_remove_rejects_partial_integrated_mechanical_assembly():
    target='Front handlebars, gauges, and fender of the black motorcycle in the foreground'
    assert partial_integrated_mechanical_assembly(target)
    assert not partial_integrated_mechanical_assembly(
        'detachable black pannier on the rear of the motorcycle'
    )
    row={'task_type':'remove','editing_instruction':'','visual_grounding':{
        'target_description':target}}
    value={'decision':'accept','reason':'The selected components form one unit.',
           'dependency_class':'none','target_description':target,
           'editing_instruction':'Remove the front handlebars, gauges, and fender of the black motorcycle.'}
    assert validate(value,row,'scope')[1] == (
        'policy_rejected_partial_integrated_mechanical_assembly'
    )


def test_attribute_rejects_semantically_unresolved_background_patch():
    grounding={
        'target_description':(
            "blurred, light-grey vertical background surface located to the "
            "right of the tennis player's raised arm"
        ),
        'reason':(
            'The selected pixels are likely part of spectator clothing or a barrier.'
        ),
        'selected_surfaces':['blurred light-grey background surface'],
    }
    assert attribute_targets_unresolved_background(grounding)
    assert not attribute_targets_unresolved_background({
        'target_description':'painted concrete wall behind the bicycle',
        'reason':'The selected pixels are a clearly visible concrete wall.',
        'selected_surfaces':['painted concrete wall'],
    })
    row={'task_type':'attribute','editing_instruction':'','visual_grounding':grounding}
    value={'decision':'accept','reason':'The selected surface is visible.',
           'dependency_class':'none','target_description':grounding['target_description'],
           'editing_instruction':(
               "Change the blurred background surface to the right of the player's arm to blue."
           )}
    assert validate(value,row,'scope')[1]=='attribute_target_is_unresolved_background'


def test_unresolved_tiny_sliver_is_not_an_edit_unit():
    assert unresolved_tiny_fragment(
        'thin brown horizontal fragment or sliver resting on denim'
    )
    assert not unresolved_tiny_fragment('small brass nameplate on a wooden door')


def test_silhouette_does_not_justify_skin_and_multicolor_is_not_made_solid():
    silhouette='surfer in the right foreground seen from behind in silhouette'
    row={'task_type':'attribute','editing_instruction':'','visual_grounding':{
        'target_description':silhouette,
        'reason':'Only a dark solid silhouette is visible.',
        'selected_surfaces':['dark silhouette of torso and legs']}}
    value={'decision':'accept','reason':'The back is selected.','dependency_class':'none',
           'target_description':silhouette,
           'editing_instruction':"Change the skin tone of the surfer in the right foreground to orange."}
    assert validate(value,row,'scope')[1]=='attribute_material_not_visually_grounded'
    dog='brown and white dog sitting behind the standing dog'
    dog_row={'task_type':'attribute','editing_instruction':'','visual_grounding':{
        'target_description':dog,'reason':'Brown and white fur is visible.',
        'selected_surfaces':['brown and white fur']}}
    dog_value={'decision':'accept','reason':'The dog is selected.','dependency_class':'none',
               'target_description':dog,
               'editing_instruction':'Change the fur color of the brown and white dog sitting behind the standing dog to solid black.'}
    assert validate(dog_value,dog_row,'scope')[1]=='attribute_would_erase_visible_pattern'


def test_attribute_preserves_weathering_and_requires_visible_color_contrast():
    assert textured_or_weathered('mottled weathered stone surface')
    target='weathered stone pillar on the right'
    row={'task_type':'attribute','editing_instruction':'','visual_grounding':{
        'target_description':target,'reason':'The surface is weathered and mottled.',
        'selected_surfaces':['mottled black and white stone']}}
    value={'decision':'accept','reason':'The pillar is selected.','dependency_class':'none',
           'target_description':target,
           'editing_instruction':'Change the weathered stone pillar on the right to uniform light beige.'}
    assert validate(value,row,'scope')[1]=='attribute_would_erase_visible_texture'
    dark={'target_description':'man in a black suit','reason':'A black jacket is visible.',
          'selected_surfaces':['black suit jacket']}
    assert attribute_color_change_is_low_contrast(
        'Change the black suit jacket to navy blue.', dark
    )
    assert not attribute_color_change_is_low_contrast(
        'Change the black suit jacket to bright red.', dark
    )
    assert attribute_recolors_whole_pattern(
        "Change the color of the man's striped shirt to bright blue.",
        'brown and white striped shirt',
    )
    assert not attribute_recolors_whole_pattern(
        "Change the white stripes on the man's shirt to bright blue.",
        'brown and white striped shirt',
    )
    assert not attribute_recolors_whole_pattern(
        'Change the yellow stadium seats to bright green.',
        'red and yellow stadium bleachers with a blue floor',
    )
    elephant={
        'target_description':'elephant standing in the water on the right',
        'reason':'Dark grey wrinkled skin is visible.',
        'selected_surfaces':['wrinkled body skin'],
    }
    assert attribute_uses_implausible_living_tissue_color(
        'Change the elephant skin color to bright orange.', elephant
    )
    hand={
        'target_description':'person hand reaching in from the right',
        'selected_surfaces':['skin of the back of the hand'],
    }
    assert attribute_uses_implausible_living_tissue_color(
        'Change the skin tone of the hand to a bright solid blue.', hand
    )
    assert not attribute_uses_implausible_living_tissue_color(
        'Change the skin tone of the hand to warm tan.', hand
    )
    assert attribute_uses_implausible_intrinsic_material_color(
        'Change the red tomato sauce to a bright blue color.'
    )
    assert attribute_uses_implausible_intrinsic_material_color(
        'Change the red tomato sauce to a bright green color.'
    )
    assert not attribute_uses_implausible_intrinsic_material_color(
        'Change the white plate to a bright blue color.'
    )


def test_replace_rejects_vague_color_only_substitute():
    target='grey dump truck bed on the left'
    row={'task_type':'replace','editing_instruction':'','visual_grounding':{
        'target_description':target,'reason':'The selected truck bed is visible.'}}
    vague={'decision':'accept','reason':'The unit is selected.','dependency_class':'none',
           'target_description':target,
           'editing_instruction':'Replace the grey dump truck bed on the left with a blue one.'}
    assert validate(vague,row,'scope')[1]=='replace_target_is_only_a_color_variant'
    concrete={**vague,
              'editing_instruction':'Replace the grey dump truck bed on the left with a red flatbed trailer.'}
    assert validate(concrete,row,'scope')[1]=='accepted'


def test_replace_rejects_named_same_category_variants():
    for target,instruction in [
        ('green toothbrush standing in the holder',
         'Replace the green toothbrush standing in the holder with a blue toothbrush.'),
        ('reticulated giraffe in the foreground facing right',
         'Replace the reticulated giraffe in the foreground facing right with a reticulated giraffe.'),
        ('green helmet on the background cosplayer',
         'Replace the green helmet on the background cosplayer with a blue helmet.'),
    ]:
        row={'task_type':'replace','editing_instruction':'','visual_grounding':{
            'target_description':target,'reason':'The target is visible.'}}
        value={'decision':'accept','reason':'The target is selected.',
               'dependency_class':'none','target_description':target,
               'editing_instruction':instruction}
        assert validate(value,row,'scope')[1]=='replace_keeps_same_object_category'


def test_replace_rejects_same_object_changed_only_by_material():
    target='small metallic silver ball on the ground'
    instruction='Replace the small metallic silver ball on the ground with a red rubber ball.'
    assert replacement_is_merely_surface_variant(target,'red rubber ball.')
    row={'task_type':'replace','editing_instruction':'','visual_grounding':{
        'target_description':target,'reason':'The ball is visible.'}}
    value={'decision':'accept','reason':'The ball is selected.',
           'dependency_class':'none','target_description':target,
           'editing_instruction':instruction}
    assert validate(value,row,'scope')[1]=='replace_keeps_same_object_category'


def test_replace_rejects_same_category_style_disguised_as_new_form():
    target='skis worn by two foreground skiers'
    replacement='wide snowboard-style skis.'
    assert replacement_is_merely_surface_variant(target,replacement)
    assert replacement_is_merely_surface_variant(
        'tan umbrella with white figures',
        'red umbrella with white polka dots.',
    )
    assert replacement_is_merely_surface_variant(
        'person in a dark blue shirt and jeans',
        'person in a bright yellow raincoat.',
    )


def test_replace_rejects_same_flag_with_only_new_decoration():
    target='blue flag with a yellow emblem on the right'
    replacement='red flag featuring a white cross.'
    assert replacement_is_merely_surface_variant(target,replacement)
    assert replacement_is_merely_surface_variant(
        target, 'vertical red banner featuring a gold star.'
    )


def test_replace_rejects_surface_carrier_changed_only_by_pattern_or_content():
    assert replacement_is_merely_surface_variant(
        'dark blue necktie with a white emblem', 'red plaid tie.'
    )
    assert replacement_is_merely_surface_variant(
        'purple sign advertising massage services', 'blue real estate agency sign.'
    )
    assert replacement_is_merely_surface_variant(
        'purple sign advertising massage services', 'wooden restaurant menu board.'
    )
    assert replacement_is_merely_surface_variant(
        'green cucumber in the basket', 'long green zucchini.'
    )
    assert replacement_is_merely_surface_variant(
        'city street mural on the back wall', 'large abstract geometric painting.'
    )
    assert replacement_is_merely_surface_variant(
        'weathered stone pillar on the left', 'smooth polished marble column.'
    )
    assert replacement_is_merely_surface_variant(
        'leftmost desktop monitor', 'large flat-screen television.'
    )
    assert replacement_is_merely_surface_variant(
        'leaf-patterned armchair beside the sofa', 'solid red leather wingback chair.'
    )
    assert replacement_is_merely_surface_variant(
        'grey aircraft in the foreground', 'red and white commercial jet.'
    )
    assert replacement_is_merely_surface_variant(
        'weathered wooden boat on the mudflats', 'small red aluminum fishing skiff.'
    )
    assert replacement_is_merely_surface_variant(
        'green bushes flanking the truck', 'low dense red-leafed shrubs.'
    )
    assert not replacement_is_merely_surface_variant(
        'wooden armchair beside the sofa', 'black swivel chair.'
    )
    assert replacement_is_merely_surface_variant(
        'light green shirt worn by the person in the background',
        'red plaid flannel shirt.',
    )
    assert replacement_is_merely_surface_variant(
        'deep red coat worn by a spectator', 'bright yellow rain jacket.'
    )
    assert replacement_is_merely_surface_variant(
        'green glass bottle with a blue cap', 'blue plastic soda bottle.'
    )
    assert replacement_is_merely_surface_variant(
        'dark metal letterbox mounted on the door', 'white ceramic mail slot.'
    )
    assert replacement_is_merely_surface_variant(
        'hot dog on the left side of the plate', 'grilled bratwurst.'
    )
    assert not replacement_is_merely_surface_variant(
        'woman wearing a deep red coat', 'elderly man wearing a yellow rain jacket.'
    )


def test_edit_rejects_selected_reflection_or_excluded_corresponding_reflection():
    assert edit_conflicts_with_reflection({
        'target_description': 'reflection of the black and white cat in the mirror',
        'outside_description': 'the actual cat standing in front of the mirror',
    })
    assert edit_conflicts_with_reflection({
        'target_description': 'black and white cat standing in front of a mirror',
        'outside_description': 'reflection of the cat in the mirror, wooden floor',
    })
    assert not edit_conflicts_with_reflection({
        'target_description': 'red vase on the table',
        'outside_description': 'reflection of a person in a distant window',
    })

    target = 'reflection of the budgerigar below the wooden perch'
    row = {'task_type': 'attribute', 'editing_instruction': '', 'visual_grounding': {
        'target_description': target,
        'outside_description': 'actual budgerigar perched above the reflection',
    }}
    value = {
        'decision': 'accept', 'reason': 'The reflected bird is selected.',
        'dependency_class': 'none', 'target_description': target,
        'editing_instruction': 'Change the yellow feathers in the reflection below the wooden perch to red.',
    }
    assert validate(value, row, 'scope')[1] == 'policy_rejected_reflection_dependency'


def test_replace_rejects_obvious_footprint_expansion():
    assert replacement_has_incompatible_footprint(
        'wooden armchair on the left', 'modern black leather sofa.'
    )
    assert replacement_has_incompatible_footprint(
        'leaf-patterned armchair beside the sofa', 'simple wooden bench.'
    )
    assert not replacement_has_incompatible_footprint(
        'wooden armchair on the left', 'black swivel chair.'
    )
    assert replacement_has_incompatible_footprint(
        'desktop monitor on the left', 'large curved ultrawide monitor.'
    )
    assert replacement_has_incompatible_footprint(
        'airplane in the sky', 'large red biplane.', mask_area=0.001
    )
    assert replacement_has_incompatible_footprint(
        'crouching person in black clothing', 'standing skier in a red jacket.'
    )
    assert replacement_has_incompatible_footprint(
        'blue bus on the left', 'red double-decker bus.'
    )
    assert replacement_has_incompatible_footprint(
        'blue bus on the left', 'red articulated bus.'
    )
    assert replacement_has_incompatible_footprint(
        'hot dog on the left side of the plate', 'grilled chicken sandwich.'
    )
    assert replacement_has_incompatible_footprint(
        'dark metal letterbox on the door', 'round brass doorbell.'
    )
    assert replacement_has_incompatible_footprint(
        'wooden dining chair with a high backrest', 'simple wooden stool.'
    )
    assert replacement_has_incompatible_footprint(
        'wooden dining chair with a high backrest', 'white upholstered armchair.'
    )
    assert replacement_has_incompatible_footprint(
        'charred pizza on the foreground plate', 'slice of pepperoni pizza.'
    )
    assert replacement_has_incompatible_footprint(
        'tan umbrella with white figures', 'large bouquet of red roses.'
    )
    assert not replacement_has_incompatible_footprint(
        'elephant in the foreground', 'large brown bear.', mask_area=0.2
    )


def test_selected_container_with_excluded_contents_is_dependency_conflict():
    grounding={
        'target_description':'large gold metallic planter pot on the right',
        'outside_description':'green coniferous plant inside the pot; pavement below',
    }
    assert explicit_outside_dependency(grounding)=='supported_by_target'
    grounding['outside_description']='green foliage inside the pot; pavement below'
    assert explicit_outside_dependency(grounding)=='supported_by_target'


def test_grounding_cannot_append_an_unsampled_outside_object_noun():
    grounding={
        'target_description':'park bench with slatted backrest and wheel',
        'marker_observations':['TARGET 1: wooden slats of the bench backrest'],
        'selected_surfaces':['slatted wooden backrest','metal bench frame'],
        'outside_description':'decorative cart with spoked wheels to the right',
    }
    assert target_promotes_unsampled_outside_noun(grounding)
    grounding['marker_observations'].append('TARGET 2: rubber wheel of the bench')
    assert not target_promotes_unsampled_outside_noun(grounding)


def test_instruction_cannot_delegate_choice_among_repeated_local_parts():
    assert instruction_uses_ambiguous_member_reference(
        'Add a ladybug to one of the purple flower spikes.'
    )
    assert not instruction_uses_ambiguous_member_reference(
        'Add a ladybug to the leftmost purple flower spike.'
    )


def test_add_accepts_apply_for_surface_graphics():
    target='wooden picket fence on the right side of the path'
    row={'task_type':'add','editing_instruction':'','visual_grounding':{
        'target_description':target,
        'reason':'The wooden pickets are selected.',
        'selected_surfaces':['vertical wooden pickets'],
        'outside_description':'vegetation behind the fence',
    }}
    value={'decision':'accept','reason':'Novelty check: no similar sticker is visible. A sticker fits one picket.',
           'dependency_class':'none','target_description':target,
           'editing_instruction':(
               'Apply a small red heart sticker to one picket on the right fence.'
           )}
    assert validate(value,row,'scope')[1]=='accepted'


def test_replace_rejects_person_changed_only_by_uniform_color():
    target='player in the yellow uniform with number 16 on the back'
    replacement='player in a red uniform.'
    assert replacement_is_merely_surface_variant(target,replacement)


def test_replace_rejects_group_changed_only_by_food_coating():
    target='group of white-frosted and powdered donuts in the center'
    replacement='cluster of chocolate-frosted donuts.'
    assert replacement_is_merely_surface_variant(target,replacement)
    row={'task_type':'replace','editing_instruction':'','visual_grounding':{
        'target_description':target,'reason':'The donut group is selected.'}}
    value={'decision':'accept','reason':'The group is selected.',
           'dependency_class':'none','target_description':target,
           'editing_instruction':(
               'Replace the group of white-frosted and powdered donuts in the '
               'center with a group of chocolate-frosted donuts.'
           )}
    assert validate(value,row,'scope')[1]=='replace_keeps_same_object_category'


def test_replace_allows_recognizable_same_supercategory_identity_changes():
    cases=[
        ('sheep in the middle of the pen',
         'Replace the sheep in the middle of the pen with a black and white Dorper sheep.'),
        ('female soccer player in a blue jersey',
         'Replace the female soccer player in a blue jersey with a male player wearing red.'),
        ('bouquet of red and orange flowers',
         'Replace the bouquet of red and orange flowers with a bouquet of white lilies.'),
    ]
    for target,instruction in cases:
        row={'task_type':'replace','editing_instruction':'','visual_grounding':{
            'target_description':target,'reason':'The target is visible.'}}
        value={'decision':'accept','reason':'The new identity is visibly distinct.',
               'dependency_class':'none','target_description':target,
               'editing_instruction':instruction}
        assert validate(value,row,'scope')[1]=='accepted'
