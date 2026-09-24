import pytest

from crispedit.mask.matching import match_single_candidate, single_object_prompt


@pytest.mark.parametrize('source,target,description,expected', [
    ('sofa','sofa with blanket','A blue blanket was added to the backrest.','blanket'),
    ('teal satin shoe','teal satin shoe with bow','A bow was added.','bow'),
    ('drawer pull','drawer pull with tassel','A tassel appeared.','tassel'),
    ('empty plate','plate with cookies','The plate is now filled with cookies.','cookies'),
    ('the empty bridge','the bridge with animals','The bridge is now populated with animals.','animals'),
    ('bare arm','arm with tattoo','A tattoo was added.','tattoo'),
    ('empty space','vase with flowers','The space is filled with a new vase.',''),
    ('empty plate','plate with cookies','The plate was replaced and filled with cookies.',''),
    ('empty red plate','blue plate with cookies','Cookies were added.',''),
    ('plate','plate with cookies','Cookies were recolored.',''),
    ('','chair with cushions','A chair was added.',''),
    ('red chair','blue chair with cushions','Cushions were added and chair recolored.',''),
    ('chair','chair with cushions','The chair was replaced and cushions added.',''),
    ('chair','chair with cushions','The chair moved.',''),
    ('chair with cushions','chair with cushions','Cushions changed color.',''),
])
def test_added_attachment_requires_explicit_addition_and_same_existing_owner(source,target,description,expected):
    from crispedit.mask.matching import added_attachment_ref
    assert added_attachment_ref(dict(source_ref=source,target_ref=target,change=description)) == expected


def test_disjoint_body_concepts_are_queried_separately_but_limb_is_not_split():
    from crispedit.mask.matching import atomic_body_refs
    assert atomic_body_refs('face and arms') == ['face','arms']
    assert atomic_body_refs('arm and hand') == ['arm and hand']
    assert atomic_body_refs('chair with cushions') == ['chair with cushions']


@pytest.mark.parametrize('ref',['orange car','orange car body','red car topper','person','black jacket','green robe','wooden chair'])
def test_single_object_can_be_matched_without_unioning_neighbors(ref):
    assert single_object_prompt(ref)


@pytest.mark.parametrize('ref',['cookies','arms and hands','exposed face skin','chairs with cushions','pair of eyes','stack of plates'])
def test_plural_and_multipart_concepts_keep_multiple_instances(ref):
    assert not single_object_prompt(ref)


def test_layout_and_sparse_hints_override_singular_spelling():
    assert not single_object_prompt('armchair',layout='nearby_group')
    assert not single_object_prompt('bird',density='sparse')
    assert not single_object_prompt('person',region_mode='aggregate_region')


def test_solid_object_groups_are_not_particle_tiles():
    from crispedit.mask.pipeline import _region_mode, _mask_density, _sam_text_prompt
    for ref in ['group of men in beige coats','desk with grey top','armchairs']:
        item = dict(ref=ref,region_layout='nearby_group',region_mode='aggregate_region',mask_density='sparse')
        assert _region_mode(item) == _mask_density(item,'object') == 'object'
    assert _sam_text_prompt('group of men in beige coats') == 'men'
    for ref in ['confetti','string of lights','flowers','tattoo on people']:
        item = dict(ref=ref,region_layout='nearby_group',region_mode='aggregate_region',mask_density='sparse')
        assert _mask_density(item,_region_mode(item)) == 'sparse'


def test_spatial_match_beats_high_scoring_neighbor():
    intended = dict(box_iou=.85,concept_score=.8)
    neighbor = dict(box_iou=.2,concept_score=.99)
    assert match_single_candidate([neighbor,intended]) == [intended]
    assert match_single_candidate([]) == []


def test_pcs_single_selection_does_not_union_neighbor_pixels():
    import numpy as np
    import torch
    from crispedit.mask.pipeline import _pcs_mask

    masks = torch.zeros((2, 1, 100, 100))
    masks[0, 0, 10:90, 10:75] = 1
    masks[1, 0, 10:30, 76:90] = 1
    output = dict(masks=masks, boxes=torch.tensor([[10,10,75,90],[76,10,90,30]]),
                  scores=torch.tensor([.85,.99]))
    class Processor:
        def reset_all_prompts(self, state): pass
        def set_text_prompt(self, **kwargs): return output
    args = (Processor(), {}, 'car', np.array([10,10,90,90]), (100,100))
    single, audit = _pcs_mask(*args, single_instance=True)
    group, _ = _pcs_mask(*args)
    assert single[50,50] and not single[20,80] and group[20,80]
    assert audit['accepted_count'] == 2 and audit['selected_count'] == 1
def test_container_prompt_keeps_whole_object_scope_without_expanding_contents():
    from crispedit.mask.matching import container_object_prompt
    assert container_object_prompt('white bowl containing pie and a fork', 'replace') == 'bowl'
    assert container_object_prompt('basket with bread', 'remove') == 'basket'
    for ref in ('slice of pie with fork', 'rim of bowl', 'food in a bowl', 'bird head with crest', 'chairs with cushions'):
        assert container_object_prompt(ref, 'replace') == ''
    for edit_type in ('add', 'color', 'motion'):
        assert container_object_prompt('bowl with cookies', edit_type) == ''
    assert container_object_prompt('bowl with cookies', 'remove', layout='nearby_group') == ''
