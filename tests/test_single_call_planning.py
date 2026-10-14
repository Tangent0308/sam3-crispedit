import json
from collections import Counter

import numpy as np
from PIL import Image
import pytest

from sa1b_pipeline.per_mask import plan_all_masks
from sa1b_pipeline.schema import Candidate, CandidateObject


@pytest.mark.parametrize('width', [5, 11])
@pytest.mark.parametrize('shape', [(1, 1), (3, 8), (29, 31)])
def test_fast_contour_matches_pillow_including_edges_and_holes(width, shape):
    from PIL import ImageFilter
    from sa1b_pipeline.per_mask import _outer_mask_band
    mask = (np.random.default_rng(7).random(shape) > .75).astype('uint8') * 255
    mask[0, 0] = 255
    image = Image.fromarray(mask)
    expected = (np.asarray(image.filter(ImageFilter.MaxFilter(width))) > 0) & (mask == 0)
    np.testing.assert_array_equal(_outer_mask_band(image, width), expected)


class Model:
    model_id = 'test-single-call'
    def __init__(self, mutation=None):
        self.calls = []
        self.mutation = mutation

    def generate_multimage(self, jobs):
        paths, prompt = jobs[0]
        self.calls.append((paths, prompt))
        assert len(paths)==2
        assert max(Image.open(paths[1]).size)<=934
        assert paths[0].endswith('scene_locator.png')
        assert paths[1].endswith('annotated_crop.png')
        context = json.loads(prompt.split('Task: ', 1)[1].splitlines()[0])
        op = context['preferred_operation']
        row = dict(feasible=True, object_id=context['object_id'], category='bag', operation=op,
                   ref_phrase='the orange bag held by the man in the striped shirt', scope='bag',
                   source_color='orange', payload={'add':'a water bottle','remove':'','replace':'a canvas tote bag','attribute':''}[op],
                   attributes=[{'property':'color','value':'blue'}] if op=='attribute' else [],
                   placement={'relation':'in','site':'side pocket','anchor_mode':'support'} if op=='add' else None)
        if self.mutation:
            self.mutation(row, context)
        return [row]


@pytest.fixture
def source(tmp_path):
    Image.new('RGB', (80,80), (30,100,200)).save(tmp_path/'source.png')
    # Tiny, border-touching masks formerly failed geometry eligibility gates.
    mask = np.zeros((80,80),dtype=np.uint8)
    mask[:3,:20] = 255
    Image.fromarray(mask).save(tmp_path/'mask.png')
    return Candidate('scene', str(tmp_path/'source.png'), [
        CandidateObject(str(i), 'noisy source label', mask_path=str(tmp_path/'mask.png')) for i in range(41)])


def test_one_call_for_every_mask_no_quality_or_overlap_shortlist(source,tmp_path):
    model=Model();counts=Counter()
    cases=plan_all_masks([source],model,work_dir=tmp_path/'run',counts=counts)
    assert len(cases)==41 and len(model.calls)==41
    assert sorted(counts.values())==[10,10,10,11]
    assert len({c.case_id for c in cases})==41
    assert all(c.selector['model_calls']==1 and not c.selector['semantic_audit_performed'] for c in cases)
    stages=[json.loads(l)['stage'] for l in (tmp_path/'run/audit.jsonl').read_text().splitlines()]
    assert Counter(stages)=={'single_call_plan':41,'terminal':41}


def test_repeated_reference_is_recorded_without_extra_calls_or_mask_rejection(source,tmp_path):
    source.objects=source.objects[:2]
    mask=np.zeros((80,80),dtype=np.uint8);mask[60:65,60:70]=255
    path=tmp_path/'other_mask.png';Image.fromarray(mask).save(path)
    source.objects[1].mask_path=str(path)
    model=Model()
    cases=plan_all_masks([source],model,work_dir=tmp_path/'run')
    assert len(cases)==len(model.calls)==2
    flags=[json.loads(l) for l in (tmp_path/'run/reference_collisions.jsonl').read_text().splitlines()]
    assert len(flags)==1 and flags[0]['needs_reference_review']
    assert not flags[0]['source_mask_rejected']


@pytest.mark.parametrize('mutation',[
    lambda r,c:r.update(object_id='wrong-id'),
    lambda r,c:r.update(feasible=False,reason='Cannot distinguish the target'),
    lambda r,c:r.update(_parse_error='truncated JSON'),
])
def test_failed_output_is_not_retried(source,tmp_path,mutation):
    source.objects=source.objects[:1];model=Model(mutation)
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run')
    assert len(model.calls)==1


@pytest.mark.parametrize('reference', [
    'the petal just inside the cyan box',
    'the leaf in the selected region',
    'the object shown in the crop',
    'the curled leaf outlined in magenta',
    'the leaf below the cyan-marked area',
    'the object beside the pink filled region',
    'the right-hand shoe displayed inside the inset frame',
    'the red sneaker shown in the right-side inset',
    'the petal left of the cyan-marked upper-right petal group',
    'the leaf pictured in the upper inset',
    'the red sneaker shown in the inset on the right side of the scene',
    'the shoe displayed inside an inset',
])
def test_annotation_dependent_reference_is_instruction_failure_without_retry(source,tmp_path,reference):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(ref_phrase=reference))
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run')
    assert len(model.calls)==1
    rows=[json.loads(l) for l in (tmp_path/'run/audit.jsonl').read_text().splitlines()]
    assert rows[-1]['reason']=='reference_depends_on_annotation'


@pytest.mark.parametrize('reference', ['the cyan gas cylinder beside the welder',
                                     'the surgical mask held by the nurse',
                                     'the crop growing behind the fence',
                                     'the window with an inset frame beside the door'])
def test_real_object_is_not_an_annotation_reference(source,tmp_path,reference):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(ref_phrase=reference))
    assert len(plan_all_masks([source],model,work_dir=tmp_path/'run'))==1


def test_in_call_fallback_keeps_quota_auditable(source,tmp_path):
    source.objects=source.objects[:1]
    def fallback(row,context):
        row.update(operation='replace',payload='a canvas tote bag',placement=None,
                   skipped_operations=[{'operation':'add','reason':'No suitable accessory for this target'}])
    model=Model(fallback);counts=Counter()
    cases=plan_all_masks([source],model,work_dir=tmp_path/'run',counts=counts,allow_operation_fallback=True)
    assert len(cases)==1 and len(model.calls)==1
    assert counts=={'replace':1}
    assert cases[0].selector['preferred_operation']=='add'


def test_fixed_type_cannot_silently_change(source,tmp_path):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(operation='remove',payload='',placement=None))
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run',operation_policy='replace')
    assert len(model.calls)==1


def test_identifiable_target_has_no_blind_point_requirement(source,tmp_path):
    source.objects=source.objects[:1]
    model=Model();cases=plan_all_masks([source],model,work_dir=tmp_path/'run',operation_policy='attribute')
    assert len(cases)==1
    assert 'orange bag held by the man in the striped shirt' in cases[0].units[0].instruction_ref
    assert 'reference_grounding' not in cases[0].selector
    assert len(model.calls)==1


def test_missing_fallback_bookkeeping_does_not_reject_valid_instruction(source,tmp_path):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(operation='remove',payload='',placement=None))
    cases=plan_all_masks([source],model,work_dir=tmp_path/'run',allow_operation_fallback=True)
    assert len(cases)==1 and cases[0].selector['operation_fallback']
    assert len(model.calls)==1


def test_balanced_default_rejects_type_fallback_without_retry(source,tmp_path):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(operation='attribute',payload='',placement=None,
                                   attributes=[{'property':'color','value':'blue'}]))
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run')
    assert len(model.calls)==1
    terminal=json.loads((tmp_path/'run/audit.jsonl').read_text().splitlines()[-1])
    assert terminal['reason']=='operation_not_allowed'
    assert 'Only the single operation' in model.calls[0][1]


def test_explicit_removal_scope_conflict_is_rejected_without_extra_call(source,tmp_path):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(removal_preserves_unselected=False))
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run',operation_policy='remove')
    assert len(model.calls)==1


@pytest.mark.parametrize('extra', [
    {'target_kind': 'anatomy'},
    {'target_kind': 'part', 'removal_is_natural': False},
])
def test_anatomically_or_physically_unreasonable_removal_rejected(source,tmp_path,extra):
    source.objects=source.objects[:1]
    model=Model(lambda row,ctx: row.update(extra))
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run',operation_policy='remove')
    assert len(model.calls)==1


@pytest.mark.parametrize('scope,ref', [
    ('arm', "the background person's arm visible behind the floral shirt sleeve"),
    ('left hand', 'the woman\u2019s visible left hand beside the bag'),
    ('forearm', 'the forearm of the worker beside the cylinder'),
    ('hand', 'the blurred hand of the out-of-focus person in the white shirt'),
    ('arm', 'the arm of the blurred background person behind the sleeve'),
])
def test_explicit_human_removal_cannot_bypass_check_with_part_kind(source,tmp_path,scope,ref):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(target_kind='part',removal_is_natural=True,
                                   scope=scope,ref_phrase=ref))
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run',operation_policy='remove')
    assert len(model.calls)==1
    terminal=json.loads((tmp_path/'run/audit.jsonl').read_text().splitlines()[-1])
    assert terminal['reason']=='unnatural_removal_of_target'


@pytest.mark.parametrize('scope,ref,kind', [
    ('arm', 'the arm of the chair beside the man', 'part'),
    ('hand', 'the minute hand of the clock behind the woman', 'part'),
    ('arm', "the arm of the man's chair beside the desk", 'part'),
    ('tattoo', "the tattoo on the person's arm", 'part'),
    ('bracelet', "the bracelet around the woman's wrist", 'object'),
    ('arm', "the printed figure of a person's arm on the shirt", 'printed_motif'),
])
def test_human_part_check_does_not_reject_objects_or_depictions(source,tmp_path,scope,ref,kind):
    source.objects=source.objects[:1]
    model=Model(lambda r,c:r.update(target_kind=kind,removal_is_natural=True,
                                   scope=scope,ref_phrase=ref))
    assert len(plan_all_masks([source],model,work_dir=tmp_path/'run',operation_policy='remove'))==1
    assert len(model.calls)==1


def test_color_only_replacement_rejected_without_an_extra_call(source,tmp_path):
    source.objects=source.objects[:1]
    model=Model(lambda row,ctx: row.update(replacement_changes_form=False))
    assert not plan_all_masks([source],model,work_dir=tmp_path/'run',operation_policy='replace')
    assert len(model.calls)==1


def test_annotation_does_not_change_scene_or_execution_mask(source,tmp_path):
    from sa1b_pipeline.per_mask import single_call_views
    paths, _, mask_path, _, box = single_call_views(source,source.objects[0],tmp_path/'views')
    panel=Image.open(paths[1]).convert('RGB')
    crop=np.asarray(panel.crop((0,38,panel.width,panel.height)))
    expected=np.asarray(Image.open(source.image_path).convert('RGB').crop(box).resize((panel.width,panel.height-38),Image.Resampling.LANCZOS))
    selected=np.asarray(Image.open(mask_path).crop(box).resize((panel.width,panel.height-38),Image.Resampling.NEAREST))>0
    assert np.array_equal(crop[selected],expected[selected])
    assert np.array_equal(np.array(Image.open(mask_path)),np.array(Image.open(source.objects[0].mask_path)))
    assert len(paths)==2


def test_custom_crop_keeps_mask_and_pixels_and_has_distinct_cache_key(source,tmp_path):
    from sa1b_pipeline.per_mask import single_call_views
    base = single_call_views(source, source.objects[0], tmp_path/'views')
    paths, geometry, mask_path, digest, box = single_call_views(
        source, source.objects[0], tmp_path/'views', crop_padding_factor=1.3,
        crop_min_fraction=.025, crop_min_pixels=12, crop_max_edge=512)
    assert digest != base[3]
    assert len(paths) == 2
    panel = Image.open(paths[1]).convert('RGB')
    size = (panel.width, panel.height-38)
    assert max(size) == 512
    source_mask = np.array(Image.open(source.objects[0].mask_path))
    assert np.array_equal(np.array(Image.open(mask_path)), source_mask)
    expected = np.array(Image.open(source.image_path).convert('RGB').crop(box).resize(size,Image.Resampling.LANCZOS))
    selected = np.array(Image.open(mask_path).crop(box).resize(size,Image.Resampling.NEAREST)) > 0
    actual = np.array(panel.crop((0,38,panel.width,panel.height)))
    assert np.array_equal(actual[selected], expected[selected])
    assert geometry == base[1]
def test_selected_pixel_evidence_excludes_unselected_white_neighbours():
    import numpy as np
    from PIL import Image
    from sa1b_pipeline.per_mask import selected_pixel_evidence
    rgb = np.full((20, 20, 3), 255, dtype=np.uint8)
    rgb[:, 10] = (210, 25, 35)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 10] = 255
    evidence = selected_pixel_evidence(Image.fromarray(rgb), mask, max_samples=7)
    assert evidence['selected_pixels'] == 20
    assert evidence['sampled_pixels'] == 7
    assert evidence['median_rgb'] == [210, 25, 35]
    assert evidence['frequent_rgb_bins'] == [{'rgb': [210, 25, 35], 'sample_fraction': 1.0}]
    assert np.all(rgb[:, 0] == 255) and np.all(mask[:, 0] == 0)
