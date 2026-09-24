import pytest
from synthesis_pipeline.collect_reviewed_cohorts import validate_review


def test_release_requires_explicit_review_not_model_acceptance():
    row=dict(editing_instruction='Remove the left cup.',new_instruction='Remove the left cup.',
             verification='model_verified_not_assistant_reviewed',
             assistant_review=dict(visual_quality='pass',instruction_match='pass'))
    with pytest.raises(ValueError):validate_review(row)
    row['verification']='assistant_reviewed_original_instruction'
    assert validate_review(row)['visual_quality']=='pass'
    row['new_instruction']='Remove the right cup.'
    with pytest.raises(ValueError):validate_review(row)


def test_release_rewrite_review_must_match_exact_text():
    row=dict(editing_instruction='Add a pin to his right cuff.',verification='assistant_reviewed_rewrite',
             assistant_review=dict(visual_quality='pass',instruction_match='fail',rewrite_match='pass',
                                   reviewed_instruction='Add a pin to his left cuff.'))
    with pytest.raises(ValueError):validate_review(row)
    row['assistant_review']['reviewed_instruction']=row['editing_instruction']
    assert validate_review(row)['rewrite_match']=='pass'
