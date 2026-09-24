from synthesis_pipeline.plan_removal_relations import PROMPT_V6, validate_relation_plan


def test_v6_requires_post_removal_support_reasoning():
    text = PROMPT_V6.lower()
    assert 'sole physical support' in text
    assert 'support-surface rule' in text
    assert 'defer' in text
    assert 'remove_together' in text


def test_deferred_plan_is_safe_even_if_model_relation_list_is_malformed():
    value = {'decision': 'defer', 'reason': 'support conflict', 'relations': ['not a relation']}
    assert validate_relation_plan(value, policy='relations-v6') == 'defer'
