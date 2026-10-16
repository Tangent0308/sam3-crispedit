from sa1b_pipeline.blind_identifiability import evaluate_probe, probe_request


def group(op="attribute", scopes=("backpack", "shirt")):
    return {
        "source_image": "clean.jpg",
        "instruction_noref": "Change the color of the selected region to red.",
        "operation": op,
        "instruction_spec": {"attributes": [{"property": "color", "value": "red"}]},
        "members": [
            {"scope": s, "mask": "secret.png", "ref": "secret ref"} for s in scopes
        ],
    }


def raw(**kw):
    return dict(
        {
            "blind_schema": 3,
            "decision": "ambiguous",
            "semantic_scope": "broad",
            "examples": ["a visible shirt", "a visible bag"],
            "evidence": "ordinary recoloring applies to several objects",
        },
        **kw,
    )


def test_no_locators_and_no_small_object_bbox_requirement():
    g = group()
    images, prompt = probe_request(g)
    assert images == ["clean.jpg"]
    assert "secret" not in prompt
    assert evaluate_probe(g, raw())["pass"]


def test_uncertain_and_specific_not_passed():
    for decision in ["specific", "uncertain"]:
        r = raw()
        r["decision"] = decision
        assert not evaluate_probe(group(), r)["pass"]


def test_payload_coverage_and_no_broad_replace_bypass():
    g = group("replace", ("window", "road marking"))
    assert not evaluate_probe(g, raw())["pass"]
    r = raw()
    r.update(semantic_scope="restricted", eligible_categories=["road marking"])
    assert not evaluate_probe(g, r)["pass"]
    g = group("replace", ("lamp head", "street lamp head"))
    r["eligible_categories"] = ["lamp head"]
    assert evaluate_probe(g, r)["pass"]


def test_missing_examples_and_truncation_fail_closed():
    for change in [
        {"examples": []},
        {"examples": ["same", "same"]},
        {"_generation_metadata": {"finish_reason": "length"}},
    ]:
        r = raw()
        r.update(change)
        assert not evaluate_probe(group(), r)["pass"]


def test_object_adjectives_do_not_create_false_category_mismatch():
    from sa1b_pipeline.blind_identifiability import _category_covered

    assert _category_covered("canvas bag", ["military bag"])
    assert _category_covered("blue storage box", ["wooden box"])
    assert not _category_covered("door handle", ["window handle"])
    assert not _category_covered("window", ["road marking"])


def test_v4_example_omission_is_diagnostic_not_incompatibility():
    r=raw(blind_schema=4, semantic_scope='restricted', eligible_categories=['turnstile','sign'], excluded_categories=[])
    g=group('add',('indicator','sign'))
    result=evaluate_probe(g,r)
    assert result['pass'] and result['coverage']==[False,True]
    assert result['requires_localized_shared_edit_audit']


def test_v4_explicit_incompatible_member_is_rejected():
    r=raw(blind_schema=4, semantic_scope='restricted', eligible_categories=['table'], excluded_categories=['chair'])
    result=evaluate_probe(group('add',('table','chair')),r)
    assert not result['pass']
    assert result['reason']=='member_explicitly_incompatible_with_payload'


def test_v4_requires_valid_exclusion_response_and_blind_ambiguity():
    r=raw(blind_schema=4, eligible_categories=[])
    assert not evaluate_probe(group(),r)['pass']
    for decision in ['uncertain','specific']:
        assert not evaluate_probe(group(),dict(r,excluded_categories=[],decision=decision))['pass']
    assert not evaluate_probe(group(),dict(r,excluded_categories=[],examples=[]))['pass']
