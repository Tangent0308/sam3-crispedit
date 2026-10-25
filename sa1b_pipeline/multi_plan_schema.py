"""Structured decoder for the single-call multi-target planner; semantics stay in validators."""


def multi_plan_schema(*, execution_contract=False):
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    text = {"type": "string"}
    boolean = {"type": "boolean"}
    grounding = obj(dict(entity=text, mask_description=text, visible_before=text, current_value=text,
                         edit_feasible=boolean, reason=text,
                         extent_match=boolean, ref_unique=boolean,
                         unsupported_after={'type':'array','items':text,'maxItems':4},
                         preserve={'type':'array','items':text,'maxItems':3}, required_change=text,
                         contact_xy={'type':'array','items':{'type':'number','minimum':0,'maximum':1},'maxItems':2}))
    placement = obj(dict(relation={"type": "string", "enum": [
        "on", "in", "inside", "to", "around", "across", "along", "over", "at",
        "attached to", "clipped to", "hanging from", "resting on", "mounted on",
        "sewn onto", "tied around", "pinned to", "hooked over", "strapped over"]},
        site=text, anchor_mode={"const": "support"}))
    attribute = {"anyOf": [
        obj(dict(property={"const": "opacity"},
                 value={"enum": ["transparent", "translucent", "opaque"]})),
        obj(dict(property={"type": "string", "enum": [
            "color", "material", "finish", "pattern", "texture", "text", "tint"]}, value=text)),
    ]}

    # Source IDs are bookkeeping, not a visual prediction. Bind from the U tag
    # after decoding, so example IDs cannot become a second conflicting key.
    target = obj(dict(unit_id=text, object_id={"const": ""}, grounding=grounding,
                     ref_phrase=text, operation={"type": "string", "enum": ["add", "remove", "replace", "attribute"]},
                     payload=text, target_part={"const": ""}, remove_mode={"const": "whole"},
                     placement={"anyOf": [placement, {"type": "null"}]},
                     attributes={"type": "array", "items": attribute, "maxItems": 1},
                     qualifier={"const": ""}, evidence=text))
    result = obj(dict(grounding_version={"const": 4}, feasible=boolean, targets={"type": "array", "items": target, "maxItems": 5},
                    generation_order={"type": "array", "items": text, "maxItems": 5},
                    edit_dependencies={"type":"array", "maxItems":10, "items":obj(dict(before=text,after=text,
                        relation={"enum":["reveals_edited_surface","uses_edited_host","changes_shared_context"]}, evidence=text))}, evidence=text))

    if not execution_contract:
        for key in ('extent_match', 'ref_unique', 'unsupported_after', 'preserve', 'required_change'):
            del grounding['properties'][key]
            grounding['required'].remove(key)
        result['properties']['grounding_version']={'const':3}
        del result['properties']['edit_dependencies']
        result['required'].remove('edit_dependencies')
    return result
