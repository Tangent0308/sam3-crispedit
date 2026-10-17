"""Structured decoder for the single-call multi-target planner; semantics stay in validators."""


def multi_plan_schema():
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    text = {"type": "string"}
    boolean = {"type": "boolean"}
    grounding = obj(dict(entity=text, mask_description=text, visible_before=text, current_value=text,
                         edit_feasible=boolean, reason=text))
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

    target = obj(dict(unit_id=text, object_id=text, grounding=grounding,
                     ref_phrase=text, operation={"type": "string", "enum": ["add", "remove", "replace", "attribute"]},
                     payload=text, target_part={"const": ""}, remove_mode={"const": "whole"},
                     placement={"anyOf": [placement, {"type": "null"}]},
                     attributes={"type": "array", "items": attribute, "maxItems": 1},
                     qualifier={"const": ""}, evidence=text))
    return obj(dict(grounding_version={"const": 3}, feasible=boolean, targets={"type": "array", "items": target, "maxItems": 5},
                    generation_order={"type": "array", "items": text, "maxItems": 5}, evidence=text))
