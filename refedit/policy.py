"""Deterministic RefEdit task routing around the paired-image MLLM planner."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict

from refedit import GROUND_PROMPT_VERSION


@dataclass(frozen=True)
class TaskInference:
    final_task: str
    reason: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


_ADD = re.compile(r"^(?:add|place|put|insert|include|introduce)\b", re.I)
_REMOVE = re.compile(
    r"^(?:remove|delete|erase|eliminate|take away|get rid of)\b", re.I
)
_REPLACE = re.compile(r"^(?:replace|swap|substitute|transform)\b", re.I)
_MOVE = re.compile(r"^move\b", re.I)
_MATERIAL = re.compile(
    r"\b(?:material|texture|finish|upholstery|fabric|wooden|wood|metallic|metal|"
    r"marble|stone|brick|glass|leather|rubber|ceramic|wool|fur)\b",
    re.I,
)
_APPEARANCE_NOUN = re.compile(r"\b(?:color|colour)\b", re.I)
_COLOR = re.compile(
    r"\b(?:red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey|"
    r"gold|golden|silver|beige|cyan|magenta)\b",
    re.I,
)


def infer_task(instruction: object) -> TaskInference:
    """Map RefEdit's synthetic instruction families to ScaleEdit task policies.

    RefEdit does not publish an edit-type column.  The generator uses highly
    regular imperative templates, so verb-first routing is both deterministic
    and auditable.  The paired-image planner remains the source of truth for
    the realized region and geometry.
    """

    text = " ".join(str(instruction or "").strip().split())
    if not text:
        raise ValueError("RefEdit instruction is empty")
    if _ADD.search(text):
        return TaskInference("object_addition", "leading_addition_verb")
    if _REMOVE.search(text):
        return TaskInference("object_removal", "leading_removal_verb")
    if _MOVE.search(text):
        return TaskInference("action_editing", "leading_move_verb")
    if _REPLACE.search(text) or re.match(r"^turn\b", text, re.I):
        return TaskInference("object_replacement", "leading_replacement_verb")
    if _MATERIAL.search(text):
        return TaskInference("material_change", "material_or_texture_lexicon")
    if _APPEARANCE_NOUN.search(text) or _COLOR.search(text):
        return TaskInference("color_change", "color_lexicon")
    return TaskInference("object_replacement", "fallback_local_object_change")


def apply_refedit_contract(payload: Dict, task: TaskInference) -> Dict:
    """Attach provenance and fail closed on routes outside RefEdit's contract."""

    result = dict(payload)
    base_prompt_version = str(result.get("prompt_version", ""))
    result["base_prompt_version"] = base_prompt_version
    result["prompt_version"] = GROUND_PROMPT_VERSION
    result["refedit_task_inference"] = task.as_dict()
    flags = list(result.get("refedit_policy_flags", []))
    mode = str(result.get("mask_mode", ""))
    if mode != "regions":
        flags.append(f"NON_LOCAL_ROUTE:{mode or 'missing'}")

    required_side = {
        "object_addition": "target",
        "object_removal": "source",
        "color_change": "source",
        "material_change": "source",
    }.get(task.final_task)
    if required_side and not result.get(required_side):
        flags.append(f"MISSING_REQUIRED_SIDE:{required_side}")
    if flags:
        result["refedit_policy_flags"] = list(dict.fromkeys(flags))
        result["ground_parse_ok"] = False
    return result


def refedit_planner_prompt(base_prompt: str) -> str:
    old = "You are auditing one ScaleEdit source/result image pair before mask labeling."
    new = (
        "You are auditing one RefEdit source/result image pair before mask labeling.\n"
        "RefEdit contains local referring-expression edits: add, remove, replace, recolor, "
        "or change material/texture. Prefer a precise regions plan; a non-local route is invalid."
    )
    if old not in base_prompt:
        raise ValueError("unexpected base planner prompt")
    return base_prompt.replace(old, new, 1)
