"""Stable, benchmark-aligned records exchanged by the pipeline.

The schema deliberately keeps source provenance and visible masks beside every
unit.  MLLM text is advisory; masks always come from the SA-1B annotations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

OPERATIONS = ("add", "replace", "remove", "attribute")
LOCATORS = ("none", "point", "box", "mask")


@dataclass
class CandidateObject:
    object_id: str
    category: str
    description: str = ""
    mask_path: str | None = None
    rle: dict[str, Any] | None = None
    bbox: list[int] | None = None
    area_fraction: float | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    source_id: str
    image_path: str
    objects: list[CandidateObject]
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    image_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditUnit:
    unit_id: str
    object_id: str
    operation: str
    ref_phrase: str
    scope: str
    payload: str
    instruction_ref: str
    instruction_noref: str
    mask_path: str
    bbox: list[int]
    point: list[int]
    attribute_kind: str | None = None
    locator: str = "mask"
    has_ref: bool = True
    completion_requirement: str = ""
    preserve: list[str] = field(default_factory=list)
    mllm_evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionCase:
    case_id: str
    source: Candidate
    units: list[EditUnit]
    difficulty: dict[str, Any]
    selector: dict[str, Any] = field(default_factory=dict)
    status: str = "selected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sa1b.samtok.selection.v1",
            "id": self.case_id,
            "source": self.source.to_dict(),
            "units": [u.to_dict() for u in self.units],
            "difficulty": self.difficulty,
            "selector": self.selector,
            "status": self.status,
        }


def validate_unit(unit: EditUnit) -> None:
    if unit.operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {unit.operation}")
    if unit.locator not in LOCATORS or (not unit.has_ref and unit.locator == "none"):
        raise ValueError("no-ref units require a point, box, or mask locator")
    if not unit.mask_path:
        raise ValueError("each unit needs an immutable source mask")
    if len(unit.bbox) != 4 or len(unit.point) != 2:
        raise ValueError("invalid unit geometry")
    if not unit.ref_phrase.strip() or not unit.scope.strip():
        raise ValueError("ref_phrase and scope are required")
    words = unit.instruction_ref.split()
    no_words = unit.instruction_noref.split()
    if not 4 <= len(words) <= 40 or not 4 <= len(no_words) <= 24:
        raise ValueError("instruction length is outside the benchmark contract")


def validate_case(case: SelectionCase, min_units: int = 2, max_units: int = 5) -> None:
    if not min_units <= len(case.units) <= max_units:
        raise ValueError(f"a case needs {min_units}-{max_units} independent targets")
    if len({u.object_id for u in case.units}) != len(case.units):
        raise ValueError("one physical object cannot be counted as multiple targets")
    if not set(u.operation for u in case.units) <= set(OPERATIONS):
        raise ValueError("unknown operation in case")
    operations = [u.operation for u in case.units]
    if len(operations) <= 4 and len(set(operations)) != len(operations):
        raise ValueError("benchmark-v2 style scenes keep operations distinct through four targets")
    if len(operations) == 5 and max(operations.count(op) for op in set(operations)) > 2:
        raise ValueError("a five-target scene may repeat at most one operation")
    for unit in case.units:
        validate_unit(unit)
