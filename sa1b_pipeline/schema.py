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
    # ``mask_path`` always points to a source-image SA-1B region.  For add
    # this is the host/attachment anchor; for the other operations it is the
    # visible target being edited.  Keeping the role explicit prevents later
    # consumers from treating an add anchor as a pixel-accurate change mask.
    mask_role: str = "target"
    mask_semantics: dict[str, Any] = field(default_factory=dict)
    support_policy: str = "target_only"
    edit_surface: str | None = None
    source_color: str | None = None
    # Construction-only fields, never substitute these for the benchmark mask.
    execution_mask_path: str | None = None
    protected_mask_path: str | None = None
    surface_type: str | None = None
    target_rgb: list[float] | None = None

    instruction_protocol_version: str | None = None
    instruction_spec: dict[str, Any] = field(default_factory=dict)
    instruction_length_advice: list[str] = field(default_factory=list)

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
    expected_role = "anchor" if unit.operation == "add" else "target"
    if unit.mask_role not in {"anchor", "target"}:
        raise ValueError("mask_role must be anchor or target")
    if unit.mask_role != expected_role:
        raise ValueError(f"{unit.operation} units require mask_role={expected_role}")
    if unit.support_policy not in {"target_only", "target_plus_collar", "anchor_plus_attachment"}:
        raise ValueError("unknown support_policy")
    if len(unit.bbox) != 4 or len(unit.point) != 2:
        raise ValueError("invalid unit geometry")
    if not unit.ref_phrase.strip() or not unit.scope.strip():
        raise ValueError("ref_phrase and scope are required")
    if not unit.instruction_ref.strip() or not unit.instruction_noref.strip():
        raise ValueError('instructions must be nonempty')
    if unit.instruction_spec or unit.instruction_protocol_version:
        from .instruction_protocol import VERSION, validate_pair
        if unit.instruction_protocol_version != VERSION or unit.instruction_spec.get('operation') != unit.operation:
            raise ValueError('instruction protocol operation/version mismatch')
        if unit.instruction_spec.get('ref') != unit.ref_phrase or unit.instruction_spec.get('scope') != unit.scope:
            raise ValueError('instruction selection mismatch')
        validate_pair(unit.instruction_spec, unit.instruction_ref, unit.instruction_noref)
        from .instruction_protocol import mask_contract
        if unit.mask_semantics and unit.mask_semantics != mask_contract(unit.instruction_spec):
            raise ValueError("source mask semantics differ from instruction protocol")


def validate_case(case: SelectionCase, min_units: int = 2, max_units: int = 5,
                  enforce_operation_mix: bool = False) -> None:
    if not min_units <= len(case.units) <= max_units:
        raise ValueError(f"a case needs {min_units}-{max_units} independent targets")
    if len({u.object_id for u in case.units}) != len(case.units):
        raise ValueError("one physical object cannot be counted as multiple targets")
    if not set(u.operation for u in case.units) <= set(OPERATIONS):
        raise ValueError("unknown operation in case")
    # Quality-first cohorts choose an operation per mask. The old pilot rule
    # forced four operation types into every five-target image and pushed
    # unsuitable targets into remove/add. Keep it only for compatibility runs.
    if enforce_operation_mix:
        operations = [u.operation for u in case.units]
        if len(operations) <= 4 and len(set(operations)) != len(operations):
            raise ValueError("benchmark-v2 style scenes keep operations distinct through four targets")
        if len(operations) == 5 and (len(set(operations)) != 4 or max(operations.count(op) for op in set(operations)) > 2):
            raise ValueError("a five-target scene may repeat at most one operation")
    for unit in case.units:
        validate_unit(unit)
