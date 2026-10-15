"""Schemas and invariant checks for counterfactual twin and joint-edit groups.

The ordinary ``SelectionCase`` schema describes one image edit.  These records
keep the group-level contract explicit so that a group cannot silently degrade
to an unrelated set of single-target rows during resume or export.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import OPERATIONS, EditUnit

GROUP_SCHEMA_VERSION = "sa1b.samtok.group.v1"
TWIN_SCOPE_LEVELS = {"object", "part_on_different_hosts", "part_on_same_host"}


@dataclass
class TwinGroup:
    twin_id: str
    source_id: str
    source_image: str
    source_provenance: dict[str, Any]
    operation: str
    instruction_spec: dict[str, Any]
    instruction_noref: str
    scope_level: str
    members: list[dict[str, Any]]
    split: str
    group_checks: dict[str, Any]
    merged_target: dict[str, Any] | None = None
    schema_version: str = GROUP_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "twin_id": self.twin_id,
            "source_id": self.source_id,
            "source_image": self.source_image,
            "source_provenance": self.source_provenance,
            "operation": self.operation,
            "instruction_spec": self.instruction_spec,
            "instruction_noref": self.instruction_noref,
            "scope_level": self.scope_level,
            "members": self.members,
            "split": self.split,
            "group_checks": self.group_checks,
            "merged_target": self.merged_target,
        }


@dataclass
class MultiGroup:
    multi_id: str
    source_id: str
    source_image: str
    source_provenance: dict[str, Any]
    n_targets: int
    operations: list[str]
    generation_order: list[str]
    adjacent_pairs: list[dict[str, Any]]
    units: list[dict[str, Any]]
    joint_target: str | None
    joint_audit: dict[str, Any]
    split: str
    schema_version: str = GROUP_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "multi_id": self.multi_id,
            "source_id": self.source_id,
            "source_image": self.source_image,
            "source_provenance": self.source_provenance,
            "n_targets": self.n_targets,
            "operations": self.operations,
            "generation_order": self.generation_order,
            "adjacent_pairs": self.adjacent_pairs,
            "units": self.units,
            "joint_target": self.joint_target,
            "joint_audit": self.joint_audit,
            "split": self.split,
        }


def _require_file(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")


def _validate_member(member: dict[str, Any], *, twin: bool = False) -> None:
    required = ("unit_id", "object_id", "ref", "instruction", "instruction_noref",
                "instruction_spec", "mask", "bbox", "point", "support_mask",
                "area_fraction")
    for key in required:
        if key not in member:
            raise ValueError(f"member missing {key}")
    for key in ("mask", "support_mask"):
        _require_file(member[key], key)
    if len(member["bbox"]) != 4 or len(member["point"]) != 2:
        raise ValueError("invalid member geometry")
    if not isinstance(member["instruction_spec"], dict):
        raise ValueError("instruction_spec must be an object")
    if member["instruction_spec"].get("ref") != member["ref"]:
        raise ValueError("member ref differs from instruction_spec")
    if not member["instruction"].strip() or not member["instruction_noref"].strip():
        raise ValueError("member instructions must be nonempty")
    if member["instruction_spec"].get("version") == "2.0":
        from .instruction_protocol import validate_pair
        validate_pair(member["instruction_spec"], member["instruction"], member["instruction_noref"])
        if member.get("blind_eligible") is not True:
            raise ValueError("unresolved protocol 2.0 source region")
    if twin and member["instruction"] == member["instruction_noref"]:
        # Ref and no-ref may coincide only if the source ref is genuinely the
        # protocol's selected placeholder; group data always has a real ref.
        raise ValueError("twin member must preserve a distinct ref instruction")


def validate_twin_record(record: dict[str, Any], *, expected_split: str | None = None) -> None:
    if record.get("schema_version") != GROUP_SCHEMA_VERSION:
        raise ValueError("unsupported twin schema")
    if not record.get("twin_id") or not record.get("source_id"):
        raise ValueError("twin_id and source_id are required")
    _require_file(record.get("source_image"), "source_image")
    if record.get("operation") not in OPERATIONS:
        raise ValueError("invalid twin operation")
    if record.get("scope_level") not in TWIN_SCOPE_LEVELS:
        raise ValueError("invalid twin scope level")
    if expected_split is not None and record.get("split") != expected_split:
        raise ValueError("unexpected split")
    members = record.get("members")
    if not isinstance(members, list) or len(members) not in (2, 3):
        raise ValueError("twins require two or three members")
    if record.get("split") == "eval" and len(members) != 2:
        raise ValueError("evaluation twins require exactly two members")
    areas = [float(m.get("area_fraction", 0)) for m in members]
    if min(areas) <= 0 or max(areas) > 3 * min(areas):
        raise ValueError("twin exact mask area ratio exceeds 3")
    if len({m.get("object_id") for m in members}) != len(members):
        raise ValueError("twin members must be distinct objects")
    if record.get("instruction_noref") != members[0].get("instruction_noref"):
        raise ValueError("group no-ref instruction mismatch")
    for member in members:
        _validate_member(member, twin=True)
        if member["instruction_noref"] != record["instruction_noref"]:
            raise ValueError("twin no-ref instructions are not identical")
        if member["instruction_spec"].get("operation") != record["operation"]:
            raise ValueError("twin operation mismatch")
    if not isinstance(record.get("group_checks"), dict):
        raise ValueError("group_checks must be an object")


def validate_multi_record(record: dict[str, Any], *, expected_split: str | None = None) -> None:
    if record.get("schema_version") != GROUP_SCHEMA_VERSION:
        raise ValueError("unsupported multi schema")
    if not record.get("multi_id") or not record.get("source_id"):
        raise ValueError("multi_id and source_id are required")
    _require_file(record.get("source_image"), "source_image")
    n = record.get("n_targets")
    if not isinstance(n, int) or not 2 <= n <= 5:
        raise ValueError("multi groups require 2-5 targets")
    units = record.get("units")
    if not isinstance(units, list) or len(units) != n:
        raise ValueError("n_targets does not match units")
    operations = record.get("operations")
    if not isinstance(operations, list) or len(operations) != n or any(o not in OPERATIONS for o in operations):
        raise ValueError("invalid operations")
    if n <= 4 and len(set(operations)) != n:
        raise ValueError("2-4 target groups require distinct operation types")
    if n == 5 and len(set(operations)) != 4:
        raise ValueError("five target groups require all four operation types")
    order = record.get("generation_order")
    ids = [u.get("unit_id") for u in units]
    if not isinstance(order, list) or set(order) != set(ids) or len(set(order)) != n:
        raise ValueError("generation_order must contain every unit exactly once")
    if expected_split is not None and record.get("split") != expected_split:
        raise ValueError("unexpected split")
    if len({u.get("object_id") for u in units}) != n:
        raise ValueError("multi targets must be distinct objects")
    for unit in units:
        _validate_member(unit)
    adjacent = record.get("adjacent_pairs")
    if not isinstance(adjacent, list) or not adjacent:
        raise ValueError("multi group requires an adjacent pair")
    if not isinstance(record.get("joint_audit"), dict):
        raise ValueError("joint_audit must be an object")


def operation_counts(records: Iterable[dict[str, Any]], *, multi: bool = False) -> Counter:
    counts: Counter = Counter()
    for row in records:
        values = row.get("operations", []) if multi else [row.get("operation")]
        counts.update(v for v in values if v in OPERATIONS)
    return counts
