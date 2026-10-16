"""Balance operation schedules for adjacent multi-target groups.

Operation labels are assigned before planning.  A fixed ``add/replace/...``
order is biased because some masks cannot naturally support every operation.
This module first enumerates the schedules that are valid for each selected
group, then chooses one schedule per group so the aggregate member counts are
as close as possible to the four-way quota.  It never relabels a mask to an
operation that fails the cheap naturalness gate.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from typing import Any, Callable, Iterable

from .schema import OPERATIONS


def _target_counts(total_members: int) -> dict[str, int]:
    base, remainder = divmod(int(total_members), len(OPERATIONS))
    return {op: base + int(index < remainder) for index, op in enumerate(OPERATIONS)}


def operation_schedules(
    objects: list[dict[str, Any]],
    image_size: tuple[int, int],
    allowed: Callable[[dict[str, Any], str, tuple[int, int]], bool],
) -> list[tuple[str, ...]]:
    """Return all protocol-valid schedules that pass ``allowed``.

    Groups with 2--4 members use distinct operation types.  A five-member
    group uses all four types and repeats exactly one type, as required by the
    data specification.
    """
    n = len(objects)
    if n not in (2, 3, 4, 5):
        return []
    if n <= 4:
        candidates: Iterable[tuple[str, ...]] = itertools.permutations(OPERATIONS, n)
    else:
        # The repeated operation may occur at ANY two target positions.
        # Appending a repeat to four distinct operations missed 144 of the
        # 240 valid schedules (e.g. add, add, remove, replace, attribute).
        candidates = (
            schedule for schedule in itertools.product(OPERATIONS, repeat=5)
            if len(set(schedule)) == 4
        )
    result = []
    for schedule in candidates:
        if all(allowed(obj, op, image_size) for obj, op in zip(objects, schedule)):
            result.append(tuple(schedule))
    return list(dict.fromkeys(result))


def _stable_key(row: dict[str, Any], index: int, seed: int) -> bytes:
    candidate = row.get("candidate") or {}
    source = str(candidate.get("source_id", ""))
    object_ids = ",".join(str(obj.get("object_id", "")) for obj in row.get("objects", []))
    return hashlib.sha256(f"{seed}:{source}:{object_ids}:{index}".encode()).digest()


def _greedy_assignment(
    rows: list[dict[str, Any]],
    options: list[list[tuple[str, ...]]],
    targets: dict[str, int],
    seed: int,
) -> list[tuple[str, ...]]:
    """Deterministic fallback used when scipy is unavailable."""
    order = sorted(range(len(rows)), key=lambda i: (len(options[i]), _stable_key(rows[i], i, seed)))
    loads: Counter[str] = Counter()
    assigned: list[tuple[str, ...] | None] = [None] * len(rows)
    for index in order:
        choices = options[index]
        if not choices:
            raise ValueError(f"no feasible operation schedule for group index {index}")

        def score(schedule: tuple[str, ...]):
            next_loads = loads + Counter(schedule)
            deviations = [abs(next_loads[op] - targets[op]) for op in OPERATIONS]
            return (max(deviations), sum(deviations), sum(x * x for x in deviations), schedule)

        chosen = min(choices, key=score)
        assigned[index] = chosen
        loads.update(chosen)
    return [value for value in assigned if value is not None]


def _milp_assignment(
    options: list[list[tuple[str, ...]]],
    targets: dict[str, int],
):
    """Solve the min-max quota deviation problem with scipy when available."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return None

    variable_count = sum(len(item) for item in options)
    # Binary schedule choices, positive/negative deviation for each operation,
    # and one variable bounding the maximum absolute deviation.
    z_index = variable_count + 8
    variable_total = z_index + 1
    matrix = lil_matrix((len(options) + 4 + 8, variable_total), dtype=float)
    objective = np.zeros(variable_total)
    integrality = np.zeros(variable_total)
    lower_bounds = np.zeros(variable_total)
    upper_bounds = np.full(variable_total, np.inf)
    column = 0
    for row_index, schedules in enumerate(options):
        for schedule in schedules:
            matrix[row_index, column] = 1.0
            integrality[column] = 1.0
            upper_bounds[column] = 1.0
            for operation in schedule:
                matrix[len(options) + OPERATIONS.index(operation), column] += 1.0
            column += 1

    # Minimize max deviation first; the small L1 term makes ties compact.
    objective[z_index] = 1000.0
    for operation_index, operation in enumerate(OPERATIONS):
        positive = variable_count + 2 * operation_index
        negative = positive + 1
        matrix[len(options) + operation_index, positive] = -1.0
        matrix[len(options) + operation_index, negative] = 1.0
        objective[positive] = 1.0
        objective[negative] = 1.0
        matrix[len(options) + 4 + 2 * operation_index, positive] = 1.0
        matrix[len(options) + 4 + 2 * operation_index, z_index] = -1.0
        matrix[len(options) + 5 + 2 * operation_index, negative] = 1.0
        matrix[len(options) + 5 + 2 * operation_index, z_index] = -1.0

    lower = np.array([1.0] * len(options) + [targets[op] for op in OPERATIONS] + [-np.inf] * 8)
    upper = np.array([1.0] * len(options) + [targets[op] for op in OPERATIONS] + [0.0] * 8)
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 180, "mip_rel_gap": 0},
    )
    if result.x is None or result.status not in (0, 1):
        return None
    assigned = []
    column = 0
    for schedules in options:
        width = len(schedules)
        selected = int(np.argmax(result.x[column:column + width]))
        assigned.append(schedules[selected])
        column += width
    return assigned


def balance_multi_operations(
    rows: list[dict[str, Any]],
    *,
    image_size_for: Callable[[dict[str, Any]], tuple[int, int]],
    allowed: Callable[[dict[str, Any], str, tuple[int, int]], bool],
    seed: int = 0,
) -> dict[str, Any]:
    """Assign feasible operation schedules and return a compact report.

    The rows are updated in place.  The report records the requested counts,
    actual counts, and the largest absolute deviation for reproducibility.
    """
    options: list[list[tuple[str, ...]]] = []
    for index, row in enumerate(rows):
        size = image_size_for(row)
        schedules = operation_schedules(row.get("objects", []), size, allowed)
        if not schedules:
            raise ValueError(f"no feasible operation schedule for group index {index}")
        options.append(schedules)
    targets = _target_counts(sum(len(row.get("objects", [])) for row in rows))
    assignment = _milp_assignment(options, targets)
    if assignment is None:
        assignment = _greedy_assignment(rows, options, targets, seed)
    loads: Counter[str] = Counter()
    for row, schedule in zip(rows, assignment):
        row["operations"] = list(schedule)
        loads.update(schedule)
    actual = {operation: int(loads[operation]) for operation in OPERATIONS}
    deviations = {operation: actual[operation] - targets[operation] for operation in OPERATIONS}
    return {
        "groups": len(rows),
        "member_count": sum(len(row.get("objects", [])) for row in rows),
        "target_counts": targets,
        "operation_counts": actual,
        "deviation": deviations,
        "max_abs_deviation": max((abs(value) for value in deviations.values()), default=0),
        "solver": "scipy_milp_or_greedy_fallback",
    }

