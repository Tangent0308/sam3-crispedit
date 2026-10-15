"""Bounded replenishment by *accepted* operation/category cells, with no model calls.

The coordinator persists frozen waves; workers may finish a wave in any order.
Attempts include planning and generation rejects. Quota overflow stays available
for inspection and never displaces a missing class or weakens a quality gate.
"""

from __future__ import annotations

import math
from collections import Counter

OPERATIONS = ("add", "remove", "replace", "attribute")


def cell(row):
    return row["operation"] + ":" + row.get("category_relation", "same")


def quotas(total, same_fraction=0.30):
    if total < 1 or not 0 <= same_fraction <= 1:
        raise ValueError("invalid target/fraction")
    by_op = {op: total // 4 + int(i < total % 4) for i, op in enumerate(OPERATIONS)}
    raw = {op: n * same_fraction for op, n in by_op.items()}
    same = {op: math.floor(v) for op, v in raw.items()}
    for op in sorted(OPERATIONS, key=lambda op: -(raw[op] - same[op]))[
        : round(total * same_fraction) - sum(same.values())
    ]:
        same[op] += 1
    return {
        op + ":" + rel: (same[op] if rel == "same" else n - same[op])
        for op, n in by_op.items()
        for rel in ("same", "cross")
    }


def accepted_groups(groups, targets):
    counts = Counter()
    result = []
    seen = set()
    for g in groups:
        gid = g["twin_id"]
        key = cell(g)
        if gid in seen:
            continue
        seen.add(gid)
        if g.get("group_checks", {}).get("group_pass") is not True:
            continue
        if counts[key] >= targets.get(key, 0):
            continue
        result.append(g)
        counts[key] += 1
    return result


def choose_wave(pool, attempted, accepted, targets, *, wave_size, max_attempts):
    """Pick new sources for missing cells; low-yield cells get more attempts.

    Called only after the previous wave completes. No re-labeling or repeated
    call for an old pair. Existing successful groups are counted on resume.
    """
    if wave_size < 1 or max_attempts < 1:
        raise ValueError("invalid attempt budget")
    attempts = Counter(cell(r) for r in attempted)
    successes = Counter(cell(g) for g in accepted)
    used = {str(r["candidate"]["source_id"]) for r in attempted}
    available = {k: [] for k in targets}
    for row in pool:
        if str(row["candidate"]["source_id"]) not in used and cell(row) in available:
            available[cell(row)].append(row)
    chosen = []
    allocated = Counter()
    budget = min(wave_size, max(0, max_attempts - len(attempted)))
    while len(chosen) < budget:
        options = []
        for key, target in targets.items():
            deficit = max(0, target - successes[key])
            while (
                available[key]
                and str(available[key][0]["candidate"]["source_id"]) in used
            ):
                available[key].pop(0)
            if not deficit or not available[key]:
                continue
            rate = (successes[key] + 2) / (attempts[key] + 4)
            demand = math.ceil(deficit / rate)
            if allocated[key] >= demand:
                continue
            priority = (demand - allocated[key]) / max(1, demand)
            options.append((priority, deficit, -allocated[key], key))
        if not options:
            break
        key = max(options)[-1]
        row = available[key].pop(0)
        chosen.append(row)
        allocated[key] += 1
        used.add(str(row["candidate"]["source_id"]))
    return chosen


def progress(groups, attempted, targets):
    accepted = accepted_groups(groups, targets)
    counts = Counter(cell(g) for g in accepted)
    attempts = Counter(cell(r) for r in attempted)
    return {
        "accepted": len(accepted),
        "target": sum(targets.values()),
        "attempted": len(attempted),
        "by_cell": {
            k: {
                "target": n,
                "accepted": counts[k],
                "remaining": max(0, n - counts[k]),
                "attempts": attempts[k],
            }
            for k, n in targets.items()
        },
        "by_operation": {
            op: sum(counts[op + ":" + r] for r in ("same", "cross"))
            for op in OPERATIONS
        },
        "complete": all(counts[k] >= n for k, n in targets.items()),
    }
