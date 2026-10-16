"""Deterministic quota scheduling for independently owned four-node shards."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

OPS = ("add", "remove", "replace", "attribute")


def allocate(total, weights):
    scale = sum(weights.values())
    raw = {k: total * v / scale for k, v in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), str(k)))[
        : total - sum(out.values())
    ]:
        out[k] += 1
    return out


def target_cells(kind, total):
    if total < 1:
        raise ValueError("positive target required")
    if kind == "twins":
        from .dynamic_twin_balance import quotas

        base = quotas(total)
        triples = allocate(round(total * 0.15), base)
        return {
            f"{k}:{n}": (triples[k] if n == 3 else v - triples[k])
            for k, v in base.items()
            for n in (2, 3)
        }
    if kind != "multi":
        raise ValueError("unknown group kind")
    counts = allocate(total, {2: 0.2, 3: 0.3, 4: 0.25, 5: 0.25})
    result = {}
    for n, count in counts.items():
        schedules = (
            list(itertools.combinations(OPS, n))
            if n <= 4
            else [(*OPS, op) for op in OPS]
        )
        result.update(allocate(count, {multi_cell(s): 1 for s in schedules}))
    return result


def split_cells(cells, world=4):
    """Round-robin units across all cells, preserving every global integer quota."""
    shards = [dict.fromkeys(cells, 0) for _ in range(world)]
    cursor = 0
    for key, count in cells.items():
        for _ in range(count):
            shards[cursor % world][key] += 1
            cursor += 1
    return shards


def multi_cell(operations):
    return str(len(operations)) + ":" + "+".join(sorted(operations))


def cell(row, kind):
    if kind == "twins":
        units = row.get("objects", row.get("members", []))
        return f"{row['operation']}:{row.get('category_relation', 'same')}:{len(units)}"
    ops = row.get("operations") or [
        u["instruction_spec"]["operation"] for u in row["units"]
    ]
    return multi_cell(ops)


def source(row):
    return str(row.get("candidate", row).get("source_id"))


def identity(row, kind):
    if row.get("twin_id") or row.get("multi_id"):
        return row.get("twin_id") or row["multi_id"]
    from scripts.run_group_pipeline import _group_id

    return _group_id(kind, row)


def proposal_key(row):
    # A rejected proposal is never retried with another operation or ordering.
    return (source(row), tuple(sorted(str(o["object_id"]) for o in row["objects"])))


def passed(group, kind):
    return (
        group.get("group_checks", {}).get("group_pass") is True
        if kind == "twins"
        else group.get("joint_audit", {}).get("pass") is True
    )


def group_scopes(group):
    return {
        str(m.get("scope") or m.get("instruction_spec", {}).get("scope", ""))
        .strip()
        .casefold()
        for m in group.get("members", [])
    } - {""}


def is_part_group(group):
    members = group.get("members", [])
    return bool(members) and all(
        m.get("selector_granularity", m.get("granularity")) == "part" for m in members
    )


def accept(groups, targets, kind):
    counts, sources = Counter(), Counter()
    seen, out = set(), []
    scope_counts = Counter()
    object_count = 0
    total = sum(targets.values())
    # Distribution constraints apply to production-sized runs. Tiny diagnostic
    # batches cannot represent twenty categories or the requested part ratio.
    profile = kind == "twins" and total >= 100
    for g in groups:
        gid = identity(g, kind)
        k = cell(g, kind)
        sid = source(g)
        if gid in seen or not passed(g, kind):
            continue
        seen.add(gid)
        if counts[k] >= targets.get(k, 0) or sources[sid] >= 2:
            continue
        scopes = group_scopes(g) if profile else set()
        if profile and (
            any(scope_counts[x] >= math.floor(total * 0.05) for x in scopes)
            or (not is_part_group(g) and object_count >= math.floor(total * 0.60))
        ):
            continue
        out.append(g)
        scope_counts.update(scopes)
        object_count += int(not is_part_group(g))
        counts[k] += 1
        sources[sid] += 1
    return out


def multi_priority(row):
    """Soft operation fit within a quota cell; visual planning remains decisive."""
    from .group_selection import twin_operation_priority
    scores = [twin_operation_priority([obj], op)
              for obj, op in zip(row["objects"], row["operations"])]
    # Prefer improving the weakest target before polishing easy targets.
    return min(scores, default=0), sum(scores)


def choose(pool, attempted, accepted, targets, kind, limit):
    """Replenish missing final cells; cap source reuse, never mutate instructions."""
    used = {proposal_key(r) for r in attempted}
    source_counts = Counter(source(g) for g in accepted)
    counts = Counter(cell(g, kind) for g in accepted)
    tries = Counter(cell(r, kind) for r in attempted)
    available = defaultdict(list)
    part_only = (
        kind == "twins"
        and sum(targets.values()) >= 100
        and sum(not is_part_group(g) for g in accepted)
        >= math.floor(sum(targets.values()) * 0.60)
    )
    scope_counts = Counter(x for g in accepted for x in group_scopes(g))
    for row in pool:
        if kind == "twins" and sum(targets.values()) >= 100:
            # add resolves to a source attachment part after generation.
            if (
                part_only
                and row["operation"] != "add"
                and row.get("candidate_granularity") != "part"
            ):
                continue
            if any(
                str(o.get("category", "")).strip().casefold() in scope_counts
                and scope_counts[str(o.get("category", "")).strip().casefold()]
                >= math.floor(sum(targets.values()) * 0.05)
                for o in row["objects"]
            ):
                continue
        k = cell(row, kind)
        if (
            proposal_key(row) not in used
            and source_counts[source(row)] < 2
            and counts[k] < targets.get(k, 0)
        ):
            available[k].append(row)
    if kind == "twins":
        from .group_selection import twin_operation_priority

        for bucket in available.values():
            bucket.sort(
                key=lambda r: twin_operation_priority(r["objects"], r["operation"])
            )
    else:
        for bucket in available.values():
            bucket.sort(key=multi_priority)
    chosen, allocated, wave_sources = [], Counter(), Counter()
    while len(chosen) < limit:
        options = []
        for k, rows in available.items():
            while rows and (
                proposal_key(rows[-1]) in used
                or source_counts[source(rows[-1])] + wave_sources[source(rows[-1])] >= 2
            ):
                rows.pop()
            deficit = targets.get(k, 0) - counts[k]
            rate = (counts[k] + 2) / (tries[k] + 4)
            demand = math.ceil(deficit / rate)
            if rows and deficit > 0 and allocated[k] < demand:
                options.append(
                    ((demand - allocated[k]) / demand, deficit, -allocated[k], k)
                )
        if not options:
            break
        k = max(options)[-1]
        row = available[k].pop()
        chosen.append(row)
        allocated[k] += 1
        wave_sources[source(row)] += 1
        used.add(proposal_key(row))
    return chosen


def describe(groups, targets, kind):
    groups = accept(groups, targets, kind)
    counts = Counter(cell(g, kind) for g in groups)
    ops, sizes, relations, scopes = Counter(), Counter(), Counter(), Counter()
    for g in groups:
        units = g.get("members", g.get("units", []))
        sizes[len(units)] += 1
        ops.update([g["operation"]] if kind == "twins" else g["operations"])
        if kind == "twins":
            relations[g.get("category_relation", "same")] += 1
            scopes[g.get("scope_level", "unknown")] += 1
    return {
        "accepted": len(groups),
        "target": sum(targets.values()),
        "by_operation": dict(ops),
        "member_counts": dict(sizes),
        "category_relations": dict(relations),
        "scope_levels": dict(scopes),
        "actual_part_groups": sum(is_part_group(g) for g in groups)
        if kind == "twins"
        else None,
        "scope_group_counts": dict(Counter(x for g in groups for x in group_scopes(g)))
        if kind == "twins"
        else None,
        "area_buckets": dict(
            Counter(
                "small"
                if min(m["area_fraction"] for m in g.get("members", g.get("units", [])))
                < 0.005
                else "medium"
                if min(m["area_fraction"] for m in g.get("members", g.get("units", [])))
                < 0.02
                else "large"
                for g in groups
                if all(
                    "area_fraction" in m for m in g.get("members", g.get("units", []))
                )
            )
        ),
        "remaining": {k: max(0, n - counts[k]) for k, n in targets.items()},
        "complete": all(counts[k] == n for k, n in targets.items()),
    }


def propose(job):
    """CPU-only proposals for one source; exact geometry still checked pre-plan."""
    candidate, kind, wanted, seed = job
    from .group_selection import (
        _cheap_operation_allowed,
        _image_size,
        select_multi_candidates,
        select_twin_candidates,
    )
    from .multi_operation_balance import operation_schedules

    rows = []
    if kind == "twins":
        for key in wanted:
            op, relation, n = key.split(":")
            rows.extend(
                select_twin_candidates(
                    [candidate],
                    target=4,
                    operation=op,
                    category_relation=relation,
                    member_count=int(n),
                    seed=seed,
                    max_groups_per_source=4,
                )
            )
    else:
        # One source returns up to two candidates for each target count; all
        # adjacency evidence uses exact RLE. Schedules are fixed before planning.
        initial = select_multi_candidates(
            [candidate],
            target=20,
            seed=seed,
            max_groups_per_source=8,
            strict_geometry=True,
        )
        for row in initial:
            for schedule in operation_schedules(
                row["objects"], _image_size(candidate), _cheap_operation_allowed
            ):
                if multi_cell(schedule) in wanted:
                    rows.append({**row, "operations": list(schedule)})
                    # One representative assignment per operation multiset.
        # Do not retain the first lexicographic operation assignment when a
        # better fit exists for the SAME group and SAME operation multiset.
        rows.sort(key=multi_priority, reverse=True)
        seen = set()
        unique = []
        for row in rows:
            key = proposal_key(row), cell(row, kind)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        rows = unique
    compact = {k: v for k, v in candidate.items() if k != "objects"}
    for row in rows:
        row["candidate"] = compact
    return rows
