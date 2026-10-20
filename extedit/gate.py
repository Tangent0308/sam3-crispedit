"""Gate: edited region and edit gates per unit, geometry, paste-back outside the supports, post-paste checks.

Per unit (rules of protocol-adapt-4 attach_edited_region / edit_gates / geometry_flags, plus the C3 gates of
docs/01 section 2.11):
  edited region   masks.edited_region on (BEFORE, aligned AFTER) around S and the execution region;
  gates           S unchanged (< 10 % of S changed), and for C3: a replaced object at least 85 % changed, a
                  recolored object at least 90 % changed, an add's anchor at most 60 % changed;
  geometry        where the new item (add) or the execution region lies relative to S.
Per row: no solid block of change outside every unit's support (old drop_C), then AFTER is pasted onto BEFORE
inside the union of the units' paste regions (clean_pairs.paste) and checked as the old GATE did.

Supports (doc 07 section 3.5): attribute S; remove S grown by 18 % of its side; replace S grown by 50 % of its side;
add S and the new item, grown by 18 % of the item's side.  A unit's paste region is its S (add: the new item) and the
pixels that changed inside its support.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import masks as mk
from . import pixels as px

SUPPORT_GROW = {"attribute": 0.0, "remove": 0.18, "replace": 0.50, "add": 0.18}
# Covered: share of S that changed.  Validation 1 (2026-10-20, 120 pairs): every accepted recolor and replacement
# covered at least 0.92 of S, and the four between 0.76 and 0.83 were complete edits whose SAM3 mask reached a little
# past the object; under 0.75 counts as half done (the C3 failure), 0.75-0.90 is flagged for stricter subsets.
EDIT_GATES = {"s_unchanged": 0.1, "max_noise": 15.0, "replace_covered": 0.75, "attribute_covered": 0.75,
              "covered_review": 0.90, "anchor_changed": 0.60}


def unit_regions(src, tgt, spec, s_mask, item, d=None):
    """(execution region, support, paste region, edited region, stats) of one unit on the aligned pair."""
    op = spec["operation"]
    if op == "add":
        base = s_mask | item
        support = mk.grow(base, max(2.0, SUPPORT_GROW["add"] * mk.side(item)))
        region, stats = mk.edited_region(src, tgt, s_mask, item, d=d)
        paste = item | (region & support if region is not None else False)
        return item, support, paste, region, stats
    grow = SUPPORT_GROW[op]
    support = mk.grow(s_mask, grow * mk.side(s_mask)) if grow else s_mask.copy()
    region, stats = mk.edited_region(src, tgt, s_mask, None if op == "attribute" else support, d=d)
    paste = s_mask | (region & support if region is not None and op != "attribute" else False)
    return paste, support, paste, region, stats


def unit_gates(spec, s_mask, item, region, stats, params=EDIT_GATES):
    """(flags, hard reasons) from the pixels that changed (edited_region) for one unit."""
    flags, hard = [], []
    if region is None:
        return flags, ["edited_region_empty"]
    op = spec["operation"]
    if stats["area_frac"] < 0.0003:
        flags.append("edited_region_tiny")
    if stats["dropped_far_frac"] > stats["area_frac"]:
        flags.append("edited_region_far_changes")
    if stats["in_anchor"] < 0.3:
        flags.append("edited_region_off_anchor")
    reliable = stats["noise"] <= params["max_noise"]
    if not reliable:
        flags.append("edited_region_noisy")
    if op != "add":
        if reliable and stats["s_covered"] < params["s_unchanged"]:
            hard.append("edited_region_misses_s")
        elif op == "replace" and stats["s_covered"] < params["replace_covered"]:
            hard.append("replace_incomplete")
        elif op == "attribute" and stats["s_covered"] < params["attribute_covered"]:
            hard.append("attribute_incomplete")
        elif op in ("replace", "attribute") and stats["s_covered"] < params["covered_review"]:
            flags.append("s_partly_changed")
    else:
        rest = s_mask & ~mk.grow(item, 2)
        changed = float((region & rest).sum()) / max(int(rest.sum()), 1)
        stats["anchor_changed"] = round(changed, 3)
        if rest.sum() >= 64 and changed > params["anchor_changed"]:
            hard.append("anchor_changed")
    return flags, hard


def gate_row(run, row, pre, plan, segs, out_dir):
    """Gate one planned and segmented row; writes the pasted AFTER when the row passes."""
    src = px.load_rgb(run / row["source_image"])
    tgt_orig = px.load_rgb(run / row["target_image"])
    shape = src.shape[:2]
    tgt = px.apply_alignment(tgt_orig, shape, pre["align"])
    valid = px.valid_mask(tgt_orig.shape, shape, pre["align"])
    d = mk.color_difference(src, tgt)
    if not valid.all():
        d = np.where(valid, d, 0.0)
    units, pastes, hard_row = [], [], []
    for unit, seg in zip(plan["units"], segs):
        entry = {"unit": seg["unit"], "flags": list(seg.get("flags", [])), "hard": []}
        units.append(entry)
        if "selection" not in seg:
            entry["hard"].append(seg.get("reason") or "no_selection")
            continue
        if seg.get("roundtrip", 0.0) < 0.5:
            entry["hard"].append("token_roundtrip")
        spec = unit["spec"]
        s_mask = mk.decode(seg["selection"])
        item = mk.decode(seg["footprint"]) if seg.get("footprint") else None
        exec_mask, support, paste, region, stats = unit_regions(src, tgt, spec, s_mask, item, d)
        flags, hard = unit_gates(spec, s_mask, item, region, stats)
        geo = mk.geometry(spec, s_mask, exec_mask, unit["slots"].get("host_scope", ""))
        geo_flags, geo_hard = mk.geometry_flags(spec, geo)
        entry["flags"] += flags + geo_flags
        entry["hard"] += hard + geo_hard
        entry.update(geometry=geo, edited_stats=stats, execution_region=mk.encode(exec_mask),
                     paste_region=mk.encode(paste))
        pastes.append(paste)
    if any(e["hard"] for e in units):
        return {"id": row["id"], "status": "rejected", "reason": "unit_gate", "units": units}
    paste_union = np.logical_or.reduce(pastes)
    if not valid.all() and valid[paste_union].mean() < px.RULES["paste_cover"]:
        hard_row.append("paste_outside_target")
    blob = px.coherent_outside(src, tgt, paste_union, valid=valid)
    change = mk.decode(pre["change"]) if pre.get("change") else None
    grown = mk.grow(paste_union, max(2, int(px.PASTE_DILATE * min(shape))))
    unexplained = float((change & ~grown).sum()) / max(int(change.sum()), 1) if change is not None else 0.0
    checks = {"blob": round(blob, 5), "unexplained": round(unexplained, 4)}
    if blob > px.GATE["blob_max"]:
        hard_row.append("unrelated_change")
    pasted, grown = px.paste(src, tgt, paste_union)
    cut, mad = px.ring_metrics(src, tgt, grown)
    final = px.stats(src, pasted, paste_union)
    checks.update(ring_cut=round(cut, 4), ring_mad=round(mad, 2), fill=round(final["fill"], 4),
                  far_out=round(final["far_out"], 4), area_kept=round(float(grown.mean()), 4))
    if final["far_out"] > px.GATE["far_out_final"]:
        hard_row.append("outside_still_changed")
    if final["fill"] < px.GATE["fill_final"]:
        hard_row.append("edit_lost_after_paste")
    if cut > px.GATE["ring_cut"]:
        hard_row.append("edit_cut_at_paste_boundary")
    if mad > px.GATE["ring_mad"]:
        hard_row.append("seam_at_paste_boundary")
    if grown.mean() > px.GATE["area_kept"]:
        hard_row.append("paste_region_too_large")
    record = {"id": row["id"], "units": units, "checks": checks}
    if hard_row:
        return {**record, "status": "rejected", "reason": ",".join(hard_row)}
    # the final edited region of every unit, on the pair as it will be stored
    for entry, unit, seg in zip(units, plan["units"], segs):
        s_mask = mk.decode(seg["selection"])
        exec_mask = mk.decode(entry["execution_region"])
        region, stats = mk.edited_region(Image.fromarray(src), Image.fromarray(pasted), s_mask,
                                         None if unit["spec"]["operation"] == "attribute" else exec_mask)
        if region is None:
            return {**record, "status": "rejected", "reason": "edited_region_empty_after_paste"}
        entry["edited_region"] = {"method": mk.EDITED_METHOD, **mk.encode(region), "area_frac": stats["area_frac"],
                                  "stats": stats}
    out = out_dir / row["id"][-2:] / f"{row['id']}.target.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pasted).save(out, compress_level=3)
    return {**record, "status": "passed", "target_pasted": str(out.relative_to(run)),
            "treatment": pre["align"]["kind"] + "+paste"}
