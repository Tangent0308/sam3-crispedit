"""Unit tests of the protocol 2.0 labeling pipeline (extedit); CPU only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from extedit import PROTOCOL_SHA256, STYLE_SHA256
from extedit import audit as au
from extedit import gate as gt
from extedit import masks as mk
from extedit import pixels as px
from extedit import plan as pl

ROOT = Path(__file__).resolve().parents[1] / "extedit"


def test_vendored_modules_are_pinned():
    assert hashlib.sha256((ROOT / "instruction_protocol.py").read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert hashlib.sha256((ROOT / "instruction_style.py").read_bytes()).hexdigest() == STYLE_SHA256


def answer(**unit):
    base = {"areas": [1], "operation": "replace", "ref": "the red cup on the left", "scope": "cup", "host_ref": "",
            "host_scope": "", "part": "", "granularity": "", "target_part": "", "payload": "a glass vase",
            "remove_mode": "whole", "attributes": [], "placement": {}, "qualifier": "", "s_box": [100, 200, 300, 400],
            "host_box": [], "item_box": [90, 180, 320, 420], "comparable": True}
    base.update(unit)
    return json.dumps({"observed": "x", "quality": "clean", "problem": "", "reason": "", "units": [base]})


def test_plan_parse_replace():
    plan, problems = pl.parse_plan(answer(), 2)
    assert not problems
    unit = plan["units"][0]
    assert unit["spec"]["operation"] == "replace" and unit["spec"]["version"] == "2.0"
    assert unit["s_box"] == [100, 200, 300, 400] and unit["areas"] == [1] and unit["comparable"]
    ref, noref, reason = au.sentences(unit["spec"])
    assert ref == "Replace the red cup on the left with a glass vase."
    assert noref == "Replace the selected region with a glass vase." and reason is None


def test_plan_parse_add_needs_contact_relation_and_host_box():
    _, problems = pl.parse_plan(answer(operation="add", payload="a small bird", granularity="object",
                                       ref="the branch on the right", scope="branch", placement={"relation": "near"}), 1)
    assert any('relation "near"' in p for p in problems)
    _, problems = pl.parse_plan(answer(operation="add", payload="a small bird", granularity="surface",
                                       ref="the road next to the man", scope="road", host_ref="the road",
                                       host_scope="road", placement={"relation": "on"}), 1)
    assert any("host_box" in p for p in problems)
    plan, problems = pl.parse_plan(answer(operation="add", payload="a small bird", granularity="object",
                                          ref="the branch on the right", scope="branch",
                                          placement={"relation": "on", "site": "", "anchor_mode": "support"}), 1)
    assert not problems
    assert au.sentences(plan["units"][0]["spec"])[1] == "Add a small bird on the selected region."


def test_plan_parse_reject_and_bad_box():
    plan, problems = pl.parse_plan(json.dumps({"observed": "x", "quality": "reject", "problem": "global",
                                               "units": []}), 0)
    assert plan["quality"] == "reject" and plan["problem"] == "global" and not problems
    _, problems = pl.parse_plan(answer(s_box=[300, 200, 100, 400]), 1)
    assert any("s_box" in p for p in problems)


def test_attribute_text_and_area_numbers():
    plan, problems = pl.parse_plan(answer(operation="attribute", payload="", ref="the sign above the door",
                                          scope="sign", attributes=[{"property": "text", "value": "OPEN"}],
                                          areas=[1, 7, 1]), 2)
    assert not problems and plan["units"][0]["areas"] == [1]
    assert au.sentences(plan["units"][0]["spec"])[0] == 'Change the text on the sign above the door to "OPEN".'


def synthetic_pair(shift=0):
    rng = np.random.default_rng(0)
    src = (rng.random((256, 320, 3)) * 255).astype(np.uint8)
    src = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(src).resize((640, 512)))
    tgt = src.copy()
    tgt[200:300, 250:380] = [250, 20, 20]
    return src, tgt


def test_change_map_and_areas():
    src, tgt = synthetic_pair()
    change, stats, _ = px.change_map(src, tgt)
    assert 0.03 < stats["area_frac"] < 0.06
    areas, more = px.change_areas(change)
    assert len(areas) == 1 and more == 0 and areas[0]["n"] == 1
    box = areas[0]["box_1000"]
    assert abs(box[0] - 390) <= 10 and abs(box[1] - 390) <= 10


def test_paste_keeps_outside_pixels():
    src, tgt = synthetic_pair()
    noisy = np.clip(tgt.astype(int) + 3, 0, 255).astype(np.uint8)  # the generator changed every pixel a little
    z = np.zeros(src.shape[:2], bool)
    z[200:300, 250:380] = True
    pasted, grown = px.paste(src, noisy, z)
    far = ~mk.grow(grown, 7)
    assert np.array_equal(pasted[far], src[far])
    assert np.array_equal(pasted[z], noisy[z])


def test_unit_gates_attribute_incomplete():
    spec = pl.sl.to_spec(pl.sl.normalize_slots({"operation": "attribute", "ref": "the cup", "scope": "cup",
                                                "attributes": [{"property": "color", "value": "blue"}]})[0])
    s_mask = np.zeros((100, 100), bool)
    s_mask[20:60, 20:60] = True
    region = np.zeros_like(s_mask)
    region[20:60, 20:28] = True  # a fifth of S changed
    stats = {"area_frac": 0.08, "dropped_far_frac": 0.0, "in_anchor": 1.0, "noise": 2.0, "s_covered": 0.2}
    _, hard = gt.unit_gates(spec, s_mask, None, region, stats)
    assert hard == ["attribute_incomplete"]
    flags, hard = gt.unit_gates(spec, s_mask, None, region, {**stats, "s_covered": 0.6})
    assert not hard and "s_partly_changed" in flags


def test_payload_of_a_new_item_has_no_definite_article():
    _, problems = pl.parse_plan(answer(operation="add", payload="the main performer", granularity="object",
                                       ref="the stage monitor", scope="monitor",
                                       placement={"relation": "on", "site": "", "anchor_mode": "support"}), 1)
    assert any("never with" in p for p in problems)


def test_site_spot_cuts_a_large_anchor():
    host = np.zeros((400, 400), bool)
    host[200:400, :] = True  # a road
    item = np.zeros_like(host)
    item[170:230, 100:140] = True
    spot, info = mk.site_spot(host, item)
    assert spot is not None and spot.sum() < 0.2 * host.sum() and (spot & host).sum() == spot.sum()


def test_tone_match_removes_a_global_shift():
    src, tgt = synthetic_pair()
    shifted = np.clip(tgt.astype(int) + 9, 0, 255).astype(np.uint8)  # AFTER rendered 9 levels brighter
    away = np.ones(src.shape[:2], bool)
    away[180:320, 230:400] = False
    out, info = px.tone_match(src, shifted, away)
    assert info["applied"] and info["after"] < 1.5 < info["before"]
    assert np.abs(out[away].astype(int) - src[away].astype(int)).mean() < 1.5


def test_paste_region_keeps_only_pieces_attached_to_s():
    s_mask = np.zeros((200, 200), bool)
    s_mask[80:120, 80:120] = True
    region = s_mask.copy()
    region[120:126, 80:120] = True     # its shadow, touching S
    region[10:14, 10:14] = True        # a speck of re-rendering noise far away
    kept = gt.attached(region, s_mask, 3)
    assert kept[122, 100] and not kept[12, 12]


def test_box_away_from_its_changed_areas_is_asked_again():
    areas = [{"n": 1, "box_1000": [400, 600, 500, 900]}]
    _, problems = pl.parse_plan(answer(s_box=[400, 330, 500, 590], item_box=[]), areas)  # y scaled by the width
    assert any("does not reach the changed areas" in p for p in problems)
    _, problems = pl.parse_plan(answer(s_box=[390, 580, 510, 910], item_box=[]), areas)
    assert not problems


def test_tail_reads_only_new_complete_lines(tmp_path):
    from extedit import llm
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b"')  # the second line is still being written
    tail = llm.Tail(tmp_path)
    assert [r["id"] for r in tail.read()] == ["a"]
    with open(path, "a") as stream:
        stream.write('}\n{"id": "c"}\n')
    assert [r["id"] for r in tail.read()] == ["b", "c"]
    assert tail.read() == []


def test_sharded_stage_is_done_when_every_shard_is(tmp_path):
    from extedit import llm
    assert not llm.is_done(tmp_path)
    llm.mark_done(tmp_path, "DONE.0-of-2")
    assert not llm.is_done(tmp_path)
    llm.mark_done(tmp_path, "DONE.1-of-2")
    assert llm.is_done(tmp_path)


def test_text_area_is_a_compact_region():
    from extedit import segment as sg
    strokes = np.zeros((100, 200), bool)
    for x in range(40, 160, 12):  # letters: thin vertical strokes
        strokes[40:60, x:x + 3] = True
    hull = sg.convex_fill(strokes)
    assert hull.sum() > 3 * strokes.sum() and (hull & strokes).sum() == strokes.sum()
    assert sg.is_text_unit(pl.sl.to_spec(pl.sl.normalize_slots({
        "operation": "attribute", "ref": "the sign", "scope": "sign",
        "attributes": [{"property": "text", "value": "OPEN"}]})[0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
