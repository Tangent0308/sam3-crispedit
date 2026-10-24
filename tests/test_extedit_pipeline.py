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


def test_four_node_doc_embeds_the_entry_script_verbatim():
    repo = ROOT.parent
    doc = (repo / "docs" / "EXTEDIT_4NODE.md").read_text()
    block = doc.split("<!-- code: scripts/extedit_bootstrap_arnold.sh -->", 1)[1]
    block = block.split("```bash\n", 1)[1].split("\n```", 1)[0]
    assert block + "\n" == (repo / "scripts" / "extedit_bootstrap_arnold.sh").read_text()


def test_four_node_doc_current_task_block_is_settings_plus_the_entry_script():
    repo = ROOT.parent
    doc = (repo / "docs" / "EXTEDIT_4NODE.md").read_text()
    block = doc.split("<!-- code: §0 settings + scripts/extedit_bootstrap_arnold.sh -->", 1)[1]
    block = block.split("```bash\n", 1)[1].split("\n```", 1)[0]
    settings, script = block.split("\n\n#!/usr/bin/env bash\n", 1)
    assert settings.startswith("# ===== 运行设置 =====") and "EXTEDIT_RUN_ID=" in settings
    assert "#!/usr/bin/env bash\n" + script + "\n" == (repo / "scripts" / "extedit_bootstrap_arnold.sh").read_text()


def test_text_check_decisions():
    from extedit import textcheck as tc
    assert tc.normalize("ELÉCTION!") == tc.normalize("election") == "election"
    assert tc.normalize("Mc Sorley's") == "mcsorleys"
    read = lambda before, after, legible=True, garbled=False: {"before": before, "after": after,
                                                               "after_legible": legible, "after_garbled": garbled}
    assert tc.decide("ELECCIÓN", read("RODRÍGUEZ", "ELÉCTION")) == (False, "ocr_mismatch")
    assert tc.decide("HANDLE", read("HANDL", "ANDLE")) == (False, "ocr_mismatch")
    assert tc.decide("PIZZAXPRESS", read("PIZZAEXPRESS", "PIZZA EXPRESS")) == (False, "ocr_mismatch")
    assert tc.decide("VORTEX", read("VORTEX", "V O R T E X")) == (False, "no_text_change")
    assert tc.decide("CENTER", read("BEACH", "BEACH CENTER")) == (False, "old_text_remains")
    assert tc.decide("ALEXA", read("ALEX", "ALEXA")) == (True, None)  # the old text is part of the new one
    assert tc.decide("Pirates", read("PIRATES", "Pirates")) == (True, None)  # a change of letter case
    assert tc.decide("BALM", read("THE BEARD", "THE BALM")) == (True, None)  # a neighboring word read along
    assert tc.decide("HOLLYWOOD", read("BLVD", "HOLLYWOOD", legible=False)) == (True, None)
    assert tc.decide("Love Forever", read("Just Married", "LOVE FOREVER")) == (True, None)


def test_text_check_parsing_and_units():
    from extedit import textcheck as tc
    value = tc.parse_reading('```json\n{"before": "BLVD", "after": "HOLLYWOOD", "after_legible": "true", '
                             '"after_garbled": false}\n```')
    assert value == {"before": "BLVD", "after": "HOLLYWOOD", "after_legible": True, "after_garbled": False}
    with pytest.raises(ValueError):
        tc.parse_reading('{"before": "", "after": "X"}')
    plan, _ = pl.parse_plan(answer(operation="attribute", payload="", ref="the sign", scope="sign",
                                   attributes=[{"property": "text", "value": "OPEN"}]), 1)
    assert tc.text_units(plan) == [(0, "OPEN")]
    assert tc.text_units(pl.parse_plan(answer(), 1)[0]) == []


def test_text_close_ups_keep_the_area_and_enlarge_it():
    from PIL import Image
    from extedit import textcheck as tc
    src = Image.new("RGB", (2000, 1000), "white")
    s_mask = np.zeros((1000, 2000), bool)
    s_mask[480:520, 900:1100] = True  # a line of text, 200 x 40 px
    box = tc.crop_box(mk.box_1000(s_mask), src.size)
    assert box[0] <= 900 - 0.5 * 200 and box[2] >= 1100 + 0.5 * 200 and box[2] - box[0] <= 2000
    before, after = tc.close_ups(src, src, s_mask)
    assert before.size == after.size and max(before.size) == 1024


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_text_payload_marks_new_objects_with_writing():
    from extedit.textcheck import has_text_payload
    assert has_text_payload({"operation": "add", "payload": 'a wooden sign reading "OPEN"'})
    assert has_text_payload({"operation": "replace", "payload": "an oval sign reading “Ivy Room”"})
    assert not has_text_payload({"operation": "replace", "payload": "a children's bicycle"})
    assert not has_text_payload({"operation": "attribute", "payload": 'reading "X"'})


def test_earlier_batches_are_excluded_with_their_source_images(tmp_path):
    from extedit.sources import load_exclusions
    (tmp_path / "ingest").mkdir()
    rows = [{"id": "crispedit-add_00001-000", "source_sha1": "a"}, {"id": "crispedit-add_00001-001", "source_sha1": "a"},
            {"id": "crispedit-add_00001-000", "source_sha1": "a"}]
    (tmp_path / "ingest" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "{broken")
    ids, per_image = load_exclusions([tmp_path])
    assert ids == {"crispedit-add_00001-000", "crispedit-add_00001-001"} and per_image == {"a": 2}


def test_naturalness_votes_and_parsing():
    from extedit import natcheck as nc
    natural, traces = {"natural": True, "traces": ""}, {"natural": False, "traces": "floating wheels"}
    assert nc.decide([natural, natural]) == (True, None)
    assert nc.decide([natural, traces], reject_votes=2) == (True, None)
    assert nc.decide([natural, traces], reject_votes=1) == (False, "before_unnatural")
    assert nc.decide([traces, traces]) == (False, "before_unnatural")
    assert nc.decide([natural]) == (False, "unreadable_answer")
    assert nc.parse_vote('{"traces": "x", "natural": "false"}') == {"natural": False, "traces": "x"}
    with pytest.raises(ValueError):
        nc.parse_vote('{"traces": "x"}')
    assert nc.needs_check({"dataset": "crispedit", "native_type": "add"})
    assert not nc.needs_check({"dataset": "scaleedit", "native_type": "object_addition"})


def test_naturalness_views_keep_the_area_with_context():
    from PIL import Image
    from extedit import natcheck as nc
    area = np.zeros((600, 800), bool)
    area[300:340, 500:560] = True
    box = nc.crop_box(area, (800, 600))
    assert box[0] <= 500 and box[1] <= 300 and box[2] >= 560 and box[3] >= 340
    assert box[2] - box[0] >= 200 and box[3] - box[1] >= 200  # at least a quarter of the long side
    image = Image.new("RGB", (800, 600), (90, 90, 90))
    full, before, after = nc.views(image, image, area)
    assert before.size == after.size and max(before.size) <= 1024 and full.size == (800, 600)


def test_payload_cap_counts_earlier_batches_and_multi_rows_first():
    from collections import Counter
    from extedit import export as ex
    spec = lambda op, payload: {"operation": op, "payload": payload}
    singles = [{"id": f"r{k}", "instruction_spec": spec("replace", "A wooden, bench")} for k in range(5)]
    singles.append({"id": "c", "instruction_spec": spec("attribute", "")})
    multis = [{"id": "m", "units": [{"instruction_spec": spec("add", "a wooden bench")}]}]
    assert ex.payload_key(spec("replace", "An  Old, Red Car")) == "old red car"
    assert ex.payload_key(spec("remove", "")) is None
    dropped = ex.cap_payloads(singles, multis, Counter({"wooden bench": 1}), cap=3)
    assert len(dropped) == 4 and "c" not in dropped  # 1 earlier + 1 multi + 1 single = 3
    assert dropped == ex.cap_payloads(list(reversed(singles)), multis, Counter({"wooden bench": 1}), cap=3)


def test_exclusions_from_a_rows_file_and_decided_rows(tmp_path):
    from extedit.export import decided_ids
    from extedit.sources import load_exclusions
    rows = tmp_path / "decided_rows.jsonl"
    rows.write_text(json.dumps({"id": "a", "source_sha1": "x"}) + "\n" + json.dumps({"id": "b", "source_sha1": "x"}) + "\n")
    assert load_exclusions([rows]) == ({"a", "b"}, {"x": 2})
    final = tmp_path / "final"
    final.mkdir()
    (final / "manifest_protocol_v2.jsonl").write_text(json.dumps({"id": "a"}) + "\n")
    (final / "manifest_multi_protocol_v2.jsonl").write_text("")
    (final / "rejected.jsonl").write_text(json.dumps({"id": "b", "reason": "drop_no_change"}) + "\n"
                                          + json.dumps({"id": "c", "reason": "missing"}) + "\n")
    assert decided_ids(final) == {"a", "b"}


def test_text_recheck_unit_kinds():
    from extedit.textrecheck import text_kind
    assert text_kind({"operation": "attribute", "attributes": [{"property": "text", "value": "OPEN"}]}) == "change"
    assert text_kind({"operation": "attribute", "attributes": [{"property": "color", "value": "red"}]}) is None
    assert text_kind({"operation": "add", "payload": "a 'LOVE' text"}) == "render"
    assert text_kind({"operation": "replace", "payload": "a banner reading Festival of Lights"}) == "render"
    assert text_kind({"operation": "add", "payload": "a man's straw hat"}) is None
    assert text_kind({"operation": "remove", "payload": "", "scope": "lettering"}) is None


def test_text_recheck_request_parsing_and_expected_text():
    from extedit import textrecheck as tr
    assert tr.parse_request('{"text": "OPEN", "garbled": "false"}') == {"text": "OPEN", "garbled": False}
    with pytest.raises(ValueError):
        tr.parse_request('{"text": "OPEN"}')
    change = {"kind": "change", "expected": "SALE"}
    render = {"kind": "render", "expected": None}
    assert tr.expected_text(change, [{"text": "x", "garbled": False}]) == "SALE"
    assert tr.expected_text(render, [{"text": "", "garbled": False}, {"text": "Poppy", "garbled": False}]) == "Poppy"
    assert tr.expected_text(render, [{"text": "", "garbled": False}] * 2) == ""


def test_text_recheck_units_from_both_formats(tmp_path):
    from PIL import Image
    from extedit import masks as mk
    from extedit import textrecheck as tr
    mask = np.zeros((40, 60), bool)
    mask[10:20, 10:30] = True
    (tmp_path / "rel").mkdir()
    Image.fromarray(mask.astype(np.uint8) * 255).save(tmp_path / "rel" / "m.png")
    rle = {k: v for k, v in mk.encode(mask).items() if k != "box_1000"}
    release = {"sample_id": "s1", "instruction": "Change the text on the sign to \"OPEN\".", "mask": "m.png",
               "source_image": "a.jpg", "edited_image": "b.png", "edited_region": rle,
               "instruction_spec": {"operation": "attribute", "attributes": [{"property": "text", "value": "OPEN"}]}}
    (tmp_path / "rel" / "r.jsonl").write_text(json.dumps(release) + "\n")
    unit = {"instruction_spec": {"operation": "add", "payload": "a sign reading 'Hi'"}, "instruction_ref": "Add ...",
            "selection": rle, "edited_region": rle}
    pair = {"id": "p1", "source_image": "a.jpg", "target_image": "b.png", "units": [unit, {
        "instruction_spec": {"operation": "remove", "payload": ""}, "selection": rle}]}
    (tmp_path / "man").mkdir()
    (tmp_path / "man" / "m.jsonl").write_text(json.dumps(pair) + "\n")
    sources = {"A": {"kind": "release", "root": str(tmp_path / "rel"), "files": [("r.jsonl", False)]},
               "B": {"kind": "manifest", "root": str(tmp_path / "man"), "files": [("m.jsonl", True)],
                     "read_changes": False}}
    counts = tr.build_units(sources, tmp_path / "units.jsonl")
    units = [json.loads(line) for line in open(tmp_path / "units.jsonl")]
    assert counts == {"A:change": 1, "B:render": 1}
    assert units[0]["id"] == "A:s1#0" and units[0]["expected"] == "OPEN" and units[0]["area"]["png"].endswith("m.png")
    assert units[1]["id"] == "B:p1#0" and units[1]["read"] and units[1]["area"]["rle_counts"] == rle["rle_counts"]
    assert tr.area_mask(units[0]["area"], (40, 60)).sum() == mask.sum() == tr.area_mask(units[1]["area"], (40, 60)).sum()


def test_text_recheck_checks_a_unit():
    import asyncio
    from extedit import textrecheck as tr

    class Client:
        def __init__(self):
            self.calls = []

        async def call(self, parts, prompt, call, seed=0, prefix=None, n=1):
            self.calls.append(call)
            if call == "request":
                return {"choices": [{"text": '{"text": "OPEN", "garbled": false}'}] * n, "tokens": 4}
            return {"text": '{"before": "CLOSED", "after": "OPEN NOW", "after_legible": true, "after_garbled": false}',
                    "tokens": 3}

    async def parts():
        return [{"type": "text", "text": "x"}]
    unit = {"id": "D3:r#0", "source": "D3", "unit": 0, "kind": "render", "instruction": "Add a sign reading OPEN.",
            "payload": "a sign reading OPEN", "expected": None, "read": True}
    client = Client()
    record = asyncio.run(tr.check_unit(client, unit, parts))
    assert client.calls == ["request", "ocr"] and record["expected"] == "OPEN"
    assert record["read_passed"] is True and record["reading"]["after"] == "OPEN NOW" and len(record["votes"]) == 2
    client = Client()
    record = asyncio.run(tr.check_unit(client, {**unit, "kind": "change", "expected": "SALE", "read": False}, parts))
    assert client.calls == ["request"] and record["reading"] is None and record["read_passed"] is None
    summary = tr.summary_of([record])
    assert summary["read"] == {"not_read": 1} and summary["garbled_votes"] == {0: 1}


def test_text_recheck_doc_block_is_settings_plus_the_entry_script():
    repo = ROOT.parent
    doc = (repo / "docs" / "EXTEDIT_TEXT_RECHECK.md").read_text()
    block = doc.split("<!-- code: text recheck settings + scripts/extedit_bootstrap_arnold.sh -->", 1)[1]
    block = block.split("```bash\n", 1)[1].split("\n```", 1)[0]
    settings, script = block.split("\n\n#!/usr/bin/env bash\n", 1)
    assert "EXTEDIT_JOB=textrecheck" in settings and "EXTEDIT_NNODES=1" in settings
    assert "#!/usr/bin/env bash\n" + script + "\n" == (repo / "scripts" / "extedit_bootstrap_arnold.sh").read_text()
