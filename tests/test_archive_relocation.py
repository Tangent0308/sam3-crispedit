import json
import pytest

from sa1b_pipeline.archive_relocation import ArchivePaths


def test_longest_prefix_preserves_formal_release_and_boundaries():
    paths = ArchivePaths({"/old/run": "/archive/run", "/old/run/release": "/old/run/release"})
    assert paths.path("/old/run/inputs/a.json") == "/archive/run/inputs/a.json"
    assert paths.path("/old/run/release/images/a.png") == "/old/run/release/images/a.png"
    assert paths.path("/old/runner/a") == "/old/runner/a"
    assert paths.path("images/a.png") == "images/a.png"


def test_nested_media_and_hash_keys_move_without_changing_hashes():
    paths = ArchivePaths({"/old/run": "/new/run"})
    original = {"image": "/old/run/a.png", "hashes": {"/old/run/a.png": "abc"},
                "frames": ["/old/run/b.png"], "fingerprint": "frozen"}
    moved = paths.record(original)
    assert moved["hashes"] == {"/new/run/a.png": "abc"}
    assert moved["frames"] == ["/new/run/b.png"]
    assert moved["fingerprint"] == "frozen"
    assert original["image"] == "/old/run/a.png"


def test_collision_is_not_silently_dropped():
    with pytest.raises(ValueError, match="collide"):
        ArchivePaths({"/a": "/b"}).record({"/a/x": 1, "/b/x": 2})


def test_materialization_keeps_original_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    original = '{"image":"/old/a.png"}\n'
    source.write_text(original)
    output = tmp_path / "view.jsonl"
    paths = ArchivePaths({"/old": "/new"})
    paths.materialize(source, output, jsonl=True)
    assert source.read_text() == original
    assert json.loads(output.read_text())["image"] == "/new/a.png"
    with pytest.raises(FileExistsError):
        paths.materialize(source, output, jsonl=True)
    with pytest.raises(ValueError, match="overwrite"):
        paths.materialize(source, source, jsonl=True)


def test_new_run_source_loading_is_explicit_and_leaves_frozen_records_untouched(tmp_path):
    from scripts.run_twins_single_node import load_source_candidates
    source = tmp_path / "train_rank0.jsonl"
    original = '{"source_id":"sa_1","image_path":"/old/run/image.png"}\n'
    source.write_text(original)
    mapping = tmp_path / "path_map.json"
    mapping.write_text(json.dumps({"/old/run": "/archive/run"}))
    assert load_source_candidates([source])[0]["image_path"] == "/old/run/image.png"
    assert load_source_candidates([source], mapping)[0]["image_path"] == "/archive/run/image.png"
    assert source.read_text() == original
