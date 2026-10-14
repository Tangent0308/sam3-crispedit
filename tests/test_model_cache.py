import json

import pytest

from sa1b_pipeline.model_cache import resolve_model, stage_model


def test_cache_reuses_files_repairs_corruption_and_preserves_model_id(tmp_path, monkeypatch):
    model = tmp_path/'source'
    model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'weights.safetensors').write_bytes(b'original weights')
    cache = stage_model(model, tmp_path/'cache')
    from pathlib import Path
    weights = Path(cache)/'weights.safetensors'
    stamp = weights.stat().st_mtime_ns
    assert stage_model(model, tmp_path/'cache') == cache
    assert weights.stat().st_mtime_ns == stamp
    weights.write_bytes(b'corrupt! weights')
    stage_model(model, tmp_path/'cache')
    assert weights.read_bytes() == b'original weights'
    monkeypatch.setenv('SA1B_LOCAL_MODEL_MAP', json.dumps({str(model): cache}))
    assert resolve_model(str(model)) == cache
    assert resolve_model('another-canonical-id') == 'another-canonical-id'
    (Path(cache).parent/'files.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest changed'):
        resolve_model(str(model))


def test_changed_source_gets_distinct_cache_and_shared_cache_is_rejected(tmp_path):
    model = tmp_path/'source'
    model.mkdir()
    (model/'config.json').write_text('{}')
    first = stage_model(model, tmp_path/'cache')
    (model/'config.json').write_text('{"changed":true}')
    assert stage_model(model, tmp_path/'cache') != first
    with pytest.raises(ValueError, match='node-local'):
        stage_model(model, '/mnt/invalid-cache')
