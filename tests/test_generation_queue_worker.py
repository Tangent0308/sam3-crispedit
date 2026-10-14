import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from sa1b_pipeline import editor
from sa1b_pipeline.specialist_editors import RecolorEditor
from sa1b_pipeline.work_queue import WorkQueue
from scripts.run_generation_queue import main


def test_worker_recovers_after_audit_publish_without_regenerating(tmp_path, monkeypatch):
    source, mask = tmp_path/'source.png', tmp_path/'mask.png'
    Image.new('RGB', (30, 30), 'gray').save(source)
    Image.new('L', (30, 30), 255).save(mask)
    calls = []

    class Model(RecolorEditor):
        def __init__(self, *args, **kwargs):
            self.pipe = SimpleNamespace(close=lambda: None)

        def generate(self, path, unit, seed=0):
            calls.append(unit['unit_id'])
            return super().generate(path, unit, seed)

    generation = WorkQueue(tmp_path/'generation.db')
    for i in range(2):
        case = {'id': f'case-{i}', 'source': {'image_path': str(source)},
                'units': [{'unit_id': f'U{i}', 'operation': 'attribute',
                           'surface_type': 'diffuse_uniform', 'mask_path': str(mask),
                           'payload': 'blue', 'instruction_ref': 'Change the fabric to blue.'}]}
        generation.enqueue({'job_id': f'case-{i}/U{i}', 'case': case})
    generation.seal()
    monkeypatch.setattr(editor, 'DiffusersQwenImageEditor', Model)
    monkeypatch.setattr(sys, 'argv', ['worker', '--root', str(tmp_path), '--worker', '0'])
    original_finish = WorkQueue.finish

    def fail_finish(*args, **kwargs):
        raise RuntimeError('crash after durable audit publish')

    monkeypatch.setattr(WorkQueue, 'finish', fail_finish)
    with pytest.raises(RuntimeError, match='crash'):
        main()
    assert calls == ['U0']
    monkeypatch.setattr(WorkQueue, 'finish', original_finish)
    main()
    assert calls == ['U0', 'U1']
    assert generation.counts() == {'done': 2}
    audit = WorkQueue(tmp_path/'audit.db')
    assert audit.counts() == {'pending': 2}
    for row in generation.results():
        assert row['job_id'] == row['row']['case_id']+'/'+row['row']['unit_id']
    main()  # Completed queue exits without loading another editor.
    assert calls == ['U0', 'U1']
    audit.close()
    generation.close()


def test_cached_generation_keeps_pixels_but_refuses_changed_instruction(tmp_path, monkeypatch):
    import hashlib
    import json

    source, mask = tmp_path/'source.png', tmp_path/'mask.png'
    Image.new('RGB', (30, 30), 'gray').save(source)
    Image.new('L', (30, 30), 255).save(mask)
    calls = []

    class Model(RecolorEditor):
        def __init__(self, *args, **kwargs):
            self.pipe = SimpleNamespace(close=lambda: None)

        def generate(self, path, unit, seed=0):
            calls.append(unit['instruction_ref'])
            return super().generate(path, unit, seed)

    case = {'id': 'case', 'source': {'image_path': str(source)},
            'units': [{'unit_id': 'U1', 'operation': 'attribute', 'surface_type': 'diffuse_uniform',
                       'mask_path': str(mask), 'payload': 'blue', 'instruction_ref': 'Change the fabric to blue.'}]}
    monkeypatch.setattr(editor, 'DiffusersQwenImageEditor', Model)
    hashes = []
    for name in ['original', 'reuse', 'changed']:
        root = tmp_path/name
        if name == 'changed':
            case['units'][0]['instruction_ref'] = 'Change the fabric to green.'
        q = WorkQueue(root/'generation.db')
        q.enqueue({'job_id': 'case/U1', 'case': json.loads(json.dumps(case))})
        q.seal()
        argv = ['worker', '--root', str(root), '--worker', '0']
        if name != 'original':
            argv += ['--reuse-generation-from', str(tmp_path/'original')]
        monkeypatch.setattr(sys, 'argv', argv)
        if name == 'changed':
            with pytest.raises(ValueError, match='different or untracked generation'):
                main()
        else:
            main()
            row = q.results()[0]
            from pathlib import Path
            hashes.append(hashlib.sha256(Path(row['row']['edited_image']).read_bytes()).hexdigest())
            assert row['reused_generation'] == (name == 'reuse')
        q.close()
    assert len(calls) == 1 and hashes[0] == hashes[1]
