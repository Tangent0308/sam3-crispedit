import json
from pathlib import Path

import pytest
from PIL import Image

from sa1b_pipeline.editor import generate_cases
from sa1b_pipeline.io import write_jsonl
from sa1b_pipeline.specialist_editors import RecolorEditor
from sa1b_pipeline.work_queue import WorkQueue


@pytest.mark.parametrize('benchmark_only', [False, True])
def test_queue_recovers_when_manifest_checkpoint_lags_images(tmp_path, benchmark_only):
    source = tmp_path/'source.png'
    mask = tmp_path/'mask.png'
    Image.new('RGB', (30, 30), 'gray').save(source)
    Image.new('L', (30, 30), 255).save(mask)
    units = [{'unit_id': f'U{i}', 'operation': 'attribute', 'surface_type': 'diffuse_uniform',
              'mask_path': str(mask), 'payload': 'blue',
              'instruction_ref': 'Change the fabric to blue.'} for i in (1, 2)]
    manifest = tmp_path/'input.jsonl'
    case = {'id': 'case', 'source': {'image_path': str(source)}, 'units': units}
    if benchmark_only:
        case.update(benchmark_only=True, benchmark_original_id='original')
    write_jsonl(manifest, [case])
    queue = WorkQueue(tmp_path/'queue.db')
    calls = []
    class Editor(RecolorEditor):
        def generate(self, image_path, unit, seed=0):
            calls.append(unit['unit_id'])
            return super().generate(image_path, unit, seed)
    def publish(row):
        assert Path(row['edited_image']).exists()
        queue.enqueue({'job_id': row['case_id']+'/'+row['unit_id'], 'row': row})
    def interrupted(row):
        publish(row)
        raise RuntimeError('interrupted after durable publish')
    with pytest.raises(RuntimeError, match='interrupted'):
        generate_cases(manifest, tmp_path/'out', Editor(), on_result=interrupted, checkpoint_every=16)
    assert not (tmp_path/'out/generated.jsonl').exists()
    rows = generate_cases(manifest, tmp_path/'out', Editor(), on_result=publish, checkpoint_every=16)
    assert calls == ['U1', 'U2']
    assert queue.counts() == {'pending': 2}
    assert len(rows) == len(list(map(json.loads, (tmp_path/'out/generated.jsonl').read_text().splitlines()))) == 2
    assert all(bool(row.get('benchmark_only')) == benchmark_only for row in rows)
    if benchmark_only:
        assert all(row['benchmark_original_id'] == 'original' for row in rows)
    queue.close()
