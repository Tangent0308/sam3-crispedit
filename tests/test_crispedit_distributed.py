"""Exercise the four-rank handoff and exact stage-2 quality-PASS join."""

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crispedit.prefilter.pair_runner import build_jobs as quality_jobs, parse_args as quality_args
from crispedit.difficulty.scene_runner import build_jobs as scene_jobs, parse_args as scene_args


SCRIPT = Path(__file__).resolve().parents[1] / "crispedit" / "distributed.py"
spec = importlib.util.spec_from_file_location("crispedit_prefilter_4node", SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_four_rank_complete_pipeline_and_resume(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = tmp_path / 'sam3.pt'
    checkpoint.touch()
    for index in range(4):
        pq.write_table(pa.table({"dummy": [index, index]}), source / f"add_{index:05d}.parquet")

    def fake_runner(args, plan, stage, names):
        local = args.run_dir / "work" / stage / f"node{args.rank}"
        if not names:
            return
        for name in names:
            (local / "audit").mkdir(parents=True, exist_ok=True)
            (local / "manifest").mkdir(parents=True, exist_ok=True)
            if stage == "quality":
                verdicts = ["FAIL", "FAIL"] if name == "add_00000.parquet" else ["PASS", "FAIL"]
                table = pa.table({"row_idx": [0, 1], "prefilter_verdict": verdicts,
                                  "prefilter_parse_ok": [True, True], 'prefilter_run_id': ['test', 'test'],
                                  'filter_decision': ['keep' if v == 'PASS' else 'drop' for v in verdicts]})
            else:
                assert pipeline.quality_pass_rows(args.quality_dir, name) == [0]
                table = pa.table({"row_idx": [0], "scene_decision": ["PASS"],
                                  "scene_model_called": [True], "scene_parse_ok": [True],
                                  'scene_pass': [True], 'source_prefilter_verdict': ['PASS'],
                                  'source_prefilter_run_id': ['test']})
            for kind in ("audit", "manifest"):
                pq.write_table(table, local / kind / name)
        pipeline.write_json(local / "run_summary.json", {
            "shards": len(names), "worker_errors": [], "worker_exit_codes": [0],
            "model_calls": len(names) if stage == "scene" else 0,
        })

    monkeypatch.setattr(pipeline, "stage_runner", fake_runner)
    def fake_labels(args, labels, stage):
        root = args.run_dir / 'work' / stage / f'node{args.rank}'
        root.mkdir(parents=True, exist_ok=True)
        for name in labels['assignments'][args.rank]:
            p = root / name
            if p.exists():
                continue
            if stage == 'mask':
                assert (args.control / 'grounding.merged').is_file()
            pq.write_table(pa.table({'row_idx': labels['rows'][name], 'mask_selection_reason': ['SELECTED'],
                'ground_parse_ok': [True], 'ground_json': ['{}'], 'qc_flag': ['OK']}), p)
    def fake_validation(args, command, stage):
        assert stage == 'validate'
        assert len(list((args.label_dir / 'mask').glob('*.parquet'))) == 3
        pipeline.write_json(args.label_dir / 'validation_summary.json', {'counts': {'rows': 3}})
    monkeypatch.setattr(pipeline, 'label_runner', fake_labels)
    monkeypatch.setattr(pipeline, 'execute', fake_validation)
    common = dict(nodes=4, run_dir=tmp_path / "run", source_dir=source,
                  quality_dir=tmp_path / "quality", scene_dir=tmp_path / "scene",
                  model_path=model, python=Path(sys.executable), download_run_dir=None,
                  devices="0,1,2,3,4,5,6,7", tensor_parallel_size=1,
                  batch_size=4, wait_seconds=30, plan_only=False, resume=False,
                  resume_token="", label_dir=None, checkpoint_path=checkpoint,
                  sam_python=Path(sys.executable), grounding_tp=2, grounding_batch_size=4, local_test=True)

    def launch(rank):
        pipeline.run(SimpleNamespace(**common, rank=rank))

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(launch, rank) for rank in range(4)]:
            future.result(timeout=40)
    quality = json.loads((common["quality_dir"] / "run_summary.json").read_text())
    scene = json.loads((common["scene_dir"] / "run_summary.json").read_text())
    assert quality["verdicts"] == {"PASS": 3, "FAIL": 5}
    assert scene["decisions"] == {"PASS": 3}
    assert scene["model_calls"] == 3
    assert pq.read_table(common["scene_dir"] / "manifest/add_00000.parquet").num_rows == 0
    assert all((common["run_dir"] / "control/initial" / f"scene.node{rank}.done").exists()
               for rank in range(4))
    assert (common['run_dir'] / 'control/initial/complete.ok').is_file()
    labels = json.loads((common['run_dir'] / 'label_plan.json').read_text())
    assert sum(labels['node_rows']) == 3 and sorted(labels['node_rows']) == [0, 1, 1, 1]
    # Restarting the same run may revisit the merger, but must not double-count calls.
    pipeline.summarize(SimpleNamespace(**common, rank=0),
                       json.loads((common["run_dir"] / "plan.json").read_text()), "scene")
    assert json.loads((common["scene_dir"] / "run_summary.json").read_text())["model_calls"] == 3
    # A new attempt revalidates shared artifacts without stale completion markers.
    common.update(resume=True, resume_token='retry2')
    def reuse_filter(args, plan, stage, names):
        pass
    monkeypatch.setattr(pipeline, 'stage_runner', reuse_filter)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(launch, rank) for rank in range(4)]:
            future.result(timeout=40)
    assert (common['run_dir'] / 'control/retry2/complete.ok').is_file()
    assert json.loads((common['scene_dir'] / 'run_summary.json').read_text())['model_calls'] == 3


def test_peer_failure_is_not_hidden_by_other_phase(tmp_path):
    pipeline.marker(tmp_path / 'mask.node2.failed', 'GPU failed')
    with pytest.raises(RuntimeError, match='Peer failed'):
        pipeline.wait_for(tmp_path / 'complete.ok', tmp_path, 1)
    with pytest.raises(RuntimeError, match='peer failed'):
        pipeline.await_nodes('grounding', tmp_path, 4, 1)


def test_scene_command_passes_manifest_subdirectory(tmp_path, monkeypatch):
    args = SimpleNamespace(run_dir=tmp_path / 'run', quality_dir=tmp_path / 'quality',
                           source_dir=tmp_path / 'source', model_path=tmp_path / 'model',
                           python=Path(sys.executable), devices='0,1', rank=0,
                           tensor_parallel_size=1, batch_size=4)
    def check_command(args, command, stage):
        assert stage == 'scene'
        assert command[command.index('--prefilter-manifest-dir') + 1] == str(args.quality_dir / 'manifest')
        raise RuntimeError('checked command')
    monkeypatch.setattr(pipeline, 'execute', check_command)
    with pytest.raises(RuntimeError, match='checked command'):
        pipeline.stage_runner(args, {}, 'scene', ['add_00000.parquet'])


def test_summary_preserves_historical_out_of_scope_results(tmp_path):
    source, quality = tmp_path / 'source', tmp_path / 'quality'
    source.mkdir()
    for name in ['add_00000.parquet', 'style_00000.parquet']:
        pq.write_table(pa.table({'dummy': [1, 2]}), source / name)
        for kind in ['audit', 'manifest']:
            (quality / kind).mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({'row_idx': [0, 1], 'prefilter_verdict': ['PASS', 'FAIL'],
                                    'prefilter_parse_ok': [True, True]}), quality / kind / name)
    args = SimpleNamespace(source_dir=source, quality_dir=quality, run_dir=tmp_path / 'run', nodes=4)
    result = pipeline.summarize(args, {'source_shards': ['add_00000.parquet'],
                                      'quality_assignments': [[], [], [], []]}, 'quality')
    assert result['rows'] == 4 and result['shards'] == 2
    assert result['verdicts'] == {'PASS': 2, 'FAIL': 2}
    assert result['active_source_shards'] == 1 and result['preserved_out_of_scope_shards'] == 1


@pytest.mark.parametrize('stage', ['quality', 'scene'])
def test_filter_resume_does_not_load_model_for_completed_shards(tmp_path, monkeypatch, stage):
    import queue
    from crispedit.prefilter import pair_runner
    from crispedit.difficulty import scene_runner
    module = pair_runner if stage == 'quality' else scene_runner
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(module, '_current_output', lambda job: True)
    monkeypatch.setattr(module, '_existing_summary', lambda job: {'rows': 2})
    def unexpected_load(*args, **kwargs):
        pytest.fail('A completed filter shard must not load the model')
    monkeypatch.setattr(module, 'Qwen38PairAuditor' if stage == 'quality' else 'Qwen38SceneAuditor', unexpected_load)
    events = queue.Queue()
    module.worker_main(0, [0], [SimpleNamespace(num_rows=2, input_path='add_00000.parquet')],
                       dict(input_dir=str(tmp_path), output_dir=str(tmp_path), model_path=str(tmp_path),
                            prefilter_manifest_dir=str(tmp_path), overwrite=False), events)
    messages = list(events.queue)
    assert any(m['kind'] == 'shard_done' for m in messages)
    assert not any(m['kind'] == 'worker_error' for m in messages)


def test_mask_merger_rejects_other_run_and_wrong_row_ids(tmp_path):
    args = SimpleNamespace(run_dir=tmp_path / 'run', label_dir=tmp_path / 'labels')
    local = args.run_dir / 'work/mask/node0'
    local.mkdir(parents=True)
    labels = {'rows': {'add_00000.parquet': [7]}, 'assignments': [['add_00000.parquet']]}
    p = local / 'add_00000.parquet'
    pq.write_table(pa.table({'row_idx': [0], 'mask_selection_reason': ['SELECTED'], 'qc_flag': ['OK']}), p)
    with pytest.raises(ValueError, match='row alignment'):
        pipeline.merge_labels(args, labels, 'mask')
    pq.write_table(pa.table({'row_idx': [7], 'mask_selection_reason': ['SELECTED'], 'qc_flag': ['OK']}), p)
    (args.label_dir / 'mask').mkdir(parents=True, exist_ok=True)
    (args.label_dir / 'mask' / p.name).write_bytes(b'other run')
    with pytest.raises(FileExistsError, match='another mask run'):
        pipeline.merge_labels(args, labels, 'mask')


def test_grounding_parse_error_is_row_level_and_mergeable(tmp_path):
    root = tmp_path / 'grounding'
    root.mkdir()
    name = 'add_00000.parquet'
    pq.write_table(pa.table({
        'row_idx': [3],
        'mask_selection_reason': ['SELECTED'],
        'ground_parse_ok': [False],
        'grounding_status': ['PARSE_ERROR'],
        'ground_json': [json.dumps({'requests': []})],
        'qc_flag': ['GROUND_FAIL'],
    }), root / name)
    labels = {'rows': {name: [3]}}
    pipeline.verify_label_shards(root, [name], labels, 'grounding')

    pq.write_table(pa.table({
        'row_idx': [3],
        'mask_selection_reason': ['SELECTED'],
        'ground_parse_ok': [False],
        'grounding_status': ['GROUND_FAIL'],
        'ground_json': [json.dumps({'runtime_error': 'engine stopped'})],
        'qc_flag': ['GROUND_FAIL'],
    }), root / name)
    with pytest.raises(ValueError, match='Grounding parse/runtime error'):
        pipeline.verify_label_shards(root, [name], labels, 'grounding')


def test_exact_shard_list_is_shared_by_both_stages(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("add_00001.parquet", "add_00002.parquet"):
        pq.write_table(pa.table({"input_img": [b"x", b"x"], "output_img": [b"x", b"x"],
                                 "instruction": ["edit", "edit"], "type": ["add", "add"]}),
                       source / name)
    names = tmp_path / "shards.txt"
    names.write_text("add_00002.parquet\n")
    quality = tmp_path / "quality"
    (quality / "manifest").mkdir(parents=True)
    pq.write_table(pa.table({"row_idx": [0, 1], "prefilter_verdict": ["PASS", "FAIL"]}),
                   quality / "manifest" / "add_00002.parquet")
    q = quality_args(["--input-dir", str(source), "--output-dir", str(quality),
                      "--shard-list-file", str(names)])
    s = scene_args(["--input-dir", str(source), "--output-dir", str(tmp_path / "scene"),
                    "--prefilter-manifest-dir", str(quality / "manifest"),
                    "--shard-list-file", str(names)])
    assert [(Path(job.input_path).name, job.num_rows) for job in quality_jobs(q)] == [("add_00002.parquet", 2)]
    assert [(Path(job.input_path).name, job.row_indices) for job in scene_jobs(s)] == [("add_00002.parquet", (0,))]
