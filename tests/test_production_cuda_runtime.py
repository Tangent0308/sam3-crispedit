from pathlib import Path

from sa1b_pipeline.production_generation import runtime_env


def test_every_role_pins_cuda129_wheel():
    root = Path(__file__).resolve().parents[1]
    for role in ('planner', 'editor', 'audit'):
        rows = (root / f'requirements/production/{role}.lock.txt').read_text().splitlines()
        vllm = [r for r in rows if r.startswith('vllm')]
        assert len(vllm) == 1
        assert vllm[0].startswith('vllm @ https://') and 'cu129' in vllm[0]
        assert 'torch==2.13.0+cu129' in rows
        assert 'nvidia-cuda-runtime-cu12==12.9.79' in rows
        assert not any(r.startswith('nvidia-cuda-runtime==') for r in rows)
        assert [r for r in rows if r.startswith('opencv-')] == ['opencv-python-headless==5.0.0.93']


def test_child_library_path_does_not_add_legacy_cuda13(tmp_path, monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/usr/local/cuda-12.9/lib64')
    libs = tmp_path / 'lib/python3.12/site-packages/nvidia'
    for name in ('cu13', 'cuda_runtime', 'cublas'):
        (libs / name / 'lib').mkdir(parents=True)
    env = runtime_env(tmp_path / 'bin/python', '3')
    assert str(libs / 'cu13/lib') not in env['LD_LIBRARY_PATH']
    assert str(libs / 'cuda_runtime/lib') in env['LD_LIBRARY_PATH']
    assert '/usr/local/cuda-12.9/lib64' in env['LD_LIBRARY_PATH']
    assert env['CUDA_VISIBLE_DEVICES'] == '3'
