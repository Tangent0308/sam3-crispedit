"""Validate the deployed CUDA 12.9 stack with actual GPU kernels before ready."""
import argparse
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def check(role):
    import torch
    import transformers
    import vllm
    from vllm import _custom_ops

    wheel = json.loads(metadata.distribution('vllm').read_text('direct_url.json') or '{}').get('url', '')
    if torch.version.cuda != '12.9' or 'cu129' not in wheel:
        raise RuntimeError(f'Expected CUDA 12.9 builds: torch={torch.__version__}, vllm={vllm.__version__}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable during runtime preflight')
    x = torch.ones((4, 128), device='cuda', dtype=torch.float16)
    torch.testing.assert_close(x @ x.T, torch.full((4, 4), 128., device='cuda', dtype=x.dtype))
    out = torch.empty_like(x)
    _custom_ops.rms_norm(out, x, torch.ones(128, device='cuda', dtype=x.dtype), 1e-6)
    torch.testing.assert_close(out, x)
    if role == 'editor':
        from vllm_omni.entrypoints.omni import Omni

        from utils.qwen21_omni_regional import RegionalQwenImage21Pipeline
        assert Omni and RegionalQwenImage21Pipeline
    torch.cuda.synchronize()
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if 'libcudart.so.' in line})
    if not libraries or any('libcudart.so.13' in p for p in libraries):
        raise RuntimeError(f'Unexpected loaded CUDA runtime: {libraries}')
    print(json.dumps({'role': role, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'vllm': vllm.__version__, 'vllm_wheel': wheel, 'transformers': transformers.__version__,
                      'gpu': torch.cuda.get_device_name(), 'cuda_libraries': libraries,
                      'matmul': 'pass', 'vllm_rms_norm': 'pass'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['planner', 'editor', 'audit'], required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        check(args.role)
        return
    # Match production child-process library resolution, before importing CUDA.
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from sa1b_pipeline.production_generation import runtime_env
    env = runtime_env(sys.executable, os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--role', args.role, '--child'],
                   env=env, cwd=repo, check=True)


if __name__ == '__main__':
    main()
