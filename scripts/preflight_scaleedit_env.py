#!/usr/bin/env python3
"""Fail before planning if native libraries, spawned imports or vLLM cannot run."""
import argparse
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def check_opencv():
    import cv2
    import numpy as np
    installed = {d.metadata['Name'].lower().replace('_', '-') for d in importlib.metadata.distributions()}
    variants = installed & {'opencv-python', 'opencv-python-headless',
                            'opencv-contrib-python', 'opencv-contrib-python-headless'}
    if variants != {'opencv-python-headless'}:
        raise RuntimeError(f'Only opencv-python-headless is allowed; installed: {sorted(variants)}')
    if not re.search(r'^\s*GUI:\s+NONE\s*$', cv2.getBuildInformation(), re.M):
        raise RuntimeError('OpenCV must be built without GUI / Qt / OpenGL dependencies')
    pixels = np.zeros((16, 16), np.uint8)
    pixels[4:12, 4:12] = 255
    assert cv2.connectedComponents(pixels)[0] == 2
    assert cv2.imdecode(cv2.imencode('.png', pixels)[1], cv2.IMREAD_GRAYSCALE).shape == pixels.shape
    import typing
    assert '/cv2/' not in str(typing.__file__), typing.__file__
    return cv2.__version__


def spawned_imports():
    check_opencv()
    from transformers import AutoProcessor
    from vllm import LLM
    from sam3.model.sam3_image_processor import Sam3Processor
    print('Spawned OpenCV / Transformers / vLLM / SAM3 imports passed', flush=True)


def validate_cluster_reports(reports):
    if len(reports) != 4 or len({r['hostname'] for r in reports}) != 4:
        raise ValueError('Four distinct physical hosts are required')
    for key in ('commit', 'code_sha256', 'versions'):
        if any(r[key] != reports[0][key] for r in reports):
            raise ValueError(f'Nodes disagree on {key}')
    if not all(r['vllm_probe'] and r['gpus'] == 8 for r in reports):
        raise ValueError('Incomplete GPU/model preflight')
    if any(r['base_python'].startswith('/mnt/') or r['python'].startswith('/mnt/') for r in reports):
        raise ValueError('Every node must use local Python and local dependencies')


def vllm_worker(model_path):
    # A real two-image request exercises the process path that failed on Arnold.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    check_opencv()
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    image = Image.new('RGB', (336, 336), 'red')
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'},
                 {'type': 'text', 'text': 'Describe the color in these images in one word.'}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    llm = LLM(model=model_path, tensor_parallel_size=1, dtype='bfloat16', max_model_len=8192,
              max_num_seqs=4, gpu_memory_utilization=0.85, limit_mm_per_prompt={'image': 2},
              enforce_eager=True, generation_config='vllm', trust_remote_code=True, seed=0)
    outputs = llm.generate([{'prompt': prompt, 'multi_modal_data': {'image': [image, image]}}],
                           SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
    assert outputs and outputs[0].outputs and outputs[0].outputs[0].text.strip(), 'Empty vLLM response'
    print('VLLM_TWO_IMAGE_PROBE_OK ' + repr(outputs[0].outputs[0].text), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--model-path', help='Also load Qwen on GPU 0 and generate a real two-image response')
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--cluster-dir', type=Path, help='Validate four completed node readiness reports')
    parser.add_argument('--vllm-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cluster_dir:
        validate_cluster_reports([json.loads((args.cluster_dir / f'node{rank}.ready.json').read_text())
                                  for rank in range(4)])
        print('Four physical nodes passed environment and real vLLM preflight', flush=True)
        return
    if args.vllm_worker:
        vllm_worker(args.model_path)
        return
    version = check_opencv()
    if not Path(sys.executable).resolve().is_relative_to(REPO / '.uv-python'):
        raise RuntimeError('Base Python must be installed in this node-local clone')
    import torch
    import numpy
    import pyarrow
    import transformers
    import vllm
    from sam3.model_builder import build_sam3_image_model
    assert torch.version.cuda == '12.9', torch.version.cuda
    assert torch.cuda.device_count() == args.gpus, (torch.cuda.device_count(), args.gpus)
    for device in range(args.gpus):
        value = torch.ones(1, device=f'cuda:{device}') + 1
        assert value.item() == 2
    child = multiprocessing.get_context('spawn').Process(target=spawned_imports)
    child.start()
    child.join(timeout=300)
    if child.is_alive():
        child.kill()
        child.join()
        raise TimeoutError('Spawned import check exceeded 300 seconds')
    if child.exitcode != 0:
        raise RuntimeError(f'Spawned import check failed with exit code {child.exitcode}')
    if args.model_path:
        print('Loading Qwen3.8 on GPU 0 for the startup probe', flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                    '--vllm-worker', '--model-path', args.model_path],
                                   env=env, start_new_session=True)
        try:
            code = process.wait(timeout=1800)
            if code:
                raise RuntimeError(f'vLLM two-image startup probe failed: exit={code}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    from scaleedit.distributed import code_digest
    report = {'hostname': socket.gethostname(), 'python': sys.executable, 'code_sha256': code_digest(),
              'base_python': str(Path(sys.executable).resolve()), 'gpus': args.gpus,
              'versions': {'torch': torch.__version__, 'vllm': vllm.__version__,
                           'transformers': transformers.__version__, 'numpy': numpy.__version__, 'cv2': version},
              'vllm_probe': bool(args.model_path),
              'commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output_json.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(args.output_json)
    print('PREFLIGHT_OK ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
