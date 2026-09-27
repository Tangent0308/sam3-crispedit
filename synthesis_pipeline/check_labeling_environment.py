"""Check each isolated runtime and record comparable cross-node package versions."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

from synthesis_pipeline.run_multinode_labeling import atomic


def check_opencv():
    """All OpenCV wheels own cv2: reject mixed installs before importing it."""
    providers = {}
    for name in ('opencv-python', 'opencv-python-headless',
                 'opencv-contrib-python', 'opencv-contrib-python-headless'):
        try:
            providers[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if set(providers) != {'opencv-python-headless'}:
        raise RuntimeError(f'Expected only opencv-python-headless; found {providers}. '
                           'OpenCV wheels overwrite the same cv2 files. Recreate the runtime '
                           'from the corrected locks; installing X11 libraries does not fix this conflict.')
    import cv2
    gui = next((line.split(':', 1)[1].strip() for line in cv2.getBuildInformation().splitlines()
                if line.strip().startswith('GUI:')), None)
    if gui != 'NONE':
        raise RuntimeError(f'Loaded cv2 is not headless (GUI={gui}); recreate this runtime')
    return {'version': cv2.__version__, 'gui': gui, 'providers': providers}


PROBE = r'''
import importlib.metadata as m,json,sys
from synthesis_pipeline.check_labeling_environment import check_opencv
opencv=check_opencv()
import torch,cv2,numpy,PIL,pycocotools
assert sys.version_info[:2]==(3,12),sys.version
assert torch.cuda.is_available(), 'CUDA unavailable'
role=sys.argv[1]
expected={'sam':{'torch':'2.13.0+cu129','torchaudio':'2.11.0+cu129','torchvision':'0.28.0+cu129','transformers':'4.57.6'},
 'mllm':{'torch':'2.13.0+cu129','vllm':'0.28.0+cu129','transformers':'5.17.0'},
 'editor':{'torch':'2.13.0+cu129','vllm':'0.29.0','transformers':'5.14.1','diffusers':'0.40.0',
           'vllm-omni':'0.29.0rc2.dev265+g44ea27c80'}}
assert str(torch.version.cuda).startswith('12.9'), (role, torch.version.cuda)
for package,version in expected[role].items():
 assert m.version(package)==version,(role,package,m.version(package),version)
names=['torch','torchvision','torchaudio','numpy','pillow','opencv-python-headless','transformers']
if role=='sam':
 import os
 sys.path.insert(0,os.environ['SAMTOK_SAM3_SOURCE'])
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 import scipy,pyarrow
 names+=['diffusers','scipy','timm','pyarrow']
elif role=='mllm':
 from vllm import LLM,SamplingParams
 from transformers import AutoProcessor
 names+=['vllm']
else:
 from vllm_omni.entrypoints.omni import Omni
 from vllm_omni.inputs.data import OmniDiffusionSamplingParams
 from vllm_omni.model_extras import build_image_to_image_prompt
 from utils.qwen21_omni_regional import RegionalQwenImage21Pipeline
 names+=['vllm','vllm-omni','diffusers','scipy']
print('ENV_JSON='+json.dumps(dict(python=sys.version.split()[0],packages={n:m.version(n) for n in names},opencv=opencv,cuda=torch.version.cuda,
 gpu_count=torch.cuda.device_count(),gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])))
'''


TRANSIENT_CUDA_MARKERS = (
    "error 802",
    "system not yet initialized",
    "cuda initialization",
    "cuda unavailable",
)


def _transient_cuda_failure(output: str) -> bool:
    text = str(output).lower()
    return any(marker in text for marker in TRANSIENT_CUDA_MARKERS)


def run_probe_with_retry(python, role, env, *, retries=3, delay_seconds=5):
    """Run one role probe, retrying only transient CUDA initialization failures.

    Arnold can expose a worker before its CUDA driver context is ready.  A fresh
    interpreter is intentional here: it clears the failed CUDA initialization
    state without changing package or model configuration.  Package/import
    errors still fail immediately, so a genuinely broken environment is never
    hidden by retries.
    """
    attempts = max(1, int(retries))
    last = None
    for attempt in range(1, attempts + 1):
        last = subprocess.run(
            [python, '-c', PROBE, role],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(last.stdout, flush=True)
        if last.returncode == 0:
            return last
        if not _transient_cuda_failure(last.stdout) or attempt == attempts:
            break
        print(
            f'{role} probe saw transient CUDA initialization failure '
            f'({attempt}/{attempts}); retrying in {delay_seconds}s',
            flush=True,
        )
        time.sleep(max(0, float(delay_seconds)))
    raise RuntimeError(
        f'{role} environment check failed after {attempts} attempt(s):\n'
        f'{last.stdout if last is not None else "no probe output"}'
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); reports={}
    retries = int(os.environ.get('SAMTOK_ENV_PROBE_RETRIES', '3'))
    delay = float(os.environ.get('SAMTOK_ENV_PROBE_DELAY', '5'))
    for role,key in [('sam','SAM'),('mllm','MLLM'),('editor','EDITOR')]:
        python=os.environ[f'SAMTOK_{key}_PYTHON']
        env=os.environ.copy()
        # Force eager module loading for the preflight subprocess only.  This
        # makes a driver readiness race visible/retryable before later workers
        # start model processes; it does not alter the editing runtime.
        env.setdefault('CUDA_MODULE_LOADING', 'EAGER')
        if role=='editor':
            site=Path(python).parent.parent/'lib/python3.12/site-packages/nvidia'
            env['LD_LIBRARY_PATH']=':'.join([str(site/'cu13/lib'),str(site/'cuda_runtime/lib'),env.get('LD_LIBRARY_PATH','')])
            env['DIFFUSION_ATTENTION_BACKEND']='TORCH_SDPA'
        proc = run_probe_with_retry(
            python, role, env, retries=retries, delay_seconds=delay,
        )
        reports[role]=json.loads(next(s.removeprefix('ENV_JSON=') for s in proc.stdout.splitlines() if s.startswith('ENV_JSON=')))
    atomic(a.out,reports)


if __name__=='__main__':main()
