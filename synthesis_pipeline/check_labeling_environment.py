"""Check each isolated runtime and record comparable cross-node package versions."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from synthesis_pipeline.run_multinode_labeling import atomic


PROBE = r'''
import importlib.metadata as m,json,sys
import torch,cv2,numpy,PIL,pycocotools
assert sys.version_info[:2]==(3,12),sys.version
assert torch.cuda.is_available(), 'CUDA unavailable'
role=sys.argv[1]
expected={'sam':{'torch':'2.8.0','torchvision':'0.23.0','transformers':'4.57.6'},
 'mllm':{'torch':'2.13.0+cu129','vllm':'0.28.0+cu129','transformers':'5.17.0'},
 'editor':{'torch':'2.13.0+cu129','vllm':'0.29.0','transformers':'5.14.1','diffusers':'0.40.0',
           'vllm-omni':'0.29.0rc2.dev265+g44ea27c80'}}
for package,version in expected[role].items():
 assert m.version(package)==version,(role,package,m.version(package),version)
names=['torch','torchvision','numpy','pillow','opencv-python-headless','transformers']
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
print('ENV_JSON='+json.dumps(dict(python=sys.version.split()[0],packages={n:m.version(n) for n in names},cuda=torch.version.cuda,
 gpu_count=torch.cuda.device_count(),gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); reports={}
    for role,key in [('sam','SAM'),('mllm','MLLM'),('editor','EDITOR')]:
        python=os.environ[f'SAMTOK_{key}_PYTHON']
        env=os.environ.copy()
        if role=='editor':
            site=Path(python).parent.parent/'lib/python3.12/site-packages/nvidia'
            env['LD_LIBRARY_PATH']=':'.join([str(site/'cu13/lib'),str(site/'cuda_runtime/lib'),env.get('LD_LIBRARY_PATH','')])
            env['DIFFUSION_ATTENTION_BACKEND']='TORCH_SDPA'
        proc=subprocess.run([python,'-c',PROBE,role],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(proc.stdout,flush=True)
        if proc.returncode:raise RuntimeError(f'{role} environment check failed')
        reports[role]=json.loads(next(s.removeprefix('ENV_JSON=') for s in proc.stdout.splitlines() if s.startswith('ENV_JSON=')))
    atomic(a.out,reports)


if __name__=='__main__':main()
