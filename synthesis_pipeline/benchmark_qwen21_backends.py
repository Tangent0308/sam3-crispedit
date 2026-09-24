"""Matched, frozen-plan Diffusers/Omni benchmark. No planning/audit changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--model-id', default='/tmp/tanyue_qwen_image_21')
    p.add_argument('--ids', default=None, help='Comma-separated numeric IDs; default all frozen plans')
    p.add_argument('--gpus', default='0,1', help='Diffusers GPU, Omni GPU')
    p.add_argument('--omni-attention-backend', default='TORCH_SDPA')
    a = p.parse_args()
    gpus = a.gpus.split(',')
    if len(gpus) != 2 or gpus[0] == gpus[1]:
        p.error('Use two different GPUs of the same model')
    rows = [json.loads(s) for s in (a.data_root / 'annotations.jsonl').read_text().splitlines() if s.strip()]
    ids = a.ids or ','.join(r['image'].split('_')[0] for r in rows)
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root / 'logs').mkdir()
    jobs = []
    for backend, gpu, env_name in zip(('diffusers', 'vllm-omni'), gpus, ('qwen_image_21', 'qwen_omni_21')):
        venv = ROOT.parent / '.venvs' / env_name
        python = str(venv / 'bin/python')
        options = ['--data-root', str(a.data_root.resolve()), '--out-root', str((a.out_root / backend).resolve()),
                   '--variant', 'context_grounded_v4_qwen21', '--ids', ids,
                   '--steps', '40', '--model-id', a.model_id, '--qwen21-backend', backend,
                   '--qwen21-prompt-policy', 'typed-v1', '--remove-composition-policy', 'adaptive-remove-v2',
                   '--latent-protection-policy', 'legacy', '--relation-geometry-policy', 'legacy']
        subprocess.run(['/usr/bin/python', '-m', 'synthesis_pipeline.experiment_edit_quality',
                        *options, '--manifest-only'], cwd=ROOT, check=True)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '8'}
        if backend == 'vllm-omni':
            site = venv / 'lib/python3.12/site-packages/nvidia'
            libs = [str(site / 'cu13/lib'), str(site / 'cuda_runtime/lib')]
            env['LD_LIBRARY_PATH'] = ':'.join(libs + [env.get('LD_LIBRARY_PATH', '')])
            env['DIFFUSION_ATTENTION_BACKEND'] = a.omni_attention_backend
        command = [python, '-u', '-m', 'synthesis_pipeline.experiment_edit_quality', *options]
        log = (a.out_root / 'logs' / f'{backend}.log').open('w')
        start = time.perf_counter()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((backend, proc, log, start))
        (a.out_root / f'{backend}_launch.json').write_text(json.dumps(
            {'command': command, 'gpu': gpu, 'pid': proc.pid,
             'attention_backend': env.get('DIFFUSION_ATTENTION_BACKEND')}, indent=2))
    results = []
    while jobs:
        for job in list(jobs):
            backend, proc, log, start = job
            if proc.poll() is not None:
                log.close()
                item = dict(backend=backend, exit_code=proc.returncode,
                            process_wall_seconds=round(time.perf_counter()-start, 3))
                results.append(item)
                print(json.dumps(item), flush=True)
                jobs.remove(job)
        if jobs:
            time.sleep(1)
    (a.out_root / 'run_summary.json').write_text(json.dumps(results, indent=2))
    if any(r['exit_code'] for r in results):
        raise RuntimeError('A backend failed: inspect saved logs')


if __name__ == '__main__':
    main()
