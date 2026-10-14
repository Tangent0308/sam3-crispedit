"""Real single-host production component smoke; never counts towards 50k."""
import argparse
import time
from pathlib import Path

from sa1b_pipeline.production_generation import run_generation
from sa1b_pipeline.production_state import (
    OPERATIONS,
    atomic_json,
    read_rows,
    write_rows,
)
from scripts.run_production_4node import emit, planner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = {'planner_python': '/tmp/sa1b-next-runtime/bin/python',
              'editor_python': '/tmp/sa1b-editor-runtime/bin/python',
              'audit_python': '/tmp/sa1b-mllm-runtime/bin/python',
              'planner_model': '/tmp/sa1b-planner-models/Qwen3.8-Flash-Next-FP8',
              'editor_model': '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1',
              'audit_model': '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B',
              'devices': list(map(str, range(8)))}
    sources = read_rows(a.manifest)
    plans = planner(a.root, sources, dict.fromkeys(OPERATIONS, 1), config, emit)
    planned = time.monotonic()
    write_rows(a.root/'generation_input.jsonl', plans)
    rows, unused = run_generation(a.root/'generation_input.jsonl', a.root/'generation',
                                 Path('/tmp/sa1b-production-smoke-queues'), dict.fromkeys(OPERATIONS, 1), config, emit)
    atomic_json(a.root/'summary.json', {'sources': len(sources), 'masks': sum(len(r['objects']) for r in sources),
                                      'accepted_plans': len(plans), 'generated_and_audited': len(rows),
                                      'unused_plans': len(unused), 'planning_wall_seconds': planned-started,
                                      'generation_audit_wall_seconds': time.monotonic()-planned,
                                      'production_quota_credit': 0})


if __name__ == '__main__':
    main()
