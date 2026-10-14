"""Recommended one-call/two-image planner with assigned operations and no fallback."""
import sys

from scripts.experiment_batched_planning import main as experiment_main


def main():
    # Put caller options last so hardware/model settings remain configurable.
    # Operation contract is appended last and cannot silently become fallback.
    sys.argv = [sys.argv[0],
                '--budgets', '256', '--batch-size', '54',
                '--model', '/tmp/sa1b-planner-models/Qwen3.8-Flash-Next-FP8',
                '--tensor-parallel-size', '4', '--gpu-memory-utilization', '.85',
                '--moe-backend', 'triton', '--disable-flashinfer-autotune', '--engram-cpu-offload',
                '--view', 'dark', '--scene-max-edge', '1280', '--wide-context',
                '--crop-max-edge', '896', '--context-min-fraction', '.025',
                '--context-min-pixels', '12', '--view-workers', '4', '--priority-policy', 'balanced',
                *sys.argv[1:], '--prompt-style', 'grounded-balanced', '--structured-output', 'typed-choice',
                '--operation-mode', 'assigned']
    experiment_main()


if __name__ == '__main__':
    main()
