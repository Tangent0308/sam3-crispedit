#!/usr/bin/env python3
"""Persistent single-GPU inference worker for committed multi-target chains."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
import os
import signal
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sa1b_pipeline.production_state import atomic_json


def watch_parent(owner_pid):
    # Each service owns a fresh process group. A killed coordinator must not
    # leave GPU model descendants running and block the next resume.
    while True:
        if os.getppid() != owner_pid:
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return
        time.sleep(1)


_HEADERS = {}


def _header(path):
    """(request_kind, priority) of a request; request files are immutable once visible."""
    if path.name not in _HEADERS:
        value = json.loads(path.read_text())
        _HEADERS[path.name] = (value.get('request_kind'), int(value.get('priority') or 0))
    return _HEADERS[path.name]


def next_batch(pending, role, batch_size):
    """FIFO within a role; short blind jobs feed editors before final audits.

    A request may carry an integer ``priority`` (opt-in priority scheduling): higher first,
    FIFO within one priority. Without it every request has priority 0 and the order is FIFO.
    """
    pending=sorted(pending,key=lambda p:(p.stat().st_mtime_ns,p.name))
    if role == 'audit':
        blind=[p for p in pending if _header(p)[0]=='blind']
        if blind:
            pending=blind
    pending=sorted(pending,key=lambda p:-_header(p)[1])  # stable sort keeps FIFO within a priority
    return pending[:batch_size if role in ('audit','planner') else 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['editor', 'twins', 'audit', 'planner', 'grounding'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--dtype', default='bf16')
    p.add_argument('--owner-pid', type=int, required=True)
    p.add_argument('--max-new-tokens', type=int, default=1600)
    p.add_argument('--thinking-token-budget', type=int)
    p.add_argument('--lama-repair-model')
    p.add_argument('--restore-color-texture', action='store_true')
    p.add_argument('--text-verify', action='store_true',
                   help='OCR-verified exact wording / no unrequested glyphs, bounded retries and glyph repair')
    p.add_argument('--editor-finish', help='Opt-in JSON finishing options (sa1b_pipeline.edit_finish / removal_effects)')
    args = p.parse_args(); root = args.directory
    threading.Thread(target=watch_parent, args=(args.owner_pid,), daemon=True).start()
    if args.role in ('editor', 'twins'):
        from sa1b_pipeline.editor import DiffusersQwenImageEditor
        model = DiffusersQwenImageEditor(args.model, steps=40, dtype=args.dtype, crop_instruction=False,
                                         finish=json.loads(args.editor_finish) if args.editor_finish else None)
        if args.lama_repair_model or args.restore_color_texture:
            from sa1b_pipeline.qwen_tool_refinement import QwenWithTools
            from sa1b_pipeline.specialist_editors import LamaEditor
            lama = LamaEditor(args.lama_repair_model) if args.lama_repair_model else None
            model = QwenWithTools(model, lama, repair_noop_remove=bool(lama),
                                  restore_texture=args.restore_color_texture,
                                  diagnostic_directory=root/'qwen_proposals')
        if args.text_verify:
            from sa1b_pipeline.text_verification import TextVerifiedEditor
            model = TextVerifiedEditor(model)
    elif args.role == 'grounding':
        from sa1b_pipeline.grounding_evidence import GroundingEvidence
        model = GroundingEvidence(args.model, root/'evidence')
    elif args.role == 'planner':
        from sa1b_pipeline.mllm import VLLMSelector
        from vllm.sampling_params import StructuredOutputsParams
        os.environ['SA1B_FORCE_DIRECT_VLLM'] = '1'
        model = VLLMSelector(args.model, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                             enable_thinking=args.thinking_token_budget is not None,
                             thinking_token_budget=args.thinking_token_budget,
                             tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        from sa1b_pipeline.twin_planning import output_schema
        model.params.structured_outputs = StructuredOutputsParams(json=output_schema())
    else:
        from sa1b_pipeline.audit_backend import VLLMAuditor
        model = VLLMAuditor(args.model, batch_size=args.batch_size, max_tokens=1000, check_operation=False)
    atomic_json(root/'ready.json', {'time': time.time(), 'role': args.role})
    handled = set()
    try:
        while not (root/'stop').exists():
            pending = sorted(p for p in (root/'requests').glob('*.json') if p.name not in handled)
            if not pending:
                time.sleep(.05); continue
            if args.role in ('audit', 'planner') and len(pending) < args.batch_size:
                time.sleep(.1)
                pending = sorted(p for p in (root/'requests').glob('*.json') if p.name not in handled)
            batch = next_batch(pending,args.role,args.batch_size)
            start = time.monotonic()
            try:
                requests = [json.loads(p.read_text()) for p in batch]
                if args.role in ('editor', 'twins') and 'group' in requests[0]:
                    # Both tasks use the same editor weights and configuration.
                    # A resident pool can switch datasets without reloading.
                    from sa1b_pipeline.group_generation import generate_twin_group
                    item = requests[0]
                    group = generate_twin_group(item['group'], item['directory'], model, seed=item.get('seed', 0))
                    atomic_json(Path(item['directory'])/'group.json', group)
                    results = [group]
                elif args.role == 'editor' and requests[0].get('request_kind') == 'removal_effect':
                    item = requests[0]
                    from PIL import Image
                    image = Image.open(item['source']).convert('RGB')
                    effect, info = model.compute_removal_effect(image, item['unit'])
                    path = None
                    if effect is not None:
                        path = item['output']
                        Image.fromarray(effect.astype('uint8') * 255, mode='L').save(path)
                    results = [{'mask': path, 'info': info}]
                elif args.role == 'editor':
                    item = requests[0]; output = model.generate(item['source'], item['unit'], seed=item['seed'])
                    path = root/'responses'/f'{batch[0].stem}.png'; output.save(path)
                    results = [{'image': str(path), 'context': model.last_generation_context}]
                elif args.role == 'grounding':
                    results = [model.inspect(**requests[0])]
                elif args.role == 'planner':
                    schemas = [j.get('schema', output_schema()) for j in requests]
                    limits = ([j.get('max_new_tokens', args.max_new_tokens) for j in requests]
                              if any('max_new_tokens' in j for j in requests) else None)
                    thinking = ([False if j.get('thinking') is False else None for j in requests]
                                if any('thinking' in j for j in requests) else None)
                    results = model.generate_multimage([(j['images'], j['prompt']) for j in requests],
                                                       schemas=schemas, token_limits=limits, thinking=thinking)
                else:
                    results = model.call_batch(requests)
                if len(results) != len(batch):
                    raise RuntimeError('incomplete service batch')
                for path, result in zip(batch, results):
                    atomic_json(root/'responses'/path.name, {'result': result})
            except Exception as exc:
                traceback.print_exc()
                for path in batch:
                    atomic_json(root/'responses'/path.name, {'error': str(exc)})
                raise  # An infrastructure failure is not an edit rejection.
            handled.update(p.name for p in batch)
            print(json.dumps({'role': args.role, 'batch': len(batch), 'seconds': time.monotonic()-start,
                              'completed': len(handled),
                              **({'model_metrics': model.last_batch_metrics} if args.role == 'planner' else {})}), flush=True)
    finally:
        if args.role in ('editor', 'twins'):
            model.pipe.close()
        else:
            model.close()


if __name__ == '__main__':
    main()
