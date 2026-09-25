"""Native ScaleEdit quality -> scene -> grounding -> mask runners (vLLM/SAM3)."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from scaleedit import policy
from scaleedit.inference import Qwen38FilterEngine, parse_device_groups, assign_jobs
from scaleedit.mask.checklist import parse_checklist_grounding

BASE = [('row_idx', pa.int64()), ('sample_id', pa.string()), ('final_task', pa.string()),
        ('final_instruction', pa.string()), ('source_relative_path', pa.string())]
FILTER_SCHEMA = pa.schema(BASE + [
    ('verdict', pa.string()), ('keep', pa.bool_()), ('reason', pa.string()),
    ('assessment_json', pa.string()), ('attempts_json', pa.string()), ('prompt', pa.string()),
    ('error', pa.string()), ('method', pa.string()), ('model', pa.string()),
])
GROUND_SCHEMA = pa.schema(BASE + [('ground_json', pa.string()), ('qc_flag', pa.string()),
                                ('error', pa.string()), ('model', pa.string())])
INSTANCE = pa.struct([
    ('instance_id', pa.string()), ('candidate_id', pa.int32()), ('ref', pa.string()),
    ('grounding_image', pa.string()), ('mask_method', pa.string()), ('mask_source', pa.string()),
    ('bbox_2d', pa.list_(pa.float64())), ('bbox_xyxy', pa.list_(pa.float64())),
    ('area', pa.int64()), ('rle_size', pa.list_(pa.int32())), ('rle_counts', pa.string()),
    ('mapped_from_target', pa.bool_()), ('audit_json', pa.string()),
])
MASK_SCHEMA = pa.schema(BASE + [
    ('ground_json', pa.string()), ('mask_png', pa.binary()), ('instance_masks', pa.list_(INSTANCE)),
    ('qc_flag', pa.string()), ('qc_flags_json', pa.string()), ('mask_source', pa.string()),
    ('area_frac', pa.float64()), ('mask_sum', pa.int64()), ('mask_height', pa.int32()),
    ('mask_width', pa.int32()), ('error', pa.string()), ('method', pa.string()),
])
SCHEMAS = {'quality': FILTER_SCHEMA, 'scene': FILTER_SCHEMA, 'grounding': GROUND_SCHEMA, 'mask': MASK_SCHEMA}
REQUIRED_COLUMNS = frozenset({'sample_id', 'final_task', 'final_instruction', 'source_image', 'edited_image'})


@dataclass
class Job:
    path: str
    indices: list[int]
    num_rows: int
    signature: str


def decode(value):
    if isinstance(value, dict):
        value = value['bytes']
    with Image.open(io.BytesIO(value)) as image:
        return image.convert('RGB')


def base_row(row_idx, row):
    return dict(row_idx=row_idx, sample_id=str(row['sample_id']), final_task=policy.task_name(row['final_task']),
                final_instruction=str(row['final_instruction'] or '').strip(),
                source_relative_path=row.get('source_relative_path', ''))


def discover(root):
    paths = sorted(set(root.glob('part-*.parquet')) | set(root.glob('expand-*.parquet')))
    if not paths:
        raise FileNotFoundError(f'No ScaleEdit data shards in {root}')
    return paths


def load_selection(path):
    if path is None:
        return None
    payload = json.loads(path.read_text())
    cases = payload['cases'] if isinstance(payload, dict) else payload
    result = {}
    for item in cases:
        name, idx = str(item['shard']), item['row_idx']
        if Path(name).name != name or not name.endswith('.parquet') or type(idx) is not int or idx < 0:
            raise ValueError(f'Invalid selection: {item}')
        result.setdefault(name, set()).add(idx)
    return {name: sorted(values) for name, values in result.items()}


def load_index(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing upstream result: {path}')
    rows = pq.read_table(path).to_pylist()
    indexed = {row['row_idx']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate upstream row_idx: {path}')
    return indexed


def passed_indices(records, upstream, label):
    result = []
    for idx, row in records:
        entry = upstream.get(idx)
        if entry is None or entry['sample_id'] != str(row['sample_id']):
            raise ValueError(f'{label} identity mismatch at row {idx}')
        if entry.get('final_instruction') != str(row['final_instruction'] or '').strip():
            raise ValueError(f'{label} instruction mismatch at row {idx}')
        if entry.get('keep') != (entry.get('verdict') == 'PASS'):
            raise ValueError(f'{label} contradictory verdict at row {idx}')
        if entry['keep']:
            result.append((idx, row))
    return result


def gate_records(records, args, name):
    if args.stage == 'quality':
        return records
    records = passed_indices(records, load_index(args.quality_dir / 'manifest' / name), 'quality')
    if args.stage in {'grounding', 'mask'}:
        records = passed_indices(records, load_index(args.scene_dir / 'manifest' / name), 'scene')
    return records


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        digest.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def signature(path, indices, args, digest):
    info = path.stat()
    payload = dict(source=str(path.resolve()), size=info.st_size, mtime_ns=info.st_mtime_ns,
                   indices=indices, code=digest, stage=args.stage, model=args.model_path,
                   checkpoint=args.checkpoint_path, max_pixels=args.max_pixels,
                   max_tokens=args.max_new_tokens, retries=args.parse_retries,
                   tensor_parallel_size=args.tensor_parallel_size, batch_size=args.batch_size,
                   model_len=args.vllm_max_model_len, max_num_seqs=args.vllm_max_num_seqs)
    upstream = []
    if args.stage != 'quality':
        upstream.append(args.quality_dir / 'manifest' / path.name)
    if args.stage in {'grounding', 'mask'}:
        upstream.append(args.scene_dir / 'manifest' / path.name)
    if args.stage == 'mask':
        upstream.append(args.grounding_dir / path.name)
    payload['upstream'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in upstream}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def output_path(args, name):
    return args.output_dir / ('audit' if args.stage in {'quality', 'scene'} else '') / name


def reusable(path, expected):
    if not path.exists():
        return False
    actual = (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b'scaleedit_signature', b'').decode()
    if actual != expected:
        raise ValueError(f'Incompatible existing result: {path}; use a fresh output directory')
    return True


def build_jobs(args):
    selected = load_selection(args.selection_file)
    paths = discover(args.input_dir)
    if selected is not None:
        missing = set(selected) - {p.name for p in paths}
        if missing:
            raise ValueError(f'Selected shards not found: {sorted(missing)}')
        paths = [p for p in paths if p.name in selected]
    digest, jobs = code_digest(), []
    for path in paths:
        pf = pq.ParquetFile(path)
        if not REQUIRED_COLUMNS <= set(pf.schema_arrow.names):
            raise ValueError(f'Not a native ScaleEdit shard: {path}')
        indices = selected[path.name] if selected is not None else list(range(pf.metadata.num_rows))
        if any(i >= pf.metadata.num_rows for i in indices):
            raise ValueError(f'Selection outside source shard: {path}')
        sig = signature(path, indices, args, digest)
        target = output_path(args, path.name)
        if reusable(target, sig):
            if args.stage in {'quality', 'scene'} and not reusable(args.output_dir / 'manifest' / path.name, sig):
                # Recover a crash between audit publication and its derived manifest.
                write_table(args.output_dir / 'manifest' / path.name, pq.read_table(target).to_pylist(), FILTER_SCHEMA, sig)
            continue
        jobs.append(Job(str(path), indices, len(indices), sig))
    return jobs


def write_table(path, rows, schema, sig):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema.metadata or {}, scaleedit_signature=sig.encode())
    table = pa.Table.from_pylist(rows, schema=schema.with_metadata(metadata))
    temp = path.with_suffix('.parquet.incomplete')
    pq.write_table(table, temp, compression='zstd')
    os.replace(temp, path)


def infer_filter(engine, records, args):
    output, requests = [], []
    for row_idx, row in records:
        base = base_row(row_idx, row)
        result = dict(base, verdict='DROP', keep=False, reason='', assessment_json='{}',
                      attempts_json='[]', prompt='', error='', method=policy.METHOD, model=args.model_path)
        output.append(result)
        if base['final_task'] not in policy.LOCAL_TASKS:
            result['reason'] = 'excluded_edit_family'
            continue
        try:
            if not base['final_instruction']:
                raise ValueError('empty final_instruction')
            images = [('SOURCE before editing', decode(row['source_image']))]
            prompt = policy.scene_prompt(base['final_task'], base['final_instruction'])
            if args.stage == 'quality':
                images.append(('TARGET edited result', decode(row['edited_image'])))
                prompt = policy.quality_prompt(base['final_task'], base['final_instruction'])
            result['prompt'] = prompt
            requests.append((len(output) - 1, policy.conversation(images, prompt)))
        except Exception as exc:
            result.update(reason='invalid_input', error=repr(exc))
    if requests:
        texts = engine.generate([conv for _, conv in requests], max_tokens=args.max_new_tokens)
        parser = policy.parse_quality if args.stage == 'quality' else policy.parse_scene
        for (index, conv), text in zip(requests, texts):
            _, assessment, error, attempts = engine._parse_with_retry(
                conv, text, parser, lambda e: f'Correct this JSON schema error: {e}. Output only complete JSON.',
                args.max_new_tokens)
            result = output[index]
            result.update(error=error, attempts_json=json.dumps(attempts, ensure_ascii=False))
            if error:
                result['reason'] = 'model_parse_error'
                continue
            keep = assessment.get('keep', assessment.get('verdict') == 'PASS')
            result.update(verdict='PASS' if keep else 'DROP', keep=keep,
                          reason=assessment.get('summary', assessment.get('reason', '')),
                          assessment_json=json.dumps(assessment, ensure_ascii=False))
    return output


def infer_grounding(engine, records, args):
    output, conversations = [], []
    images = {}
    for row_idx, row in records:
        result = dict(base_row(row_idx, row), ground_json='{}', qc_flag='GROUND_FAIL', error='', model=args.model_path)
        output.append(result)
        try:
            source, target = decode(row['source_image']), decode(row['edited_image'])
            images[len(output) - 1] = (source, target)
            prompt = policy.edit_units_prompt(result['final_task'], result['final_instruction'])
            conv = policy.conversation([('Image 1 SOURCE', source), ('Image 2 TARGET', target)], prompt)
            conversations.append((len(output) - 1, conv, prompt))
        except Exception as exc:
            result['error'] = repr(exc)
    if not conversations:
        return output
    texts = engine.generate([conv for _, conv, _ in conversations], max_tokens=args.max_new_tokens)
    payloads, requests = {}, []
    for (index, conv, prompt), text in zip(conversations, texts):
        _, observation, error, attempts = engine._parse_with_retry(
            conv, text, policy.parse_edit_units,
            lambda e: f'Return COMPLETE concise edit-unit JSON. Schema error: {e}.', args.max_new_tokens * 2)
        payload = dict(method=policy.METHOD, observation=dict(prompt=prompt, parsed=observation,
                       parse_ok=not error, error=error, attempts=attempts), requests=[], boxes={'source':[], 'target':[]})
        payloads[index] = payload
        if error or not observation['changes']:
            output[index]['error'] = error or 'no_realized_edits'
            continue
        for side in ('source', 'target'):
            context = policy.side_context(observation, side)
            if not context['changes']:
                continue
            loc_prompt = policy.locate_prompt(observation, side)
            loc_conv = policy.conversation([(f'Full {side} image', images[index][side == 'target'])], loc_prompt)
            requests.append((index, side, context, loc_conv, loc_prompt))
    for start in range(0, len(requests), args.batch_size):
        chunk = requests[start:start + args.batch_size]
        texts = engine.generate([item[3] for item in chunk], max_tokens=1536)
        for (index, side, context, conv, prompt), text in zip(chunk, texts):
            parser = lambda reply, context=context, side=side: policy.parse_located_units(reply, context, side)
            _, parsed, error, attempts = engine._parse_with_retry(
                conv, text, parser, lambda e: f'Correct JSON: {e}. Return every requested ID once.', 3072)
            boxes, unresolved = parsed if parsed is not None else ([], [])
            geometries = {u['change_id']:u['geometry'] for u in context['changes']}
            for box in boxes:
                box['geometry'] = geometries[box['change_id']]
            payloads[index]['boxes'][side] = boxes
            payloads[index]['requests'].append(dict(grounding_image=side, prompt=prompt, parse_ok=not error,
                                                   error=error, unresolved=unresolved, attempts=attempts))
    for index, payload in payloads.items():
        result = output[index]
        if not result['error']:
            requests_ok = payload['requests'] and all(r['parse_ok'] for r in payload['requests'])
            has_boxes = any(payload['boxes'].values())
            unresolved = any(r['unresolved'] for r in payload['requests'])
            result['qc_flag'] = ('MASK_REVIEW' if unresolved else 'OK') if requests_ok and has_boxes else 'GROUND_FAIL'
            if result['qc_flag'] == 'GROUND_FAIL':
                result['error'] = 'missing_or_invalid_grounding'
        result['ground_json'] = json.dumps(payload, ensure_ascii=False)
    return output


def process_job(job, engine, args, progress):
    path = Path(job.path)
    source = pq.read_table(path).to_pylist()
    records = [(idx, source[idx]) for idx in job.indices]
    records = gate_records(records, args, path.name)
    output = []
    if args.stage == 'mask':
        from scaleedit.render import render_mask
        upstream = load_index(args.grounding_dir / path.name)
        for idx, row in records:
            ground = upstream.get(idx)
            if (ground is None or ground['sample_id'] != str(row['sample_id'])
                    or ground['final_instruction'] != str(row['final_instruction'] or '').strip()):
                raise ValueError('Grounding/source identity mismatch')
            output.append(render_mask(engine, idx, row, ground))
            progress.put(1)
    else:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            output.extend(infer_grounding(engine, batch, args) if args.stage == 'grounding' else infer_filter(engine, batch, args))
            progress.put(len(batch))
    # Account for upstream-excluded rows in the source-selection progress total.
    progress.put(job.num_rows - len(records))
    write_table(output_path(args, path.name), output, SCHEMAS[args.stage], job.signature)
    if args.stage in {'quality', 'scene'}:
        write_table(args.output_dir / 'manifest' / path.name, output, FILTER_SCHEMA, job.signature)


def worker(devices, jobs, values, progress):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    args = argparse.Namespace(**values)
    # Empty/upstream-excluded shards still need typed outputs, not a loaded GPU model.
    active, empty = [], []
    for job in jobs:
        metadata = pq.read_table(job.path, columns=['sample_id', 'final_task', 'final_instruction']).to_pylist()
        records = gate_records([(i, metadata[i]) for i in job.indices], args, Path(job.path).name)
        needed = bool(records)
        if args.stage in {'quality', 'scene'}:
            needed = any(policy.task_name(row['final_task']) in policy.LOCAL_TASKS for _, row in records)
        (active if needed else empty).append(job)
    for job in empty:
        process_job(job, None, args, progress)
    jobs = active
    if not jobs:
        return
    if args.stage == 'mask':
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(device='cuda', checkpoint_path=args.checkpoint_path, load_from_HF=False)
        engine = Sam3Processor(model, device='cuda', confidence_threshold=0.3)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for job in jobs:
                process_job(job, engine, args, progress)
    else:
        engine = Qwen38FilterEngine(args)
        try:
            for job in jobs:
                process_job(job, engine, args, progress)
        finally:
            core = getattr(getattr(getattr(engine, 'model', None), 'llm_engine', None), 'engine_core', None)
            if core is not None:
                core.shutdown(timeout=30)


def summarize(args):
    counts, by_task = Counter(), {}
    root = args.output_dir / 'audit' if args.stage in {'quality', 'scene'} else args.output_dir
    columns = ['final_task', 'error', 'verdict' if args.stage in {'quality', 'scene'} else 'qc_flag']
    for path in sorted(root.glob('*.parquet')):
        for row in pq.read_table(path, columns=columns).to_pylist():
            label = row[columns[-1]]
            counts['rows'] += 1
            counts[label] += 1
            counts['errors'] += bool(row['error'])
            by_task.setdefault(row['final_task'], Counter())[label] += 1
    result = dict(stage=args.stage, counts=dict(counts), by_task=by_task, method=policy.METHOD)
    (args.output_dir / 'run_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=tuple(SCHEMAS), required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--quality-dir', type=Path)
    parser.add_argument('--scene-dir', type=Path)
    parser.add_argument('--grounding-dir', type=Path)
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--model-path', default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')
    parser.add_argument('--checkpoint-path', default='/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt')
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--max-pixels', type=int, default=1310720)
    parser.add_argument('--parse-retries', type=int, default=1)
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--vllm-max-model-len', type=int)
    parser.add_argument('--vllm-max-num-seqs', type=int, default=4)
    parser.add_argument('--vllm-enforce-eager', action='store_true')
    parser.add_argument('--max-images-per-generate', type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.parse_retries < 0:
        parser.error('Invalid batch size or retry count')
    for required in ({'quality': [], 'scene': ['quality_dir'], 'grounding': ['quality_dir', 'scene_dir'],
                      'mask': ['quality_dir', 'scene_dir', 'grounding_dir']}[args.stage]):
        if getattr(args, required) is None:
            parser.error(f'--{required.replace("_", "-")} is required')
    src, dst = args.input_dir.resolve(), args.output_dir.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        parser.error('Source and output directories must be disjoint')
    args.max_new_tokens = args.max_new_tokens or {'quality':1024, 'scene':256, 'grounding':3072, 'mask':1}[args.stage]
    args.vllm_max_model_len = args.vllm_max_model_len or (16384 if args.stage == 'grounding' else 8192)
    if args.stage == 'mask' and args.tensor_parallel_size != 1:
        parser.error('SAM3 requires TP1')
    for upstream in (args.quality_dir, args.scene_dir, args.grounding_dir):
        if upstream is not None and (dst == upstream.resolve() or dst in upstream.resolve().parents
                                    or upstream.resolve() in dst.parents):
            parser.error('Stage outputs and upstream results must be disjoint')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = load_selection(args.selection_file)
        scope = json.dumps(dict(input_dir=str(src), stage=args.stage, selection=selection), sort_keys=True, indent=2)
        scope_path = args.output_dir / 'scope.json'
        if scope_path.exists() and scope_path.read_text() != scope:
            raise ValueError('Different input selection in existing output directory; use a fresh directory')
        scope_path.write_text(scope)
        jobs = build_jobs(args)
        if jobs:
            groups = parse_device_groups(args.devices, args.tensor_parallel_size)
            ctx = mp.get_context('spawn')
            progress = ctx.Queue()
            processes = [ctx.Process(target=worker, args=(devices, bucket, vars(args), progress))
                         for devices, bucket in assign_jobs(jobs, groups)]
            for process in processes:
                process.start()
            try:
                with tqdm(total=sum(job.num_rows for job in jobs), desc=f'ScaleEdit {args.stage}', unit='pair', mininterval=5) as bar:
                    while any(p.is_alive() for p in processes):
                        if any(p.exitcode not in (None, 0) for p in processes):
                            raise RuntimeError(f'Worker failed: {[p.exitcode for p in processes]}')
                        try:
                            bar.update(progress.get(timeout=1))
                        except queue.Empty:
                            pass
                    for process in processes:
                        process.join()
                    while not progress.empty():
                        bar.update(progress.get())
                if any(p.exitcode != 0 for p in processes):
                    raise RuntimeError(f'Worker failed: {[p.exitcode for p in processes]}')
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=30)
        summarize(args)


if __name__ == '__main__':
    main()
