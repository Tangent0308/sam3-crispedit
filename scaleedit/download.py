"""Append verified, nonduplicate filtered ScaleEdit pairs using pinned HF shards.

The public source has been reordered: never trust manifest row_index alone.
Match original_instruction uniquely within its named source shard. Export the
reviewed instruction/category and both image bytes; never export an URL-only pair.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import threading
import time
import urllib.request

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from PIL import Image
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

FILTERED_REPO = 'QingyuShi/scaleedit-filtered-6m'
SOURCE_REPO = 'InternVL-U/ScaleEdit-12M'
EXCLUDED_TASKS = frozenset({
    'background_replacement', 'style_transfer', 'tone_adjustment',
    'visual_beautification', 'viewpoint_transformation', 'part_extraction',
})
LOCAL_TASKS = frozenset({
    'action_editing', 'building_surface_text_editing', 'color_change',
    'compositional_editing', 'count_change', 'gui_interface_text_editing',
    'material_change', 'movie_poster_text_editing', 'object_addition',
    'object_removal', 'object_replacement', 'object_surface_text_editing',
    'perceptual_reasoning', 'scientific_reasoning', 'size_change',
    'social_reasoning', 'symbolic_reasoning',
})
SCHEMA = pa.schema([
    ('sample_id', pa.string()), ('split', pa.string()),
    ('source_relative_path', pa.string()), ('manifest_row_index', pa.int64()),
    ('public_source_row_index', pa.int64()), ('edit_task', pa.string()),
    ('final_task', pa.string()), ('original_instruction', pa.string()),
    ('final_instruction', pa.string()), ('instruction_action', pa.string()),
    ('category_action', pa.string()), ('confidence', pa.float64()),
    ('source_image', pa.binary()), ('edited_image', pa.binary()),
    ('source_image_url', pa.string()), ('source_image_origin', pa.string()),
    ('source_image_width', pa.int64()), ('source_image_height', pa.int64()),
    ('edited_image_width', pa.int64()), ('edited_image_height', pa.int64()),
])


class DownloadProgress(tqdm):
    """Keep HF byte progress visible when stdout/stderr are redirected by tmux."""
    def __init__(self, *args, **kwargs):
        kwargs['disable'] = False
        kwargs['mininterval'] = 10.0
        super().__init__(*args, **kwargs)


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def cleanup_owned_cache(args):
    if args.cleanup_cache and args.cache_dir.exists():
        root = args.cache_dir.resolve()
        if root.name != f'scaleedit-expand-{args.run_id}' or root.parent.name != '.cache':
            raise ValueError('Refusing cleanup of an unexpected cache root')
        shutil.rmtree(root)


def finalize_approximate_state(state, existing_counts, tolerance_percent):
    """Explicit opt-in only: retain the requested target and report actual counts."""
    if not math.isfinite(tolerance_percent) or not 0 < tolerance_percent <= 1:
        raise ValueError('Finalization tolerance must be in (0, 1] percent')
    added = Counter(existing_counts)
    added.subtract(state['baseline_counts'])
    if any(n < 0 for n in added.values()):
        raise ValueError('Baseline data changed')
    actual = sum(added.values())
    requested = state['target_new_rows']
    if not requested * (1 - tolerance_percent / 100) <= actual <= requested:
        raise ValueError('Actual rows are outside the explicit finalization tolerance')
    return {**state, 'complete': True, 'completion_reason': 'within_explicit_tolerance',
            'completion_tolerance_percent': tolerance_percent, 'new_rows': actual,
            'target_shortfall_rows': requested - actual, 'new_counts': dict(added),
            'final_total_rows': sum(existing_counts.values()), 'updated_at': time.time()}


def unique_instruction_index(instructions):
    index = {}
    for row_idx, instruction in enumerate(instructions):
        if instruction:
            index[instruction] = row_idx if instruction not in index else None
    return {key: value for key, value in index.items() if value is not None}


def balanced_allocation(available, total):
    result = {key: 0 for key in sorted(available)}
    remaining = min(total, sum(available.values()))
    while remaining:
        keys = [key for key in result if result[key] < available[key]]
        portion = max(1, remaining // len(keys))
        for key in keys:
            count = min(portion, available[key] - result[key], remaining)
            result[key] += count
            remaining -= count
    return result


def available_source_groups(groups, sizes):
    """Filtered manifests can also reference shards absent from the public release."""
    usable, missing = {}, {}
    for relative, indices in groups.items():
        if sizes.get(relative, 0):
            usable[relative] = indices
        else:
            missing[relative] = len(indices)
    return usable, missing


def matched_manifest_indices(relative, indices, table, args, revision):
    """HF range-read just the instruction column before committing to image transfer."""
    cache = args.cache_dir / 'indices' / relative
    source = args.cache_dir / 'source' / relative
    if cache.exists():
        instructions = pq.read_table(cache)['edit_instruction'].to_pylist()
    else:
        if source.exists():
            indexed = pq.ParquetFile(source).read(columns=['edit_instruction'])
        else:
            for attempt in range(4):
                try:
                    fs = HfFileSystem()
                    url = f'datasets/{SOURCE_REPO}@{revision}/{relative}'
                    with fs.open(url, 'rb', block_size=256 * 1024, cache_type='readahead') as handle:
                        indexed = pq.ParquetFile(handle).read(columns=['edit_instruction'])
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** (attempt + 1))
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.parquet.incomplete')
        pq.write_table(indexed, temporary, compression='zstd')
        temporary.replace(cache)
        instructions = indexed['edit_instruction'].to_pylist()
    unique = unique_instruction_index(instructions)
    requested = table['original_instruction'].take(pa.array(indices)).to_pylist()
    return [i for i, instruction in zip(indices, requested) if instruction in unique]


def existing_state(root):
    ids, pairs, counts = set(), set(), Counter()
    for path in tqdm(sorted(root.glob('*.parquet')), desc='Index existing ScaleEdit', unit='shard'):
        table = pq.read_table(path, columns=[
            'sample_id', 'source_relative_path', 'original_instruction', 'final_task'])
        for row in table.to_pylist():
            if row['sample_id'] in ids:
                raise ValueError(f"Duplicate existing sample_id: {row['sample_id']}")
            ids.add(row['sample_id'])
            pairs.add((row['source_relative_path'], row['original_instruction']))
            counts[row['final_task']] += 1
    return ids, pairs, counts


def verified_images(row, fetch_urls):
    source, target = row.get('source_image'), row.get('edited_image')
    origin = 'embedded'
    if not source:
        if not fetch_urls or not row.get('source_image_url'):
            raise ValueError('source_bytes_missing')
        expected = row.get('source_image_sha256')
        if not expected:
            raise ValueError('source_url_missing_checksum')
        request = urllib.request.Request(row['source_image_url'], headers={'User-Agent': 'ScaleEditDataset/1.0'})
        with urllib.request.urlopen(request, timeout=15) as response:
            source = response.read(32 * 1024 * 1024 + 1)
        if len(source) > 32 * 1024 * 1024:
            raise ValueError('source_url_too_large')
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError('source_url_checksum_mismatch')
        origin = 'url_sha256_verified'
    if not target:
        raise ValueError('target_bytes_missing')
    sizes = []
    for data in (source, target):
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            sizes.append(image.size)
    return source, target, origin, sizes


def image_result(row, fetch_urls):
    try:
        return verified_images(row, fetch_urls), None
    except Exception as exc:
        # Do not log remote URLs, which can contain signed credentials.
        code = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return None, 'invalid_image:' + code[:100]


def download_shard(relative, args, revision, size):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'Unsafe HF path: {relative}')
    path = args.cache_dir / 'source' / relative
    if path.exists() and path.stat().st_size == size:
        pq.ParquetFile(path).metadata
        return path
    for attempt in range(4):
        try:
            result = Path(hf_hub_download(SOURCE_REPO, relative, repo_type='dataset',
                                         revision=revision, local_dir=args.cache_dir / 'source',
                                         tqdm_class=DownloadProgress))
            if result.stat().st_size != size:
                raise ValueError('HF shard size mismatch')
            pq.ParquetFile(result).metadata
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(30, 2 ** (attempt + 1)))


def materialize(path, candidates, limits, excluded_pairs, args, sink, total_bar):
    """Read batches, not whole multi-GB image row groups, and publish atomically."""
    pf = pq.ParquetFile(path)
    index = unique_instruction_index(pf.read(columns=['edit_instruction'])['edit_instruction'].to_pylist())
    chosen = {}
    stats = Counter()
    records = candidates.to_pylist()
    random.Random(args.seed + int(hashlib.sha256(str(path).encode()).hexdigest()[:8], 16)).shuffle(records)
    # Retain reserves: corrupt or URL-only images must not consume the quota.
    selected_per_task = Counter()
    for record in records:
        task = record['final_task']
        if limits.get(task, 0) <= 0:
            continue
        key = (record['source_relative_path'], record['original_instruction'])
        public_idx = index.get(record['original_instruction'])
        if public_idx is None:
            stats['non_unique_or_unmatched_instruction'] += 1
        elif key in excluded_pairs or public_idx in chosen:
            stats['duplicate_original_pair'] += 1
        elif selected_per_task[task] < limits[task] * 3 + 64:
            chosen[public_idx] = record
            selected_per_task[task] += 1
    accepted = Counter()
    offset = 0
    columns = [name for name in [
        'edit_instruction', 'source_image', 'source_image_url', 'source_image_sha256', 'edited_image',
    ] if name in pf.schema_arrow.names]
    with ThreadPoolExecutor(max_workers=getattr(args, 'image_workers', 8)) as image_pool, \
         tqdm(total=len(chosen), desc=f'Materialize {path.stem}', unit='pair', mininterval=10) as bar:
        for batch in pf.iter_batches(batch_size=32, columns=columns, use_threads=False):
            indices = [i for i in range(batch.num_rows) if offset + i in chosen]
            if indices:
                originals = batch.take(pa.array(indices)).to_pylist()
                results = image_pool.map(lambda row: image_result(row, args.fetch_source_urls), originals)
                for local_idx, original, (verified, error) in zip(indices, originals, results):
                    record = chosen[offset + local_idx]
                    task = record['final_task']
                    bar.update(1)
                    if accepted[task] >= limits[task]:
                        continue
                    if original['edit_instruction'] != record['original_instruction']:
                        raise ValueError('Instruction identity changed while materializing')
                    if error:
                        stats[error] += 1
                        continue
                    source, target, origin, sizes = verified
                    output = {key: record.get(key) for key in SCHEMA.names}
                    output.update(manifest_row_index=record['row_index'], public_source_row_index=offset + local_idx,
                                  source_image=source, edited_image=target, source_image_origin=origin,
                                  source_image_url=original.get('source_image_url'),
                                  source_image_width=sizes[0][0], source_image_height=sizes[0][1],
                                  edited_image_width=sizes[1][0], edited_image_height=sizes[1][1])
                    sink.append(output)
                    excluded_pairs.add((record['source_relative_path'], record['original_instruction']))
                    accepted[task] += 1
                    total_bar.update(1)
            offset += batch.num_rows
            if all(accepted[task] >= limit for task, limit in limits.items()):
                break
    sink.flush()
    return accepted, stats


class Sink:
    def __init__(self, root, run_id, rows_per_file):
        self.root, self.run_id, self.rows_per_file = root, run_id, rows_per_file
        self.buffer = []
        self.number = len(list(root.glob(f'expand-{run_id}-*.parquet')))
        self.lock = threading.RLock()

    def append(self, row):
        with self.lock:
            self.buffer.append(row)
            if len(self.buffer) >= self.rows_per_file:
                self.flush()

    def flush(self):
        with self.lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self.buffer:
            return
        path = self.root / f'expand-{self.run_id}-{self.number:05d}.parquet'
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_suffix('.parquet.incomplete')
        pq.write_table(pa.Table.from_pylist(self.buffer, schema=SCHEMA), temp, compression='zstd')
        os.replace(temp, path)
        self.buffer.clear()
        self.number += 1


def reserve_wave_limits(wave, counts, need, remaining):
    """Reserve disjoint category quotas before concurrent materialization."""
    left = dict(need)
    reserved = {}
    for rel in wave:
        limits = {}
        for task in sorted(left):
            count = min(left[task], counts[rel].get(task, 0), remaining)
            if count:
                limits[task] = count
                left[task] -= count
                remaining -= count
        reserved[rel] = limits
    return reserved


def process_source_shard(rel, indices, table, limits, pairs, args, revision, size, sink, bar):
    if not limits:
        return Counter(), Counter()
    path = download_shard(rel, args, revision, size)
    return materialize(path, table.take(pa.array(indices)), limits, pairs, args, sink, bar)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # The lock covers indexing, downloads and appending; existing shards are immutable.
    with (args.output_dir / '.scaleedit-download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ids, pairs, existing_counts = existing_state(args.output_dir)
        state_path = args.run_dir / 'download_state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['output_dir'] != str(args.output_dir.resolve()) or state['target_new_rows'] != args.target_rows:
                raise ValueError('Resume configuration differs from the saved run')
        else:
            api = HfApi()
            state = dict(output_dir=str(args.output_dir.resolve()), target_new_rows=args.target_rows,
                         baseline_rows=len(ids), baseline_counts=dict(existing_counts), processed=[],
                         filtered_revision=api.dataset_info(FILTERED_REPO).sha,
                         source_revision=api.dataset_info(SOURCE_REPO).sha, excluded_tasks=sorted(EXCLUDED_TASKS),
                         run_id=args.run_id, started_at=time.time())
            atomic_json(state_path, state)
        if state['run_id'] != args.run_id:
            raise ValueError('Different run ID cannot resume this state')
        if args.finalize_within_percent is not None:
            state = finalize_approximate_state(state, existing_counts, args.finalize_within_percent)
            atomic_json(state_path, state)
            cleanup_owned_cache(args)
            print('FINALIZED_APPROXIMATE', json.dumps({key: state[key] for key in (
                'new_rows', 'target_new_rows', 'target_shortfall_rows', 'final_total_rows', 'completion_reason')}), flush=True)
            return
        if state.get('complete'):
            cleanup_owned_cache(args)
            print('Already complete', json.dumps(state, ensure_ascii=False), flush=True)
            return
        manifest = hf_hub_download(FILTERED_REPO, 'keep_manifest.parquet', repo_type='dataset',
                                   revision=state['filtered_revision'], local_dir=args.cache_dir / 'manifest')
        print('Index filtered manifest and exclude existing IDs', flush=True)
        table = pq.read_table(manifest, filters=[('final_task', 'in', sorted(LOCAL_TASKS))])
        table = table.filter(pc.invert(pc.is_in(table['sample_id'], value_set=pa.array(sorted(ids)))))
        # Do not download raw global-edit shards merely to recover a few rerouted examples.
        rels = table['source_relative_path'].to_pylist()
        keep = [Path(rel).parent.name.split('_', 1)[-1] not in EXCLUDED_TASKS for rel in rels]
        table = table.filter(pa.array(keep))
        groups = defaultdict(list)
        for i, rel in enumerate(table['source_relative_path'].to_pylist()):
            groups[rel].append(i)
        sizes = {s.rfilename: s.size for s in HfApi().dataset_info(
            SOURCE_REPO, revision=state['source_revision'], files_metadata=True).siblings}
        groups, missing = available_source_groups(groups, sizes)
        state['missing_public_shards'] = missing
        atomic_json(state_path, state)
        print('MANIFEST_INDEX', json.dumps({'available_shards':len(groups), 'missing_public_shards':missing}), flush=True)
        processed = set(state['processed'])
        resolved = {}
        with ThreadPoolExecutor(max_workers=args.index_workers) as pool:
            futures = {pool.submit(matched_manifest_indices, rel, indices, table, args, state['source_revision']): rel
                       for rel, indices in groups.items() if rel not in processed}
            for future in tqdm(as_completed(futures), total=len(futures), desc='HF unique instruction index', unit='shard', mininterval=5):
                rel = futures[future]
                matched = future.result()
                if matched:
                    resolved[rel] = matched
        state['resolvable_candidates_by_shard'] = {rel: len(indices) for rel, indices in resolved.items()}
        atomic_json(state_path, state)
        groups = resolved
        counts = {rel: Counter(table['final_task'].take(pa.array(indices)).to_pylist()) for rel, indices in groups.items()}
        done = set(state['processed'])
        # Resume derives accepted counts from durable output, not a possibly stale checkpoint.
        added = Counter(existing_counts)
        added.subtract(state['baseline_counts'])
        if any(n < 0 for n in added.values()):
            raise ValueError('Existing dataset changed underneath the download')
        sink = Sink(args.output_dir, args.run_id, args.rows_per_file)
        downloaded_bytes = state.get('downloaded_bytes', 0)
        with tqdm(total=args.target_rows, initial=sum(added.values()), desc='ScaleEdit new verified pairs', unit='pair', mininterval=10) as bar:
            while sum(added.values()) < args.target_rows:
                available = Counter()
                for rel, values in counts.items():
                    if rel not in done:
                        available.update(values)
                allocation = balanced_allocation({key: available[key] + added[key] for key in LOCAL_TASKS}, args.target_rows)
                need = {key: max(0, allocation[key] - added[key]) for key in LOCAL_TASKS}
                candidates = [rel for rel in groups if rel not in done and sum(min(n, need.get(key, 0)) for key, n in counts[rel].items())]
                if not candidates:
                    break
                wave = []
                projected = dict(need)
                while candidates and len(wave) < args.workers:
                    rel = max(candidates, key=lambda p: sum(min(n, projected.get(key, 0)) for key, n in counts[p].items()) / sizes[p])
                    useful = sum(min(n, projected.get(key, 0)) for key, n in counts[rel].items())
                    if not useful:
                        break
                    if downloaded_bytes + sizes[rel] > args.max_download_gb * 1e9:
                        raise RuntimeError('Download safety budget reached; preserve cache and resume with a larger explicit budget')
                    downloaded_bytes += sizes[rel]
                    wave.append(rel)
                    candidates.remove(rel)
                    for key, n in counts[rel].items():
                        projected[key] = max(0, projected.get(key, 0) - n)
                print('DOWNLOAD_WAVE', json.dumps({'shards':wave,'need':need,'planned_gb':downloaded_bytes/1e9}), flush=True)
                reserved = reserve_wave_limits(wave, counts, need, args.target_rows - sum(added.values()))
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(process_source_shard, rel, groups[rel], table, reserved[rel], pairs,
                                           args, state['source_revision'], sizes[rel], sink, bar): rel for rel in wave}
                    for future in as_completed(futures):
                        rel = futures[future]
                        accepted, errors = future.result()
                        added.update(accepted)
                        for key, n in accepted.items():
                            need[key] -= n
                        done.add(rel)
                        state.update(processed=sorted(done), new_rows=sum(added.values()), new_counts=dict(added),
                                     downloaded_bytes=downloaded_bytes, updated_at=time.time())
                        state.setdefault('shard_results', {})[rel] = dict(accepted=dict(accepted), skipped=dict(errors))
                        atomic_json(state_path, state)
                        print('SHARD_COMPLETE', rel, json.dumps(state['shard_results'][rel]), flush=True)
        state.update(complete=sum(added.values()) >= args.target_rows, final_total_rows=state['baseline_rows'] + sum(added.values()))
        atomic_json(state_path, state)
        if not state['complete']:
            raise RuntimeError(f"Insufficient resolvable pairs: {sum(added.values())}/{args.target_rows}")
        # Only this run's owned staging is removed; the final shards and audit are retained.
        cleanup_owned_cache(args)
        print('COMPLETE', json.dumps(state, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--target-rows', type=int, default=200000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-workers', type=int, default=8)
    parser.add_argument('--index-workers', type=int, default=8)
    parser.add_argument('--rows-per-file', type=int, default=256)
    parser.add_argument('--max-download-gb', type=float, default=600)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--fetch-source-urls', action='store_true', help='Fetch only URLs with a matching published SHA256; default uses complete HF-embedded pairs')
    parser.add_argument('--cleanup-cache', action='store_true')
    parser.add_argument('--finalize-within-percent', type=float,
                        help='Explicitly finalize existing data within at most 1 percent of target; does not download more')
    args = parser.parse_args()
    if (min(args.target_rows, args.workers, args.image_workers, args.index_workers, args.rows_per_file) <= 0
            or not math.isfinite(args.max_download_gb) or args.max_download_gb <= 0
            or not args.run_id.replace('_', '').isalnum()):
        parser.error('Positive limits/workers and an alphanumeric run ID are required')
    run(args)


if __name__ == '__main__':
    main()
