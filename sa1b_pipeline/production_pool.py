"""Per-rank source pool of the evidence production (scripts/run_evidence_production_4node.py).

Layout of one rank's pool (``nodes/node<r>/pool``):
  sources/<sid>/                  records written by shard_feeder (image-hash claims shared by all ranks)
  shards/<shard>.json             feeder marker; categories/<shard>.json: prefiltered categories
  labeled/<shard>.json            the shard's categories have affordance labels in the rank cache
  menus/<shard>.jsonl, .json      per-image menus (evidence_selection.image_menu) and their marker

``prepare`` runs before every wave's selection. It adds shards in the rank's fixed order until the
unused menu images can serve the planned wave: feed (CPU, HDFS) -> label the new categories (27B on
the node's GPUs, text only, no reasoning; the GPUs are idle between waves) -> menus (CPU).
"""
from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from . import evidence_selection as es
from .production_state import atomic_json
from .shard_feeder import SG2, materialize_shard, shard_tables

# Per-image availability measured on 3,997 prepared sources (planning/availability_4000.json): twin group
# possible for the operation alone / multi group with >= 2 targets. Used only before menus exist.
PRIOR_TWIN = {'add': .081, 'remove': .371, 'replace': .368, 'attribute': .707}
PRIOR_MULTI = .894
IMAGES_PER_SHARD = 11_000


def table_path(root: Path, shard: int) -> Path:
    return Path(root) / 'inputs/shards' / f'{shard:05d}.json'


def write_tables(root: Path, shards) -> list:
    """Record tables of the given shards from the SG2 indexes (one pass over both pickles, ~1 min)."""
    from .production_sources import DENSE_INDEX, IMAGE_INDEX
    missing = [s for s in shards if not table_path(root, s).exists()]
    if missing:
        with open(SG2 / IMAGE_INDEX, 'rb') as f:
            image_index = pickle.load(f)
        with open(SG2 / DENSE_INDEX, 'rb') as f:
            dense_index = pickle.load(f)
        for shard, rows in shard_tables(image_index, dense_index, missing).items():
            if not rows:
                raise ValueError(f'shard {shard} has no records')
            atomic_json(table_path(root, shard), rows)
    return missing


def load_exclusion(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    return {'source_ids': set(value['source_ids']), 'image_hashes': set(value['image_hashes']),
            'benchmark_sha256': value['benchmark_sha256']}


class Pool:
    def __init__(self, root: Path, rank: int, config: dict):
        self.root, self.rank, self.config = Path(root), rank, config
        self.dir = self.root / 'nodes' / f'node{rank}' / 'pool'
        self.cache = self.root / 'nodes' / f'node{rank}' / 'affordance_cache.json'
        self.order = json.loads((self.root / 'inputs/shard_order.json').read_text())[str(rank)]
        sel = config['selection']
        self.settings = es.selection_settings(
            twin_members=sel['twin_members'], twin_members_add=sel['twin_members_add'],
            worn_check=sel['worn_check'], covering_check=sel['covering_check'],
            no_surface_recolor=sel['no_surface_recolor'], skip_grayscale=sel['skip_grayscale'])
        self.fingerprint = es.settings_fingerprint(self.settings)

    def done(self, kind: str) -> list:
        """Shards (rank order) with a marker in <pool>/<kind>/; one directory listing, not a stat per shard
        (a missing path costs ~14 ms on the shared mount)."""
        folder = self.dir / kind
        have = {int(p.stem) for p in folder.glob('*.json')} if folder.exists() else set()
        return [s for s in self.order if s in have]

    # -- feeding -------------------------------------------------------------------------------------------
    def feed(self, shards: list) -> list:
        pool = self.config['pool']
        if not (SG2 / 'sa1b_tfd_1k').is_dir():
            return self.wait_fed(shards)
        write_tables(self.root, shards)
        exclusion = load_exclusion(self.root / 'inputs/exclusion.json')

        def one(shard):
            rows = json.loads(table_path(self.root, shard).read_text())
            info = materialize_shard(shard, rows, self.dir, exclusion, streams=pool['feeder_streams'],
                                     workers=pool['feeder_workers'], claims_root=self.root / 'pool_claims')
            print(json.dumps({'stage': 'pool_feed', 'rank': self.rank, **info}), flush=True)
            return info
        with ThreadPoolExecutor(max(1, pool['feeder_parallel_shards'])) as threads:
            return list(threads.map(one, shards))

    def wait_fed(self, shards: list, timeout_hours: float = 48.0) -> list:
        """Without the HDFS mount on this worker: wait for ``prefeed`` (run where the mount exists) to feed them."""
        started, said = time.time(), 0.0
        while True:
            missing = [s for s in shards if not (self.dir / 'shards' / f'{s:05d}.json').exists()]
            if not missing:
                return [json.loads((self.dir / 'shards' / f'{s:05d}.json').read_text()) for s in shards]
            if time.time() - started > timeout_hours * 3600:
                raise TimeoutError(f'shards {missing} were not fed within {timeout_hours} h and this worker has no '
                                   f'SA-1B HDFS mount ({SG2}); run run_evidence_production_4node.py prefeed')
            if time.time() - said > 600:
                print(json.dumps({'stage': 'pool_wait_prefeed', 'rank': self.rank, 'shards': missing,
                                  'waited_seconds': round(time.time() - started)}), flush=True)
                said = time.time()
            time.sleep(60)

    # -- labelling -----------------------------------------------------------------------------------------
    def label(self, shards: list, *, mllm_python: str, planner_model: str, devices: list) -> dict:
        """Affordance labels for the prefiltered categories of ``shards`` (rank cache, seeded from the base
        cache; labels other ranks already wrote are copied instead of judged again)."""
        if not self.cache.exists():
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_text((self.root / 'inputs/affordance_cache_base.json').read_text())
        wanted = {}
        for shard in shards:
            wanted.update(json.loads((self.dir / 'categories' / f'{shard:05d}.json').read_text()))
        cache = es.load_affordance_cache(self.cache)
        pending = {c: d for c, d in wanted.items() if c not in cache}
        copied = 0
        for other in sorted((self.root / 'nodes').glob('node*/affordance_cache.json')):
            if other != self.cache and pending:
                try:
                    labels = json.loads(other.read_text())
                except ValueError:
                    continue  # being replaced by its rank right now; the model judges these instead
                for c in [c for c in pending if c in labels]:
                    cache[c] = labels[c]
                    pending.pop(c)
                    copied += 1
        if copied:
            atomic_json(self.cache, cache)
        started, judged = time.time(), 0
        if pending:
            from .inference_services import RemoteSelector, ServicePool
            service = ServicePool(python=mllm_python, role='planner', devices=devices, model=planner_model,
                                  directory=Path(self.config['pool']['service_dir']) / f'label_rank{self.rank}',
                                  batch_size=self.config['pool']['label_batch'],
                                  planner_options={'max_new_tokens': 64}, dispatch='least_loaded')
            try:
                selector = RemoteSelector(service)
                items = sorted(pending.items())
                chunk = self.config['pool']['label_chunk']
                for i in range(0, len(items), chunk):
                    es.update_affordance_cache(self.cache, dict(items[i:i + chunk]), selector)
                    judged = min(len(items), i + chunk)
                    print(json.dumps({'stage': 'pool_label', 'rank': self.rank, 'judged': judged,
                                      'pending': len(items), 'seconds': round(time.time() - started, 1)}), flush=True)
            finally:
                service.close()
        cache = es.load_affordance_cache(self.cache)
        info = {'categories': len(wanted), 'copied_from_other_ranks': copied, 'judged': judged,
                'unlabelled_after': sum(c not in cache for c in wanted), 'seconds': round(time.time() - started, 1)}
        for shard in shards:
            atomic_json(self.dir / 'labeled' / f'{shard:05d}.json', info)
        return info

    # -- menus ---------------------------------------------------------------------------------------------
    def menus(self, shard: int) -> dict:
        marker = self.dir / 'menus' / f'{shard:05d}.json'
        if marker.exists():
            info = json.loads(marker.read_text())
            if info['fingerprint'] != self.fingerprint:
                raise ValueError(f'menus of shard {shard} were computed with other selection settings')
            return info
        started = time.time()
        rows = json.loads(table_path(self.root, shard).read_text())
        records = [str(self.dir / 'sources' / r[0] / 'complete.json') for r in rows]
        with ProcessPoolExecutor(self.config['pool']['menu_workers'], initializer=_init_menu,
                                 initargs=(str(self.cache), self.settings)) as pool:
            out = list(pool.map(_menu, records, chunksize=8))
        path = marker.with_suffix('.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            for row in out:
                f.write(json.dumps(row) + '\n')
        counts = Counter(row.get('skip', 'menu') for row in out)
        twin = {op: sum(1 for r in out if r.get('twin', {}).get(op, 0) >= 2) for op in es.OPERATIONS}
        multi = sum(1 for r in out if len(r.get('multi', [])) >= 2)
        info = {'fingerprint': self.fingerprint, 'settings': self.settings, 'shard': shard, 'images': len(out),
                'counts': dict(counts), 'twin_available': twin, 'multi_available': multi,
                'seconds': round(time.time() - started, 1)}
        atomic_json(marker, info)
        print(json.dumps({'stage': 'pool_menus', 'rank': self.rank, **{k: v for k, v in info.items() if k != 'settings'}}),
              flush=True)
        return info

    # -- sufficiency ---------------------------------------------------------------------------------------
    def available(self, used: set) -> dict:
        """Unused menu images that can serve each twin operation / a multi group."""
        twin, multi, images = Counter(), 0, 0
        for path in sorted((self.dir / 'menus').glob('*.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get('skip') or row['sid'] in used:
                    continue
                images += 1
                for op in es.OPERATIONS:
                    twin[op] += row['twin'][op] >= 2
                multi += len(row['multi']) >= 2
        return {'images': images, 'twin': {op: twin[op] for op in es.OPERATIONS}, 'multi': multi}

    def shortage(self, plan: dict, used: set) -> tuple[int, dict]:
        """Shards still to add for this wave (0: enough) from the planned groups and per-image availability."""
        factor = self.config['pool']['availability_factor']
        have = self.available(used)
        menus = [json.loads(p.read_text()) for p in sorted((self.dir / 'menus').glob('*.json'))]
        imgs = sum(m['images'] for m in menus)
        rate_twin = {op: (sum(m['twin_available'][op] for m in menus) / imgs if imgs else PRIOR_TWIN[op])
                     for op in es.OPERATIONS}
        rate_multi = sum(m['multi_available'] for m in menus) / imgs if imgs else PRIOR_MULTI
        need_images = []
        twins, multi = plan['plan']['twins'], plan['plan']['multi']
        for op in es.OPERATIONS:
            want = math.ceil(factor * min(twins['groups'], math.ceil(twins['deficits'][op] / 2)))
            if want > have['twin'][op]:
                need_images.append((want - have['twin'][op]) / max(1e-3, rate_twin[op]))
        want_multi = math.ceil(factor * multi['groups'])
        if want_multi > have['multi']:
            need_images.append((want_multi - have['multi']) / max(1e-3, rate_multi))
        want_images = math.ceil(factor * (twins['groups'] + multi['groups']))
        if want_images > have['images']:
            need_images.append(want_images - have['images'])
        shards = math.ceil(max(need_images, default=0) / IMAGES_PER_SHARD)
        return shards, {'available': have, 'shards_needed': shards}

    def prepare(self, plan: dict, used: set, *, mllm_python: str, planner_model: str, devices: list) -> dict:
        """Bring shards in rank order to menus until the wave can be served (or the shard list ends).

        Each step takes the next shards without menus: feeds those not fed yet (shards fed in advance
        by ``prefeed``, or by an interrupted call, are not fed again), labels those not labelled, then
        computes their menus. Shards that are not needed yet stay untouched."""
        log = []
        limit = self.config['pool']['max_shards_per_wave']
        while True:
            missing, info = self.shortage(plan, used)
            log.append(info)
            menued = set(self.done('menus'))
            batch = [s for s in self.order if s not in menued][:min(missing, limit)]
            if not batch:
                break
            fed, labeled = set(self.done('shards')), set(self.done('labeled'))
            if [s for s in batch if s not in fed]:
                self.feed([s for s in batch if s not in fed])
            if [s for s in batch if s not in labeled]:
                log.append({'label': self.label([s for s in batch if s not in labeled], mllm_python=mllm_python,
                                                planner_model=planner_model, devices=devices)})
            for shard in batch:
                self.menus(shard)
        return {'rank': self.rank, 'shards': len(self.done('menus')), 'log': log}


_MENU = {}


def _init_menu(cache, settings):
    _MENU.update(affordance=es.load_affordance_cache(cache), settings=settings)


def _menu(record_path):
    path = Path(record_path)
    sid = path.parent.name
    if not path.exists():
        return {'sid': sid, 'record': record_path, 'skip': 'missing_record'}
    record = json.loads(path.read_text())
    if record.get('status') != 'ready':
        return {'sid': sid, 'record': record_path, 'skip': 'not_ready'}
    return {'sid': sid, 'record': record_path,
            **es.image_menu(record['candidate'], _MENU['affordance'], _MENU['settings'])}


def used_sources(waves_root: Path) -> set:
    """Source IDs already selected by earlier waves of this rank."""
    used = set()
    for path in Path(waves_root).glob('wave*/inputs/evidence_selection_summary.json'):
        for ids in json.loads(path.read_text()).get('source_ids', {}).values():
            used.update(ids)
    return used


def shard_order(seed: int, world_size: int, exclude=()) -> dict:
    """Disjoint per-rank shard lists over all 1,000 SA-1B shard pairs, in a seeded random order."""
    import random
    shards = [s for s in range(1000) if s not in set(exclude)]
    random.Random(seed).shuffle(shards)
    return {str(r): shards[r::world_size] for r in range(world_size)}
