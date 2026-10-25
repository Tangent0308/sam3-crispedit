#!/usr/bin/env python3
"""Write metadata-first twin/multi inputs for run_dual_validation (opt-in).

Reads per-image candidates (``<sources-root>/<sid>/complete.json``), excludes
reserved/used source IDs, judges category affordances once per category with a
text-only, no-reasoning model call (cached), and writes
``<inputs-root>/{twins,multi}/inputs/{selected,manifest}.jsonl``.

Twins over-generate a third candidate by default (``twin_keep=2``); run the
planning entry with ``--evidence-gates`` so two passing members are kept.

Production pools (``--menu-dir``): per-image menus (``evidence_selection.menu``, computed once per
source after its categories are labelled) say which operations an image can serve. Images that
cannot serve a needed operation are skipped without decoding, and the rows of the chosen images are
built in ``--workers`` processes. Decisions and results do not depend on timing (results are
applied in submission order, a fixed number of builds ahead).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline import evidence_selection as es  # noqa: E402
from sa1b_pipeline.production_state import atomic_json  # noqa: E402
from sa1b_pipeline.source_isolation import load_reservation  # noqa: E402


class OpenAITextSelector:
    """Minimal text-only adapter for a local OpenAI-compatible vLLM server."""
    def __init__(self, url: str, model: str, workers: int = 32):
        self.url, self.model, self.workers = url.rstrip('/'), model, workers

    def _one(self, prompt, schema, limit):
        body = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': limit,
                'temperature': 0.0, 'chat_template_kwargs': {'enable_thinking': False},
                'structured_outputs': {'json': schema}}
        request = urllib.request.Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            text = json.loads(response.read())['choices'][0]['message']['content'] or ''
        try:
            return json.loads(text)
        except ValueError:
            return {}

    def generate_multimage(self, jobs, *, schemas, token_limits, thinking=None):
        if any(images for images, _ in jobs):
            raise ValueError('text-only adapter')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda args: self._one(*args),
                                 [(prompt, schema, limit) for (_, prompt), schema, limit in zip(jobs, schemas, token_limits)]))


def _ids(value, out: set):
    if isinstance(value, dict):
        if 'source_id' in value:
            out.add(str(value['source_id']))
        for v in value.values():
            _ids(v, out)
    elif isinstance(value, list):
        for v in value:
            _ids(v, out) if not isinstance(v, str) else (out.add(v) if v.startswith('sa_') else None)


_BUILD = {}


def _init_build(affordance_path, settings):
    _BUILD.update(affordance=es.load_affordance_cache(affordance_path), settings=settings)


def _build(task):
    """Row of one image for one kind, built exactly as the sequential path builds it."""
    record_path, kind, snapshot = task
    candidate = json.loads(Path(record_path).read_text())['candidate']
    st = _BUILD['settings']
    options = dict(support_check=st['support_check'], saturated_ok=False, limits=st['limits'])
    if kind == 'twins':
        return es.select_twin(candidate, _BUILD['affordance'], Counter(snapshot), members=st['members'],
                              members_by_operation=st['members_by_operation'], **options)
    return es.select_multi(candidate, _BUILD['affordance'], Counter(snapshot), **options)


def load_menus(menu_dir: Path, fingerprint: str) -> dict:
    """sid -> menu row from <menu-dir>/<shard>.jsonl, whose marker <shard>.json must match the settings."""
    menus = {}
    for marker in sorted(menu_dir.glob('*.json')):
        info = json.loads(marker.read_text())
        if info.get('fingerprint') != fingerprint:
            raise ValueError(f'menus were computed with other selection settings: {marker}')
        for line in marker.with_suffix('.jsonl').read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                # Records sit next to the menus (<pool>/sources/<sid>), so a pool can be moved.
                row['record'] = str(menu_dir.parent / 'sources' / row['sid'] / 'complete.json')
                menus[row['sid']] = row
    return menus


def predict_multi(menu: dict, deficit: Counter) -> list:
    """Operations select_multi would use with these deficits, from the menu's multi operations
    (a prediction; the built row decides). Empty: no needed operation, or fewer than two targets."""
    ranked = sorted(es.OPERATIONS, key=lambda op: (-deficit[op], op))
    available = [op for op in ranked if op in menu['multi']]
    chosen = [op for op in available if deficit[op] > 0][:4]
    if not chosen:
        return []  # a group of saturated operations only adds nothing to the balance
    for op in available:
        if len(chosen) >= 2:
            break
        if op not in chosen:
            chosen.append(op)
    return chosen if len(chosen) >= 2 else []


def select_from_menus(args, order, menus, deficits, settings, stats):
    """Production selection over menus (targeted deficits only).

    Twins first: each operation gets its share of the wave (``twin_allocation``) and takes, scarcest
    operation first, the next images in seeded order whose menu can serve it (twin menus are exact).
    Multi then uses the remaining images, most-needed operations first. Rows are built in parallel by
    select_twin / select_multi and applied in submission order, so results do not depend on timing.
    """
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    rows = {'twins': [], 'multi': []}
    usable = [sid for sid in order if not menus[sid].get('skip')]
    for sid in order:
        if menus[sid].get('skip'):
            stats[menus[sid]['skip']] += 1
    used = set()
    allocation = es.twin_allocation(args.twins, deficits['twins'])
    available = {op: sum(1 for sid in usable if menus[sid]['twin'][op] >= 2) for op in es.OPERATIONS}
    twin_jobs = []
    for op in sorted(es.OPERATIONS, key=lambda op: (available[op] / max(1, allocation[op]), op)):
        taken = 0
        for sid in usable:
            if taken >= allocation[op]:
                break
            if sid not in used and menus[sid]['twin'][op] >= 2:
                used.add(sid)
                twin_jobs.append((sid, op))
                taken += 1
        if taken < allocation[op]:
            stats[f'twins_{op}_short'] += allocation[op] - taken
    window = 4 * args.workers
    with ProcessPoolExecutor(args.workers, initializer=_init_build,
                             initargs=(str(args.affordance_cache), settings)) as pool:
        futures = [(pool.submit(_build, (menus[sid]['record'], 'twins', {op: 1})), op) for sid, op in twin_jobs]
        for future, op in futures:
            row = future.result()
            if row and row['operation'] == op:
                rows['twins'].append(row)
                deficits['twins'][op] -= 2
                stats['twins_selected'] += 1
            else:
                stats['twins_build_rejected'] += 1
        inflight, pending = deque(), 0

        def apply(item):
            nonlocal pending
            future, predicted, snapshot = item
            row = future.result()
            pending -= 1
            for op in predicted:
                deficits['multi'][op] += 1
            ops = list(row['operations']) if row else []
            if row and any(snapshot.get(op, 0) > 0 for op in ops) and len(rows['multi']) < args.multi:
                rows['multi'].append(row)
                for op in ops:
                    deficits['multi'][op] -= 1
                stats['multi_selected'] += 1
                if ops != predicted:
                    stats['multi_operations_differ_from_menu'] += 1
            else:
                stats['multi_build_rejected'] += 1

        cursor = 0
        rest = [sid for sid in usable if sid not in used]
        while True:
            more = (cursor < len(rest) and len(rows['multi']) + pending < args.multi
                    and max(deficits['multi'].values()) > 0)
            if inflight and (len(inflight) >= window or not more):
                apply(inflight.popleft())
                continue
            if not more:
                break
            sid = rest[cursor]
            cursor += 1
            predicted = predict_multi(menus[sid], deficits['multi'])
            if not predicted:
                stats['multi_no_group'] += 1
                continue
            snapshot = dict(deficits['multi'])
            for op in predicted:
                deficits['multi'][op] -= 1
            pending += 1
            inflight.append((pool.submit(_build, (menus[sid]['record'], 'multi', snapshot)), predicted, snapshot))
    return rows, len(usable), {'images': len(menus), 'usable': len(usable), 'workers': args.workers,
                               'twin_allocation': allocation, 'twin_available': available}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources-root', type=Path, required=True)
    p.add_argument('--inputs-root', type=Path, required=True)
    p.add_argument('--reservation', type=Path, required=True)
    p.add_argument('--exclude', type=Path, nargs='*', default=[],
                   help='JSON files whose source IDs (any nesting, or plain ID lists) must not be used')
    p.add_argument('--twins', type=int, default=16)
    p.add_argument('--multi', type=int, default=12)
    p.add_argument('--twin-members', type=int, choices=[2, 3, 4], default=3)
    p.add_argument('--twin-members-add', type=int, choices=[2, 3, 4],
                   help='Over-select add twin candidates (low yield); evidence gates keep the best two')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--affordance-cache', type=Path, required=True)
    p.add_argument('--affordance-server', help='OpenAI-compatible URL of a no-reasoning text model')
    p.add_argument('--affordance-model', default='q27')
    p.add_argument('--planner-model', help='Local model path for VLLMSelector when no server is given')
    p.add_argument('--initial-deficits', type=json.loads,
                   help='JSON {"twins": {op: n}, "multi": {op: n}}: target units still to SELECT per operation '
                        '(remaining accepted need / expected yield). Saturated operations are then not chosen.')
    p.add_argument('--cache-only', action='store_true',
                   help='Use only the affordance cache; categories missing from it are ineligible (no model call)')
    p.add_argument('--prefetch-images', type=int, default=0,
                   help='Only fill the affordance cache for the next N candidate images, then exit')
    p.add_argument('--worn-check', action='store_true',
                   help='Opt-in: do not remove worn body-wear (shoe, helmet, glove, main garments) from a person')
    p.add_argument('--covering-check', action='store_true',
                   help='Opt-in: do not remove a covering layer (snow, moss, tarp, ...) lying on an object')
    p.add_argument('--no-surface-recolor', action='store_true',
                   help='Opt-in: no colour/attribute edits of sky, water or ground surfaces')
    p.add_argument('--skip-grayscale', action='store_true',
                   help='Opt-in: skip black-and-white source photos (new content comes out coloured; ~0.5%% of sources)')
    p.add_argument('--no-support-check', action='store_true',
                   help='Ablation: do not exclude remove/replace targets that support segmented neighbours')
    p.add_argument('--menu-dir', type=Path,
                   help='Production: per-image menus (<shard>.jsonl + marker); candidates are the menu images, '
                        'needs --initial-deficits and --cache-only')
    p.add_argument('--workers', type=int, default=16, help='with --menu-dir: processes building rows')
    args = p.parse_args(argv)
    if args.menu_dir and (args.initial_deficits is None or not args.cache_only):
        p.error('--menu-dir needs --initial-deficits and --cache-only')
    reserved, _ = load_reservation(args.reservation)
    excluded = set(reserved)
    for path in args.exclude:
        text = path.read_text()
        _ids([json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl'
             else json.loads(text), excluded)
    if args.cache_only:
        selector = None
    elif args.affordance_server:
        selector = OpenAITextSelector(args.affordance_server, args.affordance_model)
    else:
        if not args.planner_model:
            p.error('--affordance-server, --planner-model or --cache-only is required')
        import os
        os.environ['SA1B_FORCE_DIRECT_VLLM'] = '1'
        from sa1b_pipeline.mllm import VLLMSelector
        selector = VLLMSelector(args.planner_model, batch_size=64, max_new_tokens=80, enable_thinking=False)
    import os
    settings = es.selection_settings(twin_members=args.twin_members, twin_members_add=args.twin_members_add,
                                     worn_check=args.worn_check, covering_check=args.covering_check,
                                     no_surface_recolor=args.no_surface_recolor,
                                     support_check=not args.no_support_check, skip_grayscale=args.skip_grayscale)
    menus = load_menus(args.menu_dir, es.settings_fingerprint(settings)) if args.menu_dir else None
    # Shuffle the full pool first, then filter: later waves (more exclusions)
    # keep the same relative order, so a prefetched affordance cache stays aligned.
    ids = sorted(menus if menus is not None else
                 (name for name in os.listdir(args.sources_root) if name.startswith(('sa_', 'sa1b-'))))
    random.Random(args.seed).shuffle(ids)
    ids = [name for name in ids if name not in excluded and name.removeprefix('sa1b-') not in excluded]
    from sa1b_pipeline.archive_relocation import ArchivePaths
    archive = ArchivePaths.load()
    rows = {'twins': [], 'multi': []}
    manifests = {'twins': [], 'multi': []}
    targeted = args.initial_deficits is not None
    deficits = {kind: Counter({op: float((args.initial_deficits or {}).get(kind, {}).get(op, 0)) for op in es.OPERATIONS})
                for kind in ('twins', 'multi')}
    stats = Counter()
    started = time.monotonic()
    cursor = 0
    if args.prefetch_images:
        if selector is None:
            p.error('--prefetch-images needs a model (server or --planner-model)')
        n = 0
        while n < args.prefetch_images and cursor < len(ids):
            chunk = ids[cursor:cursor + 64]
            cursor += len(chunk)
            categories = {}
            for sid in chunk:
                path = args.sources_root / sid / 'complete.json'
                if path.exists():
                    for obj in json.loads(path.read_text()).get('candidate', {}).get('objects', []):
                        categories.setdefault(str(obj.get('category')), obj.get('description'))
                    n += 1
            cache = es.update_affordance_cache(args.affordance_cache, categories, selector)
            print(json.dumps({'prefetched_images': n, 'cache_categories': len(cache)}), flush=True)
        return
    def wanted(kind):
        limit = args.twins if kind == 'twins' else args.multi
        return len(rows[kind]) < limit and not (targeted and max(deficits[kind].values()) <= 0)

    if menus is not None:
        rows, cursor, menu_info = select_from_menus(args, ids, menus, deficits, settings, stats)
        manifests = {kind: [row['candidate'] for row in rows[kind]] for kind in rows}
    while menus is None and (wanted('twins') or wanted('multi')) and cursor < len(ids):
        if stats['not_ready'] > 64 and stats['not_ready'] > 4 * (stats['twins_selected'] + stats['multi_selected'] + 1) * 16:
            raise RuntimeError(f'source records are not readable (paths?): {dict(stats)}')
        chunk = ids[cursor:cursor + 16]
        cursor += len(chunk)
        candidates = []
        for sid in chunk:
            path = args.sources_root / sid / 'complete.json'
            if not path.exists():
                stats['missing_record'] += 1
                continue
            record = json.loads(path.read_text())
            # Cached records keep pre-migration absolute paths; map them like every
            # other reader of archived inputs (never rewrite the cached record).
            candidate = archive.record({'candidate': record.get('candidate') or {}})['candidate']
            if record.get('status', 'ready') != 'ready' or not Path(candidate.get('image_path', '')).exists():
                stats['not_ready'] += 1
                continue
            if args.skip_grayscale and es.is_grayscale(candidate['image_path']):
                stats['grayscale_skipped'] += 1
                continue
            candidates.append(candidate)
        categories = {}
        for candidate in candidates:
            for obj in candidate['objects']:
                categories.setdefault(str(obj.get('category')), obj.get('description'))
        t_afford = time.monotonic()
        affordance = (es.load_affordance_cache(args.affordance_cache) if selector is None
                      else es.update_affordance_cache(args.affordance_cache, categories, selector))
        t_select = time.monotonic()
        for candidate in candidates:
            # Alternate kinds so both draw from the same image distribution.
            for kind in (('twins', 'multi') if len(rows['twins']) * max(1, args.multi) <= len(rows['multi']) * max(1, args.twins)
                         else ('multi', 'twins')):
                if len(rows[kind]) >= (args.twins if kind == 'twins' else args.multi):
                    continue
                if targeted and max(deficits[kind].values()) <= 0:
                    continue  # every operation already has enough selected units
                extra_rules = {k: True for k, v in (('worn_check', args.worn_check), ('covering_check', args.covering_check),
                                                    ('surface_recolor_check', args.no_surface_recolor)) if v}
                options = dict(support_check=not args.no_support_check, saturated_ok=not targeted,
                               limits=dict(es.DEFAULTS, **extra_rules) if extra_rules else es.DEFAULTS)
                extra = {'members_by_operation': {'add': args.twin_members_add}} if args.twin_members_add else {}
                row = (es.select_twin(candidate, affordance, deficits[kind], members=args.twin_members, **options, **extra)
                       if kind == 'twins' else es.select_multi(candidate, affordance, deficits[kind], **options))
                stats[f'{kind}_{"selected" if row else "no_group"}'] += 1
                if row and not targeted:
                    for op in es.OPERATIONS:  # untargeted: replenish quotas evenly per selected group
                        deficits[kind][op] += (2 if kind == 'twins' else len(row['operations'])) / 4
                if row:
                    rows[kind].append(row)
                    manifests[kind].append(candidate)
                    break
        print(json.dumps({'scanned': cursor, 'twins': len(rows['twins']), 'multi': len(rows['multi']),
                          'categories': len(categories), 'affordance_seconds': round(t_select - t_afford, 1),
                          'selection_seconds': round(time.monotonic() - t_select, 1)}), flush=True)
    for kind in ('twins', 'multi'):
        out = args.inputs_root / kind / 'inputs'
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'selected.jsonl').open('w') as f:
            for row in rows[kind]:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (out / 'manifest.jsonl').open('w') as f:
            for candidate in manifests[kind]:
                f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
    summary = {'version': es.VERSION, 'seed': args.seed, 'scanned_images': cursor, 'excluded_ids': len(excluded),
               'stats': stats, 'seconds': time.monotonic() - started,
               **({'menus': menu_info} if menus is not None else {}),
               'initial_deficits': args.initial_deficits, 'remaining_deficits': {k: dict(v) for k, v in deficits.items()},
               'support_check': not args.no_support_check,
               'twins': {'groups': len(rows['twins']), 'operations': Counter(r['operation'] for r in rows['twins']),
                         'members': Counter(len(r['objects']) for r in rows['twins'])},
               'multi': {'groups': len(rows['multi']), 'operations': Counter(op for r in rows['multi'] for op in r['operations']),
                         'targets': Counter(len(r['objects']) for r in rows['multi'])},
               'source_ids': {k: [r['candidate']['source_id'] for r in v] for k, v in rows.items()}}
    atomic_json(args.inputs_root / 'evidence_selection_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_ids'}, default=dict, indent=1))


if __name__ == '__main__':
    main()
