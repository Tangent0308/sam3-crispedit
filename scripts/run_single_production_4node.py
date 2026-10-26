#!/usr/bin/env python3
"""Single-target production (single-target v2.1) on Arnold workers (4 x 8 GPUs), one experiment directory.

Each rank runs the single-target wave coordinator (scripts/run_single_waves.py) on its own SA-1B shards: before
every wave, ``prepare-pool`` adds shards until the wave can be served (shard_feeder -> affordance labels on the
idle GPUs -> per-image menus, as in the twins/multi production); every wave selects by the remaining need per
operation, so the final export has the same number of singles for each of the four operations.

  prepare    once: config.json (quota = singles per operation and rank), inputs/ (exclusions, affordance
             cache, a shard order disjoint from the shards the twins/multi production can reach)
  bundle     once after prepare: code bundle (working tree) + checksums, model inventory, entry scripts;
             bundle --hotfix REASON replaces the code of a started run (config and inputs stay frozen)
  warm-pool  optional, before submitting, on a machine with the SA-1B HDFS mount and 8 idle GPUs: feed,
             label and build the menus of the first shards of every rank, so the workers start selecting at once
  node       on every worker (deployment/arnold_entry.sh): stage models, run the rank's waves
  status     progress of all ranks (rank 0 also writes progress.json / logs/progress.log)
  merge      final/: the ranks' balanced exports; an operation one rank left short is topped up from the
             others' surplus (--partial before all ranks are done)
  prepare-pool  called by the wave coordinator before each selection
  recover-lock  release a dead worker's lock (only after that worker has stopped)

Layout: nodes/node<r>/{pool, affordance_cache.json, waves/, done.json}; pool_claims/ (image-hash claims shared by
all ranks); final/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from sa1b_pipeline.production_state import atomic_json, file_hash, immutable_json  # noqa: E402

OPERATIONS = ('add', 'remove', 'replace', 'attribute')
DATA = Path('/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline')
MODELS = Path('/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models')
EVIDENCE_RUN = DATA / '02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025'
VALIDATION = DATA / '04_current_validation/sa1b_single_target_v2_20261026'
PROTECTED = [Path('/mnt/bn/strategy-mllm-train/user/tanyue/datasets'),
             DATA / '02_production_runs/sa1b_samtok_edit_50k_20261014', EVIDENCE_RUN]
DEFAULT_INPUTS = {
    'reservation': DATA / '02_production_runs/sa1b_samtok_group_train_20k_20261016/inputs/eval_reservation.json',
    'benchmark_exclusion': DATA / '02_production_runs/sa1b_samtok_edit_50k_20261014/inputs/exclusion.json',
    # benchmark + eval reservation, the 50k release sources and the earlier validation pool
    'extra_exclusion': [VALIDATION / 'planning/exclusion.json'],
    'affordance_caches': [EVIDENCE_RUN / 'inputs/affordance_cache_base.json',
                          *[EVIDENCE_RUN / f'nodes/node{r}/affordance_cache.json' for r in range(4)],
                          VALIDATION / 'nodes/node0/affordance_cache.json']}
SELECTION = {'twin_members': 3, 'twin_members_add': 4, 'worn_check': True, 'covering_check': True,
             'no_surface_recolor': True, 'skip_grayscale': True}
SELECTION_FLAGS = ['--twin-members-add', '4', '--worn-check', '--covering-check', '--no-surface-recolor',
                   '--skip-grayscale']   # select_evidence_groups.py flags with the same settings (menu fingerprint)
CHILDREN = []


def config(root: Path) -> dict:
    return json.loads((root / 'config.json').read_text())


def merged_exclusion(benchmark: Path, reservation: dict, extra: list) -> dict:
    bench = json.loads(Path(benchmark).read_text())
    ids = set(bench['source_ids']) | set(reservation['reserved_ids'])
    hashes = set(bench['image_hashes']) | set(reservation['reserved_image_sha256'])
    for path in extra:
        value = json.loads(Path(path).read_text())
        ids |= set(value.get('source_ids', []))
        hashes |= set(value.get('image_hashes', []))
    return {'source_ids': sorted(ids), 'image_hashes': sorted(hashes),
            'benchmark_sha256': 'benchmark-200 + group-eval-reservation-1200 + 50k release sources + validation pool',
            'sources': {'benchmark': str(benchmark), 'extra': [str(p) for p in extra]}}


def merged_cache(paths: list) -> dict:
    cache = {}
    for path in paths:
        try:
            cache.update(json.loads(Path(path).read_text()))
        except (OSError, ValueError):
            pass   # missing, or being rewritten by its rank right now
    return cache


def avoided_shards(order_path: Path, first: int) -> set:
    """The first shards of every rank of another production (shards it can still reach)."""
    order = json.loads(Path(order_path).read_text())
    return {s for shards in order.values() for s in shards[:first]}


def prepare(args):
    from sa1b_pipeline.production_pool import shard_order, write_tables
    root = args.root
    if (root / 'nodes').exists() and args.replace_unstarted:
        raise ValueError('ranks have started in this root; quotas and code stay frozen (use a new root)')
    if args.replace_unstarted:
        kept = root / 'planning' / f'replaced_{time.strftime("%Y%m%dT%H%M%S")}'
        kept.mkdir(parents=True)
        for name in ('config.json', 'deployment'):
            if (root / name).exists():
                (root / name).rename(kept / name)
    (root / 'inputs').mkdir(parents=True, exist_ok=True)
    reservation = json.loads(Path(args.reservation).read_text())
    for name, make in (('eval_reservation.json', lambda: reservation),
                       ('exclusion.json', lambda: merged_exclusion(args.benchmark_exclusion, reservation, args.extra_exclusion)),
                       ('affordance_cache_base.json', lambda: merged_cache(args.affordance_caches))):
        if not (root / 'inputs' / name).exists():
            atomic_json(root / 'inputs' / name, make())
    if not (root / 'inputs/shard_order.json').exists():
        avoid = set(args.avoid_shards) | avoided_shards(args.avoid_order, args.avoid_first)
        atomic_json(root / 'inputs/shard_order.json', shard_order(args.seed, args.world_size, exclude=avoid))
        atomic_json(root / 'inputs/avoided_shards.json', {'shards': sorted(avoid), 'order': str(args.avoid_order),
                                                          'first_per_rank': args.avoid_first})
    order = json.loads((root / 'inputs/shard_order.json').read_text())
    if sorted(order) != [str(r) for r in range(args.world_size)]:
        raise ValueError('inputs/shard_order.json was made for another world size; use a new root')
    write_tables(root, sorted({s for r in range(args.world_size) for s in order[str(r)][:args.tables_per_rank]}))
    per_op = math.ceil(args.samples / args.world_size / len(OPERATIONS))
    c = {
        'schema': 'single-production-1', 'experiment': root.name, 'world_size': args.world_size, 'gpus_per_node': 8,
        'seed': args.seed, 'protocol': '2.0', 'editor_steps': 40, 'pipeline': 'single-target v2.1',
        'quota': {'unit': 'singles', 'total': args.samples, 'per_rank_per_operation': per_op,
                  'exported_total_if_complete': per_op * args.world_size * len(OPERATIONS)},
        'models': {'planner': str(MODELS / 'Qwen3.8-27B'), 'editor': str(MODELS / 'Qwen-Image-2.1'),
                   'site': str(MODELS / 'sam3')},
        'selection': SELECTION, 'selection_flags': SELECTION_FLAGS,
        'waves': {'wave_targets': args.wave_targets, 'wave_cap_groups': args.wave_cap_groups, 'stop_gap': args.stop_gap,
                  'max_waves': args.max_waves, 'max_planning_chunks': args.max_planning_chunks,
                  'groups_in_flight': args.groups_in_flight},
        'pool': {'feeder_streams': 16, 'feeder_workers': 64, 'feeder_parallel_shards': 2, 'menu_workers': 64,
                 'selection_workers': 32, 'label_batch': 128, 'label_chunk': 8192, 'availability_factor': 1.5,
                 'max_shards_per_wave': 4, 'service_dir': '/tmp/sa1b-single-label-services'},
        'inputs': {name: file_hash(root / 'inputs' / name) for name in
                   ('eval_reservation.json', 'exclusion.json', 'affordance_cache_base.json', 'shard_order.json')},
        'model_cache_root': '/opt/tiger/sa1b-evidence-model-cache'}
    immutable_json(root / 'config.json', c)
    print(json.dumps({'root': str(root), 'quota': c['quota'], 'first_shards': {r: s[:3] for r, s in order.items()},
                      'excluded_sources': len(json.loads((root / 'inputs/exclusion.json').read_text())['source_ids'])},
                     indent=1))


def bundle(args):
    """Freeze the code (working tree: tracked and untracked, not ignored) with checksums, the model inventory
    and the entry scripts."""
    root, dest = args.root, args.root / 'deployment'
    dest.mkdir(exist_ok=True)
    hotfixes = []
    if (dest / 'code.tar.gz').exists():
        if not args.hotfix:
            raise ValueError('deployment already frozen; use a new experiment root for code changes, or '
                             'bundle --hotfix REASON for a fix to a started run (config and inputs stay frozen)')
        # The previous code stays beside; ranks pick the new bundle up when they restart, and the wave coordinator
        # sets aside a wave that was started by the old code (run_single_waves.supersede_stale_run).
        previous = json.loads((dest / 'manifest.json').read_text())
        hotfixes = previous.get('hotfixes', [])
        kept = root / 'planning' / f'hotfix_{time.strftime("%Y%m%dT%H%M%S")}'
        kept.mkdir(parents=True)
        for name in ('code.tar.gz', 'code.sha256', 'manifest.json'):
            (dest / name).rename(kept / name)
        hotfixes.append({'time': time.time(), 'reason': args.hotfix,
                         'previous_code_sha256': previous['code_sha256'], 'previous_bundle': str(kept)})
    names = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '--', 'sa1b_pipeline', 'scripts',
                                     'utils', 'requirements', 'pyproject.toml', 'requirements.txt'],
                                    cwd=REPO, text=True).split()
    names = sorted(n for n in names if (REPO / n).is_file() and '__pycache__' not in n)
    for needed in ('sa1b_pipeline/instruction_protocol.py', 'sa1b_pipeline/assets/fonts/sha256.json',
                   'sa1b_pipeline/single_parallel.py', 'sa1b_pipeline/single_planning.py',
                   'scripts/run_single_waves.py', 'scripts/run_single_validation.py',
                   'requirements/group_production/mllm.lock.txt', 'scripts/single_production/launch_4node.sh',
                   'scripts/evidence_production/setup_extras.sh', 'scripts/group_production/setup_runtime.sh'):
        if needed not in names:
            raise ValueError(f'code bundle would miss {needed}')
    temp = dest / 'code.tar.gz.partial'
    import gzip
    with gzip.GzipFile(temp, 'wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for name in names:
            info = tar.gettarinfo(str(REPO / name), arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, '', ''
            with (REPO / name).open('rb') as f:
                tar.addfile(info, f)
    temp.replace(dest / 'code.tar.gz')
    (dest / 'code.sha256').write_text(file_hash(dest / 'code.tar.gz') + '\n')
    (dest / 'config.sha256').write_text(file_hash(root / 'config.json') + '\n')
    c = config(root)
    models = {}
    for path in map(Path, c['models'].values()):
        models[str(path)] = {str(p.relative_to(path)): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                             for p in sorted(path.rglob('*'))
                             if p.is_file() and not any(x.startswith('.') for x in p.relative_to(path).parts)}
    atomic_json(dest / 'models.json', models)
    shutil.copyfile(REPO / 'scripts/single_production/launch_4node.sh', dest / 'launch_4node.sh')
    import shlex
    (dest / 'arnold_entry.sh').write_text(
        '#!/usr/bin/env bash\n# Same command on every Arnold worker.\nset -euo pipefail\n'
        f'export SINGLE_RUN_ROOT={shlex.quote(str(root))}\n'
        'export SINGLE_RUNTIME_ROOT="${SINGLE_RUNTIME_ROOT:-/opt/tiger/sa1b-evidence-production-runtime}"\n'
        'exec bash "$SINGLE_RUN_ROOT/deployment/launch_4node.sh"\n')
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).count('\n')
    atomic_json(dest / 'manifest.json', {
        'code_sha256': file_hash(dest / 'code.tar.gz'), 'files': len(names), 'git_head': git,
        'uncommitted_paths_at_bundle': dirty, 'config_sha256': file_hash(root / 'config.json'),
        'model_inventory_sha256': file_hash(dest / 'models.json'),
        'instruction_protocol_sha256': file_hash(REPO / 'sa1b_pipeline/instruction_protocol.py'),
        **({'hotfixes': hotfixes} if hotfixes else {})})
    print(json.dumps({'root': str(root), 'files': len(names), 'code_sha256': file_hash(dest / 'code.tar.gz'),
                      'git_head': git, 'uncommitted_paths': dirty}))


def coordinator(root: Path, c: dict, rank: int, mllm: str, editor: str) -> list:
    node = root / 'nodes' / f'node{rank}'
    w = c['waves']
    hook = [mllm, str(Path(__file__).resolve()), '--root', str(root), 'prepare-pool', '--rank', str(rank)]
    return [mllm, str(REPO / 'scripts/run_single_waves.py'), '--root', str(node / 'waves'),
            '--sources-root', str(node / 'pool/sources'), '--reservation', str(root / 'inputs/eval_reservation.json'),
            '--exclude', str(root / 'inputs/exclusion.json'), '--affordance-cache', str(node / 'affordance_cache.json'),
            '--quota-per-op', str(c['quota']['per_rank_per_operation']), '--wave-targets', str(w['wave_targets']),
            '--wave-cap-groups', str(w['wave_cap_groups']), '--max-waves', str(w['max_waves']),
            '--stop-gap', str(w['stop_gap']), '--seed', str(c['seed'] + rank),
            '--menu-dir', str(node / 'pool/menus'), '--selection-workers', str(c['pool']['selection_workers']),
            '--selection-flags-json', json.dumps(c['selection_flags']),
            '--service-root', f'/tmp/sa1b-single-services/node{rank}',
            '--max-planning-chunks', str(w['max_planning_chunks']), '--groups-in-flight', str(w['groups_in_flight']),
            '--before-wave-json', json.dumps(hook), '--',
            '--planner-model', c['models']['planner'], '--editor-model', c['models']['editor'],
            '--site-model', c['models']['site'], '--mllm-python', mllm, '--editor-python', editor]


def node(args):
    from run_evidence_production_4node import verify_frozen
    from scripts.run_group_production_4node import owner_lock
    root, rank = args.root, args.rank
    c = config(root)
    world = int(os.environ.get('ARNOLD_WORKER_NUM', '0'))
    if world != c['world_size'] or int(os.environ.get('ARNOLD_WORKER_GPU', '0')) != 8 \
            or os.environ.get('ARNOLD_ID') != str(rank) or rank not in range(c['world_size']):
        raise ValueError(f"needs {c['world_size']} Arnold workers x 8 GPUs and ARNOLD_ID == --rank")
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('unset CUDA_VISIBLE_DEVICES; ranks use physical GPUs 0..7')
    verify_frozen(root)
    node_dir = root / 'nodes' / f'node{rank}'
    node_dir.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        for child in CHILDREN:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise RuntimeError(f'worker interrupted by signal {signum}; waves resume from their checkpoints')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    lock = root / 'control' / f'node{rank}.lock.d'
    trial = os.environ.get('ARNOLD_TRIAL_ID')
    owner = json.loads((lock / 'owner.json').read_text()) if (lock / 'owner.json').exists() else {}
    if trial and owner.get('trial') and owner['trial'] != trial:
        # Left by an earlier Arnold trial: its containers are gone, so its workers have stopped.
        lock.rename(lock.with_name(f'{lock.name}.recovered.{time.time_ns()}'))
    with owner_lock(lock):
        stop, monitor_thread = threading.Event(), None
        if rank == 0:
            monitor_thread = threading.Thread(target=monitor, args=(root, stop), daemon=True)
            monitor_thread.start()
        try:
            started = json.loads((node_dir / 'started.json').read_text()) if (node_dir / 'started.json').exists() \
                else {'first_start': time.time(), 'starts': []}
            started['starts'].append({'time': time.time(), 'host': os.uname().nodename, 'trial': trial})
            if (node_dir / 'failure.json').exists():  # a previous attempt; kept beside, no longer current
                (node_dir / 'failure.json').rename(node_dir / f'failure.{int(time.time())}.json')
            atomic_json(node_dir / 'started.json', started)
            if args.model_map:
                mapping = json.loads(Path(args.model_map).read_text())
            else:
                from sa1b_pipeline.model_cache import stage_model
                mapping = {m: stage_model(m, c['model_cache_root'], progress=lambda x: print(json.dumps(x), flush=True))
                           for m in c['models'].values()}
            atomic_json(node_dir / 'model_map.json', mapping)
            env = dict(os.environ, SA1B_LOCAL_MODEL_MAP=json.dumps(mapping), SA1B_FORCE_DIRECT_VLLM='1',
                       PYTHONUNBUFFERED='1')
            if not (node_dir / 'affordance_cache.json').exists():
                shutil.copyfile(root / 'inputs/affordance_cache_base.json', node_dir / 'affordance_cache.json')
            cmd = coordinator(root, c, rank, args.mllm_python, args.editor_python)
            atomic_json(node_dir / 'command.json', cmd)
            for attempt in range(args.retries + 1):
                # Waves resume from their checkpoints: a transient failure (a lost service worker, a mount
                # hiccup) costs one restart of the current stage instead of the worker.
                child = subprocess.Popen(cmd, env=env, start_new_session=True)
                CHILDREN.append(child)
                if child.wait() == 0:
                    break
                atomic_json(node_dir / f'failure.{int(time.time())}.json',
                            {'time': time.time(), 'attempt': attempt, 'error': f'coordinator exited with {child.returncode}'})
                if attempt == args.retries:
                    raise RuntimeError(f'wave coordinator exited with {child.returncode} ({attempt + 1} attempts)')
                time.sleep(120)
            atomic_json(node_dir / 'done.json', {'time': time.time(),
                                                 'summary': json.loads((node_dir / 'waves/final/summary.json').read_text())})
            if rank == 0:
                # Keep the monitor running until every rank is done or has failed (then merge by hand).
                while not all(rank_status(root, c, r)['state'] in ('done', 'failed') for r in range(c['world_size'])):
                    time.sleep(60)
                if all(rank_status(root, c, r)['state'] == 'done' for r in range(c['world_size'])):
                    merge(SimpleNamespace(root=root, partial=False))
        except BaseException as exc:
            atomic_json(node_dir / 'failure.json', {'time': time.time(), 'error': repr(exc)})
            raise
        finally:
            stop.set()
            if monitor_thread:
                monitor_thread.join(2)


def prepare_pool(args):
    from sa1b_pipeline.production_pool import Pool, used_sources
    c = config(args.root)
    plan = json.loads(Path(args.plan).read_text())
    pool = Pool(args.root, args.rank, c)
    node_dir = args.root / 'nodes' / f'node{args.rank}'
    started = time.time()
    info = pool.prepare(plan, used_sources(node_dir / 'waves'), mllm_python=sys.executable,
                        planner_model=c['models']['planner'], devices=[str(i) for i in range(c['gpus_per_node'])])
    info['seconds'] = round(time.time() - started, 1)
    with (node_dir / 'pool_log.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'wave': plan['wave'], **info}) + '\n')
    print(json.dumps({'stage': 'prepare_pool', 'wave': plan['wave'], 'shards': info['shards'],
                      'seconds': info['seconds'], 'available': info['log'][-1].get('available')}), flush=True)


def warm_pool(args):
    """Feed, label and build the menus of the first shards of every rank on this machine (HDFS mount and the 8
    GPUs needed). Shard k of every rank is fed before shard k+1 of any; a shard already done is skipped, so this
    can stop and resume. Labels of one rank are copied by the next instead of judged again."""
    from concurrent.futures import ThreadPoolExecutor
    from sa1b_pipeline.production_pool import Pool, write_tables
    c = config(args.root)
    ranks = args.ranks if args.ranks is not None else list(range(c['world_size']))
    pools = {r: Pool(args.root, r, c) for r in ranks}
    wanted = {r: pools[r].order[:args.shards_per_rank] for r in ranks}
    write_tables(args.root, sorted({s for shards in wanted.values() for s in shards}))
    fed = {r: set(pools[r].done('shards')) for r in ranks}
    feed = [(pools[r], wanted[r][k]) for k in range(args.shards_per_rank) for r in ranks
            if k < len(wanted[r]) and wanted[r][k] not in fed[r]]
    with ThreadPoolExecutor(max(1, args.parallel)) as threads:
        list(threads.map(lambda item: item[0].feed([item[1]]), feed))
    if args.feed_only:   # CPU and HDFS only (e.g. while the GPUs are busy); run again without it later
        print(json.dumps({'warm_pool_fed': {r: pools[r].done('shards') for r in ranks}}), flush=True)
        return
    devices = [str(i) for i in range(c['gpus_per_node'])]
    for r in ranks:
        pool = pools[r]
        todo = [s for s in wanted[r] if s not in set(pool.done('labeled'))]
        if todo:
            print(json.dumps({'stage': 'warm_label', 'rank': r,
                              **pool.label(todo, mllm_python=sys.executable, planner_model=c['models']['planner'],
                                           devices=devices)}), flush=True)
        for shard in wanted[r]:
            pool.menus(shard)
    print(json.dumps({'warm_pool': {r: {'menus': pools[r].done('menus')} for r in ranks}}), flush=True)


def rank_status(root: Path, c: dict, rank: int) -> dict:
    node_dir = root / 'nodes' / f'node{rank}'
    p = node_dir / 'waves/progress.json'
    lock = root / 'control' / f'node{rank}.lock.d'
    s = {'rank': rank, 'state': 'done' if (node_dir / 'done.json').exists() else
         'failed' if (node_dir / 'failure.json').exists() and not lock.exists()
         else 'running' if lock.exists() else 'not_started'}
    if not p.exists():
        return s
    prog = json.loads(p.read_text())
    quota = c['quota']['per_rank_per_operation']
    balanced = prog['balanced']
    done = sum(balanced.values())
    progress = done / (len(OPERATIONS) * quota)
    eta = None
    ledger = node_dir / 'waves/ledger.json'
    if ledger.exists() and done:
        # From the finished waves' own run time (+1 min for selection and the pool check), so that setup, model
        # copying, new shards and restarts do not inflate it.
        walls = [w['seconds'] + 60 for w in json.loads(ledger.read_text())['waves'] if w.get('seconds')]
        if walls:
            eta = (1 - progress) / (progress / len(walls)) * (sum(walls) / len(walls))
    first = json.loads((node_dir / 'started.json').read_text())['first_start'] if (node_dir / 'started.json').exists() else None
    s.update(waves_finished=prog['waves_finished'], balanced=balanced, exported=prog['exported'],
             progress=round(progress, 4), elapsed_hours=round((time.time() - first) / 3600, 2) if first else None,
             eta_hours=round(eta / 3600, 1) if eta is not None else None, updated=prog['updated'])
    return s


def status(root: Path) -> dict:
    c = config(root)
    ranks = [rank_status(root, c, r) for r in range(c['world_size'])]
    out = {'time': time.time(), 'quota': c['quota'], 'ranks': ranks}
    known = [r for r in ranks if r.get('progress') is not None]
    if known:
        out['progress'] = round(sum(r['progress'] for r in known) / c['world_size'], 4)
        out['balanced_total'] = sum(sum(r['balanced'].values()) for r in known)
        etas = [r['eta_hours'] for r in known if r.get('eta_hours') is not None]
        out['eta_hours_slowest_rank'] = max(etas) if etas else None
    return out


def monitor(root: Path, stop: threading.Event):
    while not stop.is_set():
        try:
            s = status(root)
            atomic_json(root / 'progress.json', s)
            line = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' | ' + ' | '.join(
                f"r{r['rank']} {r['state']} w{r.get('waves_finished', 0)} "
                f"{100 * (r.get('progress') or 0):.1f}% eta {r.get('eta_hours')}h" for r in s['ranks'])
            (root / 'logs').mkdir(exist_ok=True)
            with (root / 'logs/progress.log').open('a') as f:
                f.write(line + '\n')
        except (OSError, ValueError, KeyError) as exc:  # a rank file being replaced; try again next round
            print(json.dumps({'monitor_error': repr(exc)}), flush=True)
        stop.wait(600)


def merge(args):
    """final/: every rank's balanced singles; an operation that ends short of its total (a rank stopped by the
    gap, or a rank not finished with --partial) is topped up from the ranks' surplus of that operation."""
    import run_single_waves as rw
    root = args.root
    c = config(root)
    quota = c['quota']['per_rank_per_operation']
    total = quota * c['world_size']
    out = root / ('final_partial' if args.partial else 'final')
    out.mkdir(exist_ok=True)
    balanced, surplus, ranks = [], [], []
    for r in range(c['world_size']):
        waves = root / 'nodes' / f'node{r}' / 'waves'
        if not args.partial and not (root / 'nodes' / f'node{r}' / 'done.json').exists():
            raise FileNotFoundError(f'rank {r} has not finished; use --partial')
        if not waves.exists():
            continue
        rank_out = waves / 'final' if not args.partial else out / f'rank{r}'
        summary = (json.loads((waves / 'final/summary.json').read_text()) if not args.partial
                   else rw.export(waves, quota, out=rank_out))
        balanced += [dict(s, rank=r) for s in rw.rows(rank_out / 'singles_balanced.jsonl')]
        surplus += [dict(s, rank=r) for s in rw.rows(rank_out / 'surplus.jsonl')]
        ranks.append({'rank': r, **summary})
    count = Counter(s['operation'] for s in balanced)
    topped, rest = Counter(), []
    for s in surplus:
        if count[s['operation']] < total:
            balanced.append(dict(s, topped_up=True))
            count[s['operation']] += 1
            topped[s['operation']] += 1
        else:
            rest.append(s)
    for name, data in (('singles_balanced.jsonl', balanced), ('surplus.jsonl', rest)):
        temp = out / f'{name}.partial'
        with temp.open('w') as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        temp.replace(out / name)
    report = {'quota': c['quota'], 'complete': not args.partial, 'balanced': {op: count[op] for op in OPERATIONS},
              'balanced_total': len(balanced), 'topped_up_from_surplus': dict(topped),
              'shortfall': {op: max(0, total - count[op]) for op in OPERATIONS},
              'surplus_left': dict(Counter(s['operation'] for s in rest)), 'ranks': ranks,
              'note': 'one single-target edit sample per line (source -> edited image, one mask, one instruction); '
                      'paths are absolute'}
    atomic_json(out / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'ranks'}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--samples', type=int, default=100_000, help='total singles, the same number per operation')
    prep.add_argument('--world-size', type=int, default=4)
    prep.add_argument('--seed', type=int, default=20261027)
    prep.add_argument('--wave-targets', type=float, default=1700, help='planned targets per wave and rank (~1,000 singles)')
    prep.add_argument('--wave-cap-groups', type=int, default=3000)
    prep.add_argument('--stop-gap', type=int, default=2,
                      help='per rank: no further wave when at most this many singles are missing (merge tops up)')
    prep.add_argument('--max-waves', type=int, default=1000, help='per rank (tests: a few)')
    prep.add_argument('--max-planning-chunks', type=int, default=4)
    prep.add_argument('--groups-in-flight', type=int, default=24)
    prep.add_argument('--tables-per-rank', type=int, default=6, help='shard tables written now (more on demand)')
    prep.add_argument('--replace-unstarted', action='store_true',
                      help='before any rank has started: set config.json and deployment/ aside and prepare again')
    prep.add_argument('--reservation', type=Path, default=DEFAULT_INPUTS['reservation'])
    prep.add_argument('--benchmark-exclusion', type=Path, default=DEFAULT_INPUTS['benchmark_exclusion'])
    prep.add_argument('--extra-exclusion', type=Path, nargs='*', default=DEFAULT_INPUTS['extra_exclusion'])
    prep.add_argument('--affordance-caches', type=Path, nargs='*', default=DEFAULT_INPUTS['affordance_caches'],
                      help='merged into the base affordance cache (fewer categories to label on the workers)')
    prep.add_argument('--avoid-order', type=Path, default=EVIDENCE_RUN / 'inputs/shard_order.json',
                      help='shard order of the twins/multi production, whose first shards are not used here')
    prep.add_argument('--avoid-first', type=int, default=48, help='shards per rank of --avoid-order left out')
    prep.add_argument('--avoid-shards', type=int, nargs='*', default=[667],
                      help='further shards left out (667: single-target validation pool)')
    bun = sub.add_parser('bundle')
    bun.add_argument('--hotfix', help='replace the code of a started run (reason recorded; previous bundle kept)')
    w = sub.add_parser('warm-pool', help='feed, label and build menus of the first shards of every rank here')
    w.add_argument('--shards-per-rank', type=int, default=2)
    w.add_argument('--ranks', type=int, nargs='*')
    w.add_argument('--parallel', type=int, default=2, help='shards fed at the same time')
    w.add_argument('--feed-only', action='store_true', help='only feed (no GPUs); label and menus in a later call')
    n = sub.add_parser('node')
    n.add_argument('--rank', type=int, required=True)
    n.add_argument('--mllm-python', required=True)
    n.add_argument('--editor-python', required=True)
    n.add_argument('--model-map', type=Path, help='existing {model: local copy} map instead of staging (tests)')
    n.add_argument('--retries', type=int, default=3, help='restarts of the wave coordinator after a failure')
    pp = sub.add_parser('prepare-pool', help='called by the wave coordinator before each selection')
    pp.add_argument('--rank', type=int, required=True)
    pp.add_argument('--plan', type=Path, required=True)
    sub.add_parser('status')
    rec = sub.add_parser('recover-lock', help='after a worker died without cleanup (e.g. Arnold restart on a new host)')
    rec.add_argument('--rank', type=int, required=True)
    rec.add_argument('--previous-workers-stopped', action='store_true', required=True)
    m = sub.add_parser('merge')
    m.add_argument('--partial', action='store_true', help='export what is finished so far into final_partial/')
    args = p.parse_args()
    args.root = args.root.resolve()
    if any(args.root == x or x in args.root.parents for x in PROTECTED):
        raise ValueError('protected location: new production runs go under 02_production_runs/ in a new directory')
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'bundle':
        bundle(args)
    elif args.command == 'warm-pool':
        warm_pool(args)
    elif args.command == 'node':
        node(args)
    elif args.command == 'prepare-pool':
        prepare_pool(args)
    elif args.command == 'status':
        print(json.dumps(status(args.root), indent=1))
    elif args.command == 'recover-lock':
        lock = args.root / 'control' / f'node{args.rank}.lock.d'
        if lock.exists():
            kept = lock.with_name(f'{lock.name}.recovered.{time.time_ns()}')
            lock.rename(kept)
            print(f'kept the previous lock at {kept}')
    else:
        merge(args)


if __name__ == '__main__':
    main()
