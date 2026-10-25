"""Checkpointed, bounded operation retry for the lean planner.

Only explicit semantic rejection triggers a new call. Malformed/truncated
responses are not disguised as unsuitable operations. Accepted multi targets
are immutable; dependency metadata is refreshed only for changed targets.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from .lean_planning import (choose_retry, merge_retry, normalize, prompt,
                            validate_response, schema)


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def _requests(planner, requests):
    pending = []
    for req in requests:
        identity = [req['views'], req['prompt'], req['schema']]
        if 'cache_context' in req:
            identity.append(req['cache_context'])
        if 'token_limit' in req:
            identity.append({'token_limit': req['token_limit']})
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        req['fingerprint'] = fingerprint
        if req['path'].exists():
            record = json.loads(req['path'].read_text())
            if record['fingerprint'] != fingerprint:
                raise ValueError('lean request checkpoint does not match current prompt/input')
            req['raw'] = record['raw']
        else:
            pending.append(req)
    if pending:
        options = {}
        if any('token_limit' in r for r in pending):
            if not all('token_limit' in r for r in pending):
                raise ValueError('mixed explicit/default token limits in request batch')
            options['token_limits'] = [r['token_limit'] for r in pending]
        raws = planner.generate_multimage([(r['views'], r['prompt']) for r in pending],
                                         schemas=[r['schema'] for r in pending], **options)
        if len(raws) != len(pending):
            raise RuntimeError('incomplete lean planner batch')
        for req, raw in zip(pending, raws):
            if req.get('decoder'):
                raw = req['decoder'](raw)
            req['raw'] = raw
            _save(req['path'], {k: req[k] for k in ('fingerprint', 'prompt', 'views', 'schema', 'raw')})


def plan_batch(planner, batch, *, kind, root, retry, accepted_counts=None, existing_groups=(), minimal=False, scene=False, content=False):
    if (scene or content) and not minimal:
        raise ValueError('scene planning requires the minimal output contract')
    module = ('content_planning' if content else
              ('scene_planning' if scene else ('minimal_planning' if minimal else 'lean_planning')))
    contract = __import__('sa1b_pipeline.' + module, fromlist=['prompt'])
    schema, prompt, normalize, validate_response = contract.schema, contract.prompt, contract.normalize, contract.validate_response
    # External counts represent published/accepted data, not model proposals.
    # When unavailable this is explicitly a local retained-plan proxy.
    counts = Counter(accepted_counts or {})
    if accepted_counts is None:
        for g in existing_groups:
            if kind == 'twins': counts[g['operation']] += 1
            else: counts.update(t['operation'] for t in g['units'])
    first = []
    for item in batch:
        assignments = {f'U{i+1}': item['selection']['operation'] if kind == 'twins' else item['selection']['operations'][i]
                       for i in range(len(item['objects']))}
        first.append(dict(schema=schema(kind, assignments), path=root/'planning_attempts'/item['gid']/'first.json',
                          views=item['views'], prompt=item['prompt'], item=item,
                          decoder=(lambda raw, a=assignments: contract.decode(raw, kind, a)) if minimal else None))
    _requests(planner, first)
    retried = []
    for req in first:
        item, raw = req['item'], req['raw']
        selection = item['selection']
        assignments = {f'U{i+1}': selection['operation'] if kind == 'twins' else selection['operations'][i]
                       for i in range(len(item['objects']))}
        req['assignments'] = assignments
        try:
            validate_response(raw, kind, assignments)
        except (ValueError, KeyError, TypeError) as exc:
            req['error'] = str(exc)
            continue
        if kind == 'twins':
            rejected = list(assignments) if not raw['feasible'] else []
            if not rejected: counts[next(iter(assignments.values()))] += 1
        else:
            rejected = [t['unit_id'] for t in raw['targets'] if not t['feasible'] and not t.get('_format_error')]
            counts.update(assignments[t['unit_id']] for t in raw['targets'] if t['feasible'])
        if not retry or not rejected:
            continue
        retry_path = req['path'].with_name('retry.json')
        decision_path = req['path'].with_name('retry_assignment.json')
        if decision_path.exists():
            new = json.loads(decision_path.read_text())['assignments']
        else:
            new = choose_retry(assignments, rejected[:1] if kind == 'twins' else rejected, counts)
            if kind == 'twins':
                op = new[rejected[0]]
                new = {uid: op for uid in assignments}
            _save(decision_path, dict(assignments=new, original=assignments,
                                     count_basis='final_accepted' if accepted_counts is not None else 'local_plan_proxy',
                                     counts=dict(counts)))
        counts.update([next(iter(new.values()))] if kind == 'twins' else [new[uid] for uid in rejected])
        locked = [] if kind == 'twins' else [{**t, 'operation':assignments[t['unit_id']]} for t in raw['targets'] if t['feasible']]
        reasons = raw.get('reason') if kind == 'twins' else {t['unit_id']:t['reason'] for t in raw['targets'] if not t['feasible']}
        # Reuse the exact original location metadata, without duplicating the lengthy base prompt.
        location_marker = 'PERCENT [0,100] (not editable extents): ' if minimal else 'PERCENT [0,100] (not contact_xy scale): '
        locations = (json.loads(item['prompt'].split(location_marker)[1].split('\n')[0])
                     if location_marker in item['prompt'] else None)
        retry_prompt = prompt(kind, new, locations=locations, retry_targets=rejected,
                              locked=locked, previous_reasons=reasons,
                              **({'view_style': item.get('planning_view_style', 'aligned'),
                                  'source_hints': item.get('planning_source_hints')} if content else {}))
        req['retry'] = dict(path=retry_path, views=item['views'], prompt=retry_prompt,
                            assignments=new, rejected=rejected, schema=schema(kind, new, rejected),
                            decoder=(lambda raw, a={uid:new[uid] for uid in rejected}: contract.decode(raw,kind,a)) if minimal else None)
        retried.append(req['retry'])
    _requests(planner, retried)
    results = []
    for req in first:
        item, raw = req['item'], deepcopy(req['raw'])
        assignments = req['assignments']
        history = [dict(stage='first', assignments=assignments, raw=deepcopy(raw))]
        if 'retry' in req:
            second = req['retry']
            history.append(dict(stage='operation_retry', assignments=second['assignments'], raw=deepcopy(second['raw'])))
            try:
                if kind == 'twins':
                    validate_response(second['raw'], kind, second['assignments'])
                    raw = deepcopy(second['raw'])
                else:
                    raw = merge_retry(raw, second['raw'], second['rejected'], second['assignments'])
                assignments = second['assignments']
            except (ValueError, KeyError, TypeError) as exc:
                # Keep originally usable edits if the optional retry is malformed.
                history[-1]['error'] = str(exc)
        try:
            raw = normalize(raw, kind, assignments, item['objects'])
        except (ValueError, KeyError, TypeError) as exc:
            raw = {'feasible':False, 'reason':str(exc), 'failure_reasons':['malformed_mllm_json'],
                   'evidence':str(exc), '_generation_metadata':raw.get('_generation_metadata',{})}
        raw['_planning_attempts'] = history
        raw['_planning_calls'] = len(history)
        item['selection'] = deepcopy(item['selection'])
        if kind == 'twins':
            item['selection']['operation'] = next(iter(assignments.values()))
        else:
            item['selection']['operations'] = list(assignments.values())
            item['assignments'] = {str(o['object_id']):assignments[f'U{i+1}'] for i,o in enumerate(item['objects'])}
        results.append(raw)
    return results
