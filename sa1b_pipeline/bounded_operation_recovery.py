"""Opt-in bounded operation reassignment for per-mask planning.

Recover semantic incompatibility, never a parse/grounding failure. Counts refer
only to accepted data; provisional reservations do not count as accepted output.
Twins change operation atomically across all members. No mask is changed.
"""
from collections import Counter
from copy import deepcopy
from . import per_mask_planning as pm
from .content_guard import incompatible_edit
from .lean_retry import _requests

VERSION = 'bounded-operation-recovery-1'
OPERATIONS = ('add', 'remove', 'replace', 'attribute')


def _eligible(req, target):
    if any(k in req for k in ('first_raw', 'bounded_repair', 'pre_reference_raw', 'operation_recovery')):
        return False
    if target['feasible'] or target.get('_format_error'):
        return False
    reason = target.get('reason', '')
    return (isinstance(req['raw'].get('reject'), str) or
            'attached human anatomy' in reason or
            reason.startswith('whole_person_flat_color:') or
            reason.startswith('replacement_only_changes_appearance:') or
            reason == 'explicit_host_removal_conflicts_with_retained_attachment')


def recover(planner, requests, *, kind, max_extra_calls, accepted_counts=None):
    if kind not in {'twins','multi'} or not isinstance(max_extra_calls,int) or max_extra_calls<0:
        raise ValueError('invalid recovery contract')
    counts=Counter(accepted_counts or {})
    if any(k not in OPERATIONS or not isinstance(v,int) or v<0 for k,v in counts.items()):
        raise ValueError('accepted_counts must contain nonnegative operation counts')
    reservations=Counter();items=[]
    for req in requests:
        if not any(req['item'] is item for item in items):items.append(req['item'])
    jobs=[];transactions=[]
    for item in items:
        own=[r for r in requests if r['item'] is item]
        ass=item['_mask_assignments']
        if [r['unit_id'] for r in own]!=list(ass):raise ValueError('recovery requires every member in binding order')
        decoded=pm.combine([r['raw'] for r in own],kind,ass,item['objects'])
        targets=decoded['members' if kind=='twins' else 'targets']
        eligible=[r for r,t in zip(own,targets) if _eligible(r,t)]
        subsets=[own] if kind=='twins' and eligible else [[r] for r in eligible]
        for subset in subsets:
            if len(jobs)+len(subset)>max_extra_calls:continue
            if any(any(k in r for k in ('first_raw','bounded_repair','pre_reference_raw','operation_recovery')) for r in subset):continue
            previous={ass[r['unit_id']] for r in subset}
            choices=[op for op in OPERATIONS if op not in previous and
                     all(not incompatible_edit(r['raw'].get('ref',''),op) for r in subset)]
            if not choices:continue
            operation=min(choices,key=lambda op:(counts[op]+reservations[op],OPERATIONS.index(op)))
            reservations[operation]+=1
            tx={'item':item,'requests':subset,'operation':operation,'jobs':[]}
            for req in subset:
                retry={**req,'path':req['path'].with_name(req['path'].stem+'_operation_v1_'+operation+'.json'),
                       'prompt':pm.request_prompt(operation,req['views']), 'schema':pm.schema(operation),
                       'cache_context':{**req.get('cache_context',{}),'operation_recovery':VERSION,
                                        'original_operation':ass[req['unit_id']], 'new_operation':operation},
                       'original_request':req}
                retry.pop('raw',None);jobs.append(retry);tx['jobs'].append(retry)
            transactions.append(tx)
    for start in range(0,len(jobs),max(1,planner.batch_size)):
        _requests(planner,jobs[start:start+max(1,planner.batch_size)])
    for tx in transactions:
        item=tx['item'];own=[r for r in requests if r['item'] is item]
        replacements={j['unit_id']:j['raw'] for j in tx['jobs']}
        ass={uid:tx['operation'] if uid in replacements else op for uid,op in item['_mask_assignments'].items()}
        candidate=pm.combine([replacements.get(r['unit_id'],r['raw']) for r in own],kind,ass,item['objects'])
        ts=candidate['members' if kind=='twins' else 'targets']
        # Do not accept a retry that newly breaks a previously retained member.
        prior=pm.combine([r['raw'] for r in own],kind,item['_mask_assignments'],item['objects'])
        pts=prior['members' if kind=='twins' else 'targets']
        applied=all(t['feasible'] for t in ts if t['unit_id'] in replacements)
        applied &= all(not p['feasible'] or t['feasible'] for p,t in zip(pts,ts))
        tx['applied']=bool(applied)
        for j in tx['jobs']:
            req=j['original_request'];j['applied']=bool(applied)
            req['operation_recovery']={'operation':tx['operation'],'raw':deepcopy(j['raw']),'applied':bool(applied)}
            if applied:req['pre_operation_raw']=deepcopy(req['raw']);req['raw']=j['raw']
        if applied:item['_mask_assignments']=ass
    return jobs
