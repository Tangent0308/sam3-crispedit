#!/usr/bin/env python3
"""Refine grouped source selectors in a new directory without regenerating images.

Changed masks invalidate semantic acceptance. Re-audit or explicitly review the
new revision; this command never publishes a training release.
"""
import argparse
import copy
import json
import time
from pathlib import Path

from sa1b_pipeline.source_regions import resolve_source_region, blind_group_check
from sa1b_pipeline.site_segmentation import Sam3SiteResolver
from sa1b_pipeline.group_generation import _row
from sa1b_pipeline.validate import deterministic_check
from sa1b_pipeline.production_state import atomic_json
from sa1b_pipeline.io import read_jsonl, write_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--site-model',required=True)
    args=p.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();resolver=Sam3SiteResolver(args.site_model)
    rows=[];changed=resolved=failed=0
    for old in read_jsonl(args.manifest):
        g=copy.deepcopy(old);gid=g.get('twin_id') or g['multi_id'];members=g.get('members',g.get('units',[]))
        revised=False
        for m in members:
            spec=m['instruction_spec']
            if spec['operation']!='add' or not spec['placement'].get('site'):continue
            revised=True;changed+=1
            for key in ('mask','bbox','point'):
                m[key+'_before_site_refinement']=copy.deepcopy(m[key])
            source=m.get('step_input_image') or g['source_image']
            target=m.get('edited_image') or m['step_output_image']
            ok=resolve_source_region(m,source,target,args.output_root/gid/m['unit_id'],site_resolver=resolver)
            resolved+=int(ok);failed+=int(not ok)
            row=_row(g,m,source,target,m['support_mask'])
            check=deterministic_check(row)
            m['deterministic' if 'twin_id' in g else 'step_audit']=check
            m['semantic_audit']={'decision':'pending','reason':'source_mask_changed'}
        if revised:
            blind=blind_group_check(g)
            if 'twin_id' in g:
                checks=g['group_checks'];g['group_checks_before_site_refinement']=copy.deepcopy(checks)
                checks.update(blind_protocol=blind,group_pass=False,semantic_audit_pass=False,
                              member_semantic_audit_pass=False,semantic_audit={'decision':'pending','reason':'source_mask_changed'})
                checks['deterministic']={'pass':blind['pass'] and all(m['deterministic']['decision']!='fail' for m in members),
                                         'members':[m['deterministic'] for m in members]}
            else:
                g['joint_audit_before_site_refinement']=copy.deepcopy(g['joint_audit'])
                g['joint_audit'].update(blind_protocol=blind)
                g['joint_audit']['pass']=False
                g['joint_audit']['semantic_audit']={'decision':'pending','reason':'source_mask_changed'}
            g['source_region_revision']={'method':'sam3_site_constraint','upstream_manifest':str(args.manifest.resolve()),'targets_regenerated':False}
        rows.append(g)
        atomic_json(args.output_root/gid/'group.json',g)
    write_jsonl(args.output_root/'groups.jsonl',rows)
    atomic_json(args.output_root/'summary.json',{'groups':len(rows),'site_members':changed,'resolved':resolved,'failed':failed,
                'seconds':time.perf_counter()-started,'model_load_seconds':resolver.load_seconds,'published':False})


if __name__=='__main__':main()
