"""Compare saved audit decisions with independently recorded assistant reviews."""
import argparse
import json
from pathlib import Path
from collections import Counter
from synthesis_pipeline.audit_quality_v3 import read
from synthesis_pipeline.audit_edit_pairs import write_jsonl
from synthesis_pipeline.audit_quality_v4 import corrected_annotation


def ratio(n,d): return n/d if d else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--reviews',type=Path,required=True)
    p.add_argument('--rewrite-reviews',type=Path,help='Separate post-model review of the exact reconstructed instruction')
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
    reviews={r['image']:r for r in read(a.reviews)}
    final=read(a.audit_root/'edit_audit.jsonl')
    rows={r['image']:r for r in read(a.annotations)}
    if any(r['image'] not in reviews for r in final):
        raise ValueError('Every audited case must be manually reviewed before scoring')
    q={r['image']:r for r in read(a.audit_root/'quality.jsonl')}
    candidates={r['image']:r for r in read(a.audit_root/'model_accepted_annotations.jsonl')}
    if a.rewrite_reviews:
        for reviewed in read(a.rewrite_reviews):
            name=reviewed['image']
            if name in candidates and candidates[name]['editing_instruction']!=reviewed['reviewed_instruction']:
                raise ValueError(f'Stale rewrite review: {name}')
            reviews[name]={**reviews[name],**reviewed}
    stats={}
    for stage in ['baseline','quality','verified_quality']:
        path=a.audit_root/f'{stage}.jsonl'
        if stage=='verified_quality':
            decisions={r['image']:'pass' if (r.get('audit') or {}).get('visual_quality')=='pass'
                and (r.get('verification') or {}).get('quality')=='pass' else 'fail' for r in final}
        elif path.exists():
            decisions={r['image']:'pass' if (r.get('audit') or {}).get('visual_quality')=='pass' else 'fail' for r in read(path)}
        else:continue
        good=[k for k in decisions if reviews[k]['visual_quality']=='pass']
        bad=[k for k in decisions if reviews[k]['visual_quality']=='fail']
        missed=[k for k in bad if decisions[k]=='pass']
        rejected_good=[k for k in good if decisions[k]!='pass']
        stats[stage]=dict(cases=len(decisions),manual_good=len(good),manual_bad=len(bad),
            false_pass=missed,false_reject=rejected_good,
            bad_recall=ratio(len(bad)-len(missed),len(bad)),good_retention=ratio(len(good)-len(rejected_good),len(good)))
        if path.exists():
            stats[stage]['parse_errors']=sum(r.get('parsed') is None for r in read(path))
    accepted=[r for r in final if r['quality']=='pass']
    accepted_correct=[];accepted_wrong=[];unchecked=[];rewrites=[];assistant_export=[]
    for record in final:
        name=record['image'];review=reviews[name];row=rows[name]
        if review['visual_quality']=='pass' and review['instruction_match']=='pass':
            assistant_export.append({**row,'assistant_review':review,'verification':'assistant_reviewed_original_instruction'})
        elif record['decision']=='accept_rewrite' and review['visual_quality']=='pass' and review.get('rewrite_match')=='pass':
            candidate={k:candidates[name][k] for k in ('task_type','editing_instruction')}
            assistant_export.append({**corrected_annotation(row,candidate,'accept_rewrite'),
                'assistant_review':review,'verification':'assistant_reviewed_rewrite'})
        if record['quality']!='pass':continue
        if record['decision']=='accept_rewrite':
            label_match=review.get('rewrite_match')
            rewrites.append(dict(image=name,assistant_visual=review['visual_quality'],assistant_rewrite=label_match))
        else:label_match=review['instruction_match']
        if label_match is None:unchecked.append(name)
        elif review['visual_quality']=='pass' and label_match=='pass':accepted_correct.append(name)
        else:accepted_wrong.append(name)
    original_good=[r['image'] for r in final if reviews[r['image']]['visual_quality']==reviews[r['image']]['instruction_match']=='pass']
    stats['final_admission']=dict(decisions=dict(Counter(r['decision'] for r in final)),accepted=len(accepted),
        correct=accepted_correct,incorrect=accepted_wrong,unchecked=unchecked,
        precision=ratio(len(accepted_correct),len(accepted)) if not unchecked else None,
        original_good_retention=ratio(len(set(original_good)&set(accepted_correct)),len(original_good)),
        rewrites=rewrites,assistant_verified_export=len(assistant_export))
    write_jsonl(a.out_root/'assistant_verified_annotations.jsonl',assistant_export)
    write_jsonl(a.out_root/'reviews_with_rewrites.jsonl',[reviews[r['image']] for r in final])
    (a.out_root/'evaluation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
