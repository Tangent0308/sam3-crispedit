"""Freeze a provisional candidate only after explicit development thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
from synthesis_pipeline.audit_quality_v4 import CONTOUR_QUALITY,RECONSTRUCT,CANDIDATE_VERIFY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--reference-image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ev=json.loads(a.evaluation.read_text())
    q=ev['quality'];f=ev['final_admission']
    for key,value,threshold in [('bad_recall',q['bad_recall'],.8),('good_retention',q['good_retention'],.7),
                                ('precision',f['precision'],.9),('original_good_retention',f['original_good_retention'],.7)]:
        if value is None or value<threshold:raise ValueError(f'Not ready for holdout: {key}={value} < {threshold}')
    if a.output.exists():raise FileExistsError(a.output)
    root=Path(__file__).resolve().parents[1]
    files=[root/'synthesis_pipeline/audit_quality_v4.py',root/'utils/vlm_utils.py',a.reference_image]
    config=dict(status='provisional_for_unseen_holdout_not_unattended_production',
        model='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B',
        thinking=True,reasoning_effort='low',quality_max_tokens=2048,other_max_tokens=3072,
        quality_scope='tight',quality_policy='contour-aware',verification_policy='candidate-only',
        quality_reference_image=str(a.reference_image),
        quality_images='2 aligned target crops plus 1 fixed development reference panel',
        reconstruction_and_verification_images='2 clean-full/outlined-detail panels',
        code_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
        prompts=dict(quality=CONTOUR_QUALITY,reconstruction=RECONSTRUCT,verification=CANDIDATE_VERIFY),
        development_evaluation=str(a.evaluation),development_metrics=ev,
        known_limitations='Small sample; one remaining ambiguous-instance label and false rejections. Every subsequent case still requires assistant review.')
    a.output.write_text(json.dumps(config,ensure_ascii=False,indent=2));print(a.output)


if __name__=='__main__':main()
