"""Evaluate a lighting-only correction on retained raw outputs, without diffusion."""
import argparse,json,time
from pathlib import Path
from PIL import Image
import numpy as np
from synthesis_pipeline.prepare_samtok_data import load_jsonl,write_jsonl
from utils.removal_harmonization import compose_harmonized_removal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--method',choices=['harmonic','poisson-diagnostic'],default='harmonic',
                   help='Poisson is an unpromoted geometry-boundary ablation, not a default')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=False)
    (a.out_root/'edited').mkdir();(a.out_root/'diagnostics').mkdir()
    rows=[];results=[];started=time.perf_counter()
    for row in load_jsonl(a.raw_root/'annotations.jsonl'):
        name=row['image'];d=a.raw_root/'diagnostics'/Path(name).stem
        if not (a.raw_root/'edited'/name).exists():continue
        request=json.loads((d/'generation_request.json').read_text())
        src=Image.open(d/'source_crop.png').convert('RGB')
        raw=Image.open(d/'raw_edited_crop.png').convert('RGB')
        alpha=Image.open(d/'composition_alpha.png').convert('L')
        result=Image.open(a.raw_root/'edited'/name).convert('RGB')
        if row['task_type']=='remove':
            crop,meta=compose_harmonized_removal(src,raw,alpha)
            if a.method=='poisson-diagnostic':
                import cv2
                # Padding allows a diagnostic for targets touching the frame.
                # Never write outside the original composition support.
                pad=8
                donor=cv2.copyMakeBorder(np.asarray(raw),pad,pad,pad,pad,cv2.BORDER_REFLECT)
                receiver=cv2.copyMakeBorder(np.asarray(src),pad,pad,pad,pad,cv2.BORDER_REFLECT)
                writable=(np.asarray(alpha)>0).astype('uint8')*255
                binary=cv2.copyMakeBorder(writable,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
                x,y,w,h=cv2.boundingRect(binary)
                cloned=cv2.seamlessClone(donor,receiver,binary.copy(),(x+w//2,y+h//2),cv2.NORMAL_CLONE)
                crop=Image.composite(Image.fromarray(cloned[pad:-pad,pad:-pad]),src,Image.fromarray(writable))
                meta=dict(method='poisson-diagnostic',applied=True,production_approved=False)
            assert np.array_equal(np.asarray(crop)[np.asarray(alpha)==0],np.asarray(src)[np.asarray(alpha)==0])
            result.paste(crop,tuple(request['crop_bbox'][:2]))
        else:meta=dict(applied=False,reason='non_removal')
        result.save(a.out_root/'edited'/name)
        target=a.out_root/'diagnostics'/Path(name).stem;target.mkdir()
        for file in d.iterdir():
            if file.name!='generation_request.json':(target/file.name).symlink_to(file.resolve())
        request['harmonization']=meta
        (target/'generation_request.json').write_text(json.dumps(request,indent=2))
        rows.append({**row,'harmonization':meta});results.append(dict(image=name,**meta))
    write_jsonl(a.out_root/'annotations.jsonl',rows)
    write_jsonl(a.out_root/'harmonization.jsonl',results)
    summary=dict(cases=len(rows),diffusion_calls=0,wall_seconds=time.perf_counter()-started)
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)


if __name__=='__main__':main()
