"""Compare bounded seam composition on the SAME raw outputs; no diffusion."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.prepare_samtok_data import load_jsonl,write_jsonl
from utils.context_edit import compose_grounded_crop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--policy',choices=['adaptive-remove-v4','adaptive-remove-v5'],default='adaptive-remove-v4')
    p.add_argument('--ids',default='')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=False)
    (a.out_root/'edited').mkdir();(a.out_root/'diagnostics').mkdir()
    (a.out_root/'sources').symlink_to((a.data_root/'sources').resolve())
    started=time.perf_counter();results=[];records=[]
    for row in load_jsonl(a.data_root/'annotations.jsonl'):
        if a.ids and int(row['image'].split('_')[0]) not in {int(x) for x in a.ids.split(',')}:continue
        d=a.raw_root/'diagnostics'/Path(row['image']).stem
        if not (a.raw_root/'edited'/row['image']).exists():continue
        source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
        raw=Image.open(d/'raw_edited_crop.png').convert('RGB')
        request=json.loads((d/'generation_request.json').read_text());bbox=tuple(request['crop_bbox'])
        target=mask_array(source.size,row['execution_region']['mask']).astype(bool)
        guard=mask_array(source.size,row['region_contract']['protected_mask']).astype(bool)
        begin=time.perf_counter()
        result,alpha=compose_grounded_crop(source,raw,target,'remove',bbox,guard,remove_composition_policy=a.policy)
        seconds=time.perf_counter()-begin;x1,y1,x2,y2=bbox
        writable=np.zeros(target.shape,bool);writable[y1:y2,x1:x2]=np.asarray(alpha)>0
        old=np.asarray(source);new=np.asarray(result)
        assert np.array_equal(old[~writable],new[~writable])
        assert np.array_equal(old[guard&~target],new[guard&~target])
        assert np.all(np.asarray(alpha)[target[y1:y2,x1:x2]]==255)
        result.save(a.out_root/'edited'/row['image'])
        dest=a.out_root/'diagnostics'/d.name;dest.mkdir()
        for file in d.iterdir():
            if file.name not in {'composition_alpha.png','generation_request.json'}:
                (dest/file.name).symlink_to(file.resolve())
        alpha.save(dest/'composition_alpha.png')
        request.update(remove_composition_policy=a.policy,composition_seconds=seconds,raw_reused_from=str(d.resolve()))
        (dest/'generation_request.json').write_text(json.dumps(request,indent=2))
        records.append(dict(image=row['image'],seconds=seconds,invariants_passed=True));results.append(row)
    write_jsonl(a.out_root/'annotations.jsonl',results);write_jsonl(a.out_root/'composition.jsonl',records)
    summary=dict(cases=len(results),diffusion_calls=0,wall_seconds=time.perf_counter()-started,
        mean_composition_seconds=float(np.mean([r['seconds'] for r in records])) if records else None)
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
