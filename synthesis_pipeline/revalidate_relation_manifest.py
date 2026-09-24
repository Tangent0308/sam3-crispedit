"""Revalidate saved plans under their declared schema without model calls."""
import argparse
from pathlib import Path
from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.plan_removal_relations import validate_relation_plan
from synthesis_pipeline.prepare_samtok_data import load_jsonl, write_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=False)
    (a.out_root/'sources').symlink_to((a.data_root/'sources').resolve())
    valid=[];deferred=[]
    for row in load_jsonl(a.data_root/'annotations.jsonl'):
        with Image.open(a.data_root/'sources'/row['source_image']) as source:
            mask=mask_array(source.size,row['mask'])
        status=validate_relation_plan(row['relation_plan'],row['relation_policy'],mask)
        if status=='accepted':valid.append(row)
        else:deferred.append(dict(image=row['image'],status=status))
    write_jsonl(a.out_root/'annotations.jsonl',valid)
    write_jsonl(a.out_root/'deferred.jsonl',deferred)
    print(dict(accepted=len(valid),deferred=deferred),flush=True)


if __name__=='__main__':main()
