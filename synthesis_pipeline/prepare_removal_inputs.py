"""Materialize all positive GRES/VER regions; no MLLM, candidate screening or mask audit."""
import argparse
import io
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from synthesis_pipeline.prepare_samtok_data import (
    build_positive_index, decode_rle, encode_rle, qwen_canvas_size,
    resize_mask, write_jsonl,
)
from synthesis_pipeline.reference_binding import bind_reference
from synthesis_pipeline.run_multinode_labeling import check_peer_failure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--limit-sources',type=int,default=0,help='0=all; positive values only for a bounded smoke run')
    p.add_argument('--run-root',type=Path,help='Stop source preparation when a peer fails')
    a=p.parse_args()
    if a.limit_sources<0:p.error('limit-sources must be nonnegative')
    check_peer_failure(a.run_root)
    a.out_root.mkdir(parents=True,exist_ok=False)
    rows,index_summary=build_positive_index(a.parquet,a.out_root/'positive_rows.jsonl',False)
    if a.limit_sources:rows=rows[:a.limit_sources]
    import pyarrow.parquet as pq
    # Read one parquet record batch at a time, not all embedded image bytes.
    selected={int(r['parquet_row_index']):r for r in rows}
    source_dir=a.out_root/'sources';source_dir.mkdir()
    records=[];offset=0
    with tqdm(total=len(rows),desc='materialize positive source images') as bar:
        for batch in pq.ParquetFile(a.parquet).iter_batches(batch_size=64,columns=['images']):
            check_peer_failure(a.run_root)
            for local,cell in enumerate(batch.column(0)):
                index=offset+local
                if index not in selected:continue
                row=selected[index];name=f"source_{row['source_subset']}_r{index}.png"
                with Image.open(io.BytesIO(cell.as_py()[0]['bytes'])) as im:
                    source=im.convert('RGB');size=qwen_canvas_size(*source.size)
                    source.resize(size,Image.Resampling.LANCZOS).save(source_dir/name)
                for mi,mask in enumerate(row['masks']):
                    records.append(dict(image=f"{len(records):06d}_{row['source_subset']}_r{index}_m{mi}_remove.png",
                        source_image=name,source_subset=row['source_subset'],parquet_row_index=index,
                        mask_index=mi,num_masks=row['num_masks'],mask=encode_rle(resize_mask(decode_rle(mask),size)),
                        task_type='remove',editing_instruction='',problem=row['problem'],answer=row['answer'],
                        reference_binding=bind_reference(row['answer'],mi,row['num_masks'])))
                bar.update(1)
            offset+=batch.num_rows
    if len({r['parquet_row_index'] for r in records})!=len(rows):raise ValueError('Incomplete source materialization')
    check_peer_failure(a.run_root)
    write_jsonl(a.out_root/'annotations.jsonl',records)
    summary=dict(source_images=len(rows),cases=len(records),task_type='remove',limit_sources=a.limit_sources,
                 every_region_used=True,index_summary=index_summary)
    (a.out_root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
