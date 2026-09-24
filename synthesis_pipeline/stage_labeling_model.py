"""Byte-verified node-local model staging; never transform model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from synthesis_pipeline.run_multinode_labeling import atomic, digest


def stage(source, destination):
    source=Path(source).resolve();destination=Path(destination).absolute()
    if not (source/'model_index.json').is_file():raise FileNotFoundError(source/'model_index.json')
    if destination==source or source in destination.parents:
        raise ValueError('Model cache must be separate from source')
    destination.mkdir(parents=True,exist_ok=False)
    records=[]
    files=sorted(p for p in source.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(source).parts))
    for i,file in enumerate(files):
        rel=file.relative_to(source);out=destination/rel;out.parent.mkdir(parents=True,exist_ok=True)
        temporary=out.with_name(out.name+'.staging')
        h=hashlib.sha256();size=0
        with file.open('rb') as inp,temporary.open('xb') as dst:
            while block:=inp.read(8<<20):
                dst.write(block);h.update(block);size+=len(block)
        sha=h.hexdigest()
        if size!=file.stat().st_size or digest(temporary)!=sha:
            raise ValueError(f'Model copy verification failed: {rel}')
        os.replace(temporary,out)
        records.append(dict(file=str(rel),bytes=size,sha256=sha))
        print(f'staged {i+1}/{len(files)}: {rel} ({size} bytes)',flush=True)
    atomic(destination/'staging_manifest.json',dict(source=str(source),files=records,
        bytes=sum(r['bytes'] for r in records),byte_verified=True))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();stage(a.source,a.destination)


if __name__=='__main__':main()
