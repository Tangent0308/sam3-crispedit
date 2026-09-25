"""Byte-verified node-local model staging; never transform model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from synthesis_pipeline.run_multinode_labeling import atomic, check_peer_failure


def stage(source, destination, run_root=None):
    check_peer_failure(run_root)
    source=Path(source).resolve();destination=Path(destination).absolute()
    if not (source/'model_index.json').is_file():raise FileNotFoundError(source/'model_index.json')
    if destination==source or source in destination.parents:
        raise ValueError('Model cache must be separate from source')
    destination.mkdir(parents=True,exist_ok=False)
    records=[]
    files=sorted(p for p in source.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(source).parts))
    for i,file in enumerate(files):
        check_peer_failure(run_root)
        rel=file.relative_to(source);out=destination/rel;out.parent.mkdir(parents=True,exist_ok=True)
        temporary=out.with_name(out.name+'.staging')
        h=hashlib.sha256();size=0
        with file.open('rb') as inp,temporary.open('xb') as dst:
            while block:=inp.read(8<<20):
                check_peer_failure(run_root)
                dst.write(block);h.update(block);size+=len(block)
        sha=h.hexdigest()
        verified=hashlib.sha256()
        with temporary.open('rb') as copied:
            while block:=copied.read(8<<20):
                check_peer_failure(run_root)
                verified.update(block)
        if size!=file.stat().st_size or verified.hexdigest()!=sha:
            raise ValueError(f'Model copy verification failed: {rel}')
        os.replace(temporary,out)
        records.append(dict(file=str(rel),bytes=size,sha256=sha))
        print(f'staged {i+1}/{len(files)}: {rel} ({size} bytes)',flush=True)
    check_peer_failure(run_root)
    atomic(destination/'staging_manifest.json',dict(source=str(source),files=records,
        bytes=sum(r['bytes'] for r in records),byte_verified=True))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--run-root',type=Path,help='Abort copy/verification when a peer fails')
    a=p.parse_args();stage(a.source,a.destination,a.run_root)


if __name__=='__main__':main()
