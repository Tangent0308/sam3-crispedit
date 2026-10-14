"""Run with each tested interpreter to freeze its installed inference stack."""
import json
import sys
from importlib import metadata
from pathlib import Path

rows = []
for dist in metadata.distributions():
    name = dist.metadata['Name']
    if name.lower() in {'sa1b-samtok-edit-data', 'vllm-omni'}:
        continue
    direct = dist.read_text('direct_url.json')
    if direct:
        url = json.loads(direct)['url']
        if not url.startswith('https://'):
            raise ValueError(f'nonportable dependency: {name}')
        rows.append(f'{name} @ {url}')
    else:
        rows.append(f'{name}=={dist.version}')
Path(sys.argv[1]).write_text('# Frozen from the tested 2026-10-13 runtime; install with --no-deps.\n' + '\n'.join(sorted(set(rows), key=str.lower)) + '\n')
