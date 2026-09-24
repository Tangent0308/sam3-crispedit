"""Ablate late composition on cached diffusion outputs, without another model call."""

import argparse
import json
import time
from pathlib import Path
from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from utils.context_edit import compose_guarded_crop, compose_grounded_crop, protected_neighbors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--composition', choices=['wide', 'connected', 'connected_poisson','grounded','grounded_poisson'], default='wide')
    args = p.parse_args()
    started = time.perf_counter()
    (args.out_root / 'edited').mkdir(parents=True, exist_ok=True)
    rows = []
    for line in (args.data_root / 'annotations.jsonl').read_text().splitlines():
        row = json.loads(line)
        root = args.raw_root / 'diagnostics' / Path(row['image']).stem
        request_path = root / 'generation_request.json'
        if not request_path.exists():
            continue
        request = json.loads(request_path.read_text())
        source = Image.open(args.data_root / 'sources' / row['source_image']).convert('RGB')
        raw = Image.open(root / 'raw_edited_crop.png').convert('RGB')
        if args.composition.startswith('grounded'):
            contract=row.get('region_contract',{})
            if contract.get('status') not in {'original','segmented_candidate','visually_verified'}:
                raise ValueError(f'Unresolved contract for {row["image"]}')
            protected=mask_array(source.size,contract['protected_mask'])
            result,alpha=compose_grounded_crop(source,raw,mask_array(source.size,row['mask']),
                row['task_type'],tuple(request['crop_bbox']),protected,
                poisson_blend=args.composition=='grounded_poisson')
        else:
            result, alpha = compose_guarded_crop(source, raw, mask_array(source.size, row['mask']),
                row['task_type'], tuple(request['crop_bbox']), protected_neighbors(args.data_root, row, source.size),
                replacement_context_window=True, connected_support=args.composition.startswith('connected'),
                poisson_blend=args.composition == 'connected_poisson')
        destination = args.out_root / 'edited' / row['image']
        if destination.exists():
            # Resume only identical deterministic outputs; never replace different pixels.
            if Image.open(destination).tobytes() != result.tobytes():
                raise FileExistsError(destination)
        else:
            result.save(destination)
        alpha_dir = args.out_root / 'support'
        alpha_dir.mkdir(exist_ok=True)
        alpha.save(alpha_dir / row['image'])
        rows.append({**row, 'composition_revision': {'raw_request': str(request_path),
            'method': 'guarded_' + args.composition, 'crop_bbox': request['crop_bbox'],
            'changed_from_original': Image.open(args.raw_root / 'edited' / row['image']).tobytes() != result.tobytes(),
            'alpha': str(alpha_dir / row['image'])}})
    (args.out_root / 'annotations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps(dict(cases=len(rows), diffusion_calls=0, seconds=time.perf_counter()-started)))


if __name__ == '__main__':
    main()
