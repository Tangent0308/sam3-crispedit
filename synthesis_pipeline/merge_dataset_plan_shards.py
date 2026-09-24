"""Merge source-disjoint dataset-mask planner shards into one executable cohort."""
import argparse
import json
from pathlib import Path
import shutil


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def keyed(rows):
    result = {}
    for row in rows:
        name = row['image']
        if name in result:
            raise ValueError(f'duplicate case across shards: {name}')
        result[name] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--shard-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    original = read(args.data_root / 'annotations.jsonl')
    expected = {row['image'] for row in original}
    shards = sorted(
        path for path in args.shard_root.iterdir()
        if path.is_dir() and (path / 'dataset_planning_summary.json').exists()
    )
    if not shards:
        raise ValueError('no completed planner shards')

    plan_rows, scope_rows, region_rows = [], [], []
    plan_responses, scope_responses, summaries = [], [], []
    initial_cases = []
    for shard in shards:
        summary = json.loads((shard / 'dataset_planning_summary.json').read_text())
        summaries.append(summary)
        plan_rows.extend(read(shard / 'plan/annotations.jsonl'))
        scope_rows.extend(read(shard / 'scope/annotations.jsonl'))
        region_rows.extend(read(shard / 'regions/annotations.jsonl'))
        ground = read(shard / 'plan/responses.jsonl')
        scope = read(shard / 'scope/responses.jsonl')
        plan_responses.extend(ground)
        scope_responses.extend(scope)
        initial_cases.extend(
            response['image'] for response in ground if response['attempt'] == 1
        )
    if len(initial_cases) != len(set(initial_cases)):
        raise ValueError('a case was assigned to more than one shard')
    if set(initial_cases) != expected:
        missing = sorted(expected - set(initial_cases))
        extra = sorted(set(initial_cases) - expected)
        raise ValueError(f'shard coverage mismatch missing={missing} extra={extra}')

    plans = keyed(plan_rows)
    scopes = keyed(scope_rows)
    regions = keyed(region_rows)
    if set(scopes) != set(regions):
        raise ValueError('scope and executable region sets differ')
    if not set(scopes).issubset(plans):
        raise ValueError('scope output is not a subset of visual grounding output')

    args.out_root.mkdir(parents=True, exist_ok=True)
    for directory, annotations, responses, stage_name in [
        ('plan', plans, plan_responses, 'ground'),
        ('scope', scopes, scope_responses, 'scope'),
    ]:
        root = args.out_root / directory
        root.mkdir(exist_ok=False)
        (root / 'inputs').mkdir()
        (root / 'sources').symlink_to((args.data_root / 'sources').resolve())
        write(root / 'input_annotations.jsonl', original)
        write(root / 'annotations.jsonl', [annotations[name] for name in sorted(annotations)])
        write(root / 'responses.jsonl', sorted(
            responses, key=lambda row: (row['image'], row['attempt'])
        ))
        stage_summaries = [summary['stages'][stage_name] for summary in summaries]
        stage_summary = {
            'input_cases': len(expected) if stage_name == 'ground' else len(plans),
            'accepted': len(annotations),
            'calls': sum(item['calls'] for item in stage_summaries),
            'inference_gpu_seconds': sum(item['inference_seconds'] for item in stage_summaries),
            'parallel_wall_seconds_upper_bound': max(item['wall_seconds'] for item in stage_summaries),
            'version': summaries[0]['version'],
            'shards': len(shards),
        }
        (root / 'summary.json').write_text(json.dumps(stage_summary, indent=2))
        copied = set()
        for shard in shards:
            for source in (shard / directory / 'inputs').iterdir():
                if source.name in copied:
                    raise ValueError(f'duplicate visual input across shards: {source.name}')
                shutil.copy2(source, root / 'inputs' / source.name)
                copied.add(source.name)
        expected_inputs = set(plans) if directory == 'scope' else expected
        if copied != expected_inputs:
            raise ValueError(
                f'{directory} visual input coverage mismatch: '
                f'missing={sorted(expected_inputs-copied)} extra={sorted(copied-expected_inputs)}'
            )

    root = args.out_root / 'regions'
    root.mkdir(exist_ok=False)
    (root / 'sources').symlink_to((args.data_root / 'sources').resolve())
    write(root / 'input_annotations.jsonl', original)
    write(root / 'annotations.jsonl', [regions[name] for name in sorted(regions)])
    (root / 'summary.json').write_text(json.dumps({
        'cases': len(regions), 'unresolved': 0, 'source_sam_calls': 0,
        'mask_policy': 'original_dataset', 'shards': len(shards),
    }, indent=2))
    merged = {
        'version': summaries[0]['version'],
        'input_cases': len(expected),
        'grounded': len(plans),
        'accepted': len(regions),
        'shards': len(shards),
        'aggregate_gpu_seconds': sum(item['wall_seconds'] for item in summaries),
        'parallel_wall_seconds_upper_bound': max(item['wall_seconds'] for item in summaries),
        'source_sam_calls': 0,
    }
    (args.out_root / 'dataset_planning_summary.json').write_text(
        json.dumps(merged, indent=2)
    )
    print(json.dumps(merged), flush=True)


if __name__ == '__main__':
    main()
