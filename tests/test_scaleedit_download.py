import io
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scaleedit.download import (
    EXCLUDED_TASKS, LOCAL_TASKS, SCHEMA, Sink, balanced_allocation,
    unique_instruction_index, verified_images, materialize, available_source_groups,
    matched_manifest_indices, reserve_wave_limits, finalize_approximate_state,
)


def test_reordered_duplicate_instructions_are_not_joined_by_row_number():
    assert unique_instruction_index(['second', 'duplicate', 'first', 'duplicate', None]) == {'second': 0, 'first': 2}


def test_unpublished_manifest_shards_are_audited_and_skipped():
    assert available_source_groups({'a': [1], 'b': [2, 3]}, {'a': 42}) == ({'a': [1]}, {'b': 2})


def test_metadata_index_reuses_cached_source_and_excludes_ambiguous_matches(tmp_path):
    source = tmp_path / 'source' / 'local.parquet'
    source.parent.mkdir()
    pq.write_table(pa.table({'edit_instruction': ['b', 'a', 'dup', 'dup']}), source)
    manifest = pa.table({'original_instruction': ['a', 'missing', 'dup', 'b']})
    args = SimpleNamespace(cache_dir=tmp_path)
    assert matched_manifest_indices('local.parquet', [0, 1, 2, 3], manifest, args, 'pin') == [0, 3]
    source.unlink()
    assert matched_manifest_indices('local.parquet', [3, 0], manifest, args, 'pin') == [3, 0]


def test_balance_redistributes_exhausted_rare_categories():
    assert balanced_allocation({'rare': 2, 'a': 100, 'b': 100}, 12) == {'a': 5, 'b': 5, 'rare': 2}
    assert sum(balanced_allocation({'a': 2, 'b': 1}, 100).values()) == 3
    assert balanced_allocation({'a': 0}, 5) == {'a': 0}
    assert not (LOCAL_TASKS & EXCLUDED_TASKS)


def test_parallel_wave_cannot_exceed_global_or_category_quota():
    limits = reserve_wave_limits(['one', 'two'], {'one': {'a': 10, 'b': 10}, 'two': {'a': 10, 'b': 10}},
                                 {'a': 5, 'b': 12}, 14)
    assert sum(sum(values.values()) for values in limits.values()) == 14
    assert sum(values.get('a', 0) for values in limits.values()) <= 5
    assert sum(values.get('b', 0) for values in limits.values()) <= 12


def test_explicit_approximation_preserves_requested_and_actual_counts():
    state = {'baseline_counts': {'a': 100000}, 'target_new_rows': 200000}
    result = finalize_approximate_state(state, {'a': 299633}, 1)
    assert result['target_new_rows'] == 200000
    assert result['new_rows'] == 199633
    assert result['target_shortfall_rows'] == 367
    assert result['complete']
    with pytest.raises(ValueError):
        finalize_approximate_state(state, {'a': 290000}, 1)
    with pytest.raises(ValueError):
        finalize_approximate_state(state, {'a': 299633}, 5)


def test_concurrent_sink_is_lossless_and_atomic(tmp_path):
    sink = Sink(tmp_path, 'parallel', 16)
    def append(i):
        sink.append({'sample_id': str(i)})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(300)))
    sink.flush()
    values = []
    for path in tmp_path.glob('*.parquet'):
        values.extend(pq.read_table(path)['sample_id'].to_pylist())
    assert len(values) == len(set(values)) == 300
    assert not list(tmp_path.glob('*.incomplete'))


def test_only_complete_decodable_pairs_are_exported():
    buf = io.BytesIO()
    Image.new('RGB', (4, 5)).save(buf, format='PNG')
    data = buf.getvalue()
    assert verified_images({'source_image': data, 'edited_image': data}, False)[3] == [(4, 5), (4, 5)]
    with pytest.raises(ValueError, match='source_bytes_missing'):
        verified_images({'source_image_url': 'https://example.org/a', 'edited_image': data}, False)
    with pytest.raises(ValueError, match='checksum'):
        verified_images({'source_image_url': 'https://example.org/a', 'edited_image': data}, True)


def test_append_sink_preserves_existing_shards(tmp_path):
    old = tmp_path / 'part-00000.parquet'
    old.write_bytes(b'old immutable data')
    sink = Sink(tmp_path, 'test', 2)
    row = dict.fromkeys(SCHEMA.names)
    sink.append(dict(row, sample_id='one'))
    sink.append(dict(row, sample_id='two'))
    assert old.read_bytes() == b'old immutable data'
    assert pq.read_table(tmp_path / 'expand-test-00000.parquet')['sample_id'].to_pylist() == ['one', 'two']
    resumed = Sink(tmp_path, 'test', 2)
    resumed.append(dict(row, sample_id='three'))
    resumed.flush()
    assert pq.read_table(tmp_path / 'expand-test-00001.parquet').num_rows == 1


def test_materialize_matches_instructions_and_preserves_reviewed_fields(tmp_path):
    buf = io.BytesIO()
    Image.new('RGB', (4, 5)).save(buf, format='PNG')
    data = buf.getvalue()
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist([
        {'edit_instruction': 'second', 'source_image': data, 'edited_image': data},
        {'edit_instruction': 'first', 'source_image': data, 'edited_image': data},
        {'edit_instruction': 'duplicate', 'source_image': data, 'edited_image': data},
        {'edit_instruction': 'duplicate', 'source_image': data, 'edited_image': data},
    ]), source)
    candidates = pa.Table.from_pylist([
        {'sample_id': f'shard#{i}', 'source_relative_path': 'shard.parquet', 'row_index': i,
         'original_instruction': instruction, 'final_instruction': 'reviewed ' + instruction,
         'final_task': 'color_change', 'edit_task': 'material_change', 'split': 'train'}
        for i, instruction in enumerate(['first', 'second', 'duplicate'])
    ])
    output = tmp_path / 'output'
    output.mkdir()
    sink = Sink(output, 'test', 2)
    class Bar:
        def update(self, _): pass
    accepted, skipped = materialize(source, candidates, {'color_change': 3}, set(),
        SimpleNamespace(seed=1, fetch_source_urls=False), sink, Bar())
    rows = pq.read_table(next(output.glob('*.parquet'))).to_pylist()
    assert accepted == {'color_change': 2}
    assert skipped == {'non_unique_or_unmatched_instruction': 1}
    assert [(r['sample_id'], r['manifest_row_index'], r['public_source_row_index']) for r in rows] == [
        ('shard#1', 1, 0), ('shard#0', 0, 1)]
    assert [r['final_instruction'] for r in rows] == ['reviewed second', 'reviewed first']
