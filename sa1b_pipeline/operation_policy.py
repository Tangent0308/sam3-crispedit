"""Assign operation quotas before visual selection or instruction generation."""
from __future__ import annotations
from collections import Counter
from .schema import OPERATIONS

# Retrieval hints only. The VLM still has to confirm identity and feasibility.
OPERATION_HINTS = {
    'add': {'chair', 'seat', 'bench', 'sofa', 'bag', 'backpack', 'suitcase', 'table'},
    'replace': {'chair', 'pot', 'flowerpot', 'vase', 'bag', 'backpack', 'bottle', 'lamp', 'sign', 'bench', 'cup'},
    'remove': {'chair', 'pot', 'flowerpot', 'vase', 'bottle', 'bag', 'box', 'cup', 'sign', 'trash', 'cone', 'bin'},
    'attribute': {'shirt', 'jacket', 'coat', 'sleeve', 'pants', 'jeans', 'bag', 'cap', 'hat', 'pot', 'awning'},
}

def operation_affinity(category: str, operation: str) -> bool:
    return bool(set(category.lower().replace('-', ' ').replace(',', ' ').split()) & OPERATION_HINTS[operation])


def assign_operations(source_ids: list[str], *, policy: str = 'balanced',
                      accepted_counts: dict[str, int] | None = None) -> dict[str, str]:
    """Deterministic source quotas, independent of batching and worker sharding.

    Each source is assigned one operation. Failed sources keep their assigned
    type: callers can replenish that type with new sources, never relabel a
    failed add/remove as an easy attribute task.
    """
    if policy != 'balanced' and policy not in OPERATIONS:
        raise ValueError(f'unknown operation policy: {policy}')
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('source IDs must be unique before operation assignment')
    counts = Counter(accepted_counts or {})
    if any(counts[op] < 0 for op in OPERATIONS):
        raise ValueError('operation counts must be nonnegative')
    assignments = {}
    for sid in source_ids:
        op = min(OPERATIONS, key=lambda x: (counts[x], OPERATIONS.index(x))) if policy == 'balanced' else policy
        assignments[sid] = op
        counts[op] += 1
    return assignments
