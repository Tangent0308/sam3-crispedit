"""Input-balanced operation preferences, independent of model fallback outputs."""
from __future__ import annotations

import hashlib


def fixed_balanced_schedule(keys, seed=20261012):
    """Assign every source/mask once; first-choice counts differ by at most one.

    Hash ordering avoids coupling source traversal order to operation type.
    Preferences permit a natural fallback in the same model response. This is
    input balance, not a claim that retained outputs are balanced.
    """
    keys = [(str(source), str(mask)) for source, mask in keys]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate source/mask in operation schedule')
    operations = ('add', 'replace', 'remove', 'attribute')

    def digest(key, suffix=''):
        # Length prefixes keep source/mask boundaries unambiguous.
        raw = f'{seed}:{len(key[0])}:{key[0]}:{len(key[1])}:{key[1]}:{suffix}'
        return hashlib.sha256(raw.encode()).digest()

    result = {}
    for index, key in enumerate(sorted(keys, key=lambda k: (digest(k), k))):
        first = operations[index % len(operations)]
        rest = sorted((op for op in operations if op != first), key=lambda op: digest(key, op))
        result[key] = [first, *rest]
    return result
