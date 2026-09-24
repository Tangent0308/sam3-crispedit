"""Deterministic composition diagnostics; never a semantic quality verdict."""
import re

import cv2
import numpy as np


def replacement_phrase(row):
    """Use a frozen explicit replacement or separate the source noun phrase.

    In 'replace the pot with the hose by a bucket', WITH belongs to the
    source. In 'replace the man by the door with a woman', BY belongs to the
    source. The last replacement connector distinguishes these forms. Keep
    WITH descriptors on the replacement when the source phrase is available.
    """
    if row.get('replacement_target'):
        return row['replacement_target'].strip().rstrip(' .')
    text = row['editing_instruction'].strip().rstrip(' .')
    target = row.get('segmentation_target') or row.get('masked_content', '')
    if target:
        match = re.search(re.escape(target), text, re.I)
        if match:
            tail = text[match.end():]
            connector = re.match(r'\s+(?:with|by|into|for)\s+', tail, re.I)
            if connector:
                return tail[connector.end():].strip()
    connectors = list(re.finditer(r'\s+(?:with|by)\s+(?=(?:a|an|the)\s)', text, re.I))
    if connectors:
        text = text[connectors[-1].end():]
    else:
        parts = re.split(r'\s+(?:with|into|for)\s+', text, maxsplit=1, flags=re.I)
        text = parts[-1]
    return re.split(r',\s*(?:while|keeping|preserving|maintaining)\b', text, flags=re.I)[0].strip()


def change_map(source, edited):
    if source.size != edited.size:
        raise ValueError('Source and output must have identical dimensions')
    # Suppress resampling noise, retaining actual small-object changes.
    a = cv2.GaussianBlur(np.asarray(source).astype(np.float32), (3, 3), .6)
    b = cv2.GaussianBlur(np.asarray(edited).astype(np.float32), (3, 3), .6)
    return np.abs(b - a).mean(axis=2)


def removal_change_evidence(source, edited, target):
    """Conservative removal-only veto evidence, never a quality pass.

    Almost unchanged target pixels cannot support accepting a claimed complete
    removal. Low-contrast true edits can also be flagged: retain the evidence
    and do not interpret this as a semantic proof or use it on attributes.
    """
    mask=np.asarray(target,dtype=bool)
    if mask.shape!=(source.height,source.width) or not mask.any():
        raise ValueError('Nonempty aligned original target required')
    delta=change_map(source,edited)[mask]
    fraction=float((delta>=12).mean());mean=float(delta.mean())
    return dict(mean_target_change=mean,changed_target_fraction=fraction,
        change_threshold=12,weak_change_fraction_threshold=.10,weak_change_mean_threshold=8,
        insufficient_change=bool(fraction<.10 and mean<8),semantic_quality='not_checked')


def raw_locality(source, raw, target, protected):
    target = np.asarray(target, dtype=bool)
    protected = np.asarray(protected, dtype=bool) & ~target
    if target.shape != (source.height, source.width) or protected.shape != target.shape:
        raise ValueError('Mask dimensions differ from source')
    ys, xs = np.nonzero(target)
    if not len(xs):
        raise ValueError('Empty target')
    # A narrow collar, not the whole rectangular crop. No permission to
    # relocate a new item onto an unrelated nearby instance.
    radius = int(np.clip(min(np.ptp(xs) + 1, np.ptp(ys) + 1) * .06, 4, 16))
    allowed = cv2.dilate(target.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
    delta = change_map(source, raw)
    changed = delta >= 12
    count = int(changed.sum())
    inside_count = int((changed & target).sum())
    outside_count = int((changed & ~allowed).sum())
    protected_count = int((changed & protected).sum())
    energy = float(delta[changed].sum())
    outside_energy = float(delta[changed & ~allowed].sum()) / max(energy, 1)
    safe = (inside_count >= 8 and count >= 12
            and (outside_energy <= .02 or outside_count <= 12)
            and outside_count <= max(12, count * .02)
            and protected_count <= 4)
    return dict(locality_pass=bool(safe), changed_pixels=count,
                changed_target_pixels=inside_count, outside_pixels=outside_count,
                outside_energy_fraction=outside_energy,
                protected_changed_pixels=protected_count, collar_pixels=radius,
                threshold=12, semantic_quality='not_checked'), changed & allowed


def composition_retention(source, raw, final, target):
    before = change_map(source, raw)
    after = change_map(source, final)
    significant = (before >= 12) & np.asarray(target, dtype=bool)
    count = int(significant.sum())
    retained = float(np.minimum(after[significant], before[significant]).sum()) / max(
        float(before[significant].sum()), 1)
    collapsed = count >= 8 and retained < .35
    return dict(raw_target_changed_pixels=count, retained_change_fraction=retained,
                collapsed=bool(collapsed), threshold=.35)
