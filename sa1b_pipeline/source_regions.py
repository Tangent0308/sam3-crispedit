"""Protocol 2.0 source selectors, resolved without additional MLLM calls.

Generation still uses the original host/execution geometry. New selectors are
written to separate files, with history retained; they never overwrite inputs.
"""
from __future__ import annotations

from pathlib import Path
import math
import os

_SITE_RESOLVER = None
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, maximum_filter1d

from .group_geometry import precise_point
from .instruction_protocol import mask_contract, validate_noref


def _mask(path, size):
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError('source-region mask size mismatch')
        return np.asarray(image.convert('L')) > 0


def localize_add_site(source, target, host, *, require_raw_contact=True):
    """§6.2: significant difference dilated by 30% short side, intersect host.

    The difference is evidence of the contact site, NOT the exported selector.
    All thresholds and overlap diagnostics are recorded for visual review.
    """
    a, b = np.asarray(source, dtype=np.float32), np.asarray(target, dtype=np.float32)
    if a.shape != b.shape or a.shape[:2] != host.shape:
        raise ValueError('source/target/host size mismatch')
    # Suppress pixel/texture reconstruction noise before estimating the new
    # object's extent. Otherwise scattered changed pixels span the whole host
    # and their expanded bbox incorrectly selects the entire host again.
    sigma = min(2.0, max(.5, min(host.shape)/750))
    delta = np.max(np.abs(gaussian_filter(a, (sigma,sigma,0)) -
                          gaussian_filter(b, (sigma,sigma,0))), axis=2) > 24
    components, count = label(delta)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    minimum = max(8, round(delta.size * .000002), math.ceil(float(sizes.max())*.03))
    keep = sizes >= minimum
    keep[0] = False
    delta = keep[components]
    info = {'method': 'difference_dilation_intersect_host', 'delta_threshold': 24, 'blur_sigma': sigma,
            'component_relative_min': .03,
            'component_min_pixels': minimum, 'difference_pixels': int(delta.sum()),
            'host_pixels': int(host.sum()), 'sam_fallback': 'not_configured'}
    if not delta.any():
        return None, {**info, 'reason': 'no_significant_difference'}
    ys, xs = np.where(delta)
    short = min(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
    radius = max(16, math.ceil(.3 * short))
    overlap = int((delta & host).sum())
    expanded = maximum_filter1d(delta, 2*radius+1, axis=0, mode='constant', cval=0)
    expanded = maximum_filter1d(expanded, 2*radius+1, axis=1, mode='constant', cval=0)
    site = expanded & host
    info.update(dilation_radius=radius, raw_overlap_pixels=overlap,
                site_pixels=int(site.sum()), host_fraction=float(site.sum()/max(1,host.sum())))
    # With virtually no direct contact evidence, a nearest patch would be an
    # invented anchor. The caller can supply a separately verified site mask.
    if require_raw_contact and overlap < max(8, math.ceil(.001 * int(delta.sum()))):
        return None, {**info, 'reason': 'difference_host_overlap_too_small'}
    if not site.any():
        return None, {**info, 'reason': 'site_not_resolved_from_host'}
    return site, info


def resolve_source_region(member, source_path, target_path, directory, *, site_resolver=None):
    """Resolve a member after generation; failure is explicitly ref-only.

    For chains, source_path is the step input (isolates this step's difference),
    while the host mask retains original-source coordinates and identity.
    """
    spec = member.get('instruction_spec') or {}
    if spec.get('version') != '2.0':
        raise ValueError('group generation requires instruction protocol 2.0')
    validate_noref(spec, member['instruction_noref'])
    member['mask_semantics'] = mask_contract(spec)
    member['instruction_protocol_version'] = '2.0'
    member['instruction_style_version'] = 'benchmark-concise-3-blind-1'
    unit = member.get('unit') or {}
    source = Image.open(source_path).convert('RGB')
    host_path = member.get('mask_v1') or member['mask']
    host = _mask(host_path, source.size)
    operation = spec['operation']
    mode = spec.get('remove_mode', 'whole')
    needs_site = operation == 'add'
    needs_part = bool(spec.get('target_part')) or (operation == 'remove' and mode != 'whole')
    selector, info = host, {'method': 'existing_actual_region'}
    if needs_site:
        target = Image.open(target_path).convert('RGB')
        selector, info = localize_add_site(source, target, host)
        named_site = bool(spec['placement'].get('site'))
        # Protocol 2.0 §6.2 prefers observed contact. A text segmenter often
        # cannot segment an abstract "top edge"; that is not contrary evidence
        # when the difference already identifies the host patch reliably.
        fallback = (selector is None and named_site and
                    info.get('reason') == 'difference_host_overlap_too_small' and
                    info.get('site_pixels', 0) > 0)
        if fallback and site_resolver is None and os.environ.get('SA1B_SITE_MODEL'):
            global _SITE_RESOLVER
            if _SITE_RESOLVER is None:
                from .site_segmentation import Sam3SiteResolver
                _SITE_RESOLVER = Sam3SiteResolver(os.environ['SA1B_SITE_MODEL'])
            site_resolver = _SITE_RESOLVER
        if fallback and site_resolver is not None:
            nearby_site, _ = localize_add_site(source, target, host, require_raw_contact=False)
            selector, segmentation = site_resolver(source, host, spec, nearby_site)
            # Even the fallback must be adjacent to actual changed pixels.
            if selector is not None:
                selector = selector & host & nearby_site
            info = {**info, 'segmentation': segmentation,
                    'sam_fallback': 'configured',
                    'method': 'sam3_site_constrained_difference',
                    'site_pixels': int(selector.sum()) if selector is not None else 0,
                    'host_fraction': float(selector.sum()/max(1,host.sum())) if selector is not None else 0}
        info['resolution_version'] = 'difference-first-3'
        if selector is None and unit.get('verified_site_mask_path'):
            selector = _mask(unit['verified_site_mask_path'], source.size) & host
            if not selector.any():
                selector = None
            else:
                info = {**info, 'method': 'verified_site_mask', 'fallback_verified': True}
    elif needs_part:
        path = unit.get('execution_mask_path') or member.get('execution_mask_path')
        if not path:
            selector, info = None, {'reason': 'resolved_part_mask_missing'}
        else:
            selector = _mask(path, source.size)
            if mode == 'contents':
                selector = selector | host
            info = {'method': 'container_union_contents' if mode == 'contents' else 'resolved_part_mask'}
    member['selector_granularity'] = 'part' if needs_site or needs_part else member.get('granularity', 'object')
    member['source_region_resolution'] = info
    member['source_region_review_required'] = bool(needs_site)
    if selector is None or not selector.any():
        member.update(blind_eligible=False, blind_ineligible_reason='part_mask_missing')
        if unit:
            unit.update(blind_eligible=False, blind_ineligible_reason='part_mask_missing')
        return False
    if needs_site or needs_part:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'source_region_v2.png'
        Image.fromarray(selector.astype(np.uint8)*255).save(path)
        ys, xs = np.where(selector)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
        point = precise_point(selector)
        for key in ('mask', 'bbox', 'point', 'area_fraction'):
            member.setdefault(key+'_v1', member.get(key))
        member.update(mask=str(path), bbox=bbox, point=point, area_fraction=float(selector.mean()))
        if unit:
            unit['mask_path_v1'] = unit.get('mask_path_v1') or unit['mask_path']
            unit['bbox_v1'] = unit.get('bbox_v1') or unit['bbox']
            unit['point_v1'] = unit.get('point_v1') or unit['point']
            unit.update(mask_path=str(path), bbox=bbox, point=point)
    dropped = spec.get('qualifier', '')
    from .instruction_protocol import blind_qualifier
    if dropped and not blind_qualifier(spec):
        member['qualifier_dropped'] = dropped
    member['blind_eligible'] = True
    member.pop('blind_ineligible_reason', None)
    if unit:
        unit.update(mask_semantics=member['mask_semantics'], blind_eligible=True)
    return True


def blind_group_check(group):
    members = group.get('members', group.get('units', []))
    eligible = bool(members) and all(m.get('blind_eligible') is True for m in members)
    sentences = [m['instruction_noref'] for m in members]
    same = 'twin_id' not in group or len(set(sentences)) == 1
    # Category identity is deliberately not a gate under blind instructions.
    comparable = 'twin_id' not in group or len({m.get('selector_granularity', m.get('granularity', 'object')) for m in members}) == 1
    areas = [m['area_fraction'] for m in members]
    ratio = max(areas)/min(areas) if areas and min(areas)>0 else float('inf')
    area_ok = 'twin_id' not in group or ratio <= 3
    return {'pass': bool(eligible and same and comparable and area_ok), 'all_regions_resolved': eligible,
            'shared_noref': same, 'comparable_granularity': comparable, 'final_area_ratio': ratio,
            'final_area_ratio_pass': area_ok}
