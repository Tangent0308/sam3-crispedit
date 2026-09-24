"""Bounded, evidence-driven late removal support; source labels stay immutable."""
import cv2
import numpy as np
from PIL import Image


def boundary_seam_alpha(source, raw, target, alpha, protected=None):
    """Choose a low-disagreement seam only inside the existing repair band.

    Graph-cut cannot restore the target, modify a retained entity, or enlarge
    the old writable envelope. It is a photometric seam, NOT object grounding.
    """
    target=np.asarray(target,dtype=bool)
    original=np.asarray(alpha.convert('L'))
    guard=np.zeros_like(target) if protected is None else np.asarray(protected,dtype=bool)&~target
    if source.size!=raw.size or target.shape!=original.shape or target.shape!=(source.height,source.width):
        raise ValueError('Aligned seam inputs required')
    allowed=(original>0)&~guard
    if np.any(target&~allowed):raise ValueError('Seam cannot exclude the execution target')
    if allowed.all() or not target.any():return alpha.copy()
    source_mask=(~target).astype(np.uint8)*255
    donor_mask=allowed.astype(np.uint8)*255
    masks=[cv2.UMat(source_mask),cv2.UMat(donor_mask)]
    finder=cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    finder.find([np.asarray(source.convert('RGB'),dtype=np.float32),
                 np.asarray(raw.convert('RGB'),dtype=np.float32)],[(0,0),(0,0)],masks)
    donor=(masks[1].get()>0)&allowed
    donor|=target
    smooth=cv2.GaussianBlur(donor.astype(np.float32),(5,5),.8)
    # Never increase opacity outside the previously permitted envelope.
    weight=np.minimum(smooth,original.astype(np.float32)/255)
    weight[target]=1.;weight[~allowed]=0.
    return Image.fromarray(np.rint(weight*255).astype(np.uint8))


def erased_removal_condition(crop, execution_mask, protected=None, policy='erase-neutral-v1'):
    """A bounded inpainting proposal, not a source label or accepted final edit.

    Remove semantic evidence of the selected instance before the generative
    completion. Keep protected neighbors and excluded holes; never box-fill.
    """
    mask=np.asarray(execution_mask,dtype=bool)
    if mask.shape != (crop.height,crop.width) or not mask.any():
        raise ValueError('Nonempty aligned execution mask required')
    guard=np.zeros_like(mask) if protected is None else np.asarray(protected,dtype=bool)&~mask
    if guard.shape != mask.shape:raise ValueError('Protected mask shape mismatch')
    # Small contour collar removes source edge/antialias traces, not semantic
    # attachment expansion. External accessories need a resolved plan/mask.
    support=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))).astype(bool)&~guard
    arr=np.asarray(crop.convert('RGB')).copy()
    if policy in {'erase-neutral-v1','erase-neutral-v2'}:
        arr[support]=127
    elif policy == 'erase-prefill-v1':
        arr=cv2.inpaint(arr,support.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
    else:raise ValueError('Unknown erased conditioning policy')
    return Image.fromarray(arr),dict(policy=policy,execution_pixels=int(mask.sum()),
        erased_pixels=int(support.sum()),protected_overlap_pixels=int((support&guard).sum()),
        source_mask_modified=False)


def adaptive_remove_support(source_crop, raw_crop, target, protected=None, policy='adaptive-remove-v1'):
    """Keep changed pixels touching the target inside a narrow repair band.

    This is a compositing support, NOT a semantic segmentation or a quality
    verdict. Never expand across a protected neighbor or to a distant edit.
    Untouched background and large excluded interiors remain excluded.
    """
    if policy not in {'adaptive-remove-v1', 'adaptive-remove-v2'}:
        raise ValueError('Unknown removal support policy')
    target=np.asarray(target,dtype=bool)
    if target.shape != (source_crop.height,source_crop.width) or raw_crop.size != source_crop.size:
        raise ValueError('Removal support inputs must be aligned')
    ys,xs=np.nonzero(target)
    if not len(xs):raise ValueError('Empty removal target')
    guard=np.zeros_like(target) if protected is None else np.asarray(protected,dtype=bool)&~target
    if guard.shape != target.shape:raise ValueError('Protected mask shape mismatch')
    span=min(np.ptp(xs)+1,np.ptp(ys)+1)
    radius=int(np.clip(span*.12,8,40))
    distance=cv2.distanceTransform((~target).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    a=cv2.GaussianBlur(np.asarray(source_crop,dtype=np.float32),(3,3),.6)
    b=cv2.GaussianBlur(np.asarray(raw_crop,dtype=np.float32),(3,3),.6)
    delta=np.abs(a-b).mean(axis=2)
    candidate=(delta>=12)&(distance<=radius)&~guard
    # Close tiny gaps in difference evidence, not holes in the source mask.
    candidate=cv2.morphologyEx(candidate.astype(np.uint8),cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
    candidate &= (distance<=radius)&~guard
    count,labels=cv2.connectedComponents((candidate|target).astype(np.uint8),connectivity=8)
    touching=np.unique(labels[target]);touching=touching[touching!=0]
    support=target|(candidate&np.isin(labels,touching))
    interior_added = 0
    interior_rejected = 0
    if policy == 'adaptive-remove-v2':
        # A distance collar can cut through a changed pocket between visible
        # target parts. Repair only small, supported interior pockets; never
        # turn the convex hull itself into a writable mask. No labels, object
        # names, source IDs, or instruction text participate in this rule.
        hull = np.zeros_like(target, dtype=np.uint8)
        points = np.column_stack((xs, ys)).astype(np.int32)
        cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
        changed = cv2.morphologyEx((delta >= 12).astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
        # Explicitly excluded large closed interiors may be independent
        # objects, not missing target fragments. Do not complete those holes.
        _, excluded_labels, excluded_stats, _ = cv2.connectedComponentsWithStats(
            (~target).astype(np.uint8), 8)
        exterior = np.unique(np.concatenate((excluded_labels[0], excluded_labels[-1],
            excluded_labels[:, 0], excluded_labels[:, -1])))
        large_holes = np.flatnonzero(excluded_stats[:, cv2.CC_STAT_AREA] > .10 * int(target.sum()))
        large_holes = np.setdiff1d(large_holes, np.append(exterior, 0))
        excluded_interiors = np.isin(excluded_labels, large_holes)
        pockets = changed & hull.astype(bool) & ~support & ~guard & ~excluded_interiors
        count, labels, stats, _ = cv2.connectedComponentsWithStats(pockets.astype(np.uint8), 8)
        contact = cv2.dilate(support.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        # Select whole components by label, without materializing one full
        # image-sized boolean array per speckle. Same rule, bounded memory.
        accepted = np.unique(labels[contact])
        accepted = accepted[(accepted != 0) &
            (stats[accepted, cv2.CC_STAT_AREA] <= .10 * int(target.sum()))]
        interior_rejected = count - 1 - len(accepted)
        proposed = int(stats[accepted, cv2.CC_STAT_AREA].sum())
        if proposed <= .15 * int(target.sum()):
            support |= np.isin(labels, accepted)
            interior_added = proposed
        else:
            interior_rejected += len(accepted)
    return support,dict(policy=policy,radius_pixels=radius,
        target_pixels=int(target.sum()),added_support_pixels=int((support&~target).sum()),
        interior_added_pixels=interior_added,interior_rejected_components=interior_rejected,
        protected_overlap_pixels=int((support&guard).sum()),difference_threshold=12,
        source_mask_modified=False,semantic_quality='not_checked')
