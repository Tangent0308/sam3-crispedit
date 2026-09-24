"""Match removal-fill lighting to the retained source without widening its mask."""
import time
import cv2
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve


def harmonize_removal(source, raw, alpha, max_side=192, max_correction=16.):
    """Extend reliable exterior RGB offsets harmonically into the writable area.

    Only a smooth offset changes; raw texture/geometry is retained. The original
    composition alpha remains authoritative, including protected holes. Downsampled
    estimates exclude source target pixels to avoid transferring the deleted object.
    """
    started = time.perf_counter()
    src = np.asarray(source.convert('RGB'), dtype=np.float32)
    edit = np.asarray(raw.convert('RGB'), dtype=np.float32)
    weight = np.asarray(alpha.convert('L'), dtype=np.float32) / 255.
    if src.shape != edit.shape or weight.shape != src.shape[:2]:
        raise ValueError('Aligned RGB images and composition alpha required')
    writable = weight > 0
    meta = dict(method='harmonic-offset-v1', applied=False)
    if not writable.any() or writable.all():
        return raw.copy(), {**meta, 'reason': 'no_exterior_boundary'}
    h, w = weight.shape
    scale = min(1., max_side / max(h, w))
    size = (max(2, round(w*scale)), max(2, round(h*scale)))
    diff = src-edit
    valid = (~writable) & (np.max(np.abs(diff), axis=2) < 40)
    # Exclude a small edge band from measurements (resampling/object halos).
    valid &= cv2.erode((~writable).astype('uint8'), np.ones((3,3),'uint8')) > 0
    sigma = max(2., 2./scale)
    density = cv2.GaussianBlur(valid.astype('float32'), (0,0), sigma)
    estimates = np.stack([cv2.GaussianBlur(diff[:,:,c]*valid, (0,0), sigma)
                         / np.maximum(density, 1e-6) for c in range(3)], axis=-1)
    coarse = cv2.resize(estimates, size, interpolation=cv2.INTER_AREA)
    reliable = cv2.resize(density, size, interpolation=cv2.INTER_AREA) > .15
    occupied = cv2.resize(writable.astype('float32'), size,
                          interpolation=cv2.INTER_AREA) > 0
    occupied = cv2.dilate(occupied.astype('uint8'), np.ones((3,3),'uint8')) > 0
    unknown = occupied | ~reliable
    if unknown.all():
        return raw.copy(), {**meta, 'reason': 'insufficient_reliable_boundary'}
    coarse = np.clip(coarse, -max_correction, max_correction)
    ids = np.full(unknown.shape, -1, dtype=np.int32)
    yy, xx = np.nonzero(unknown); n=len(yy)
    ids[yy,xx] = np.arange(n)
    rows=[]; cols=[]; values=[]; rhs=np.zeros((n,3), dtype=np.float64)
    degree=np.zeros(n, dtype=np.float64)
    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
        ny,nx=yy+dy,xx+dx
        valid_edge=(ny>=0)&(ny<unknown.shape[0])&(nx>=0)&(nx<unknown.shape[1])
        index=np.nonzero(valid_edge)[0]; ny,nx=ny[valid_edge],nx[valid_edge]
        degree[index]+=1
        adjacent=ids[ny,nx]; internal=adjacent>=0
        rows.extend(index[internal]);cols.extend(adjacent[internal]);values.extend([-1.]*int(internal.sum()))
        rhs[index[~internal]]+=coarse[ny[~internal],nx[~internal]]
    rows.extend(range(n));cols.extend(range(n));values.extend(degree)
    system=sparse.csr_matrix((values,(rows,cols)),shape=(n,n))
    field=coarse.copy()
    if n:field[yy,xx]=spsolve(system,rhs)
    correction=cv2.resize(field,(w,h),interpolation=cv2.INTER_LINEAR)
    corrected=np.clip(np.rint(edit+correction),0,255).astype('uint8')
    meta.update(applied=True, mean_abs_correction=float(np.abs(correction[writable]).mean()),
                max_abs_correction=float(np.abs(correction[writable]).max()),
                seconds=time.perf_counter()-started)
    return Image.fromarray(corrected),meta


def compose_harmonized_removal(source, raw, alpha):
    corrected, meta=harmonize_removal(source,raw,alpha)
    return Image.composite(corrected,source,alpha),meta


def correct_boundary_band(source, raw, alpha, band_width=12, max_correction=24.):
    """Native-resolution screened offset in a bounded INNER seam band.

    Complements the coarse lighting field without diffusing source texture or
    target pixels into the fill. Core raw geometry and outside pixels stay fixed.
    """
    src=np.asarray(source.convert('RGB'),dtype=np.float32)
    donor=np.asarray(raw.convert('RGB'),dtype=np.float32)
    writable=np.asarray(alpha.convert('L'))>0
    if src.shape!=donor.shape or writable.shape!=src.shape[:2]:raise ValueError('Aligned boundary inputs required')
    if not writable.any() or writable.all():return raw.copy()
    distance=cv2.distanceTransform(writable.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    band=writable&(distance<=band_width)
    yy,xx=np.nonzero(band);n=len(yy)
    if not n:return raw.copy()
    ids=np.full(writable.shape,-1,np.int32);ids[yy,xx]=np.arange(n)
    residual=src-donor
    reliable=(~writable)&(np.abs(residual).max(axis=2)<40)
    boundary=np.clip(residual,-max_correction,max_correction)*reliable[:,:,None]
    rows=[];cols=[];values=[];rhs=np.zeros((n,3));degree=np.full(n,.04)
    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
        ny,nx=yy+dy,xx+dx;valid=(ny>=0)&(ny<writable.shape[0])&(nx>=0)&(nx<writable.shape[1])
        index=np.flatnonzero(valid);ny,nx=ny[valid],nx[valid];degree[index]+=1
        neighbors=ids[ny,nx];internal=neighbors>=0
        rows.extend(index[internal]);cols.extend(neighbors[internal]);values.extend([-1.]*int(internal.sum()))
        rhs[index[~internal]]+=boundary[ny[~internal],nx[~internal]]
    rows.extend(range(n));cols.extend(range(n));values.extend(degree)
    matrix=sparse.csr_matrix((values,(rows,cols)),shape=(n,n))
    corrected=donor.copy();corrected[yy,xx]+=np.clip(spsolve(matrix,rhs),-max_correction,max_correction)
    return Image.fromarray(np.clip(np.rint(corrected),0,255).astype(np.uint8))
