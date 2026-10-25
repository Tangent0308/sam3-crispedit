"""Operation-specific editors with explicit construction-only writable masks.

Source ``mask_path`` always remains the benchmark selector. These adapters
never infer an add placement by erasing the entire host mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
from PIL import Image, ImageColor

from .editor import edit_prompt, region_support_mask, _feather


def read_binary_mask(path: str, size: tuple[int, int], *, allow_empty=False) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f'mask is not aligned to source: {path}')
        mask = np.asarray(image.convert('L')) > 127
    if not mask.any() and not allow_empty:
        raise ValueError(f'empty mask: {path}')
    return mask


def writable_mask(unit: dict[str, Any], size: tuple[int, int], *, require_add_placement=True) -> np.ndarray:
    source = read_binary_mask(unit['mask_path'], size)
    operation = unit['operation']
    path = unit.get('execution_mask_path')
    from .instruction_protocol import needs_local_execution
    spec = unit.get('instruction_spec') or {}
    partial = needs_local_execution(spec) and spec.get('remove_mode') != 'contents'
    relative_add = spec.get('placement', {}).get('anchor_mode') == 'reference'
    if (needs_local_execution(spec) or relative_add) and not path:
        raise ValueError('this protocol task requires a resolved execution_mask_path')
    if operation == 'add' and require_add_placement and not path:
        raise ValueError('add requires an explicit execution_mask_path for item, contact and shadow; host is only an anchor')
    if not path and operation == 'add' and unit.get('generation_contact'):
        from .edit_quality import contact_support
        support = contact_support(source, unit['generation_contact'])
    else:
        support = read_binary_mask(path, size) if path else region_support_mask(source, operation, unit.get('support_policy'))
    if operation == 'attribute' and (support & ~source).any():
        raise ValueError('attribute execution mask must be within the selected source region')
    if operation in {'remove', 'replace'} and not partial and (source & ~support).any():
        raise ValueError('remove/replace execution mask must cover the complete selected target')
    if partial and (not (source & support).any() or not (source & ~support).any()):
        raise ValueError('part execution mask must select a proper subregion of the host')
    if operation == 'add' and not relative_add and not (source & support).any():
        raise ValueError('add placement must contact the selected host')
    if unit.get('protected_mask_path'):
        protected = read_binary_mask(unit['protected_mask_path'], size, allow_empty=True)
        if (support & protected).any():
            raise ValueError('execution mask overlaps an explicitly protected region')
    return support


def crop_box(mask: np.ndarray, margin: float = .65) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('empty execution mask')
    pad = max(32, round(max(np.ptp(xs)+1, np.ptp(ys)+1)*margin))
    return max(0,int(xs.min())-pad), max(0,int(ys.min())-pad), min(mask.shape[1],int(xs.max())+pad+1), min(mask.shape[0],int(ys.max())+pad+1)


def composite(source: Image.Image, edited: Image.Image, support: np.ndarray, feather: int = 2) -> Image.Image:
    if source.size != edited.size or support.shape != (source.height, source.width):
        raise ValueError('composite sizes differ')
    alpha = Image.fromarray(np.rint(_feather(support, feather)*255).astype('uint8'))
    return Image.composite(edited.convert('RGB'), source.convert('RGB'), alpha)


class LamaEditor:
    backend_name = 'lama-torchscript'

    def __init__(self, model_id: str, *, device='cuda', max_side=1536):
        import torch
        if not Path(model_id).is_file():
            raise FileNotFoundError(f'LaMa TorchScript checkpoint required: {model_id}')
        self.torch = torch
        self.device = device
        self.model = torch.jit.load(model_id, map_location=device).eval()
        self.config = dict(backend=self.backend_name, model_id=str(Path(model_id).resolve()), max_side=max_side, compositor_version=3)
        self.max_side = max_side

    def prompt(self, unit):
        return ''  # LaMa is conditioned on image + mask, not on text.

    def support_mask(self, image, unit):
        if unit['operation'] != 'remove':
            raise ValueError('LaMa only supports remove')
        return writable_mask(unit, image.size)

    def generate(self, image_path, unit, seed=0):
        source = Image.open(image_path).convert('RGB')
        support = self.support_mask(source, unit)
        box = crop_box(support, 1.0)
        crop = source.crop(box)
        mask = Image.fromarray(support[box[1]:box[3],box[0]:box[2]].astype('uint8')*255)
        size = crop.size
        scale = min(1., self.max_side/max(size))
        work_size = (max(8,round(size[0]*scale)), max(8,round(size[1]*scale)))
        crop = crop.resize(work_size, Image.Resampling.LANCZOS)
        mask = mask.resize(work_size, Image.Resampling.NEAREST)
        rgb = np.asarray(crop,dtype='float32').transpose(2,0,1)/255
        m = (np.asarray(mask)>127).astype('float32')[None]
        pads=((0,0),(0,(-work_size[1])%8),(0,(-work_size[0])%8))
        rgb = np.pad(rgb,pads,mode='symmetric'); m=np.pad(m,pads,mode='symmetric')
        with self.torch.inference_mode():
            result = self.model(self.torch.from_numpy(rgb[None]).to(self.device),self.torch.from_numpy(m[None]).to(self.device))
        arr=result[0,:,:work_size[1],:work_size[0]].permute(1,2,0).cpu().numpy()
        patch=Image.fromarray(np.clip(arr*255,0,255).astype('uint8')).resize(size,Image.Resampling.LANCZOS)
        edited=source.copy();edited.paste(patch,box[:2])
        return composite(source,edited,support)


class FluxFillEditor:
    backend_name = 'flux-fill-diffusers'

    def __init__(self, model_id: str, *, device='cuda', steps=40, guidance=30., max_side=1024):
        import torch
        from diffusers import FluxFillPipeline
        self.torch = torch
        self.device = device
        self.pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        self.steps, self.guidance, self.max_side = steps, guidance, max_side
        self.config=dict(backend=self.backend_name,model_id=model_id,steps=steps,guidance=guidance,max_side=max_side,compositor_version=3)

    def support_mask(self, image, unit):
        if unit['operation'] not in {'add','replace','remove'}:
            raise ValueError('Flux Fill route is for add/replace/remove; attribute requires a separate policy')
        return writable_mask(unit,image.size)

    def prompt(self, unit):
        # Fill models describe the desired contents of the hole. Unit payload
        # supplies that content; the edit contract constrains scene consistency.
        if unit['operation']=='remove':
            return 'Natural continuation of the surrounding background, with the selected object absent. '+edit_prompt(unit)
        return 'A natural photograph showing '+unit['payload']+'. '+edit_prompt(unit)

    def generate(self, image_path, unit, seed=0):
        source=Image.open(image_path).convert('RGB');support=self.support_mask(source,unit)
        box=crop_box(support);crop=source.crop(box)
        mask=Image.fromarray(support[box[1]:box[3],box[0]:box[2]].astype('uint8')*255)
        scale=min(1.,self.max_side/max(crop.size))
        width=max(64,int(np.ceil(crop.width*scale/32))*32);height=max(64,int(np.ceil(crop.height*scale/32))*32)
        generated=self.pipe(prompt=self.prompt(unit),image=crop.resize((width,height),Image.Resampling.LANCZOS),
            mask_image=mask.resize((width,height),Image.Resampling.NEAREST),height=height,width=width,
            guidance_scale=self.guidance,num_inference_steps=self.steps,max_sequence_length=512,
            generator=self.torch.Generator(device=self.device).manual_seed(seed)).images[0]
        edited=source.copy();edited.paste(generated.convert('RGB').resize(crop.size,Image.Resampling.LANCZOS),box[:2])
        return composite(source,edited,support)


COLOR_ALIASES={'navy blue':'#23395d','light blue':'#8ab6d6','dark blue':'#234c80','deep blue':'#284d8b',
              'dark grey':'#555555','dark gray':'#555555','grey':'#808080','matte black':'#303030',
              'deep red':'#842d35','dark green':'#315843','warm yellow':'#d8bb68',
              # A photographed black surface still reflects light. Literal
              # RGB zero would multiply every shading/detail value by zero.
              'black':'#202020', 'white':'#dedede'}


def target_color(unit):
    spec = unit.get('instruction_spec') or {}
    if spec:
        attributes = spec.get('attributes', [])
        if (spec.get('operation') != 'attribute' or len(attributes) != 1
                or attributes[0].get('property') != 'color'):
            raise ValueError('recolor requires exactly one color attribute in instruction_spec')
        color_name = attributes[0].get('value', '')
    else:
        color_name = unit.get('payload', '')
    explicit=unit.get('target_rgb')
    if explicit is not None:
        arr=np.asarray(explicit,dtype=float)
        if arr.shape!=(3,) or not np.isfinite(arr).all() or (arr<0).any() or (arr>255).any():
            raise ValueError('target_rgb must contain three finite values in [0,255]')
        return arr/255.
    name=re.sub(r'^(?:a|an)\s+','',str(color_name).strip().lower()).strip()
    try:
        return np.asarray(ImageColor.getrgb(COLOR_ALIASES.get(name,name)),dtype=float)/255.
    except ValueError as exc:
        raise ValueError(f'unsupported color {name!r}; supply a reviewed target_rgb') from exc


def to_linear(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def to_srgb(rgb):
    rgb=np.maximum(rgb,0)
    return np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)


class RecolorEditor:
    """Deterministic diffuse-surface baseline, not an intrinsic decomposition model.

    Preserve the source luminance ratios and fine detail, then change the base
    chromaticity. Explicit eligibility avoids applying this approximation to
    glass, mirrors, patterned/mixed surfaces, or skin.
    """
    backend_name='recolor-luminance'

    def __init__(self):
        self.config=dict(backend=self.backend_name,version=4,compositor_version=3)

    def prompt(self,unit):
        return unit['instruction_ref']

    def support_mask(self,image,unit):
        if unit['operation']!='attribute' or unit.get('attribute_kind','color')!='color':
            raise ValueError('recolor only supports color attributes')
        if unit.get('surface_type')!='diffuse_uniform':
            raise ValueError('recolor requires reviewed surface_type=diffuse_uniform')
        return writable_mask(unit,image.size)

    def generate(self,image_path,unit,seed=0):
        source=Image.open(image_path).convert('RGB');support=self.support_mask(source,unit)
        rgb=np.asarray(source,dtype=float)/255.;linear=to_linear(rgb)
        luminance=linear@np.array([.2126,.7152,.0722])
        base=float(np.median(luminance[support]))
        if base<1e-5:
            raise ValueError('source surface is too dark for reliable recoloring')
        target=to_linear(target_color(unit))
        shading=luminance/max(base,1e-5)
        result=target[None,None,:]*shading[:,:,None]
        # This baseline does not estimate specularity or physical reflectance.
        result=np.clip(to_srgb(result),0,1)
        edited=Image.fromarray(np.rint(result*255).astype('uint8'))
        return composite(source,edited,support)


class IntrinsicRecolorEditor(RecolorEditor):
    """Research adapter: change diffuse albedo and preserve predicted lighting.

    Intrinsic's upstream code is academic-use-only. Mixed/printed surfaces still
    require reviewed execution masks; decomposition does not provide segmentation.
    """
    backend_name = 'intrinsic-recolor'

    def __init__(self, model_id='v2', *, device='cuda', max_side=1024):
        from intrinsic.pipeline import load_models, run_pipeline
        self.models = load_models(model_id, device=device)
        self.run_pipeline = run_pipeline
        self.device, self.max_side = device, max_side
        self.config = dict(backend=self.backend_name, version=4, compositor_version=3, model_id=model_id,
                           max_side=max_side, upstream_commit='1a93aafaaf7dbe11be7ec86323d9a06e5950966d')

    def generate(self, image_path, unit, seed=0):
        import cv2
        source = Image.open(image_path).convert('RGB')
        support = self.support_mask(source, unit)
        box = crop_box(support, 1.0)
        crop = source.crop(box)
        work = crop.copy()
        work.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
        pred = self.run_pipeline(self.models, np.asarray(work, dtype=np.float32)/255.,
                                 stage=3, resize_conf=None, device=self.device)
        alb = cv2.resize(pred['hr_alb'], crop.size, interpolation=cv2.INTER_LINEAR)
        local = support[box[1]:box[3], box[0]:box[2]]
        albedo_luma = alb @ np.array([.2126, .7152, .0722])
        base = float(np.median(albedo_luma[local]))
        if base < 1e-5:
            raise ValueError('intrinsic albedo too dark for reliable recoloring')
        # Upstream uses gamma 2.2 rather than the piecewise sRGB transfer.
        original = (np.asarray(crop, dtype=np.float32)/255.)**2.2
        # Recompute implied shading at source resolution. Signed diffuse
        # residuals amplified noise/edge artifacts in the first real probe.
        shading = original / np.maximum(alb, .01)
        exposure = float(np.median((shading @ np.array([.2126, .7152, .0722]))[local]))
        new_alb = (albedo_luma/base)[:, :, None] * (target_color(unit)**2.2)[None, None, :]
        result = np.clip(new_alb*shading/max(exposure, 1e-5), 0, 1)**(1/2.2)
        edited = source.copy()
        edited.paste(Image.fromarray(np.rint(result*255).astype('uint8')), box[:2])
        return composite(source, edited, support)
