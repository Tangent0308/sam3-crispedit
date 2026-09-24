"""Experimental single-branch regional denoising using the installed Qwen API.

Anchor context to the source's noise trajectory during diffusion, not only at
the final composite. The full target and a repair collar remain freely editable.
No outlines or color overlays are supplied to the generation model.
"""
import cv2
import numpy as np
import torch


def editable_token_weights(mask, token_size, task_type, protected=None, protection_policy='legacy'):
    if protection_policy not in {'legacy', 'guard-any-v1', 'guard-fraction-v1'}:
        raise ValueError('Unknown latent protection policy')
    inside=mask.astype(bool)
    ys,xs=np.nonzero(inside)
    if not len(xs):raise ValueError('Empty edit region')
    support=inside.copy()
    if task_type in {'add','replace'}:
        px=max(12,round((np.ptp(xs)+1)*.5));py=max(12,round((np.ptp(ys)+1)*.5))
        support[max(0,ys.min()-py):min(mask.shape[0],ys.max()+py+1),
                max(0,xs.min()-px):min(mask.shape[1],xs.max()+px+1)]=True
    radius=int(np.clip(min(np.ptp(xs)+1,np.ptp(ys)+1)*.15,12,64))
    support=cv2.dilate(support.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
    width,height=token_size
    # Area pooling plus one token of margin keeps thin structures editable.
    pooled=cv2.resize(support.astype(np.float32),(width,height),interpolation=cv2.INTER_AREA)>0
    pooled=cv2.dilate(pooled.astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32)
    if protected is not None:
        guard=cv2.resize((protected.astype(bool)&~inside).astype(np.float32),(width,height),interpolation=cv2.INTER_AREA)
        coverage=cv2.resize(inside.astype(np.float32),(width,height),interpolation=cv2.INTER_AREA)
        target=coverage>0
        if protection_policy == 'legacy':
            pooled[(guard>.5)&~target]=0
        else:
            # A thin known neighbor must not disappear merely because it
            # occupies less than half a token. Target-only tokens remain free.
            pooled[(guard>0)&~target]=0
            if protection_policy == 'guard-fraction-v1':
                mixed=target&(guard>0)
                pooled[mixed]*=coverage[mixed]/(coverage[mixed]+guard[mixed])
    return pooled


def anchor_step(latents,reference,noise,weight,sigma):
    """Flow-matching source trajectory at the *next* scheduler sigma."""
    source=(1-sigma)*reference+sigma*noise
    return latents*weight+source*(1-weight)


@torch.inference_mode()
def region_callback_kwargs(pipe,crop,mask,task_type,generator,protected=None,protection_policy='legacy'):
    if getattr(pipe, 'backend_name', None) == 'vllm-omni':
        return pipe.region_callback_kwargs(crop, mask, task_type, protected, protection_policy)
    if pipe.__class__.__name__ == 'QwenImage21Pipeline':
        return qwen21_region_callback_kwargs(
            pipe,crop,mask,task_type,generator,protected,protection_policy)
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions,VAE_IMAGE_SIZE
    width,height=calculate_dimensions(VAE_IMAGE_SIZE,crop.width/crop.height)
    multiple=pipe.vae_scale_factor*2
    width=int(width)//multiple*multiple;height=int(height)//multiple*multiple
    device=pipe._execution_device;dtype=pipe.transformer.dtype
    tensor=pipe.image_processor.preprocess(crop,height=height,width=width).unsqueeze(2)
    noise,reference=pipe.prepare_latents([tensor],1,pipe.transformer.config.in_channels//4,
                                       height,width,dtype,device,generator)
    if noise.shape!=reference.shape:raise ValueError('Source and generated latent grids differ')
    weights=editable_token_weights(mask,(width//multiple,height//multiple),task_type,protected,protection_policy)
    weight=torch.from_numpy(weights.reshape(1,-1,1)).to(device=device,dtype=dtype)
    if weight.shape[1]!=noise.shape[1]:raise ValueError('Mask token grid mismatch')
    def callback(pipeline,step,timestep,kwargs):
        sigma=pipeline.scheduler.sigmas[step+1].to(device=device,dtype=dtype)
        kwargs['latents']=anchor_step(kwargs['latents'],reference,noise,weight,sigma)
        return kwargs
    return dict(latents=noise,callback_on_step_end=callback,height=height,width=width),weights


@torch.inference_mode()
def qwen21_region_callback_kwargs(pipe,crop,mask,task_type,generator,protected=None,protection_policy='legacy'):
    """Build the same source-trajectory anchor on Qwen-Image-2.1 latents.

    Qwen-Image-2.1 has a 16x spatial VAE, 64-channel unpatched latent tokens,
    and a unified condition/generation pipeline.  The official pipeline still
    accepts initial latents and an end-of-step callback, but the old 2511
    `prepare_latents` layout is not compatible with it.
    """
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    from diffusers.utils.torch_utils import randn_tensor

    width,height,_=calculate_dimensions(1024*1024,crop.width/crop.height)
    multiple=pipe.vae_scale_factor*2
    width=int(width)//multiple*multiple;height=int(height)//multiple*multiple
    device=pipe._execution_device;dtype=pipe.transformer.dtype
    # The official pipeline converts condition images to RGBA before this same
    # preprocessing path.  Match it so the reference target trajectory has the
    # exact 2.1 latent format.
    tensor=pipe.image_processor.preprocess(
        crop.convert('RGBA'),height=height,width=width).unsqueeze(2)
    tensor=tensor.to(device=device,dtype=dtype)
    reference=pipe._encode_vae_image(tensor,generator)
    latent_height,latent_width=reference.shape[3:]
    channels=pipe.transformer.config.in_channels
    reference=pipe._pack_latents(
        reference,1,channels,latent_height,latent_width)
    noise=randn_tensor(
        (1,1,channels,latent_height,latent_width),generator=generator,
        device=device,dtype=dtype)
    noise=pipe._pack_latents(noise,1,channels,latent_height,latent_width)
    if noise.shape!=reference.shape:
        raise ValueError('Qwen-Image-2.1 source and generated latent grids differ')
    weights=editable_token_weights(
        mask,(latent_width,latent_height),task_type,protected,protection_policy)
    weight=torch.from_numpy(weights.reshape(1,-1,1)).to(
        device=device,dtype=dtype)
    if weight.shape[1]!=noise.shape[1]:
        raise ValueError('Qwen-Image-2.1 mask token grid mismatch')
    def callback(pipeline,step,timestep,kwargs):
        sigma=pipeline.scheduler.sigmas[step+1].to(device=device,dtype=dtype)
        kwargs['latents']=anchor_step(
            kwargs['latents'],reference,noise,weight,sigma)
        return kwargs
    return dict(
        latents=noise,callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=['latents'],
        height=height,width=width,output_resolution=1024,use_kv_cache=True),weights
