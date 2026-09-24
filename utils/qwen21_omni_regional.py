"""Keep source-trajectory anchoring via Omni's official custom-pipeline API.

Only the scheduler output is extended. Prompt encoding, transformer, exact KV
cache, CUDA graphs and VAE execution remain the upstream implementations.
Request-level, single-image execution only: reject unsupported batching rather
than silently removing a quality protection.
"""
import logging

import torch
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

from utils.region_denoise import anchor_step

logger = logging.getLogger(__name__)


class RegionalQwenImage21Pipeline(QwenImage21Pipeline):
    def forward(self, req):
        if len(req.sampling_params_list) != 1:
            raise ValueError('Regional editor currently supports one request at a time')
        self._regional_weights = req.sampling_params_list[0].extra_args.get('regional_weights')
        self._regional_anchor = None
        self._regional_steps = 0
        try:
            result = super().forward(req)
            if self._regional_weights is not None:
                expected = req.sampling_params_list[0].num_inference_steps
                if self._regional_steps != expected:
                    raise RuntimeError(f'Regional anchor ran {self._regional_steps}/{expected} steps')
                logger.info('Regional source anchor applied at all %d steps', self._regional_steps)
            return result
        finally:
            self._regional_anchor = None
            self._regional_weights = None

    def _prepare_generation_context(self, **kwargs):
        ctx = super()._prepare_generation_context(**kwargs)
        if self._regional_weights is not None:
            # The baseline VAE uses mode/argmax, not stochastic sampling. The
            # clean image is encoded at exactly the output dimensions, so its
            # already computed condition latents are the same source anchor.
            reference, noise = ctx['image_latents'], ctx['latents']
            sizes = kwargs['per_request_images'][0]['input_image_sizes'] if 'per_request_images' in kwargs else None
            if sizes is not None and sizes[0] != (kwargs['width'], kwargs['height']):
                raise ValueError('Clean anchor image and output dimensions must match')
            if reference is not None and reference.shape[1] == noise.shape[1]*2:
                # Optional second condition is a location guide, never an
                # anchor: reinject only the clean first image's latents.
                reference = reference[:, :noise.shape[1]]
            if reference is None or reference.shape != noise.shape:
                raise ValueError('Source/output grids must match for regional anchoring')
            weight = torch.as_tensor(self._regional_weights, device=noise.device,
                                     dtype=noise.dtype).reshape(1, -1, 1)
            if weight.shape[1] != noise.shape[1]:
                raise ValueError('Regional weight grid does not match output latent grid')
            self._regional_anchor = (reference, noise.clone(), weight)
        return ctx

    def scheduler_step_maybe_with_cfg(self, *args, **kwargs):
        latents = super().scheduler_step_maybe_with_cfg(*args, **kwargs)
        if self._regional_anchor is not None:
            reference, noise, weight = self._regional_anchor
            sigma = self.scheduler.sigmas[self._regional_steps + 1].to(
                device=latents.device, dtype=latents.dtype)
            latents = anchor_step(latents, reference, noise, weight, sigma)
            self._regional_steps += 1
        return latents

    def prepare_encode(self, *args, **kwargs):
        raise NotImplementedError('Regional anchoring requires request-level execution, not step batching')
