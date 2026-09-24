"""Official vLLM-Omni adapter for Qwen-Image-2.1 image-conditioned generation.

The adapter deliberately exposes the small ``pipe(image=..., prompt=...)``
surface used by the existing editor.  All prompt construction and late
composition stay in ``utils.context_edit``; vLLM-Omni only replaces the
diffusion execution backend.

Qwen-Image-2.1 support is currently provided by the vLLM-Omni PR #7759
branch.  That branch's official API is ``Omni.generate`` with
``build_image_to_image_prompt`` and ``OmniDiffusionSamplingParams``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import math

from PIL import Image


def _seed_from_generator(generator: Any) -> int | None:
    if generator is None:
        return None
    try:
        return int(generator.initial_seed())
    except (AttributeError, TypeError, RuntimeError):
        return None


def _floor32(value: int) -> int:
    """Match the official 2.1 serving path's multiple-of-32 output contract."""
    return max(32, (int(value) // 32) * 32)


class QwenImage21OmniPipeline:
    """Diffusers-like facade over the official offline vLLM-Omni API."""

    backend_name = "vllm-omni"
    is_qwen21 = True

    def __init__(self, model_id: str, *, tensor_parallel_size: int = 1, **kwargs: Any):
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.model_extras import get_model_class_name

        omni_kwargs = {
            "model": model_id,
            "diffusion_load_format": "dummy",
            "custom_pipeline_args": {
                "pipeline_class": "utils.qwen21_omni_regional.RegionalQwenImage21Pipeline"
            },
        }
        if tensor_parallel_size != 1:
            omni_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
        omni_kwargs.update(kwargs)
        self.omni = Omni(**omni_kwargs)
        self.model_id = model_id
        self.model_class_name = get_model_class_name(self.omni)

    @staticmethod
    def region_callback_kwargs(crop, mask, task_type, protected=None, protection_policy='legacy'):
        from utils.region_denoise import editable_token_weights

        # Same calculate_dimensions(1024**2, ratio) and 16x VAE as Diffusers.
        width = round(math.sqrt(1024 * 1024 * crop.width / crop.height) / 32) * 32
        height = round(math.sqrt(1024 * 1024 * crop.height / crop.width) / 32) * 32
        weights = editable_token_weights(
            mask, (width // 16, height // 16), task_type, protected, protection_policy)
        return dict(height=height, width=width, regional_weights=weights,
                    output_resolution=1024, use_kv_cache=True), weights

    def __call__(
        self,
        *,
        image: Image.Image | list[Image.Image],
        prompt: str,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 1.0,
        negative_prompt: str | None = None,
        generator: Any = None,
        height: int | None = None,
        width: int | None = None,
        regional_weights: Any = None,
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
    ) -> SimpleNamespace:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.model_extras import build_image_to_image_prompt

        images = image if isinstance(image, list) else [image]
        if output_resolution != 1024 or not use_kv_cache:
            raise ValueError('Official Omni backend currently requires 1024 resolution and KV cache enabled')
        if regional_weights is not None and (len(images) not in (1, 2) or
                any(im.size != images[0].size for im in images)):
            raise ValueError('Regional protection requires a clean image and at most one aligned guide')
        # The official Qwen-Image-2.1 recipe uses RGBA condition images.  Keep
        # the same semantic input for both one-image and guide-image paths.
        condition = [im.convert("RGBA") for im in images]
        condition_input: Image.Image | list[Image.Image]
        condition_input = condition[0] if len(condition) == 1 else condition

        prompt_dict = build_image_to_image_prompt(
            model_class_name=self.model_class_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=condition_input,
            height=_floor32(height) if height is not None else None,
            width=_floor32(width) if width is not None else None,
        )
        seed = _seed_from_generator(generator)
        sampling = OmniDiffusionSamplingParams(
            seed=seed,
            generator=generator,
            true_cfg_scale=float(true_cfg_scale),
            num_inference_steps=int(num_inference_steps),
            num_outputs_per_prompt=1,
            extra_args=({"regional_weights": regional_weights.tolist()}
                        if regional_weights is not None else {}),
            height=_floor32(height) if height is not None else None,
            width=_floor32(width) if width is not None else None,
        )
        outputs = self.omni.generate(prompt_dict, sampling_params_list=[sampling])
        if not outputs:
            raise RuntimeError("vLLM-Omni returned no request output")

        output_image = None
        for output in outputs:
            output_image = getattr(output, "images", None)
            if output_image:
                output_image = output_image[0]
                break
            request_output = getattr(output, "request_output", None)
            output_image = getattr(request_output, "images", None) if request_output else None
            if output_image:
                output_image = output_image[0]
                break
        if output_image is None:
            raise RuntimeError("vLLM-Omni request output contains no image")
        if not isinstance(output_image, Image.Image):
            output_image = Image.fromarray(output_image)
        return SimpleNamespace(images=[output_image])

    def close(self) -> None:
        close = getattr(self.omni, "close", None)
        if close is not None:
            close()
