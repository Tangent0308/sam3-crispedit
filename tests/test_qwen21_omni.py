"""Lightweight API-contract tests: no Omni installation or GPU required."""
import sys
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from utils.qwen21_omni import QwenImage21OmniPipeline
from utils.region_denoise import editable_token_weights, region_callback_kwargs


def test_region_weights_and_resolution_match_baseline():
    image = Image.new('RGB', (513, 681))
    mask = np.zeros((681, 513), bool)
    mask[280:340, 230:300] = True
    pipe = object.__new__(QwenImage21OmniPipeline)
    kwargs, weights = region_callback_kwargs(pipe, image, mask, 'remove', None)
    assert (kwargs['width'], kwargs['height']) == (896, 1184)
    assert weights.shape == (74, 56)
    np.testing.assert_array_equal(weights, editable_token_weights(mask, (56, 74), 'remove'))


def test_sampling_preserves_anchor_and_rejects_silently_ignored_options(monkeypatch):
    captured = {}
    inputs = ModuleType('vllm_omni.inputs.data')
    inputs.OmniDiffusionSamplingParams = lambda **kw: SimpleNamespace(**kw)
    extras = ModuleType('vllm_omni.model_extras')
    def build(**kwargs):
        captured['prompt'] = kwargs
        return kwargs
    extras.build_image_to_image_prompt = build
    monkeypatch.setitem(sys.modules, inputs.__name__, inputs)
    monkeypatch.setitem(sys.modules, extras.__name__, extras)
    image = Image.new('RGBA', (64, 96))
    def generate(prompt, sampling_params_list):
        captured['sampling'] = sampling_params_list[0]
        return [SimpleNamespace(images=[image])]
    pipe = object.__new__(QwenImage21OmniPipeline)
    pipe.model_class_name = 'QwenImage21Pipeline'
    pipe.omni = SimpleNamespace(generate=generate)
    weights = np.ones((6, 4), np.float32)
    assert pipe(image=image, prompt='Remove the target.', regional_weights=weights,
                height=96, width=64).images[0] is image
    params = captured['sampling']
    assert params.num_inference_steps == 40
    assert params.true_cfg_scale == 1
    assert params.extra_args['regional_weights'] == weights.tolist()
    assert 'extra_args' not in captured['prompt']
    with pytest.raises(TypeError):
        pipe(image=image, prompt='test', callback_on_step_end=lambda: None)
    with pytest.raises(ValueError):
        pipe(image=image, prompt='test', use_kv_cache=False)


def test_no_implicit_crop_size_override(monkeypatch):
    # No regional callback => official aspect-based ~1MP default, not crop pixels.
    captured = {}
    inputs = ModuleType('vllm_omni.inputs.data')
    inputs.OmniDiffusionSamplingParams = lambda **kw: SimpleNamespace(**kw)
    extras = ModuleType('vllm_omni.model_extras')
    extras.build_image_to_image_prompt = lambda **kw: kw
    monkeypatch.setitem(sys.modules, inputs.__name__, inputs)
    monkeypatch.setitem(sys.modules, extras.__name__, extras)
    pipe = object.__new__(QwenImage21OmniPipeline)
    pipe.model_class_name = 'QwenImage21Pipeline'
    def generate(prompt, sampling_params_list):
        captured['params'] = sampling_params_list[0]
        return [SimpleNamespace(images=[Image.new('RGB', (1024, 1024))])]
    pipe.omni = SimpleNamespace(generate=generate)
    pipe(image=Image.new('RGB', (200, 300)), prompt='test')
    assert captured['params'].width is None
    assert captured['params'].height is None


def test_regional_scheduler_matches_original_anchor_at_every_step(monkeypatch):
    import torch
    from utils.region_denoise import anchor_step

    upstream = ModuleType('vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21')
    reference = torch.tensor([[[2.], [3.], [4.]]])
    noise = torch.tensor([[[.1], [.2], [.3]]])
    class Base:
        def _prepare_generation_context(self, **kwargs):
            return {'image_latents': reference, 'latents': noise}
        def scheduler_step_maybe_with_cfg(self, latents):
            return latents
    upstream.QwenImage21Pipeline = Base
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    path = Path(__file__).resolve().parents[1] / 'utils/qwen21_omni_regional.py'
    spec = importlib.util.spec_from_file_location('regional_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipe = module.RegionalQwenImage21Pipeline()
    pipe._regional_weights = [[0., .5, 1.]]
    pipe._regional_steps = 0
    pipe.scheduler = SimpleNamespace(sigmas=torch.tensor([1., .7, .2, 0.]))
    pipe._prepare_generation_context()
    weights = torch.tensor([[[0.], [.5], [1.]]])
    for sigma in pipe.scheduler.sigmas[1:]:
        output = torch.ones_like(noise)
        torch.testing.assert_close(pipe.scheduler_step_maybe_with_cfg(output),
                                   anchor_step(output, reference, noise, weights, sigma))
    assert pipe._regional_steps == 3
    with pytest.raises(NotImplementedError):
        pipe.prepare_encode()
