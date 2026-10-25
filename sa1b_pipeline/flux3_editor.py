"""Experimental FLUX 3 Image Klein editor using native vLLM-Omni.

Requires the pinned FLUX 3 Omni branch in an isolated CUDA-compatible runtime.
An externally generated full-scene caption is mandatory; it is construction
metadata, never a replacement for the public Ref/no-ref pair.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .editor import _feather, _support_radius
from .specialist_editors import writable_mask, read_binary_mask


class Flux3OmniEditor:
    backend_name = "flux3-native-vllm-omni"

    def __init__(self, model_id: str, *, steps: int = 16):
        os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        self.params_class = OmniDiffusionSamplingParams
        self.steps = steps
        self.omni = Omni(
            model=model_id, enforce_eager=True, cache_backend="none",
            init_timeout=1800, stage_init_timeout=1800,
        )
        self.config = dict(
            backend=self.backend_name, model_id=model_id, steps=steps,
            prompt_version="full-scene-caption-1", compositor_version=3,
            approximate_cache=False, experimental=True,
        )

    @staticmethod
    def prompt(unit):
        caption = str(unit.get("generation_caption", "")).strip()
        if not caption:
            raise ValueError("FLUX 3 requires a separately recorded full-scene generation_caption")
        return caption

    def generate(self, image_path, unit, seed=0):
        from .instruction_protocol import validate_pair
        if unit.get("instruction_spec"):
            errors = validate_pair(unit["instruction_spec"], unit["instruction_ref"],
                                   unit["instruction_noref"])
            if errors:
                raise ValueError(errors)
        caption = self.prompt(unit)
        source = Image.open(image_path).convert("RGB")
        mask = read_binary_mask(unit["mask_path"], source.size)
        support = writable_mask(unit, source.size)
        ys, xs = np.nonzero(mask | support)
        margin = max(16, round(max(np.ptp(xs) + 1, np.ptp(ys) + 1) * .45))
        box = unit.get("generation_crop_bbox") or (
            max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(source.width, int(xs.max()) + margin + 1),
            min(source.height, int(ys.max()) + margin + 1),
        )
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
            raise ValueError("invalid generation crop")
        covered = np.zeros_like(support)
        covered[y0:y1, x0:x1] = True
        if (support & ~covered).any():
            raise ValueError("generation crop must cover writable support")
        crop = source.crop(box)
        width = max(32, round(math.sqrt(1024 * 1024 * crop.width / crop.height) / 32) * 32)
        height = max(32, round(math.sqrt(1024 * 1024 * crop.height / crop.width) / 32) * 32)
        params = self.params_class(
            height=height, width=width, num_inference_steps=self.steps, seed=seed,
        )
        outputs = self.omni.generate(
            {"prompt": caption, "multi_modal_data": {"image": [crop]}}, params,
        )
        if len(outputs) != 1 or not outputs[0].images:
            raise RuntimeError("FLUX 3 did not return one image")
        raw = outputs[0].images[0].convert("RGB")
        edited = raw.resize(crop.size, Image.Resampling.LANCZOS)
        local = mask[y0:y1, x0:x1]
        local_support = support[y0:y1, x0:x1]
        alpha_path = unit.get("generation_alpha_path")
        if alpha_path:
            alpha = Image.open(alpha_path).convert("L")
            if alpha.size != crop.size or (np.asarray(alpha)[~local_support] > 0).any():
                raise ValueError("comparison alpha must be aligned and bounded by support")
        else:
            radius = _support_radius(local, unit["operation"], unit.get("support_policy"))
            alpha = Image.fromarray(
                (_feather(local_support, max(2, radius // 3)) * 255).astype("uint8"))
        result = source.copy()
        result.paste(edited, (x0, y0), alpha)
        # Hard preservation is independently checkable, not a model guarantee.
        if not np.array_equal(np.asarray(result)[~support], np.asarray(source)[~support]):
            raise RuntimeError("FLUX composite changed pixels outside execution support")
        self.last_generation_context = dict(
            editor=self.config, model_prompt=caption, crop_bbox=list(box),
            native_output_size=list(raw.size), outside_support_exact=True,
            public_instruction_unchanged=True,
        )
        if unit.get("diagnostic_dir"):
            root = Path(unit["diagnostic_dir"])
            root.mkdir(parents=True, exist_ok=True)
            for name, im in (("source_crop", crop), ("raw_crop", raw),
                             ("resized_crop", edited), ("alpha", alpha)):
                im.save(root / (name + ".png"))
        return result

    def close(self):
        self.omni.close()
