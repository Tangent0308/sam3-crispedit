"""Qwen-Image-2.1 target generation with checkpoint-safe local edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import load_mask
from .io import read_jsonl, write_jsonl


def edit_prompt(unit: dict[str, Any], *, target_hint: str | None = None,
                guided: bool = False) -> str:
    locator = ""
    if target_hint:
        locator = f" The selected target is located at {target_hint} in the provided crop."
    guide = (
        " Image 2 is an aligned location guide with a thin black/white boundary; use it only "
        "to locate the target, never copy the boundary or any annotation into the result."
        if guided else ""
    )
    if unit["operation"] == "remove":
        return (
            f"{unit['instruction_ref']} Completely erase the specified target from the image. "
            "Reconstruct the newly exposed background using the surrounding texture, perspective, lighting and occlusion; "
            "leave no recolored silhouette, outline, placeholder, cyan/gray patch or duplicate. "
            "Keep every unrequested object, instance, part, boundary, occluder and background unchanged."
            + locator + guide
        )
    return (
        f"{unit['instruction_ref']} Render a natural photographic result. "
        "Keep every unrequested object, instance, part, boundary, occluder, lighting cue and background unchanged; "
        "make only physically necessary local attachment, revealed-background, shadow and contact changes. "
        "Preserve the target silhouette and photographic texture for attribute edits; for replace edits, "
        "remove the old identity completely; for add edits, make the new item clearly visible and physically attached."
        + locator + guide
    )


class DiffusersQwenImageEditor:
    """Lazy diffusers adapter; generation is isolated so it can resume per unit."""

    def __init__(self, model_id: str, *, device: str = "cuda", dtype: str = "bf16", steps: int = 40,
                 tensor_parallel_size: int = 1):
        import torch
        from utils.qwen_pipeline_loader import load_qwen21_omni_pipeline
        self.torch = torch
        torch_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                       "fp16": torch.float16, "float16": torch.float16,
                       "fp32": torch.float32, "float32": torch.float32}[dtype]
        self.pipe = load_qwen21_omni_pipeline(model_id, tensor_parallel_size=tensor_parallel_size)
        self.steps = steps

    def generate(self, image_path: str, unit: dict[str, Any], seed: int = 0) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        mask = load_mask({"mask_path": unit["mask_path"]}, image.size)
        ys, xs = mask.nonzero()
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        margin = max(16, round(max(x1 - x0, y1 - y0) * 0.45))
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(image.width, x1 + margin), min(image.height, y1 + margin)
        crop = image.crop((x0, y0, x1, y1))
        local = mask[y0:y1, x0:x1]
        target_hint = _target_hint(local)
        guided = hasattr(self.pipe, "region_callback_kwargs")
        generator = self.torch.Generator(device="cuda").manual_seed(seed)
        # The Omni regional adapter anchors the editable latent tokens itself.
        # Passing a second boundary image through the generic Qwen image list
        # is ambiguous for the model and can be copied as a cyan/white patch,
        # so keep the condition image to the clean source crop.
        prompt = edit_prompt(unit, target_hint=target_hint, guided=False)
        call_kwargs = dict(
            prompt=prompt, negative_prompt=" ", true_cfg_scale=1.0,
            num_inference_steps=self.steps, generator=generator,
        )
        if guided:
            # The official Qwen-Image-2.1 Omni adapter accepts regional latent
            # weights. They keep the clean source trajectory outside the
            # editable support without feeding a second boundary image that
            # the generator may copy into the result.
            regional, _weights = self.pipe.region_callback_kwargs(
                crop, local, unit["operation"], None, "guard-any-v1"
            )
            call_kwargs.update(regional)
            call_kwargs["image"] = crop
        else:
            # Fallback for a plain Diffusers-compatible pipeline.
            scale = min(1.0, (1024 * 1024 / (crop.width * crop.height)) ** 0.5)
            gen_crop = crop.resize((max(32, round(crop.width * scale / 32) * 32),
                                    max(32, round(crop.height * scale / 32) * 32))) if scale < 1.0 else crop
            call_kwargs.update(image=gen_crop, guidance_scale=1.0,
                               height=gen_crop.height, width=gen_crop.width)
        result = self.pipe(**call_kwargs)
        raw = result.images[0]
        if raw.mode == "RGBA":
            layer = raw.resize(crop.size, Image.Resampling.LANCZOS)
            edited_crop = Image.alpha_composite(crop.convert("RGBA"), layer).convert("RGB")
        else:
            edited_crop = raw.convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
        # Use a bounded operation-specific writable support. Attribute remains
        # on the original mask; add/replace get a collar for new silhouettes;
        # remove gets only a narrow boundary repair band.
        radius = _support_radius(local, unit["operation"])
        expanded = _dilate(local, radius) if radius else local
        alpha = Image.fromarray((_feather(expanded, max(2, radius // 3)) * 255).astype("uint8"), mode="L")
        output = image.copy()
        output.paste(edited_crop, (x0, y0), alpha)
        return output


def _target_hint(mask: np.ndarray) -> str:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty target mask")
    return (
        f"x={xs.min() / mask.shape[1]:.0%}..{(xs.max() + 1) / mask.shape[1]:.0%}, "
        f"y={ys.min() / mask.shape[0]:.0%}..{(ys.max() + 1) / mask.shape[0]:.0%}"
    )


def _target_guide(crop: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create a clean aligned source guide with a thin target boundary."""
    edge = mask.copy()
    # Four-neighbour erosion without an OpenCV dependency.
    padded = np.pad(mask, 1, mode="constant")
    eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
              & padded[1:-1, :-2] & padded[1:-1, 2:])
    edge &= ~eroded
    array = np.asarray(crop.convert("RGB")).copy()
    array[edge] = (255, 255, 255)
    return Image.fromarray(array, mode="RGB")


def _support_radius(mask: np.ndarray, operation: str) -> int:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty target mask")
    extent = max(1, min(np.ptp(xs) + 1, np.ptp(ys) + 1))
    if operation == "attribute":
        return 0
    if operation in {"add", "replace"}:
        return int(np.clip(round(extent * 0.22), 8, 48))
    return int(np.clip(round(extent * 0.06), 3, 16))


def _dilate(mask, radius: int):
    import numpy as np
    if radius <= 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), radius, mode="constant")
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return result


def _feather(mask, radius: int):
    import numpy as np
    if radius <= 0:
        return mask.astype(float)
    # Distance-to-background approximation from repeated erosions. This is
    # deterministic and avoids introducing an OpenCV dependency in generation.
    value = mask.astype(float)
    eroded = mask.astype(bool)
    for step in range(radius):
        padded = np.pad(eroded, 1, mode="constant")
        eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                  & padded[1:-1, :-2] & padded[1:-1, 2:])
        value[mask & ~eroded] = np.minimum(value[mask & ~eroded], 1.0 - (step + 1) / (radius + 1))
    return value


def generate_cases(manifest: Path, output_dir: Path, editor: DiffusersQwenImageEditor, seed: int = 0) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for case in read_jsonl(manifest):
        source = case["source"]["image_path"]
        for unit in case["units"]:
            case_id = f"{case['id']}--{unit['unit_id']}"
            path = output_dir / f"{case_id}.png"
            if not path.exists():
                editor.generate(source, unit, seed=seed).save(path)
            outputs.append({"case_id": case["id"], "unit_id": unit["unit_id"], "operation": unit["operation"],
                            "source_id": case["source"].get("source_id"),
                            "source_image": source, "edited_image": str(path), "instruction": unit["instruction_ref"],
                            "instruction_noref": unit.get("instruction_noref"), "mask": unit["mask_path"],
                            "ref_phrase": unit.get("ref_phrase"), "scope": unit.get("scope"), "payload": unit.get("payload"),
                            "bbox": unit.get("bbox"), "point": unit.get("point"),
                            "difficulty": case.get("difficulty", {}), "source_provenance": case["source"].get("provenance", {}),
                            "prompt": edit_prompt(unit), "seed": seed})
    write_jsonl(output_dir / "generated.jsonl", outputs)
    return outputs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    generate_cases(a.manifest, a.output_dir, DiffusersQwenImageEditor(a.model_id, device=a.device, dtype=a.dtype, steps=a.steps, tensor_parallel_size=a.tensor_parallel_size), a.seed)


if __name__ == "__main__":
    main()
