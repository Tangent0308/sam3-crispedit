"""Qwen-Image-2.1 target generation with checkpoint-safe local edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .geometry import load_mask
from .io import read_jsonl, write_jsonl


def edit_prompt(unit: dict[str, Any]) -> str:
    if unit["operation"] == "remove":
        return (
            f"{unit['instruction_ref']} Completely erase the specified target from the image. "
            "Reconstruct the newly exposed background using the surrounding texture, perspective, lighting and occlusion; "
            "leave no recolored silhouette, outline, placeholder, cyan/gray patch or duplicate. "
            "Keep every unrequested object, instance, part, boundary, occluder and background unchanged."
        )
    return (
        f"{unit['instruction_ref']} Render a natural photographic result. "
        "Keep every unrequested object, instance, part, boundary, occluder, lighting cue and background unchanged; "
        "make only physically necessary local attachment, revealed-background, shadow and contact changes."
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
        # Qwen-Image-2.1's supported 1024-pixel recipe is much cheaper and
        # more stable than sending full SA-1B panoramas to Omni. Resize only
        # the generation canvas; the final composite returns to source size.
        scale = min(1.0, (1024 * 1024 / (crop.width * crop.height)) ** 0.5)
        gen_crop = crop.resize((max(32, round(crop.width * scale / 32) * 32),
                                max(32, round(crop.height * scale / 32) * 32))) if scale < 1.0 else crop
        # For removal, hide the target before asking Qwen to reconstruct the
        # exposed background.  Sending the untouched target back as context
        # makes the diffusion edit preserve it or recolor it into a mask-like
        # artifact, especially for small text and thin structures.
        if unit["operation"] == "remove":
            gen_input = _erase_target(crop, mask[y0:y1, x0:x1])
            if gen_input.size != gen_crop.size:
                gen_input = gen_input.resize(gen_crop.size)
        else:
            gen_input = gen_crop
        generator = self.torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(image=gen_input, prompt=edit_prompt(unit), negative_prompt=" ", true_cfg_scale=1.0,
                           guidance_scale=1.0, num_inference_steps=self.steps, generator=generator,
                           height=gen_input.height, width=gen_input.width)
        edited_crop = result.images[0].convert("RGB").resize(crop.size)
        # Feather an expanded region to retain contact shadows and prevent
        # local edits from producing a rectangular paste seam. Add/replace
        # receive a larger footprint because new content may extend beyond
        # the source support mask; attribute stays tightly scoped.
        radius = max(3, round(margin * (0.35 if unit["operation"] in {"add", "replace"} else 0.18)))
        local = mask[y0:y1, x0:x1]
        expanded = _dilate(local, radius)
        alpha = Image.fromarray((_feather(expanded, max(2, radius // 3)) * 255).astype("uint8"), mode="L")
        output = image.copy()
        output.paste(edited_crop, (x0, y0), alpha)
        return output


def _erase_target(crop: Image.Image, local_mask) -> Image.Image:
    """Inpaint the target from a remove crop before diffusion generation."""
    import numpy as np
    array = np.asarray(crop.convert("RGB"))
    mask = (np.asarray(local_mask).astype("uint8") * 255)
    try:
        import cv2
        filled = cv2.inpaint(array, mask, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(filled, mode="RGB")
    except ImportError:  # pragma: no cover - isolated editor runtime has cv2
        # Conservative fallback: use the crop median so the model still sees
        # a missing target rather than the original object.
        fallback = array.copy()
        fallback[mask > 0] = np.median(array[mask == 0], axis=0) if np.any(mask == 0) else 127
        return Image.fromarray(fallback, mode="RGB")


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
