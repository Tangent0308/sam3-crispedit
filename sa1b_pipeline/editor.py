"""Operation-specific target generation with checkpoint-safe local edits."""
from __future__ import annotations

import argparse
import json
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import load_mask
from .io import read_jsonl, write_jsonl


def edit_prompt(unit: dict[str, Any], *, target_hint: str | None = None,
                guided: bool = False, crop_instruction: bool = False) -> str:
    locator = ""
    if target_hint:
        locator = f" The selected target is located at {target_hint} in the provided crop."
    guide = (
        " Image 2 is an aligned location guide with a thin black/white boundary; use it only "
        "to locate the target, never copy the boundary or any annotation into the result."
        if guided else ""
    )
    operation = unit["operation"]
    spec = unit.get('instruction_spec') or {}
    if spec:
        from .instruction_protocol import validate_pair
        validate_pair(spec, unit['instruction_ref'], unit['instruction_noref'])
    instruction = unit['instruction_noref'] if crop_instruction else unit['instruction_ref']
    if crop_instruction and not instruction:
        raise ValueError('crop instruction requires an existing protocol No-ref instruction')
    relative_add = spec.get('placement', {}).get('anchor_mode') == 'reference'
    if operation == "add":
        return (
            f"{instruction} The selected mask identifies the host object or attachment site, "
            "not the new item's final segmentation. Keep the host object recognizable and present; "
            "add exactly the requested item count, clearly visible, correctly placed, and consistent with "
            "the host's perspective, scale, lighting and occlusion. Do not erase, repaint, replace or "
            "duplicate the host. Preserve the host's existing color, material, texture, silhouette, "
            "text and visible parts outside the new item's necessary occlusion; only the new item and its physically necessary contact shadow "
            "may introduce pixels. "
            + ("Use the selected source instance as the spatial reference; obey the requested relation, "
               "and establish plausible contact with the named supporting surface. " if relative_add else
               "Place the new item on the visible pixels of the mask-selected host instance, with its contact point intersecting that host. "
               "Do not attach it to an unrelated neighboring instance. ") +
            "Use a conventional size and functional placement for the requested accessory; it must not "
            "block the host's use or look oversized merely to be conspicuous. Do not add unrelated objects. "
            "Keep every unrequested object, instance, boundary and background unchanged."
            + locator + guide
        )
    if operation == "remove":
        return (
            f"{instruction} Completely erase only the explicitly named object, part or section. "
            "For subpart removal or trimming, retain the remaining host; the source mask may select that host. "
            "Reconstruct the newly exposed background using the surrounding texture, perspective, lighting and occlusion; "
            "leave no recolored silhouette, outline, placeholder, cyan/gray patch or duplicate. "
            "Keep every unrequested object, instance, part, boundary, occluder and background unchanged."
            + locator + guide
        )
    if operation == "replace":
        return (
            f"{instruction} Render a natural photographic result. Replace only the selected target "
            "with the requested replacement and count, keeping its position, scale, viewpoint, contact "
            "with the scene and surrounding occlusion consistent. The replacement must fit the available space "
            "and existing support without redesigning the scene. Remove the old identity completely and do "
            "not leave fragments or create a duplicate beside it. Keep every other object, instance, boundary, "
            "lighting cue and background unchanged."
            + locator + guide
        )
    if operation == "attribute":
        return (
            f"{instruction} Render a natural photographic result. Change only the requested visible "
            "properties of the named object or part. Preserve its identity, silhouette, geometry, "
            "pose, edges, unrequested text, unrequested printed markings and all surrounding objects. "
            "For text/pattern edits perform the requested content change; tint preserves transparency, "
            "while an explicit opacity edit must realize the requested transparency. Preserve illumination direction "
            "and shading structure; for a material edit, allow only physically appropriate local texture "
            "and reflectance changes, including highlights, on the selected surface. Keep other materials "
            "and components intact; do not apply a flat uniform color wash. Do not recolor neighboring "
            "instances or invent structural changes."
            + locator + guide
        )
    return (
        f"{instruction} Render a natural photographic result. "
        "Keep every unrequested object, instance, part, boundary, occluder, lighting cue and background unchanged; "
        "make only physically necessary local attachment, revealed-background, shadow and contact changes."
        + locator + guide
    )


class DiffusersQwenImageEditor:
    """Lazy diffusers adapter; generation is isolated so it can resume per unit."""

    def __init__(self, model_id: str, *, device: str = "cuda", dtype: str = "bf16", steps: int = 40,
                 tensor_parallel_size: int = 1, crop_instruction: bool = False):
        import torch
        from utils.qwen_pipeline_loader import load_qwen21_omni_pipeline
        self.torch = torch
        torch_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                       "fp16": torch.float16, "float16": torch.float16,
                       "fp32": torch.float32, "float32": torch.float32}[dtype]
        self.pipe = load_qwen21_omni_pipeline(model_id, tensor_parallel_size=tensor_parallel_size)
        self.steps = steps
        self.config = dict(backend="qwen-omni", model_id=model_id, steps=steps, dtype=dtype, tensor_parallel_size=tensor_parallel_size, compositor_version=3)
        self.crop_instruction = crop_instruction
        if crop_instruction:
            self.config['instruction_frame'] = 'experimental-crop-noref-v1'

    def prompt(self, unit: dict[str, Any]) -> str:
        return edit_prompt(unit, crop_instruction=getattr(self, 'crop_instruction', False))

    def generate(self, image_path: str, unit: dict[str, Any], seed: int = 0) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        mask = load_mask({"mask_path": unit["mask_path"]}, image.size)
        from .specialist_editors import writable_mask
        support = writable_mask(unit, image.size, require_add_placement=False)
        ys, xs = support.nonzero()
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        margin = max(16, round(max(x1 - x0, y1 - y0) * 0.45))
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(image.width, x1 + margin), min(image.height, y1 + margin)
        crop = image.crop((x0, y0, x1, y1))
        local = mask[y0:y1, x0:x1]
        local_support = support[y0:y1, x0:x1]
        target_hint = _target_hint(local)
        guided = hasattr(self.pipe, "region_callback_kwargs")
        generator = self.torch.Generator(device="cuda").manual_seed(seed)
        # The Omni regional adapter anchors the editable latent tokens itself.
        # Passing a second boundary image through the generic Qwen image list
        # is ambiguous for the model and can be copied as a cyan/white patch,
        # so keep the condition image to the clean source crop.
        prompt = edit_prompt(unit, target_hint=target_hint, guided=False,
                             crop_instruction=getattr(self, 'crop_instruction', False))
        self.last_generation_context = {'model_prompt': prompt, 'crop_bbox': [x0, y0, x1, y1],
                                        'target_hint_in_crop': target_hint}
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
                crop, local, unit["operation"], None, "guard-any-v1",
                local_support
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
        # Use a bounded operation-specific writable support. The source mask
        # is an anchor for add, so the writable region is an attachment collar
        # around that anchor rather than an unrestricted full-image edit.
        radius = _support_radius(local, unit["operation"], unit.get("support_policy"))
        expanded = local_support
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


def _support_radius(mask: np.ndarray, operation: str, support_policy: str | None = None) -> int:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty target mask")
    extent = max(1, min(np.ptp(xs) + 1, np.ptp(ys) + 1))
    if support_policy == "target_only" or operation == "attribute":
        return 0
    if support_policy == "anchor_plus_attachment" or operation == "add":
        return int(np.clip(round(extent * 0.28), 12, 64))
    if operation == "replace":
        return int(np.clip(round(extent * 0.24), 12, 56))
    # Removal needs enough context to reconstruct occluded texture and edges;
    # a 3--16 px collar was systematically too narrow for the previous run.
    return int(np.clip(round(extent * 0.14), 8, 40))


def region_support_mask(mask: np.ndarray, operation: str,
                        support_policy: str | None = None) -> np.ndarray:
    """Return the pixel support that may differ from the source image.

    This is deliberately separate from the source ``mask``.  In particular,
    an add mask is the host anchor, while its editable support includes an
    attachment collar.  The support mask is useful for hard compositing,
    deterministic leakage checks, and later human review; it is not exposed
    as the benchmark's input locator.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("empty support mask input")
    radius = _support_radius(mask, operation, support_policy)
    return _dilate(mask, radius) if radius else mask.copy()


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
    """Feather actual support boundaries, not the edge of the photograph."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.astype(float)
    value = np.zeros(mask.shape, dtype=float)
    eroded = mask.copy()
    for step in range(radius):
        # At a clipped image edge there are no source pixels beyond the frame
        # to blend with. Zero padding would mix the removed object back into
        # every such edge (e.g. a blue shoebox stripe at the top of the photo).
        padded = np.pad(eroded, 1, mode="edge")
        interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                    & padded[1:-1, :-2] & padded[1:-1, 2:])
        value[eroded & ~interior] = (step + 1) / (radius + 1)
        eroded = interior
    value[eroded] = 1.0
    return value


def generate_cases(manifest: Path, output_dir: Path, editor: DiffusersQwenImageEditor, seed: int = 0,
                   *, on_result=None, checkpoint_every: int = 1) -> list[dict[str, Any]]:
    if checkpoint_every < 1:
        raise ValueError('checkpoint_every must be positive')
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    from .specialist_editors import writable_mask
    config = getattr(editor, 'config', {'backend': type(editor).__name__})
    for case in read_jsonl(manifest):
        source = case["source"]["image_path"]
        for unit in case["units"]:
            case_id = f"{case['id']}--{unit['unit_id']}"
            path = output_dir / f"{case_id}.png"
            support_path = output_dir / "support_masks" / f"{case_id}.png"
            prompt = editor.prompt(unit) if hasattr(editor, 'prompt') else edit_prompt(unit)
            fingerprint = hashlib.sha256(json.dumps(
                {'source': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                 'unit': unit, 'editor': config, 'prompt': prompt, 'seed': seed,
                 'masks': {key: hashlib.sha256(Path(unit[key]).read_bytes()).hexdigest()
                           for key in ('mask_path', 'execution_mask_path', 'protected_mask_path') if unit.get(key)}},
                sort_keys=True).encode()).hexdigest()
            state_path = path.with_suffix('.state.json')
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if path.exists() and state.get('fingerprint') != fingerprint:
                raise ValueError(f'output exists with different or untracked generation settings: {path}; use a new output directory')
            if not path.exists():
                started = time.perf_counter()
                result = editor.generate(source, unit, seed=seed)
                temp = path.with_suffix('.tmp.png')
                result.save(temp)
                temp.replace(path)
                state = {'fingerprint': fingerprint, 'generation_seconds': time.perf_counter()-started,
                         'generation_context': getattr(editor, 'last_generation_context', {})}
                state_path.write_text(json.dumps(state, sort_keys=True))
            if not support_path.exists():
                source_image = Image.open(source).convert("RGB")
                source_mask = load_mask({"mask_path": unit["mask_path"]}, source_image.size)
                policy = unit.get("support_policy") or {
                    "add": "anchor_plus_attachment", "replace": "target_plus_collar",
                    "remove": "target_plus_collar", "attribute": "target_only",
                }[unit["operation"]]
                support = (editor.support_mask(source_image, unit) if hasattr(editor, 'support_mask')
                           else writable_mask(unit, source_image.size, require_add_placement=False))
                support_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((support.astype("uint8") * 255), mode="L").save(support_path)
            outputs.append({"case_id": case["id"], "unit_id": unit["unit_id"], "operation": unit["operation"],
                            "object_id": unit.get("object_id"),
                            "source_id": case["source"].get("source_id"),
                            "source_image": source, "edited_image": str(path), "instruction": unit["instruction_ref"],
                            "instruction_noref": unit.get("instruction_noref"), "mask": unit["mask_path"],
                            "ref_phrase": unit.get("ref_phrase"), "scope": unit.get("scope"), "payload": unit.get("payload"),
                            "bbox": unit.get("bbox"), "point": unit.get("point"),
                            "difficulty": case.get("difficulty", {}), "source_provenance": case["source"].get("provenance", {}),
                            "prompt": prompt, "seed": seed, "editor_config": config,
                            "generation_fingerprint": fingerprint, "generation_seconds": state.get('generation_seconds'),
                            "generation_context": state.get('generation_context', {}),
                            "instruction_protocol_version": unit.get('instruction_protocol_version'),
                            "instruction_spec": unit.get('instruction_spec', {}),
                            "attribute_kind": unit.get('attribute_kind'), "edit_surface": unit.get('edit_surface'),
                            "source_color": unit.get('source_color'),
                            "execution_mask_path": unit.get('execution_mask_path'),
                            "surface_type": unit.get('surface_type'), "target_rgb": unit.get('target_rgb'),
                            # Keep the benchmark input semantics explicit in
                            # every generated row, including older manifests
                            # that predate these fields.
                            "mask_role": unit.get("mask_role", "anchor" if unit["operation"] == "add" else "target"),
                            "support_policy": unit.get(
                                "support_policy",
                                {"add": "anchor_plus_attachment", "replace": "target_plus_collar",
                                 "remove": "target_plus_collar", "attribute": "target_only"}[unit["operation"]]),
                            "support_mask": str(support_path),
                            "change_mask": None,
                            "new_object_mask": None})
            if case.get('benchmark_only'):
                outputs[-1].update(benchmark_only=True, benchmark_original_id=case['benchmark_original_id'])
            if len(outputs) % checkpoint_every == 0:
                checkpoint = output_dir / "generated.jsonl.tmp"
                write_jsonl(checkpoint, outputs)
                checkpoint.replace(output_dir / "generated.jsonl")
            if on_result is not None:
                # Image and state are durable before publishing to consumers.
                # Resume also republishes existing rows; the queue deduplicates.
                on_result(outputs[-1])
    checkpoint = output_dir / "generated.jsonl.tmp"
    write_jsonl(checkpoint, outputs)
    checkpoint.replace(output_dir / "generated.jsonl")
    return outputs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backend", choices=("qwen", "flux-fill", "lama", "recolor", "intrinsic-recolor"), default="qwen")
    p.add_argument("--model-id", help="Qwen/Flux model directory or LaMa TorchScript file")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop-instruction", action="store_true",
                   help="Experimental: use protocol No-ref wording in the editor's local crop")
    a = p.parse_args(argv)
    model_root = Path('/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models')
    if a.backend == 'qwen':
        editor = DiffusersQwenImageEditor(a.model_id or str(model_root/'Qwen-Image-2.1'), device=a.device, dtype=a.dtype, steps=a.steps, tensor_parallel_size=a.tensor_parallel_size, crop_instruction=a.crop_instruction)
    else:
        from .specialist_editors import FluxFillEditor, LamaEditor, RecolorEditor
        if a.backend == 'flux-fill':
            editor = FluxFillEditor(a.model_id or 'black-forest-labs/FLUX.1-Fill-dev', device=a.device, steps=a.steps)
        elif a.backend == 'lama':
            editor = LamaEditor(a.model_id or str(model_root/'LaMa/big-lama.pt'), device=a.device)
        elif a.backend == 'intrinsic-recolor':
            from .specialist_editors import IntrinsicRecolorEditor
            editor = IntrinsicRecolorEditor(a.model_id or 'v2', device=a.device)
        else:
            editor = RecolorEditor()
    generate_cases(a.manifest, a.output_dir, editor, a.seed)


if __name__ == "__main__":
    main()
