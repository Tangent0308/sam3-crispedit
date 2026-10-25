"""Operation-specific target generation with checkpoint-safe local edits."""
from __future__ import annotations

import argparse
import json
import hashlib
import re
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
    from .multi_execution_contract import dependency_prompt
    guide += " " + dependency_prompt(unit.get("dependency_context", [])) if unit.get("dependency_context") else ""
    operation = unit["operation"]
    spec = unit.get('instruction_spec') or {}
    if spec:
        from .instruction_protocol import validate_pair
        validate_pair(spec, unit['instruction_ref'], unit['instruction_noref'])
    instruction = unit['instruction_noref'] if crop_instruction else unit['instruction_ref']
    if unit.get('generation_reference_frame') == 'crop-entity-v1':
        # Internal generator wording only. Published ref/no-ref remain the
        # validated protocol pair. Scene ordinals do not survive cropping.
        scope = str(spec.get('scope') or unit.get('scope') or '').strip()
        if not scope or not spec or unit['instruction_noref'].count('the selected region') != 1:
            raise ValueError('local generation requires a grounded protocol source scope')
        from .instruction_protocol import render, noun
        instruction = render({**spec, 'ref':noun(scope)}, has_ref=True)
    if crop_instruction and not instruction:
        raise ValueError('crop instruction requires an existing protocol No-ref instruction')
    if operation in ('add','replace') and spec.get('payload'):
        # Source landmarks may contain "with". Explicit content slots prevent
        # their accidental reinterpretation as additional replacement objects.
        instruction += (' Requested NEW content only: '+spec['payload']+'. '
                        'Other objects or landmarks in the source description are not requested new content.')
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
        # Keep the verified glove restoration guidance only for glove targets.
        # Avoid priming unrelated hand content for independent objects.
        scope = str(unit.get("scope") or spec.get("scope") or unit.get("category", ""))
        host_rule = (
            "If the item is worn on or covers a body or supporting object, reveal that underlying body or host, "
            "rather than cutting a hole through it: removing a glove must reveal a natural bare hand that still grips any held object. "
            "Never erase the underlying limb, leave a floating held object, or substitute background through a solid body. "
            if re.search(r"\bgloves?\b", scope, re.IGNORECASE) else
            "Preserve the remaining host and its structural continuity. Reveal the underlying host where the source shows one; "
            "do not substitute background through it. Keep existing contact and support relationships plausible "
            "without inventing a new supporting object. "
        )
        return (
            f"{instruction} Completely erase only the explicitly named object, part or section. "
            "For subpart removal or trimming, retain the remaining host; the source mask selects the removed region. "
            "Reconstruct what was physically behind the removed item, using the surrounding texture, perspective, lighting and occlusion. "
            + host_rule +
            "For independent objects, reconstruct the exposed background; "
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
            "realize the requested physical form/construction, not just repaint the old silhouette. "
            "Apply the replacement to the entire named selected entity, not only its easiest accessory. Do "
            "not leave fragments or create a duplicate beside it. Keep every other object, instance, boundary, "
            "lighting cue and background unchanged."
            + locator + guide
        )
    if operation == "attribute":
        return (
            f"{instruction} Render a natural photographic result. Change only the requested visible "
            "properties of the named object or part. Preserve its identity, silhouette, geometry, "
            "pose, edges, unrequested text, unrequested printed markings and all surrounding objects. "
            "Keep the exact original instance count and component layout: two sticks remain exactly two, "
            "not multiple parallel rods. Preserve foreground occluders and carried objects even inside the target bounding box. "
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
                 tensor_parallel_size: int = 1, crop_instruction: bool = False, finish: dict | None = None):
        import torch
        from utils.qwen_pipeline_loader import load_qwen21_omni_pipeline
        self.torch = torch
        torch_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                       "fp16": torch.float16, "float16": torch.float16,
                       "fp32": torch.float32, "float32": torch.float32}[dtype]
        self.pipe = load_qwen21_omni_pipeline(model_id, tensor_parallel_size=tensor_parallel_size)
        self.steps = steps
        self.config = dict(backend="qwen-omni", model_id=model_id, steps=steps, dtype=dtype, tensor_parallel_size=tensor_parallel_size, compositor_version=3, prompt_version="region-edit-8-explicit-content")
        self.crop_instruction = crop_instruction
        if crop_instruction:
            self.config['instruction_frame'] = 'experimental-crop-noref-v1'
        self.finish = None
        if finish:
            self.configure_finish(finish)

    FINISH_OPTIONS = {'removal_effects', 'regrain', 'tone_match', 'add_cleanup', 'prompt_rules', 'min_context',
                      'defocus_match', 'focus_match', 'material_rule', 'lighting_rule', 'detail_restore'}

    def configure_finish(self, finish: dict[str, Any]) -> None:
        """Opt-in finishing (edit_finish / removal_effects); never active by default."""
        unknown = set(finish) - self.FINISH_OPTIONS
        if unknown:
            raise ValueError(f'unknown finish options: {sorted(unknown)}')
        if finish.get('defocus_match') and finish.get('focus_match'):
            raise ValueError('defocus_match and focus_match both blur new content; enable one')
        from .edit_finish import VERSION as finish_version
        self.finish = dict(finish)
        self.effect_finder = None
        self._effect_cache = {}
        if self.finish.get('removal_effects'):
            from .removal_effects import EffectFinder
            self.effect_finder = EffectFinder(self.finish['removal_effects'])
        config = {'version': finish_version, **self.finish}
        if self.effect_finder is not None:
            config['removal_effects'] = self.effect_finder.config()
        self.config = {**self.config, 'finish': config}

    def compute_removal_effect(self, image: Image.Image, unit: dict[str, Any]):
        """(effect support or None, info) for a whole-object removal; cached per image and mask."""
        if getattr(self, 'effect_finder', None) is None or unit['operation'] != 'remove':
            return None, None
        spec = unit.get('instruction_spec') or {}
        if spec.get('remove_mode', 'whole') != 'whole' or spec.get('target_part'):
            return None, None
        key = (hashlib.sha1(np.asarray(image).tobytes()).hexdigest(), str(unit['mask_path']),
               str(unit.get('protected_mask_path') or ''))
        if key not in self._effect_cache:
            from .removal_effects import effect_support
            target = load_mask({"mask_path": unit["mask_path"]}, image.size)
            protected = (load_mask({"mask_path": unit["protected_mask_path"]}, image.size)
                         if unit.get('protected_mask_path') else None)
            effect, info = self.effect_finder(image, target, protected)
            support = effect_support(effect, target) if effect.any() else None
            if support is not None and protected is not None:
                support &= ~protected
            self._effect_cache[key] = (support, info)
        return self._effect_cache[key]

    def _removal_effect(self, image: Image.Image, unit: dict[str, Any]):
        """Effect support for this unit and whether it is already inside the unit's execution mask."""
        if not getattr(self, 'finish', None):
            return None, None, False
        if unit.get('removal_effect_mask_path'):
            # Resolved before generation (multi orchestrator) and already part of the execution mask.
            effect = load_mask({"mask_path": unit['removal_effect_mask_path']}, image.size)
            return effect, unit.get('removal_effects'), True
        if unit.get('execution_mask_path'):
            return None, None, False
        effect, info = self.compute_removal_effect(image, unit)
        return effect, info, False

    def support_mask(self, image: Image.Image, unit: dict[str, Any]) -> np.ndarray:
        """Writable support; identical to writable_mask unless removal effects are enabled."""
        from .specialist_editors import writable_mask
        support = writable_mask(unit, image.size, require_add_placement=False)
        effect, _, included = self._removal_effect(image, unit)
        if effect is not None and not included:
            support = support | effect
        return support

    def _finish_rules(self, unit: dict[str, Any], effect: bool) -> str:
        # prompt_rules: removal sentences only. The add/replace material sentence is a
        # separate opt-in (material_rule): neutral for add and slightly negative for
        # replace in the 2026-10-23 paired A/B.
        from . import edit_finish as ef
        operation = unit['operation']
        if operation in ('add', 'replace'):
            rule = ef.MATERIAL_RULE if self.finish.get('material_rule') else ''
            if self.finish.get('lighting_rule'):
                # Needs the planning lighting label on the unit (site_lighting); see edit_finish.lighting_rule.
                rule += ef.lighting_rule(unit)
            return rule
        if operation == 'remove' and self.finish.get('prompt_rules'):
            return ef.REMOVE_RULE + (ef.EFFECT_RULE if effect else '')
        return ''

    def _finish_crop(self, image, crop, edited_crop, local, local_support, alpha, unit, seed, local_effect):
        from . import edit_finish as ef
        finish, operation = self.finish, unit['operation']
        spec = unit.get('instruction_spec') or {}
        source = np.asarray(crop.convert('RGB'))
        out = np.asarray(edited_crop.convert('RGB')).copy()
        info = {'version': ef.VERSION}
        payload = str(spec.get('payload') or unit.get('payload') or '')
        if operation == 'add' and finish.get('add_cleanup') and not ef.PLURAL.search(payload):
            out, info['add_cleanup'] = ef.add_cleanup(source, out, alpha)
        if operation == 'remove' and finish.get('tone_match'):
            out, info['tone_match'] = ef.tone_match(source, out, local, local_support, alpha, local_effect)
        if operation in ('add', 'replace') and finish.get('defocus_match'):
            out, info['defocus_match'] = ef.defocus_match(source, out, alpha, local)
        if operation in ('add', 'replace') and finish.get('focus_match'):
            out, info['focus_match'] = ef.focus_match(source, out, alpha, local, operation)
        if operation == 'add' and finish.get('detail_restore'):
            # Diagnostic, not in the --edit-finish preset: a 24-pair blind A/B preferred it 10:9
            # (5 same). Additions only: around a removal or replacement the generator also cleans thin
            # remnants of the old object (twigs outside the mask) that a coarse test cannot tell from blur.
            out, info['detail_restore'] = ef.detail_restore(source, out, alpha)
        if finish.get('regrain'):
            weight = ef.changed_weight(source, out, alpha)
            out, info['regrain'] = ef.regrain(source, out, weight, seed=seed, untouched=alpha <= 0)
        return Image.fromarray(out), info

    def prompt(self, unit: dict[str, Any]) -> str:
        return edit_prompt(unit, crop_instruction=getattr(self, 'crop_instruction', False))

    def generate(self, image_path: str, unit: dict[str, Any], seed: int = 0) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        mask = load_mask({"mask_path": unit["mask_path"]}, image.size)
        finish = getattr(self, 'finish', None) or {}
        # Identical to writable_mask(...) unless opt-in removal effects widen it.
        support = self.support_mask(image, unit)
        effect, effect_info, _ = self._removal_effect(image, unit) if finish else (None, None, False)
        # Read enough host context even when the writable patch is local.
        ys, xs = (support | mask).nonzero()
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        margin = max(16, round(max(x1 - x0, y1 - y0) * 0.45))
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(image.width, x1 + margin), min(image.height, y1 + margin)
        if finish.get('min_context'):
            # More surrounding photo at the same ~1MP generation size: a small
            # target is no longer upsampled 4-15x, so texture keeps photo scale.
            x0, x1 = _grow_span(x0, x1, image.width, int(finish['min_context']))
            y0, y1 = _grow_span(y0, y1, image.height, int(finish['min_context']))
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
        # Only set by the opt-in text-verification wrapper (exact wording / no lettering).
        prompt += unit.get('generator_text_rule', '')
        if finish:
            prompt += self._finish_rules(unit, effect is not None)
        contact = unit.get('generation_contact')
        if contact and unit['operation'] == 'add':
            px, py = contact['point']
            sy, sx = np.nonzero(local_support)
            prompt += (f" The new item's actual contact point is x={(px-x0)/crop.width:.1%}, "
                       f"y={(py-y0)/crop.height:.1%} in this crop, on the selected host. "
                       f"Its entire silhouette must fit inside x={sx.min()/crop.width:.1%}..{(sx.max()+1)/crop.width:.1%}, "
                       f"y={sy.min()/crop.height:.1%}..{(sy.max()+1)/crop.height:.1%}. "
                       "Keep it compact, fully visible, and attached at that point; do not move it to a similar nearby part. "
                       "Match the original local blur and scale; a distant accessory must not look like a sharp foreground object.")
        observations = unit.get('execution_observations') or {}
        if observations.get('required_change'):
            prompt += ' Visible completion: ' + str(observations['required_change']) + '.'
        if observations.get('preserve'):
            prompt += ' Preserve these visible source details: ' + '; '.join(observations['preserve']) + '.'
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
        if unit['operation'] == 'add':
            from .edit_quality import paste_diagnostics
            self.last_generation_context['paste_diagnostics'] = paste_diagnostics(crop, edited_crop, expanded)
        finished_crop = edited_crop
        if finish:
            local_effect = effect[y0:y1, x0:x1] if effect is not None else None
            finished_crop, finish_info = self._finish_crop(
                image, crop, edited_crop, local, local_support,
                np.asarray(alpha, dtype=np.float32) / 255., unit, seed, local_effect)
            self.last_generation_context['finish'] = finish_info
            if effect_info is not None:
                self.last_generation_context['removal_effects'] = effect_info
        if unit.get("diagnostic_dir"):
            directory = Path(unit["diagnostic_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            evidence = {}
            pictures = [("source_crop", crop), ("raw_crop", raw), ("resized_crop", edited_crop), ("alpha", alpha)]
            if finished_crop is not edited_crop:
                pictures.append(("finished_crop", finished_crop))
            for name, picture in pictures:
                path = directory / (name + ".png")
                picture.save(path)
                evidence[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.last_generation_context["crop_evidence"] = evidence
        output = image.copy()
        output.paste(finished_crop, (x0, y0), alpha)
        return output


def _grow_span(a: int, b: int, limit: int, side: int) -> tuple[int, int]:
    """Widen [a, b) to ``side`` pixels (clamped to the frame) while still containing it."""
    side = min(side, limit)
    if b - a >= side:
        return a, b
    start = int(round((a + b) / 2 - side / 2))
    start = max(0, min(start, limit - side))
    return start, start + side


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
    # A square dilation is separable. The previous nested offsets scanned
    # the full SA-1B canvas (2r+1)^2 times, although the output is identical
    # to two linear-time maximum filters. Preserve the exact square kernel
    # and zero boundary convention; this is not a mask approximation.
    from scipy.ndimage import maximum_filter1d
    result = maximum_filter1d(np.asarray(mask, dtype=bool), 2 * radius + 1,
                              axis=0, mode="constant", cval=0)
    return maximum_filter1d(result, 2 * radius + 1, axis=1, mode="constant", cval=0)


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
