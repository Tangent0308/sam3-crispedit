"""Qwen3.8-27B vLLM adapter with resumable, structured JSON output.

The import is lazy so geometry and unit tests work on CPU-only machines.
"""
from __future__ import annotations
from .instruction_style import INSTRUCTION_STYLE

import json
import os
import re
from pathlib import Path
from typing import Any
from PIL import Image


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("MLLM output did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("MLLM output must be a JSON object")
    return value


def _safe_parse(text: str, *, stage: str) -> dict[str, Any]:
    """Turn one malformed generation into an auditable rejection.

    vLLM returns a batch atomically, but a single vision-language generation
    can still truncate or emit prose.  The caller must be able to keep the
    other batch members and let the stage-specific contract reject this one.
    """
    try:
        return parse_json(text)
    except (ValueError, json.JSONDecodeError) as exc:
        if stage == "audit":
            return {"decision": "fail", "failure_reasons": ["malformed_mllm_json"], "evidence": str(exc)}
        return {"accept": False, "_parse_error": str(exc), "_raw_output": text}


class VLLMSelector:
    """Continuous-batching selector/planner for Qwen3.8-27B."""

    def __init__(self, model_id: str, *, tensor_parallel_size: int = 1, max_model_len: int = 16384,
                 max_new_tokens: int = 1200, temperature: float = 0.0, batch_size: int = 8,
                 disable_custom_all_reduce: bool = False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        self.batch_size = batch_size
        # Reuse the parent repository's production Qwen3.8 backend. It pins
        # the multimodal processor settings and the non-thinking chat template
        # used by the completed derived-edit runs.
        # The inherited production adapter is useful for a one-GPU smoke, but
        # it intentionally hard-codes tensor_parallel_size=1. Full selection
        # runs must honor the caller's TP setting and use the direct vLLM path.
        self._inherited_backend = False
        if tensor_parallel_size == 1 and os.environ.get("SA1B_FORCE_DIRECT_VLLM") != "1":
            try:
                from utils import vlm_utils
                vlm_utils.configure_backend("qwen38-vllm", model_id=model_id, device="cuda:0", dtype="bf16")
                self.backend = vlm_utils.get_backend()
                self._inherited_backend = True
                self.params = None
                self.max_new_tokens = max_new_tokens
                return
            except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, ValueError):
                pass
        try:
            from transformers import AutoProcessor  # type: ignore
            from vllm import LLM, SamplingParams  # type: ignore
            import vllm
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.8 selection requires the pinned vLLM/Transformers runtime; "
                "install requirements/mllm-vllm-cu129.txt"
            ) from exc
        version = tuple(int(x) for x in re.findall(r"\d+", vllm.__version__)[:2])
        if version < (0, 28):
            raise RuntimeError(
                f"vLLM {vllm.__version__} does not support Qwen3_5ForConditionalGeneration; "
                "install requirements/mllm-vllm-cu129.txt"
            )
        self.SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        from .mllm_runtime import architecture_options
        llm_kwargs = dict(model=model_id, dtype="bfloat16", tensor_parallel_size=tensor_parallel_size,
                          max_model_len=max_model_len, trust_remote_code=True,
                          max_num_seqs=8, max_num_batched_tokens=8192,
                          limit_mm_per_prompt={"image": 3}, enable_prefix_caching=True,
                          gpu_memory_utilization=0.85, enforce_eager=True,
                          disable_custom_all_reduce=disable_custom_all_reduce)
        # Older CUDA drivers cannot execute the cu13 FlashAttention-3 wheel
        # used by the pinned runtime.  Keep the production default unchanged,
        # but allow a documented runtime override for the vision encoder.
        mm_backend = os.environ.get("SA1B_MM_ENCODER_ATTN_BACKEND")
        if mm_backend:
            llm_kwargs["mm_encoder_attn_backend"] = mm_backend
        llm_kwargs.update(architecture_options(model_id, tensor_parallel_size))
        self.llm = LLM(**llm_kwargs)
        self.params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens,
                                     top_p=1.0, stop=["<|im_end|>"])

    def close(self) -> None:
        """Release vLLM's worker processes and CUDA allocations explicitly."""
        engine = getattr(getattr(self, "llm", None), "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def _messages(self, image_path: str, prompt: str) -> tuple[str, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
        return rendered, image

    def _multi_messages(self, image_paths: list[str], prompt: str) -> tuple[str, list[Image.Image]]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        rendered = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
        return rendered, images

    def generate(self, jobs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if len(jobs) > self.batch_size:
            results = []
            for start in range(0, len(jobs), self.batch_size):
                results.extend(self.generate(jobs[start:start + self.batch_size]))
                print(f"MLLM single-image requests completed: {len(results)}/{len(jobs)}", flush=True)
            return results
        if self._inherited_backend:
            messages = []
            for image_path, prompt in jobs:
                messages.append([{"role": "user", "content": [
                    {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    {"type": "text", "text": prompt},
                ]}])
            return [_safe_parse(x, stage="selection") for x in self.backend.chat_batch(messages, max_new_tokens=self.max_new_tokens)]
        requests = []
        images = []
        for image_path, text in jobs:
            prompt, image = self._messages(image_path, text)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": [image]}})
            images.append(image)
        outputs = self.llm.generate(requests, self.params, use_tqdm=False)
        import os
        log_path = os.environ.get('SA1B_MLLM_LOG')
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as stream:
                for job, output in zip(jobs, outputs):
                    stream.write(json.dumps({'image': job[0], 'prompt': job[1], 'text': output.outputs[0].text if output.outputs else '', 'finish_reason': output.outputs[0].finish_reason if output.outputs else None}, ensure_ascii=False) + '\n')
        result = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            result.append(_safe_parse(text, stage="selection"))
        return result

    def generate_multimage(self, jobs: list[tuple[list[str], str]]) -> list[dict[str, Any]]:
        if len(jobs) > self.batch_size:
            results = []
            for start in range(0, len(jobs), self.batch_size):
                results.extend(self.generate_multimage(jobs[start:start + self.batch_size]))
                print(f"MLLM multi-image requests completed: {len(results)}/{len(jobs)}", flush=True)
            return results
        if self._inherited_backend:
            messages = []
            for image_paths, prompt in jobs:
                content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
                content.append({"type": "text", "text": prompt})
                messages.append([{"role": "user", "content": content}])
            return [_safe_parse(x, stage="audit") for x in self.backend.chat_batch(messages, max_new_tokens=self.max_new_tokens)]
        requests = []
        for paths, text in jobs:
            prompt, images = self._multi_messages(paths, text)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        outputs = self.llm.generate(requests, self.params, use_tqdm=False)
        return [_safe_parse(output.outputs[0].text if output.outputs else "", stage="audit") for output in outputs]


def _required_operation_rule(operation: str | None) -> str:
    if operation is None:
        return "No operation was assigned (legacy unconstrained planning)."
    from .schema import OPERATIONS
    if operation not in OPERATIONS:
        raise ValueError(f"invalid required operation: {operation}")
    criteria = {
        'add': 'Select existing hosts with a visible usable attachment/resting surface for the requested conventional accessories. Keep the host. The source mask is the host anchor, not the new item. Avoid hosts with no natural accessory.',
        'replace': 'Select complete, compact objects or detachable parts that admit a visibly different design with compatible size, support and function. A recolor is not a replacement.',
        'remove': 'Select self-contained removable objects or detachable decorations whose absence leaves a plausible scene. Avoid deleting body parts, worn shoes/clothes wholesale, load-bearing supports or mixed-background masks. A named removable garment part is allowed only with a prepared execution region.',
        'attribute': 'Select a specific editable surface or single-material object with a credible visible appearance change. Preserve geometry, unrequested markings and other materials. Avoid already-present target colors and whole-person masks.',
    }
    return (f"HARD OPERATION ASSIGNMENT: {operation}. This was assigned BEFORE target selection. "
            f"Select targets and design instructions ONLY for {operation}; every returned unit.operation MUST equal '{operation}'. "
            "Do not choose, relabel, or fall back to any other operation, including attribute. "
            "If a target cannot support a natural edit of the assigned type, skip it with a concrete reason. "
            "If fewer than the required number of independent targets remain, reject the scene. "
            "Reject nested/overlapping targets and mutually conflicting instructions. " + criteria[operation])


def selection_prompt(candidate: dict[str, Any], *, min_targets: int = 2, max_targets: int = 5, required_operation: str | None = None) -> str:
    operation_rule = _required_operation_rule(required_operation)
    objects = candidate.get("objects", [])
    # Keep the candidate index well below the 8k context budget on crowded
    # SA-1B scenes. The image and numbered mask overlay carry the visual
    # description; category/geometry metadata is sufficient for object IDs.
    compact = [
        [str(o.get("object_id", "")), str(o.get("category", ""))[:48], o.get("bbox"), o.get("area_fraction")]
        for o in objects
    ]
    return f"""You are curating hard SA-1B source scenes for a fine-grained interactive image-editing training set.
{operation_rule}
The supplied panel shows the clean photograph on the LEFT and a mask-only object_id map on the RIGHT. Read appearance only from the clean photograph.
Inspect the image, then use the supplied dense-object candidates only as identity/mask hints. Select {min_targets} to {max_targets} DIFFERENT physical objects or parts in this same scene.
Prefer multiple near-duplicate instances, occlusion, touching boundaries, thin straps/rods, holes, disconnected visible fragments, small parts, and mixed object/part granularity. Reject background blobs, duplicate fragments of one parent, ambiguous or badly segmented masks, and targets too small to judge.
Return JSON only with keys: accept (bool), scene_summary, difficulty_tags (array), selected_object_ids (array), rejected_object_ids (array), protection_notes (array), evidence (string).
Candidates (object_id, category, bbox, area_fraction): {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}"""


def planning_prompt(candidate: dict[str, Any], selected_ids: list[str], difficulty: dict[str, Any], *, required_operation: str | None = None) -> str:
    operation_rule = _required_operation_rule(required_operation)
    selected = [
        {k: o.get(k) for k in ("object_id", "category", "bbox", "area_fraction") if o.get(k) is not None}
        for o in candidate.get("objects", []) if str(o.get("object_id")) in set(selected_ids)
    ]
    for item in selected:
        obj = next(o for o in candidate.get('objects', []) if str(o.get('object_id')) == str(item['object_id']))
        item['resolved_execution_region_available'] = bool(obj.get('metadata', {}).get('execution_mask_path'))
    # Earlier selection may have seen a colored overlay. Do not recycle its
    # appearance descriptions as visual evidence for the clean-image planner.
    difficulty = {k: v for k, v in difficulty.items()
                  if k in {'difficulty_score', 'mechanisms', 'quality_target_ids', 'target_features', 'target_count'}}
    prompt = f"""You are designing benchmark-v2-aligned local edits for a selected SA-1B scene.
{operation_rule}
The supplied panel has TWO aligned halves: LEFT is the clean source photograph, RIGHT is a grayscale mask-only map with object_id labels. Read ALL colors, materials, textures, object identities and scene context from the LEFT photograph only. The RIGHT half is only for locating masks. Never interpret guide pixels, labels or annotation descriptions as source appearance; visible clean-image evidence overrides metadata. If appearance is uncertain, omit the color from ref_phrase instead of guessing.
ATTRIBUTE SURFACE CONTRACT: Identify the real editable object/part, never silently reinterpret a person mask as clothing or a window mask as only its frame. Return edit_surface and clean-source source_color when relevant. A host selector may support target_part only when resolved_execution_region_available is true; otherwise select a precise part mask or skip the task. Color, material, finish, pattern, texture, text, tint and opacity are distinct. Material must name a real material; matte black is color/finish. Preserve unrequested geometry and appearance. Tint preserves transparency; change opacity only when explicitly requested. Never invent filesystem paths.


Each selected object can become at most one independent unit. Choose exactly one suitable operation for each accepted object from add, replace, remove, attribute, using the operation_quality table below. If no natural, locally feasible edit exists, omit that object from units and report its object_id and reason in skipped_objects; never force a payload to fill a quota. Prefer different operation types when the mask is eligible, but never force an unsuitable operation merely to balance a scene; global operation assignments are made before selection; preserve the assigned type even when rejecting a target. Use the visible selected mask as the source semantic selector; never invent a mask or count fragments as separate objects.
UNIFIED INSTRUCTION PROTOCOL 1.0: Return object_id, operation, ref_phrase (source-instance phrase), scope (visually verified short source-instance noun, not a blindly copied metadata alias), payload, target_part (optional named part of the selected host), remove_mode (whole/part/contents/contents_only/lowerlegs/hem), placement (for add: relation, optional site, anchor_mode=support/reference), attributes (for attribute: list of property/value records), edit_surface, source_color, evidence. For simple attribute, attribute_kind plus payload is accepted. For other operations attributes must be empty. Defaults: no target_part; remove_mode=whole; add relation=to. For remove leave payload empty; encode the part and extent in target_part/remove_mode/qualifier. With placement.site, choose a concrete spatial relation rather than to. Ref/no-ref are deterministically rendered from these SAME slots, replacing only the source-instance phrase with the selected <scope>; retain all part, count, result and placement semantics. Do not return two independently authored instructions.
{INSTRUCTION_STYLE}
Forms: Add <item> <relation> <anchor>. Replace <target> with <replacement>. Remove <target>. Remove <part> from <host>. Remove <target>, including its contents. Remove the contents from <host>. Remove the lower section of <target> to make it knee-length. Change the <property> of <target> to <value>. Tint <target> <color>. Make <target> translucent/opaque. Text values are exact strings; do not add quotation marks to the stored value. One source unit has one operation; compatible attributes on that same target may be combined, but do not mix operations. Keep one short sentence; aim for ref<=26 and no-ref<=16 words without dropping necessary meaning. No minimum word count.
ADD PLACEMENT: payload is only the new physical item plus necessary count/appearance. Put relation and site in placement instead of embedding ambiguous its/it references in payload. Example: payload=a beige cushion, placement relation=on, site=seat, anchor_mode=support. Preserve the source anchor. A reference placement such as beside requires resolved_execution_region_available=true; otherwise skip. Add a physical patch = add; change a printed surface pattern = attribute; erase an existing logo/text = remove; change the existing text content = attribute.text. No moving objects or pose editing under these four labels.
EXECUTABILITY: target_part, contents_only, lowerlegs and hem need resolved_execution_region_available=true. Use them only when available; never replace a missing part mask with the entire host. remove_mode=part requires target_part. For attributes supply one or more of color/material/finish/pattern/texture/text; tint and opacity each stand alone. Preserve source identity, credible scene role and real support relationships.

CONTENT DESIGN: Treat these as ordinary photo-retouching requests, not demonstrations of arbitrary model capabilities. Before writing a payload, inspect the visible object category, function, current appearance, surrounding scene, scale, and attachment/support relationships. Consider a few feasible edits internally and choose the most contextually convincing one. The result should look like a plausible alternate photograph of the same scene. Do not invent an unseen backstory to justify an odd edit. Visible change is necessary, but novelty, saturation, and maximum contrast are not goals. Preserve independent single-target tasks; one unit must not rely on another unit being applied first.
ADD: Choose one kind of absent, useful or conventional accessory, with the requested count and a visible, physically suitable attachment/resting site on the selected host. Specify enough about the item to determine normal size and placement, within the short instruction budget. Conditional examples: a seat cushion on a bare chair seat, a luggage tag on a suitcase with an accessible handle. These are illustrations, not defaults. Reject arbitrary bows/tags/badges on unrelated architecture, tires, plants or tools; decorative additions require a visible scene-based reason. Do not obscure the host's main function, attach to a neighboring object, require new supports, or enlarge the addition merely to make it noticeable. If it cannot be both naturally sized and judgeable, skip it; never switch an assigned operation.
REPLACE: Choose a recognizable alternative that serves a compatible scene role and fits the old object's footprint, orientation, support/contact points and local editable region. Conditional examples: replace a standalone dining chair with another dining-chair design, or a freestanding flower pot with a different pot design. A same-category replacement must change visible design/form, not merely color or material (that is attribute). Avoid unrelated category swaps, blocked access, floating objects, and replacements needing changes to surrounding furniture, people, hands, supports or shadows outside the editable region. Do not demand large shape expansion beyond the old mask and narrow collar.
ATTRIBUTE: Prefer a plausible product color, paint finish, fabric color, or manufactured material variant appropriate to this particular object and scene. Use a visibly different but credible value; bright colors are allowed where conventional, not as a universal default. Preserve unrequested markings and component boundaries; do not treat mixed-material objects as a uniform painted blob. Use material only when the same geometry can plausibly be manufactured in that material and appearance alone conveys it. Avoid implausible metallic/glass/wood transformations of skin, food, foliage or flexible fabric. Do not change thickness or construction. Change transparency only for an explicit opacity request; tint preserves it. If a natural surface change cannot be isolated by this mask, skip it; never switch an assigned operation. Do not choose a value already present.
Keep ref_phrase grounded in the SOURCE appearance and target identity. Keep payload concise and concrete; put rationale in evidence, not in the benchmark instruction. Ref and no-ref must request identical content; the mask supplies the instance selector, not missing edit semantics.
For remove, choose a self-contained target or an explicitly named removable part with a resolved execution region; preserve the remaining host. For replace, the source selector or resolved part region must identify the complete visible old identity. For add, use a sufficiently large, stable host and a clearly visible attached item; avoid tiny stickers, droplets, bells, or other changes that will be below the generation resolution. For repeated same-category instances, ground every positional phrase (leftmost, rightmost, far right, to the right of center, and similar) against the selected object_id and visible mask overlay; never assign a phrase to a neighboring instance or reuse contradictory superlatives for different masks. Return JSON only: {{\"skipped_objects\":[{{\"object_id\":...,\"reason\":...}}],\"units\":[{{\"object_id\":...,\"operation\":...,\"ref_phrase\":...,\"scope\":...,\"payload\":...,\"attribute_kind\":null,\"target_part\":null,\"remove_mode\":\"whole\",\"placement\":null,\"attributes\":[],\"qualifier\":\"\",\"evidence\":...}}]}}.
Difficulty evidence: {json.dumps(difficulty, ensure_ascii=False)}
Selected candidates: {json.dumps(selected, ensure_ascii=False)}"""

    if required_operation is not None:
        prompt = prompt.replace("Choose exactly one suitable operation for each accepted object from add, replace, remove, attribute", "Use only the assigned operation for every accepted object")
        prompt = prompt.replace("Prefer different operation types when the mask is eligible, but never force an unsuitable operation merely to balance a scene;", "Reject targets unsuitable for the assigned operation;")
    return prompt
