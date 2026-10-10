"""Qwen3.8-27B vLLM adapter with resumable, structured JSON output.

The import is lazy so geometry and unit tests work on CPU-only machines.
"""
from __future__ import annotations

import json
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
                 max_new_tokens: int = 1200, temperature: float = 0.0):
        # Reuse the parent repository's production Qwen3.8 backend. It pins
        # the multimodal processor settings and the non-thinking chat template
        # used by the completed derived-edit runs.
        # The inherited production adapter is useful for a one-GPU smoke, but
        # it intentionally hard-codes tensor_parallel_size=1. Full selection
        # runs must honor the caller's TP setting and use the direct vLLM path.
        self._inherited_backend = False
        if tensor_parallel_size == 1:
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
        self.llm = LLM(model=model_id, dtype="bfloat16", tensor_parallel_size=tensor_parallel_size,
                       max_model_len=max_model_len, trust_remote_code=True,
                       max_num_seqs=8, max_num_batched_tokens=8192,
                       limit_mm_per_prompt={"image": 3}, enable_prefix_caching=True,
                       gpu_memory_utilization=0.85, enforce_eager=True)
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


def selection_prompt(candidate: dict[str, Any], *, min_targets: int = 2, max_targets: int = 5) -> str:
    objects = candidate.get("objects", [])
    # Keep the candidate index well below the 8k context budget on crowded
    # SA-1B scenes. The image and numbered mask overlay carry the visual
    # description; category/geometry metadata is sufficient for object IDs.
    compact = [
        [str(o.get("object_id", "")), str(o.get("category", ""))[:48], o.get("bbox"), o.get("area_fraction")]
        for o in objects
    ]
    return f"""You are curating hard SA-1B source scenes for a fine-grained interactive image-editing training set.
Inspect the image, then use the supplied dense-object candidates only as identity/mask hints. Select {min_targets} to {max_targets} DIFFERENT physical objects or parts in this same scene.
Prefer multiple near-duplicate instances, occlusion, touching boundaries, thin straps/rods, holes, disconnected visible fragments, small parts, and mixed object/part granularity. Reject background blobs, duplicate fragments of one parent, ambiguous or badly segmented masks, and targets too small to judge.
Return JSON only with keys: accept (bool), scene_summary, difficulty_tags (array), selected_object_ids (array), rejected_object_ids (array), protection_notes (array), evidence (string).
Candidates (object_id, category, bbox, area_fraction): {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}"""


def planning_prompt(candidate: dict[str, Any], selected_ids: list[str], difficulty: dict[str, Any]) -> str:
    selected = [
        {k: o.get(k) for k in ("object_id", "category", "description", "bbox", "area_fraction") if o.get(k) is not None}
        for o in candidate.get("objects", []) if str(o.get("object_id")) in set(selected_ids)
    ]
    return f"""You are designing benchmark-v2-aligned local edits for a selected SA-1B scene.
Each selected object becomes one independent unit. Assign a balanced mixture of exactly these operations: add, replace, remove, attribute. For 2–4 targets do not repeat an operation; for 5 targets repeat at most one operation. Use the visible selected mask as the source semantic selector; never invent a mask or count fragments as separate objects.
For each unit return object_id, operation, ref_phrase (a concise image-grounded referring phrase beginning with a lowercase article when needed), scope (a short noun phrase for the visible object/part, at most four words, without location words), payload (a concrete, visibly judgeable local change), attribute_kind (color/material for attribute, otherwise null), evidence. Instructions must follow the benchmark-v2 forms: Add <new item> to <target>. / Replace <target> with <new object/part>. / Remove <target>. / Change the color/material of <target> to <value>. The no-ref form is generated as Add <new item> to the selected <scope>. / Replace the selected <scope> with <new object/part>. / Remove the selected <scope>. / Change the color/material of the selected <scope> to <value>. For add, payload MUST be a new physical item absent from the source (for example, a small pocket, badge, cushion, tag, or attached accessory); never write an existing surface, top, center, or section of the selected host as the payload. Additions must attach to the selected host; replacement must change identity/form; attribute preserves geometry; removal must be self-contained.
Return JSON only: {{\"units\":[{{\"object_id\":...,\"operation\":...,\"ref_phrase\":...,\"scope\":...,\"payload\":...,\"attribute_kind\":null,\"evidence\":...}}]}}.
Difficulty evidence: {json.dumps(difficulty, ensure_ascii=False)}
Selected candidates: {json.dumps(selected, ensure_ascii=False)}"""
