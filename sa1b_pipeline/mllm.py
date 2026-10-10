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


class VLLMSelector:
    """Continuous-batching selector/planner for Qwen3.8-27B."""

    def __init__(self, model_id: str, *, tensor_parallel_size: int = 1, max_model_len: int = 8192,
                 max_new_tokens: int = 1200, temperature: float = 0.0):
        from transformers import AutoProcessor  # type: ignore
        from vllm import LLM, SamplingParams  # type: ignore
        self.SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.llm = LLM(model=model_id, dtype="bfloat16", tensor_parallel_size=tensor_parallel_size,
                       max_model_len=max_model_len, trust_remote_code=True,
                       limit_mm_per_prompt={"image": 3}, enable_prefix_caching=True)
        self.params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens,
                                     top_p=1.0, stop=["<|im_end|>"])

    def _messages(self, image_path: str, prompt: str) -> tuple[str, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return rendered, image

    def _multi_messages(self, image_paths: list[str], prompt: str) -> tuple[str, list[Image.Image]]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        rendered = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        return rendered, images

    def generate(self, jobs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        requests = []
        images = []
        for image_path, text in jobs:
            prompt, image = self._messages(image_path, text)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": [image]}})
            images.append(image)
        outputs = self.llm.generate(requests, self.params, use_tqdm=False)
        result = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            result.append(parse_json(text))
        return result

    def generate_multimage(self, jobs: list[tuple[list[str], str]]) -> list[dict[str, Any]]:
        requests = []
        for paths, text in jobs:
            prompt, images = self._multi_messages(paths, text)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        outputs = self.llm.generate(requests, self.params, use_tqdm=False)
        return [parse_json(output.outputs[0].text if output.outputs else "") for output in outputs]


def selection_prompt(candidate: dict[str, Any], *, min_targets: int = 2, max_targets: int = 5) -> str:
    objects = candidate.get("objects", [])
    compact = [{k: o.get(k) for k in ("object_id", "category", "description", "bbox", "area_fraction") if o.get(k) is not None} for o in objects]
    return f"""You are curating hard SA-1B source scenes for a fine-grained interactive image-editing training set.
Inspect the image, then use the supplied dense-object candidates only as identity/mask hints. Select {min_targets} to {max_targets} DIFFERENT physical objects or parts in this same scene.
Prefer multiple near-duplicate instances, occlusion, touching boundaries, thin straps/rods, holes, disconnected visible fragments, small parts, and mixed object/part granularity. Reject background blobs, duplicate fragments of one parent, ambiguous or badly segmented masks, and targets too small to judge.
Return JSON only with keys: accept (bool), scene_summary, difficulty_tags (array), selected_object_ids (array), rejected_object_ids (array), protection_notes (array), evidence (string).
Candidates: {json.dumps(compact, ensure_ascii=False)}"""


def planning_prompt(candidate: dict[str, Any], selected_ids: list[str], difficulty: dict[str, Any]) -> str:
    selected = [
        {k: o.get(k) for k in ("object_id", "category", "description", "bbox", "area_fraction") if o.get(k) is not None}
        for o in candidate.get("objects", []) if str(o.get("object_id")) in set(selected_ids)
    ]
    return f"""You are designing benchmark-v2-aligned local edits for a selected SA-1B scene.
Each selected object becomes one independent unit. Assign a balanced mixture of exactly these operations: add, replace, remove, attribute. For 2–4 targets do not repeat an operation; for 5 targets repeat at most one operation. Use the visible selected mask as the source semantic selector; never invent a mask or count fragments as separate objects.
For each unit return object_id, operation, ref_phrase (a concise image-grounded referring phrase), scope (the visible object/part), payload (a concrete, visibly judgeable local change), attribute_kind (color/material for attribute, otherwise null), evidence. Instructions must follow: Add <new item> to <target>. / Replace <target> with <new object/part>. / Remove <target>. / Change the color/material of <target> to <value>. Additions must attach to the selected host; replacement must change identity/form; attribute preserves geometry; removal must be self-contained.
Return JSON only: {{\"units\":[{{\"object_id\":...,\"operation\":...,\"ref_phrase\":...,\"scope\":...,\"payload\":...,\"attribute_kind\":null,\"evidence\":...}}]}}.
Difficulty evidence: {json.dumps(difficulty, ensure_ascii=False)}
Selected candidates: {json.dumps(selected, ensure_ascii=False)}"""
