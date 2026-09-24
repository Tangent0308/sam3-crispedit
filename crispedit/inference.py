"""Qwen3.8/vLLM inference shared by CrispEdit quality and scene filters."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from PIL import Image

def parse_device_groups(spec: str, tensor_parallel_size: int) -> List[List[int]]:
    devices = [
        int(part.strip().removeprefix("cuda:"))
        for part in spec.split(",")
        if part.strip()
    ]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    if tensor_parallel_size <= 0 or len(devices) % tensor_parallel_size:
        raise ValueError(
            f"{len(devices)} devices cannot be divided into TP={tensor_parallel_size} groups"
        )
    return [
        devices[index : index + tensor_parallel_size]
        for index in range(0, len(devices), tensor_parallel_size)
    ]


def assign_jobs(
    jobs: Sequence[object], groups: Sequence[Sequence[int]]
) -> List[Tuple[List[int], List[object]]]:
    buckets = [{"devices": list(group), "rows": 0, "jobs": []} for group in groups]
    for job in sorted(jobs, key=lambda item: item.num_rows, reverse=True):
        bucket = min(buckets, key=lambda item: item["rows"])
        bucket["jobs"].append(job)
        bucket["rows"] += job.num_rows
    return [
        (item["devices"], item["jobs"])
        for item in buckets
        if item["jobs"]
    ]


def _chunks_by_image_budget(
    conversations: Sequence[List[Dict]], max_images: int
) -> List[List[List[Dict]]]:
    result: List[List[List[Dict]]] = []
    pending: List[List[Dict]] = []
    image_count = 0
    for conversation in conversations:
        count = sum(
            1
            for message in conversation
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image"
        )
        if pending and image_count + count > max_images:
            result.append(pending)
            pending, image_count = [], 0
        pending.append(conversation)
        image_count += count
    if pending:
        result.append(pending)
    return result


class Qwen38FilterEngine:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.torch = torch
        self.args = args
        self.inference_backend = "vllm"
        visible_count = torch.cuda.device_count()
        if visible_count != args.tensor_parallel_size:
            raise RuntimeError(
                f"worker sees {visible_count} GPUs, expected TP={args.tensor_parallel_size}"
            )
        self._init_vllm()


    def _init_vllm(self) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # vLLM/FlashInfer may JIT-compile a kernel by invoking the environment's
        # `ninja` executable. Directly calling venv/bin/python does not prepend
        # venv/bin to PATH, so make that invocation mode self-contained.
        # Do not resolve the interpreter symlink: uv-managed virtualenvs point
        # at the base Python, while console scripts live beside sys.executable.
        environment_bin = str(Path(sys.executable).parent)
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(
            self.args.model_path, trust_remote_code=True, local_files_only=True
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            image_processor.size.longest_edge = int(self.args.max_pixels)
        optional_engine_args = {}
        max_num_seqs = getattr(self.args, "vllm_max_num_seqs", None)
        if max_num_seqs is not None:
            optional_engine_args["max_num_seqs"] = int(max_num_seqs)
        if bool(getattr(self.args, "vllm_enforce_eager", False)):
            optional_engine_args["enforce_eager"] = True
        self.model = LLM(
            model=self.args.model_path,
            tensor_parallel_size=self.args.tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            seed=0,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.vllm_max_model_len,
            limit_mm_per_prompt={"image": 2},
            mm_processor_kwargs={"max_pixels": int(self.args.max_pixels)},
            generation_config="vllm",
            **optional_engine_args,
        )
        # This is the vLLM equivalent of transformers.generate with
        # do_sample=False and all sampling warpers disabled.
        self.SamplingParams = SamplingParams
        self.sampling_params_by_max_tokens = {}
        self.sampling_params = self._sampling_params(self._locator_max_tokens())


    def _locator_max_tokens(self) -> int:
        value = getattr(self.args, "locator_max_new_tokens", None)
        if value is None:
            value = getattr(self.args, "max_new_tokens", 1024)
        return int(value)


    def _sampling_params(self, max_tokens: int):
        max_tokens = int(max_tokens)
        cached = self.sampling_params_by_max_tokens.get(max_tokens)
        if cached is not None:
            return cached
        params = self.SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            seed=0,
            skip_special_tokens=True,
        )
        self.sampling_params_by_max_tokens[max_tokens] = params
        return params


    @staticmethod
    def _conversation_images(conversation: List[Dict]) -> List[Image.Image]:
        return [
            part["image"]
            for message in conversation
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image"
        ]


    def _generate_vllm(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        requests = []
        for conversation in conversations:
            images = self._conversation_images(conversation)
            if len(images) not in (1, 2):
                raise ValueError(
                    f"CrispEdit vLLM request requires one or two images, got {len(images)}"
                )
            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            requests.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images},
                    "mm_processor_kwargs": {
                        "max_pixels": int(self.args.max_pixels),
                    },
                }
            )
        sampling_params = (
            self.sampling_params
            if max_tokens is None
            else self._sampling_params(int(max_tokens))
        )
        results = self.model.generate(
            requests,
            sampling_params,
            use_tqdm=False,
        )
        if len(results) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(results)} results for {len(requests)} requests"
            )
        outputs = []
        for result in results:
            if len(result.outputs) != 1:
                raise RuntimeError(
                    f"vLLM returned {len(result.outputs)} candidates; expected exactly one"
                )
            outputs.append(result.outputs[0].text)
        return outputs


    def _generate_once(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        return self._generate_vllm(conversations, max_tokens=max_tokens)


    def _generate_backoff(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        try:
            return self._generate_once(conversations, max_tokens=max_tokens)
        except self.torch.cuda.OutOfMemoryError:
            if len(conversations) <= 1:
                raise
            for index in range(self.torch.cuda.device_count()):
                with self.torch.cuda.device(index):
                    self.torch.cuda.empty_cache()
            midpoint = len(conversations) // 2
            return self._generate_backoff(
                conversations[:midpoint], max_tokens=max_tokens
            ) + self._generate_backoff(
                conversations[midpoint:], max_tokens=max_tokens
            )


    def generate(
        self, conversations: Sequence[List[Dict]], max_tokens: int | None = None
    ) -> List[str]:
        outputs: List[str] = []
        for chunk in _chunks_by_image_budget(
            conversations, self.args.max_images_per_generate
        ):
            outputs.extend(self._generate_backoff(chunk, max_tokens=max_tokens))
        return outputs


    @staticmethod
    def _corrective_conversation(
        conversation: List[Dict], invalid_text: str, correction_prompt: str
    ) -> List[Dict]:
        return list(conversation) + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(invalid_text)}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": correction_prompt}],
            },
        ]


    def _parse_with_retry(
        self,
        conversation: List[Dict],
        text: str,
        parser,
        correction_builder,
        max_tokens: int,
    ):
        error = ""
        parsed = None
        attempts = []
        for attempt in range(self.args.parse_retries + 1):
            try:
                parsed = parser(text)
                error = ""
            except Exception as exc:
                error = repr(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_text": text,
                    "parse_ok": not error,
                    "error": error,
                }
            )
            if not error:
                break
            if attempt < self.args.parse_retries:
                correction_prompt = correction_builder(error)
                attempts[-1]["correction_prompt"] = correction_prompt
                retry_conversation = self._corrective_conversation(
                    conversation, text, correction_prompt
                )
                text = self.generate(
                    [retry_conversation], max_tokens=max_tokens
                )[0]
        return text, parsed, error, attempts

