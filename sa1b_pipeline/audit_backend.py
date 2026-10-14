"""Batched audit adapter retaining the Transformers audit's visual inputs."""
from __future__ import annotations

from PIL import Image

from .mllm import parse_json


def audit_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert('RGB')
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        images.append(image)
    return images


def parse_audit(raw):
    try:
        return parse_json(raw)
    except ValueError as exc:
        return {'_parse_error': str(exc), 'decision': 'fail',
                'failure_reasons': ['malformed_mllm_json']}


class VLLMAuditor:
    def __init__(self, model_id, *, batch_size=4, max_model_len=16384,
                 gpu_memory_utilization=.88, max_tokens=512, cache_inputs=True, check_operation=False):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from .model_cache import resolve_model
        model_id = resolve_model(model_id)
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.processor = AutoProcessor.from_pretrained(model_id)
        cache_options = {} if cache_inputs else {'mm_processor_cache_gb': 0}
        self.llm = LLM(
            model=model_id, dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=max_model_len, max_num_seqs=batch_size,
            max_num_batched_tokens=8192, gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True, trust_remote_code=True,
            limit_mm_per_prompt={'image': 3, 'video': 0},
            mm_encoder_attn_backend='TORCH_SDPA', enable_prefix_caching=cache_inputs,
            **cache_options)
        extra_sampling = {}
        if check_operation:
            from vllm.sampling_params import StructuredOutputsParams

            from .validate import OPERATION_AUDIT_SCHEMA
            extra_sampling['structured_outputs'] = StructuredOutputsParams(json=OPERATION_AUDIT_SCHEMA)
        self.params = SamplingParams(temperature=0, max_tokens=max_tokens, top_p=1, **extra_sampling)
        self.batch_size = batch_size

    def call_batch(self, jobs):
        if not 0 < len(jobs) <= self.batch_size:
            raise ValueError('audit batch outside configured capacity')
        requests = []
        for job in jobs:
            images = audit_images(job['images'])
            content = [{'type': 'image', 'image': im} for im in images]
            content.append({'type': 'text', 'text': job['prompt']})
            prompt = self.processor.apply_chat_template(
                [{'role': 'user', 'content': content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            requests.append({'prompt': prompt, 'multi_modal_data': {'image': images}})
        outputs = self.llm.generate(requests, self.params, use_tqdm=False)
        if len(outputs) != len(jobs):
            raise RuntimeError('audit backend returned an incomplete batch')
        # Offline LLM.generate returns outputs in input order.
        result = []
        for job, output in zip(jobs, outputs):
            completion = output.outputs[0]
            raw = completion.text
            audit = parse_audit(raw)
            if completion.finish_reason == 'length':
                audit = {**audit, 'decision': 'fail', 'failure_reasons':
                         list(audit.get('failure_reasons', [])) + ['truncated_audit']}
            result.append({'job_id': job['job_id'], 'semantic_audit': audit,
                           'semantic_audit_raw': raw,
                           'input_tokens': len(output.prompt_token_ids or []),
                           'output_tokens': len(completion.token_ids),
                           'finish_reason': completion.finish_reason})
        return result

    def close(self):
        shutdown = getattr(getattr(self.llm, 'llm_engine', None), 'shutdown', None)
        if callable(shutdown):
            shutdown()


class TransformersAuditor:
    def __init__(self, model_id, **kwargs):
        from .hf_mllm import HFSelector
        self.model = HFSelector(model_id=model_id)
        self.max_tokens = kwargs.get('max_tokens', 512)

    def call_batch(self, jobs):
        result = []
        for job in jobs:
            audit, raw = self.model.call(job['images'], job['prompt'], self.max_tokens)
            result.append({'job_id': job['job_id'], 'semantic_audit': audit,
                           'semantic_audit_raw': raw})
        return result

    def close(self):
        pass
