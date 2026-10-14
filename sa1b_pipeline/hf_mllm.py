"""Local Transformers VLM adapter for single- and multi-image planning."""
import time
from PIL import Image
from .mllm import parse_json


class HFSelector:
    model_id = '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B'

    def __init__(self, model_id=None, max_new_tokens=2400):
        self.model_id = model_id or self.model_id
        self.max_new_tokens = max_new_tokens
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            low_cpu_mem_usage=True, trust_remote_code=True)
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(self.model_id)

    def call(self, paths, prompt, tokens):
        images = []
        for p in paths:
            im = Image.open(p).convert('RGB')
            im.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            images.append({'type': 'image', 'image': im})
        inputs = self.proc.apply_chat_template(
            [{'role': 'user', 'content': images + [{'type': 'text', 'text': prompt}]}],
            tokenize=True, add_generation_prompt=True, return_tensors='pt',
            return_dict=True, enable_thinking=False)
        inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=tokens, do_sample=False, use_cache=True)
        raw = self.proc.batch_decode([out[0][inputs['input_ids'].shape[1]:]], skip_special_tokens=True)[0]
        try:
            return parse_json(raw), raw
        except ValueError as exc:
            return {'_parse_error': str(exc), 'decision': 'fail', 'failure_reasons': ['malformed_mllm_json']}, raw

    def generate(self, jobs):
        outputs = []
        for path, prompt in jobs:
            start = time.time()
            result, raw = self.call([path], prompt, 2400)
            result['_raw_output'] = raw
            outputs.append(result)
            print(f'planned {len(outputs)}/{len(jobs)} {time.time()-start:.1f}s', flush=True)
        return outputs

    def generate_multimage(self, jobs):
        outputs = []
        for paths, prompt in jobs:
            result, raw = self.call(paths, prompt, self.max_new_tokens)
            result['_raw_output'] = raw
            outputs.append(result)
        return outputs
