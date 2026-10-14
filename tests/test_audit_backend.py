from types import SimpleNamespace

import pytest
from PIL import Image

from sa1b_pipeline.audit_backend import VLLMAuditor, audit_images


def test_same_thumbnail_and_request_binding(tmp_path):
    paths = []
    for i, size in enumerate([(2000, 1000), (600, 800), (1536, 512)]):
        path = tmp_path/f'{i}.png'
        Image.new('RGB', size, (20*i, 40, 60)).save(path)
        paths.append(str(path))
    actual = audit_images(paths)
    for path, image in zip(paths, actual):
        expected = Image.open(path).convert('RGB')
        expected.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        assert image.size == expected.size
        assert image.tobytes() == expected.tobytes()

    auditor = VLLMAuditor.__new__(VLLMAuditor)
    auditor.batch_size = 2
    auditor.params = object()
    prompts = []

    def template(messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        prompts.append(messages[0]['content'][-1]['text'])
        return prompts[-1]

    auditor.processor = SimpleNamespace(apply_chat_template=template)
    def generate(requests, params, **kwargs):
        assert [r['prompt'] for r in requests] == ['first', 'second']
        return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
            text=text, token_ids=[2], finish_reason=reason)]) for text, reason in
            [('not json', 'stop'), ('{"decision":"pass"}', 'length')]]
    auditor.llm = SimpleNamespace(generate=generate)
    outputs = auditor.call_batch([{'job_id': str(i), 'images': paths, 'prompt': prompt}
                                  for i, prompt in enumerate(['first', 'second'])])
    assert [o['job_id'] for o in outputs] == ['0', '1']
    assert outputs[0]['semantic_audit']['failure_reasons'] == ['malformed_mllm_json']
    assert outputs[1]['semantic_audit']['decision'] == 'fail'
    assert outputs[1]['semantic_audit']['failure_reasons'] == ['truncated_audit']
    with pytest.raises(ValueError):
        auditor.call_batch([])
