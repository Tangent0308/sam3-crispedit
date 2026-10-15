import json

import pytest

from sa1b_pipeline.mllm_runtime import architecture_options


def test_dense_runtime_is_unchanged(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"]}))
    assert architecture_options(str(tmp_path), 1) == {}


def test_flashnext_profiles_and_unsupported_parallelism(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"]}))
    tp4 = architecture_options(str(tmp_path), 4)
    assert tp4["engram_config"] == {"cpu_offload": True}
    assert tp4["moe_backend"] == "triton"
    assert "enable_expert_parallel" not in tp4
    assert architecture_options(str(tmp_path), 8)["enable_expert_parallel"] is True
    with pytest.raises(ValueError, match="TP4 or TEP8"):
        architecture_options(str(tmp_path), 1)
