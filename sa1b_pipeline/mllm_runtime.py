"""Architecture-specific options shared by the planner and semantic auditor."""
from __future__ import annotations

import json
from pathlib import Path


def architecture_options(model_id: str, tensor_parallel_size: int) -> dict:
    """Keep dense defaults; use the validated Flash-Next FP8 CPU offload path."""
    path = Path(model_id) / "config.json"
    if not path.is_file():
        return {}
    config = json.loads(path.read_text())
    if "Qwen4ExpForConditionalGeneration" not in config.get("architectures", []):
        return {}
    if tensor_parallel_size not in (4, 8):
        raise ValueError("Flash-Next requires the validated TP4 or TEP8 profile")
    options = {"moe_backend": "triton", "engram_config": {"cpu_offload": True},
               "enable_flashinfer_autotune": False}
    if tensor_parallel_size == 8:
        options["enable_expert_parallel"] = True
    return options
