"""Load a Qwen image editor with the model's official Diffusers pipeline.

Qwen-Image-Edit-2511 and Qwen-Image-2.1 use different pipeline classes and
sampling APIs.  Keep backend selection explicit and local so upgrading the 2.1
environment cannot silently change the frozen 2511 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path


QWEN21_CLASS = "QwenImage21Pipeline"
QWEN21_OMNI_CLASS = "QwenImage21OmniPipeline"


def detect_qwen_family(model_id: str, requested: str = "auto") -> str:
    if requested not in {"auto", "qwen2511", "qwen21"}:
        raise ValueError(f"Unknown Qwen image family: {requested}")
    if requested != "auto":
        return requested
    path = Path(model_id)
    if path.is_dir() and (path / "model_index.json").is_file():
        model_index = json.loads((path / "model_index.json").read_text())
        if model_index.get("_class_name") == QWEN21_CLASS:
            return "qwen21"
    return "qwen21" if "qwen-image-2.1" in model_id.lower() else "qwen2511"


def is_qwen21_pipeline(pipe) -> bool:
    return pipe.__class__.__name__ in {QWEN21_CLASS, QWEN21_OMNI_CLASS} or bool(
        getattr(pipe, "is_qwen21", False)
    )


def is_qwen21_omni_pipeline(pipe) -> bool:
    return pipe.__class__.__name__ == QWEN21_OMNI_CLASS or getattr(
        pipe, "backend_name", None
    ) == "vllm-omni"


def load_qwen_pipeline(
    model_id: str,
    device: str,
    torch_dtype,
    cpu_offload: str,
    model_family: str = "auto",
):
    family = detect_qwen_family(model_id, model_family)
    if family == "qwen21":
        from diffusers import QwenImage21Pipeline

        # `dtype` is the official 2.1 Diffusers argument.  Its recommended
        # sampler is 40 steps without classifier-free guidance.
        pipeline = QwenImage21Pipeline.from_pretrained(model_id, dtype=torch_dtype)
    else:
        from diffusers import QwenImageEditPlusPipeline

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )

    if cpu_offload != "none":
        if cpu_offload == "sequential":
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    return pipeline


def load_qwen21_omni_pipeline(
    model_id: str,
    *,
    tensor_parallel_size: int = 1,
    **omni_kwargs,
):
    """Load Qwen-Image-2.1 through vLLM-Omni's official ``Omni`` API."""
    from utils.qwen21_omni import QwenImage21OmniPipeline

    return QwenImage21OmniPipeline(
        model_id, tensor_parallel_size=tensor_parallel_size, **omni_kwargs
    )
