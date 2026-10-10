"""Check data mounts and the exact Qwen3.8/vLLM runtime before a long run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(model: Path, handoff: Path) -> dict:
    result = {"model": str(model), "handoff": str(handoff), "checks": {}}
    config = model / "config.json"
    if config.exists():
        data = json.loads(config.read_text())
        result["architecture"] = data.get("architectures", [])
    result["checks"]["model"] = model.is_dir() and config.exists()
    result["checks"]["handoff"] = (handoff / "sa1b_tfd_1k").is_dir() and (handoff / "sa1b_dense_label/annotation").is_dir()
    try:
        import torch
        result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count()}
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        result["checks"]["torch"] = False
    try:
        import vllm
        result["vllm"] = vllm.__version__
        numbers = tuple(int(x) for x in __import__("re").findall(r"\d+", vllm.__version__)[:2])
        result["checks"]["vllm_version"] = numbers >= (0, 28)
        try:
            from vllm.model_executor.models.registry import ModelRegistry
            result["checks"]["qwen_architecture"] = "Qwen3_5ForConditionalGeneration" in ModelRegistry.models
        except Exception as exc:
            result["registry_error"] = f"{type(exc).__name__}: {exc}"
            result["checks"]["qwen_architecture"] = False
    except Exception as exc:
        result["vllm_error"] = f"{type(exc).__name__}: {exc}"
        result["checks"]["vllm_version"] = False
        result["checks"]["qwen_architecture"] = False
    result["ready_for_full_run"] = all(result["checks"].values())
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"))
    p.add_argument("--handoff", type=Path, default=Path("/mnt/hdfs/byte_ttlive_strategy_llm/user/haobo.yuan/datasets"))
    a = p.parse_args(argv)
    result = inspect(a.model, a.handoff)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ready_for_full_run"] else 2)


if __name__ == "__main__":
    main()
