"""Regression coverage for large-node audit memory and durable batch resume."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import pytest

from synthesis_pipeline import audit_edit_pairs as audit
from synthesis_pipeline.run_multinode_multitype_labeling import merge_results, run_stage
from synthesis_pipeline.stage_labeling_model import stage


def payload():
    return {
        **dict.fromkeys(audit.AUDIT_FAILURE_KEYS), "quality": "pass",
        "source_inventory": "gray square", "edited_inventory": "blue square",
        "reason": "The selected square changed color cleanly.",
    }


def test_interrupted_batch_resumes_without_preloading_images(tmp_path, monkeypatch):
    source = tmp_path / "source"
    edited = tmp_path / "edited"
    source.mkdir()
    edited.mkdir()
    Image.new("RGB", (32, 32), "gray").save(source / "source.png")
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode()
    rows = []
    for i in range(5):
        name = f"{i:03d}_case.png"
        Image.new("RGB", (32, 32), "blue").save(edited / name)
        rows.append(dict(image=name, source_image="source.png", mask=rle,
                         task_type="attribute", editing_instruction="Recolor the square blue."))
    manifest = tmp_path / "annotations.jsonl"
    audit.write_jsonl(manifest, rows)
    args = SimpleNamespace(
        annotations_jsonl=manifest, source_dir=source, edited_dir=edited,
        out_dir=tmp_path / "audit", max_items=None, resume=True,
        batch_size=2, guard_pixels=24, vlm="qwen38-vllm", vlm_model_id="local",
        vlm_model_identity="original", vlm_device="cuda:0", vlm_dtype="bf16",
        max_new_tokens=1024,
    )
    monkeypatch.setattr(audit, "parse_args", lambda: args)
    monkeypatch.setattr(audit.vlm, "configure_backend", lambda **kwargs: None)
    monkeypatch.setattr(audit.vlm, "shutdown_backend", lambda: None)
    prepared = []
    model_ready = False
    original_prepare = audit.prepare_audit_task

    def prepare(task, config):
        assert model_ready, "Images must not be retained before model initialization"
        prepared.append(task["row"]["image"])
        return original_prepare(task, config)

    monkeypatch.setattr(audit, "prepare_audit_task", prepare)

    class Backend:
        calls = 0
        interrupt = True

        def chat_batch(self, messages, **kwargs):
            self.calls += 1
            assert len(messages) <= 2
            if self.interrupt and self.calls == 2:
                raise RuntimeError("injected interruption")
            return [json.dumps(payload()) for _ in messages]

    backend = Backend()

    def load_backend():
        nonlocal model_ready
        model_ready = True
        return backend

    monkeypatch.setattr(audit.vlm, "get_backend", load_backend)
    with pytest.raises(RuntimeError, match="injected interruption"):
        audit.main()
    checkpoint = args.out_dir / "edit_audit.jsonl"
    first_batch = audit.load_jsonl(checkpoint)
    assert [r["image"] for r in first_batch] == [r["image"] for r in rows[:2]]
    prepared.clear()
    backend.interrupt = False
    audit.main()
    assert prepared == [r["image"] for r in rows[2:]]
    final = audit.load_jsonl(checkpoint)
    assert final[:2] == first_batch
    assert len(final) == 5
    assert all(audit.reusable_audit_record(r) for r in final)
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["reused_cases"] == 2 and summary["audited_cases"] == 3
    prepared.clear()
    monkeypatch.setattr(audit.vlm, "get_backend", lambda: pytest.fail("complete run loaded model"))
    audit.main()
    assert prepared == []


def test_transformers_staging_repairs_only_bad_files(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights.safetensors").write_bytes(b"original")
    destination = tmp_path / "cache"
    stage(source, destination)
    (destination / "weights.safetensors").write_bytes(b"corrupted")
    stage(source, destination, resume=True)
    assert (destination / "weights.safetensors").read_bytes() == b"original"


def test_multitype_merge_includes_valid_nested_audit(tmp_path):
    row = dict(image="000_case.png", source_image="source.png", task_type="attribute", mask={})
    audit.write_jsonl(tmp_path / "inputs/node0/annotations.jsonl", [row])
    root = tmp_path / "nodes/node0"
    audit.write_jsonl(root / "planning/regions/annotations.jsonl", [row])
    audit.write_jsonl(root / "generation/context_grounded_v4_qwen21/annotations.jsonl", [row])
    audit.write_jsonl(root / "audit/edit_audit.jsonl", [
        dict(image=row["image"], quality="pass", audit=payload(), input_fingerprint="fp")
    ])
    summary = merge_results(tmp_path, 1)
    assert summary["audited_cases"] == 1
    assert len(audit.load_jsonl(tmp_path / "results/model_pass.jsonl")) == 1


def test_audit_watchdog_times_out_without_progress(tmp_path):
    import sys
    with pytest.raises(TimeoutError, match="No audit stage/batch progress"):
        run_stage([sys.executable, "-c", "import time; time.sleep(30)"],
                  tmp_path / "audit.log", tmp_path, timeout=30,
                  progress_path=tmp_path / "progress.json", progress_timeout=0.1)
