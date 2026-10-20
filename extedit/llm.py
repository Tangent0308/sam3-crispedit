"""Qwen3.8-27B behind a vLLM OpenAI-compatible server: the serve command and an async client.

Settings as in the protocol adaptation and the edit check that produced D3 / D4 (samtok_edit
src/samtok_edit21/preparation/edit_check.py and protocol_adapt.Client): thinking calls use the model card's
thinking preset with reasoning effort "low"; transport errors are retried, then raised.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

MODEL = "/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
# call: (thinking, reasoning effort, max output tokens)
CALLS = {"plan": (True, "low", 12000), "audit": (True, "low", 8000)}
TRANSPORT_ATTEMPTS = 6


def serve_command(vllm, model=MODEL, gpus=8, port=8000, max_model_len=20000, max_num_seqs=96,
                  gpu_memory_utilization=0.9, name="qwen38-27b"):
    """One data-parallel server per node (one engine per GPU), prefix caching on, up to 6 images per prompt."""
    return [str(vllm), "serve", str(model), "--served-model-name", name, "--data-parallel-size", str(gpus),
            "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-num-seqs", str(max_num_seqs), "--enable-prefix-caching", "--reasoning-parser", "qwen3",
            "--limit-mm-per-prompt", json.dumps({"image": 6}), "--host", "127.0.0.1", "--port", str(port)]


def parse_json(text):
    raw = text.rsplit("</think>", 1)[-1].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {raw[:200]}")
    return value


class Client:
    """OpenAI-compatible chat endpoint."""

    def __init__(self, session, url, model="qwen38-27b"):
        self.session, self.url, self.model = session, url, model

    async def call(self, parts, prompt, call, seed=0):
        thinking, effort, max_tokens = CALLS[call]
        body = {"model": self.model, "max_tokens": max_tokens, "seed": seed,
                "messages": [{"role": "user", "content": parts + [{"type": "text", "text": prompt}]}],
                "chat_template_kwargs": {"enable_thinking": thinking, **({"reasoning_effort": effort} if effort else {})},
                **THINKING}
        error = None
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
                async with self.session.post(self.url, json=body) as response:
                    data = await response.json(content_type=None)
                if "choices" not in data:
                    raise RuntimeError(str(data)[:500])
                choice = data["choices"][0]
                return {"text": choice["message"].get("content") or "", "finish": choice.get("finish_reason"),
                        "tokens": (data.get("usage") or {}).get("completion_tokens", 0)}
            except Exception as exc:  # noqa: BLE001 - retried, then raised below
                error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2 + 5 * attempt)
        raise RuntimeError(f"{call} request failed {TRANSPORT_ATTEMPTS} times: {error}")


def read_records(path):
    """{id: record} of a JSONL stage file, the last line of an id winning; records that failed with an error are
    left out, so a resumed run retries them."""
    path = Path(path)
    out = {}
    if not path.exists():
        return out
    for line in path.open():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in record:
            out.pop(record.get("id"), None)
        else:
            out[record["id"]] = record
    return out


async def run_jobs(jobs, worker, concurrency, out_path, progress_every=20):
    """Run `worker(job)` over jobs with bounded concurrency, appending each result as one JSON line."""
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = 0
    with open(out_path, "a") as stream:
        async def one(job):
            nonlocal done
            async with semaphore:
                try:
                    record = await worker(job)
                except Exception as exc:  # noqa: BLE001 - recorded, the row is retried on the next run
                    record = {"id": job["id"], "error": f"{type(exc).__name__}: {exc}"[:500]}
            async with lock:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                done += 1
                if done % progress_every == 0:
                    print(f"  {done}/{len(jobs)}", flush=True)
        await asyncio.gather(*(one(job) for job in jobs))
