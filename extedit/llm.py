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
GREEDY = {"temperature": 0.0}  # reading text back: the most likely reading, the same on every run
# call: (thinking, reasoning effort, max output tokens)
CALLS = {"plan": (True, "low", 12000), "audit": (True, "low", 8000), "ocr": (False, None, 400),
         "natcheck": (True, "low", 8000), "request": (True, "low", 4000)}
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
    """OpenAI-compatible chat endpoint.

    The content is [prefix text] + parts + [prompt text]: a prefix shared by every request (the static rules) comes
    before the images so that vLLM's prefix cache serves it once per engine instead of prefilling it per request."""

    def __init__(self, session, url, model="qwen38-27b"):
        self.session, self.url, self.model = session, url, model

    async def call(self, parts, prompt, call, seed=0, prefix=None, n=1):
        """One request; with n > 1 several samples of the same prompt (one prefill), returned as "choices"."""
        thinking, effort, max_tokens = CALLS[call]
        content = ([{"type": "text", "text": prefix}] if prefix else []) + parts + [{"type": "text", "text": prompt}]
        body = {"model": self.model, "max_tokens": max_tokens, "seed": seed, "n": n,
                "messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"enable_thinking": thinking, **({"reasoning_effort": effort} if effort else {})},
                **(THINKING if thinking else GREEDY)}
        error = None
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
                async with self.session.post(self.url, json=body) as response:
                    data = await response.json(content_type=None)
                if "choices" not in data:
                    raise RuntimeError(str(data)[:500])
                choices = [{"text": c["message"].get("content") or "", "finish": c.get("finish_reason")}
                           for c in data["choices"]]
                tokens = (data.get("usage") or {}).get("completion_tokens", 0)
                if n == 1:
                    return {**choices[0], "tokens": tokens}
                return {"choices": choices, "tokens": tokens}
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


class Tail:
    """New records of a stage's JSONL files since the last call, while other processes append to them.

    Reading only what was added keeps a following stage cheap on runs of 100k rows; a half-written last line is kept
    for the next call."""

    def __init__(self, directory, pattern="records*.jsonl"):
        self.dir, self.pattern = Path(directory), pattern
        self.pos, self.partial = {}, {}

    def read(self):
        out = []
        for path in sorted(self.dir.glob(self.pattern)):
            pos = self.pos.get(path, 0)
            with open(path, "rb") as stream:
                stream.seek(pos)
                data = stream.read()
            if not data:
                continue
            self.pos[path] = pos + len(data)
            lines = (self.partial.pop(path, b"") + data).split(b"\n")
            if lines[-1]:
                self.partial[path] = lines[-1]
            for line in lines[:-1]:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def mark_done(directory, name="DONE"):
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / name).write_text("")


def is_done(directory):
    """A stage is done when its DONE marker exists, or, for a sharded stage, all of its DONE.<k>-of-<n> markers."""
    directory = Path(directory)
    if (directory / "DONE").exists():
        return True
    shards = list(directory.glob("DONE.*-of-*"))
    if not shards:
        return False
    count = int(shards[0].name.rsplit("-of-", 1)[1])
    return len({p.name for p in shards}) >= count


async def stream_jobs(new_jobs, worker, concurrency, out_path, upstream_done=None, poll=15.0, progress_every=50):
    """Run `worker(job)` with bounded concurrency over the jobs `new_jobs()` returns, appending each result as one
    JSON line.  Without `upstream_done` the first batch is all there is; with it, new jobs are polled for until the
    upstream stage is done and nothing is left (a following stage)."""
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    running, counter = set(), {"done": 0, "queued": 0}
    with open(out_path, "a") as stream:
        async def one(job):
            async with semaphore:
                try:
                    record = await worker(job)
                except Exception as exc:  # noqa: BLE001 - recorded, the row is retried on the next run
                    record = {"id": job["id"], "error": f"{type(exc).__name__}: {exc}"[:500]}
            async with lock:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                counter["done"] += 1
                if counter["done"] % progress_every == 0:
                    print(f"  {counter['done']}/{counter['queued']}", flush=True)
        while True:
            finished = upstream_done is None or upstream_done()
            jobs = new_jobs()
            for job in jobs:
                task = asyncio.create_task(one(job))
                running.add(task)
                task.add_done_callback(running.discard)
            counter["queued"] += len(jobs)
            if finished and not jobs:
                break
            if upstream_done is None:
                continue  # the next new_jobs() is empty and ends the loop
            await asyncio.sleep(poll)
        if running:
            await asyncio.gather(*running)
