"""Minimal LLM clients (standard library only): Anthropic Messages API (sync + Message Batches)
and any OpenAI-compatible chat endpoint (Ollama, vLLM, LM Studio) for local open models."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

# USD per million tokens: (input, cache_write_5m, cache_read, output). Checked 31 Oct 2026 on
# platform.claude.com/docs/en/about-claude/pricing. Batch API = 50% of these.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50),
    "claude-sonnet-5-5": (2.0, 2.5, 0.10, 10.0),
    "claude-opus-5-5": (4.0, 5.0, 0.20, 20.0),
    "claude-sonnet-5": (2.0, 2.5, 0.20, 10.0),
    "claude-haiku-4-5": (1.0, 1.25, 0.10, 5.0),
    "claude-sonnet-4-5": (3.0, 3.75, 0.30, 15.0),
    "claude-sonnet-4-6": (3.0, 3.75, 0.30, 15.0),
}
API = "https://api.anthropic.com/v1"
# models that reject the temperature parameter (sampling defaults are used)
NO_TEMPERATURE = {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5",
                  "claude-opus-5"}


def load_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(here, ".env"), os.path.expanduser("~/.anthropic_key")):
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    if line.startswith("sk-ant-"):
                        return line
    raise RuntimeError("ANTHROPIC_API_KEY not set (env var, or a .env file next to the soraeval folder)")


def cost_usd(model: str, usage: dict, batch: bool = False) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    inp = usage.get("input_tokens", 0) or 0
    cw = usage.get("cache_creation_input_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    c = (inp * p[0] + cw * p[1] + cr * p[2] + out * p[3]) / 1e6
    return c * (0.5 if batch else 1.0)


def _post(url, payload, headers, timeout=180, method="POST"):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def anthropic_request(model, system_blocks, user, max_tokens, temperature, effort=None):
    sys = [{"type": "text", "text": s} for s in system_blocks]
    sys[-1]["cache_control"] = {"type": "ephemeral"}
    req = {"model": model, "max_tokens": max_tokens, "system": sys,
           "messages": [{"role": "user", "content": user}]}
    if temperature is not None and model not in NO_TEMPERATURE:
        req["temperature"] = temperature
    if effort:
        req["output_config"] = {"effort": effort}
    return req


class Anthropic:
    def __init__(self, model: str, temperature=0.0, max_tokens=32000, effort=None):
        self.model, self.temperature, self.max_tokens, self.effort = model, temperature, max_tokens, effort
        self.key = load_key()
        self.headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                        "content-type": "application/json"}

    def complete(self, system_blocks, user, retries=6):
        req = anthropic_request(self.model, system_blocks, user, self.max_tokens, self.temperature, self.effort)
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = _post(f"{API}/messages", req, self.headers, timeout=600)
                text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
                return {"text": text, "usage": r.get("usage", {}), "stop_reason": r.get("stop_reason"),
                        "latency_s": round(time.time() - t0, 2), "error": None,
                        "temperature_used": req.get("temperature")}
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:500]
                last = f"HTTP {e.code}: {body}"
                if e.code == 400 and "temperature" in body and "temperature" in req:
                    req.pop("temperature")      # some models reject the parameter
                    continue
                if e.code in (429, 500, 502, 503, 529):
                    time.sleep(min(60, 2 ** attempt + random.random()))
                    continue
                break
            except Exception as e:  # network
                last = repr(e)
                time.sleep(min(60, 2 ** attempt + random.random()))
        return {"text": "", "usage": {}, "stop_reason": None, "latency_s": None, "error": last}

    # ---------------------------------------------------------------- Message Batches (50% price)
    def batch_submit(self, items):
        """items: list of (custom_id, system_blocks, user). Returns batch id."""
        reqs = [{"custom_id": cid, "params": anthropic_request(self.model, sb, u, self.max_tokens, self.temperature,
                                                                 self.effort)}
                for cid, sb, u in items]
        r = _post(f"{API}/messages/batches", {"requests": reqs}, self.headers, timeout=300)
        return r["id"]

    def batch_status(self, batch_id):
        return _post(f"{API}/messages/batches/{batch_id}", None, self.headers, method="GET")

    def batch_results(self, batch_id):
        st = self.batch_status(batch_id)
        url = st.get("results_url")
        req = urllib.request.Request(url, headers=self.headers)
        out = {}
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r.read().decode().splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                res = o["result"]
                if res["type"] == "succeeded":
                    m = res["message"]
                    text = "".join(b.get("text", "") for b in m.get("content", []) if b.get("type") == "text")
                    out[o["custom_id"]] = {"text": text, "usage": m.get("usage", {}),
                                           "stop_reason": m.get("stop_reason"), "latency_s": None, "error": None}
                else:
                    out[o["custom_id"]] = {"text": "", "usage": {}, "stop_reason": None, "latency_s": None,
                                           "error": json.dumps(res)[:500]}
        return out


class Ollama:
    """Ollama native /api/chat (Amendment 2). Unlike the OpenAI-compatible endpoint it lets us set the context size,
    the thinking mode and the seed. Sampling (temperature, top-p, top-k) is left at the model's shipped defaults."""

    def __init__(self, model: str, base_url: str, max_tokens=32000, num_ctx=40960, seed=2026, think=None):
        self.model, self.base, self.max_tokens = model, base_url.rstrip("/"), max_tokens
        self.num_ctx, self.seed, self.think = num_ctx, seed, think
        self.headers = {"content-type": "application/json"}
        tags = _post(f"{self.base}/api/tags", None, self.headers, method="GET")
        self.digest = next((m.get("digest") for m in tags.get("models", []) if m.get("name") == model), None)
        if self.digest is None:
            raise RuntimeError(f"model {model} not found on {self.base} (ollama pull it first)")

    def request(self, system_blocks, user):
        req = {"model": self.model, "stream": False,
               "messages": [{"role": "system", "content": "\n\n".join(system_blocks)},
                            {"role": "user", "content": user}],
               "options": {"num_ctx": self.num_ctx, "num_predict": self.max_tokens, "seed": self.seed}}
        if self.think is not None:
            req["think"] = self.think
        return req

    def complete(self, system_blocks, user, retries=4):
        req = self.request(system_blocks, user)
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = _post(f"{self.base}/api/chat", req, self.headers, timeout=3600)
                msg = r.get("message", {})
                # "length" = hit num_predict; renamed so analyze.py counts it as truncated, like the Anthropic runs
                stop = "max_tokens" if r.get("done_reason") == "length" else r.get("done_reason")
                return {"text": msg.get("content") or "", "thinking": msg.get("thinking") or "",
                        "usage": {"input_tokens": r.get("prompt_eval_count", 0),
                                  "output_tokens": r.get("eval_count", 0)},
                        "stop_reason": stop, "latency_s": round(time.time() - t0, 2), "error": None}
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
                time.sleep(2 ** attempt)
            except Exception as e:
                last = repr(e)
                time.sleep(2 ** attempt)
        return {"text": "", "usage": {}, "stop_reason": None, "latency_s": None, "error": last}


class OpenAICompat:
    """OpenAI-compatible /v1/chat/completions (Ollama: http://localhost:11434/v1)."""

    def __init__(self, model: str, base_url: str, temperature=0.0, max_tokens=3000, api_key="none"):
        self.model, self.base, self.temperature, self.max_tokens = model, base_url.rstrip("/"), temperature, max_tokens
        self.headers = {"content-type": "application/json", "authorization": f"Bearer {api_key}"}

    def complete(self, system_blocks, user, retries=4):
        req = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
               "messages": [{"role": "system", "content": "\n\n".join(system_blocks)},
                            {"role": "user", "content": user}]}
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = _post(f"{self.base}/chat/completions", req, self.headers, timeout=900)
                text = r["choices"][0]["message"].get("content") or ""
                u = r.get("usage", {})
                return {"text": text, "usage": {"input_tokens": u.get("prompt_tokens", 0),
                                                "output_tokens": u.get("completion_tokens", 0)},
                        "stop_reason": r["choices"][0].get("finish_reason"),
                        "latency_s": round(time.time() - t0, 2), "error": None}
            except Exception as e:
                last = repr(e)
                time.sleep(2 ** attempt)
        return {"text": "", "usage": {}, "stop_reason": None, "latency_s": None, "error": last}
