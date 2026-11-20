"""Model-agnostic structured-output LLM calls, with a disk cache.

Two adapters behind one interface:
  * anthropic      -> official `anthropic` SDK, Pydantic structured outputs
  * openai_compat  -> any OpenAI-compatible endpoint (Gemini free tier, Groq,
                      or an API Kuku provides), JSON mode + Pydantic validation

Swapping models is a config/CLI change only. Every response is cached on
disk keyed by a hash of (provider, model, effort, prompts, schema), so a
re-run of the same input costs nothing and survives rate limits.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str = "anthropic"  # anthropic | openai_compat | gemini
    model: str = "claude-sonnet-5-5"
    effort: str | None = "low"  # anthropic only: low | medium | high | xhigh | max
    max_tokens: int = 16000
    base_url: str | None = None  # openai_compat only
    api_key_env: str | None = None  # env var holding the key (openai_compat)
    api_key: str | None = None  # explicit key (e.g. typed into the web app); else read from env
    fallbacks: bool = True  # anthropic server-side refusal fallback
    cache_dir: Path = Path(".cache/llm")
    use_cache: bool = True


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached_input_tokens: int = 0  # tokens the cached responses originally cost
    cached_output_tokens: int = 0
    per_task: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, task: str, inp: int, out: int, secs: float, hit: bool) -> None:
        self.calls += 1
        self.cache_hits += int(hit)
        if hit:
            self.cached_input_tokens += inp
            self.cached_output_tokens += out
        else:
            self.input_tokens += inp
            self.output_tokens += out
            self.seconds += secs
        t = self.per_task.setdefault(task, {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0})
        t["calls"] += 1
        t["cache_hits"] += int(hit)
        if not hit:
            t["input_tokens"] += inp
            t["output_tokens"] += out


@dataclass
class LLMResult:
    parsed: BaseModel
    raw: str
    cached: bool


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.usage = Usage()
        self._client = None

    # ------------------------------------------------------------ public

    def structured(self, *, task: str, system: str, user: str, schema: type[T]) -> LLMResult:
        """Return a validated `schema` instance. Retries once on invalid output."""
        key = self._cache_key(system, user, schema)
        path = self.cfg.cache_dir / f"{key}.json"
        if self.cfg.use_cache and path.exists():
            blob = json.loads(path.read_text())
            try:
                parsed = schema.model_validate_json(blob["raw"])
                self.usage.add(task, blob.get("input_tokens", 0), blob.get("output_tokens", 0), 0.0, hit=True)
                return LLMResult(parsed=parsed, raw=blob["raw"], cached=True)
            except ValidationError:
                pass  # schema changed since cached; fall through and re-call

        last_err: str | None = None
        for attempt in range(2):
            prompt = user if last_err is None else (
                f"{user}\n\nYour previous answer failed validation:\n{last_err}\nReturn corrected JSON only."
            )
            t0 = time.time()
            raw, inp, out = self._call(system, prompt, schema)
            secs = time.time() - t0
            try:
                parsed = schema.model_validate_json(_strip_fences(raw))
            except ValidationError as exc:
                self.usage.add(task, inp, out, secs, hit=False)
                last_err = str(exc)[:2000]
                continue
            self.usage.add(task, inp, out, secs, hit=False)
            raw = _strip_fences(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "task": task, "provider": self.cfg.provider, "model": self.cfg.model,
                "effort": self.cfg.effort, "raw": raw, "input_tokens": inp, "output_tokens": out,
            }, indent=1))
            return LLMResult(parsed=parsed, raw=raw, cached=False)
        raise LLMError(f"{task}: output failed validation twice: {last_err}")

    # ------------------------------------------------------------ internals

    def _cache_key(self, system: str, user: str, schema: type[BaseModel]) -> str:
        payload = json.dumps({
            "provider": self.cfg.provider, "model": self.cfg.model, "effort": self.cfg.effort,
            "system": system, "user": user,
            "schema": schema.model_json_schema(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def _call(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        if self.cfg.provider == "anthropic":
            return self._call_anthropic(system, user, schema)
        if self.cfg.provider in ("openai_compat", "gemini"):
            return self._call_openai_compat(system, user, schema)
        raise LLMError(f"unknown provider {self.cfg.provider!r}")

    def _call_anthropic(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        import anthropic

        if self._client is None:
            key = self.cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("missing API key: set ANTHROPIC_API_KEY (or use --provider gemini for the free path)")
            self._client = anthropic.Anthropic(api_key=key, max_retries=4)
        kwargs: dict = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        # Server-side refusal fallback (not offered for Haiku).
        use_fallback = self.cfg.fallbacks and "haiku" not in self.cfg.model
        try:
            if use_fallback:
                resp = self._client.beta.messages.parse(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                resp = self._client.messages.parse(**kwargs)
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"connection error: {exc}") from exc

        if resp.stop_reason == "refusal":
            raise LLMError(f"model refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise LLMError("hit max_tokens before finishing the JSON")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def _call_openai_compat(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        from openai import OpenAI, OpenAIError

        if self._client is None:
            if self.cfg.provider == "gemini":
                base_url = self.cfg.base_url or GEMINI_OPENAI_BASE
                key_env = self.cfg.api_key_env or "GEMINI_API_KEY"
            else:
                base_url = self.cfg.base_url or os.environ.get("OPENAI_COMPAT_BASE_URL")
                key_env = self.cfg.api_key_env or "OPENAI_COMPAT_API_KEY"
            key = self.cfg.api_key or os.environ.get(key_env)
            if not key:
                raise LLMError(f"missing API key: set {key_env}")
            self._client = OpenAI(base_url=base_url, api_key=key, max_retries=4)
        schema_hint = json.dumps(schema.model_json_schema())
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": f"{system}\n\nRespond with JSON only, matching this JSON Schema:\n{schema_hint}"},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=self.cfg.max_tokens,
            )
        except OpenAIError as exc:
            raise LLMError(f"{self.cfg.provider} error: {exc}") from exc
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()
