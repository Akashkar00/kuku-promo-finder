"""The end-to-end run, shared by the CLI and the Streamlit app."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from .assemble import Rules, assemble_all
from .export import write_json, write_promos
from .judge import JudgeConfig, judge_promos
from .llm import LLM, LLMConfig
from .miner import build_show_brief, mine_episodes
from .parser import parse_show
from .rank import RankConfig, dedupe, order
from .schemas import Episode, ParseAnomaly, Promo, Rejection, ShowBrief

# USD per million tokens (input, output). Used only for the cost estimate;
# verify against current pricing before quoting.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}


@dataclass
class Settings:
    llm: LLMConfig
    rules: Rules = field(default_factory=Rules)
    rank: RankConfig = field(default_factory=RankConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    include_camera: bool = False


def load_settings(config: Path, judge: bool | None = None, judge_provider: str | None = None,
                  judge_model: str | None = None, **overrides) -> Settings:
    """Read config.yaml (if present) and apply non-None overrides to the llm and judge sections."""
    cfg = yaml.safe_load(config.read_text()) if config.exists() else {}
    jc = {**cfg.get("judge", {}), **{k: v for k, v in
                                     {"enabled": judge, "provider": judge_provider, "model": judge_model}.items()
                                     if v is not None}}
    lc = {**cfg.get("llm", {}), **{k: v for k, v in overrides.items() if v is not None}}
    llm_cfg = LLMConfig(
        provider=lc.get("provider", "anthropic"),
        model=lc.get("model", "claude-sonnet-5-5"),
        effort=lc.get("effort"),
        max_tokens=lc.get("max_tokens", 16000),
        base_url=lc.get("base_url"),
        api_key_env=lc.get("api_key_env"),
        api_key=lc.get("api_key"),
        fallbacks=lc.get("fallbacks", True),
        use_cache=lc.get("use_cache", True),
    )
    if llm_cfg.provider != "anthropic":
        llm_cfg.effort = None
    return Settings(
        llm=llm_cfg,
        rules=Rules(**cfg.get("rules", {})),
        rank=RankConfig(**cfg.get("rank", {})),
        judge=JudgeConfig(**jc),
        include_camera=cfg.get("pipeline", {}).get("include_camera", False),
    )


@dataclass
class RunResult:
    episodes: list[Episode]
    anomalies: list[ParseAnomaly]
    brief: ShowBrief
    promos: list[Promo]
    rejected: list[Rejection]
    dupes: list[tuple[Promo, Promo]]
    errors: dict[int, str]
    meta: dict


def run_pipeline(
    screenplays: Path,
    settings: Settings,
    out: Path,
    episodes: set[int] | None = None,
    title: str | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> RunResult:
    """Parse -> brief -> mine -> assemble -> rank -> write outputs to `out`.

    Raises LLMError if the show brief cannot be built (nothing useful can run then);
    per-episode LLM failures are collected in `errors` instead.
    """
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    title = title or screenplays.name

    all_eps, anomalies = parse_show(screenplays)
    eps = [e for e in all_eps if episodes is None or e.number in episodes]
    write_json(out / "parse_report.json", {
        "episodes": len(all_eps), "events": sum(len(e.events) for e in all_eps),
        "anomalies": [a.model_dump() for a in anomalies],
    })

    llm = LLM(settings.llm)
    brief = build_show_brief(llm, all_eps)
    write_json(out / "show_brief.json", brief.model_dump())

    mined, errors = mine_episodes(llm, eps, brief, out / "raw", include_camera=settings.include_camera,
                                  on_progress=on_progress)
    promos, rejected = assemble_all(eps, mined, settings.rules)
    kept, dupes = dedupe(promos, len(all_eps), settings.rank)

    judge_stats: dict = {"ran": False, "reason": "disabled in config"}
    judge_llm = None
    jc = settings.judge
    if jc.enabled and kept:
        judge_llm = LLM(LLMConfig(provider=jc.provider, model=jc.model,
                                  effort=jc.effort if jc.provider == "anthropic" else None, max_tokens=8000,
                                  use_cache=settings.llm.use_cache,
                                  api_key=settings.llm.api_key if jc.provider == settings.llm.provider else None))
        judge_stats = judge_promos(judge_llm, kept, jc)
    ranked = order(kept, len(all_eps), settings.rank)
    write_promos(out, ranked, rejected, dupes, title=title, top_n=settings.rank.top_n)

    u, cfg = llm.usage, settings.llm
    price = PRICES.get(cfg.model)

    def usd(inp: int, outp: int) -> float | None:
        return round((inp * price[0] + outp * price[1]) / 1e6, 4) if price else None

    n_cands = sum(len(c) for c in mined.values())
    meta = {
        "screenplays": title, "episodes_mined": [e.number for e in eps],
        "provider": cfg.provider, "model": cfg.model, "effort": cfg.effort,
        "candidates": n_cands, "promos": len(ranked), "rejected": len(rejected), "duplicates": len(dupes),
        "validator_pass_rate": round(len(promos) / n_cands, 3) if n_cands else None,
        "episode_errors": errors,
        "usage": {"calls": u.calls, "cache_hits": u.cache_hits, "input_tokens": u.input_tokens,
                  "output_tokens": u.output_tokens, "cached_input_tokens": u.cached_input_tokens,
                  "cached_output_tokens": u.cached_output_tokens, "llm_seconds": round(u.seconds, 1),
                  "per_task": u.per_task},
        "rank_config": dataclasses.asdict(settings.rank),
        "judge": {**judge_stats, **({"usage": {"calls": judge_llm.usage.calls, "cache_hits": judge_llm.usage.cache_hits,
                                              "input_tokens": judge_llm.usage.input_tokens + judge_llm.usage.cached_input_tokens,
                                              "output_tokens": judge_llm.usage.output_tokens + judge_llm.usage.cached_output_tokens}}
                                   if judge_llm else {})},
        "estimated_cost_usd": usd(u.input_tokens, u.output_tokens),
        "estimated_judge_cost_usd": (round((judge_llm.usage.input_tokens + judge_llm.usage.cached_input_tokens) * PRICES[jc.model][0] / 1e6
                                           + (judge_llm.usage.output_tokens + judge_llm.usage.cached_output_tokens) * PRICES[jc.model][1] / 1e6, 4)
                                     if judge_llm and jc.model in PRICES else None),
        "estimated_cost_uncached_usd": usd(u.input_tokens + u.cached_input_tokens,
                                           u.output_tokens + u.cached_output_tokens),
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(out / "run_meta.json", meta)
    return RunResult(all_eps, anomalies, brief, ranked, rejected, dupes, errors, meta)
