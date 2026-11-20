"""Cold-viewer judge: an independent, listwise re-ranking of validated promos.

Why it exists: the miner scores each clip 1-5 while looking at one episode at a
time, so its scores have no anchor across episodes and bunch at 3-4. The judge
sees only the clip text (no show brief, no episode numbers, no miner scores)
and must ORDER a group of clips, which forces it to discriminate. It runs on a
different, stronger model than the miner (default: Claude Opus 5.5 judging
Sonnet's picks), so the miner is not grading its own work. Any provider works.

Position bias: each pass shuffles clips into groups of <= group_size; two passes
with different seeds are averaged, and their agreement is reported.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from importlib import resources

from .llm import LLM, LLMError
from .schemas import JudgeOrdering, Promo


@dataclass
class JudgeConfig:
    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"  # a stronger model than the miner, as a second opinion
    effort: str | None = "low"  # anthropic only
    group_size: int = 30
    passes: int = 2
    seed: int = 11


def clip_text(p: Promo) -> str:
    """What a cold viewer gets: duration, cut style and the clip's lines (IDs and names kept, nothing else)."""
    return f"({p.duration:.0f}s, {p.cut_style})\n" + "\n".join(p.transcript)


def _one_pass(llm: LLM, promos: list[Promo], cfg: JudgeConfig, seed: int, system: str) -> tuple[dict[str, float], dict[str, str]]:
    rng = random.Random(seed)
    order = promos[:]
    rng.shuffle(order)
    pct: dict[str, float] = {}
    reasons: dict[str, str] = {}
    n_groups = max(1, -(-len(order) // cfg.group_size))
    size = -(-len(order) // n_groups)  # even group sizes
    for g in range(n_groups):
        group = order[g * size:(g + 1) * size]
        labels = {f"C{i + 1:02d}": p for i, p in enumerate(group)}
        user = "\n\n".join(f"=== {lab} ===\n{clip_text(p)}" for lab, p in labels.items())
        res = llm.structured(task="judge", system=system, user=user, schema=JudgeOrdering)
        ranking: list[str] = []
        for lab in res.parsed.ranking:  # type: ignore[attr-defined]
            lab = lab.strip().upper()
            if lab in labels and lab not in ranking:
                ranking.append(lab)
        ranking += [lab for lab in labels if lab not in ranking]  # omitted clips go last
        n = len(ranking)
        for pos, lab in enumerate(ranking):
            pct[labels[lab].promo_id] = 1.0 - pos / (n - 1) if n > 1 else 1.0
        for r in res.parsed.top_reasons:  # type: ignore[attr-defined]
            lab = r.clip.strip().upper()
            if lab in labels:
                reasons.setdefault(labels[lab].promo_id, r.reason.strip())
    return pct, reasons


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(x: list[float]) -> list[float]:
        idx = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(idx):
            r[i] = float(k)
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return 1.0
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def judge_promos(llm: LLM, promos: list[Promo], cfg: JudgeConfig) -> dict:
    """Set judge_score / judge_reason on each promo in place. Returns run stats for run_meta."""
    system = resources.files("promo_finder").joinpath("prompts/judge.md").read_text(encoding="utf-8")
    passes: list[dict[str, float]] = []
    reasons: dict[str, str] = {}
    try:
        for k in range(cfg.passes):
            pct, rs = _one_pass(llm, promos, cfg, cfg.seed + k, system)
            passes.append(pct)
            for pid, r in rs.items():
                reasons.setdefault(pid, r)
    except LLMError as exc:
        if not passes:
            return {"ran": False, "error": str(exc)}
    ids = [p.promo_id for p in promos]
    for p in promos:
        vals = [ps[p.promo_id] for ps in passes if p.promo_id in ps]
        p.judge_score = round(sum(vals) / len(vals), 4) if vals else None
        p.judge_reason = reasons.get(p.promo_id)
    stats: dict = {"ran": True, "provider": cfg.provider, "model": cfg.model, "passes": len(passes),
                   "group_size": cfg.group_size}
    if len(passes) >= 2:
        a, b = [passes[0][i] for i in ids], [passes[1][i] for i in ids]
        top = max(1, min(20, len(ids)))
        ta = set(sorted(ids, key=lambda i: -passes[0][i])[:top])
        tb = set(sorted(ids, key=lambda i: -passes[1][i])[:top])
        stats["pass_agreement_spearman"] = round(_spearman(a, b), 3)
        stats[f"pass_top{top}_overlap"] = round(len(ta & tb) / top, 3)
    return stats
