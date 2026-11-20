"""Score, de-duplicate and order promos. Pure code; weights live in config.

Order of operations (see pipeline.py):
  1. dedupe()  - group cuts of the same scene; keep the best-scored cut, list the others as alt cuts
  2. (optional) the cold-viewer judge scores the kept promos (judge.py)
  3. order()   - final score, diversity cap, and the "post first" set at ranks 1..post_first_n
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Promo


@dataclass
class RankConfig:
    w_hook: float = 0.35
    w_cliffhanger: float = 0.30
    w_standalone: float = 0.25
    w_early_arc: float = 0.10
    spoiler_penalty: dict[str, float] = field(default_factory=lambda: {"none": 0.0, "low": 0.03, "high": 0.10})
    # Coherence: the brief says a promo is usually one continuous stretch, and jump cuts are
    # judged blind (text only), so each extra segment and each long skipped gap costs a little.
    extra_segment_penalty: float = 0.02
    gap_penalty: float = 0.04  # scaled by min(1, max_gap / 60 s)
    # The validator extended the clip past the LLM's cliffhanger to reach 20 s: the ending may resolve.
    extended_end_penalty: float = 0.05
    w_judge: float = 0.6  # weight of the cold-viewer judge when it ran (0 = self-scores only)
    overlap_iou: float = 0.5  # same-episode promos overlapping more than this are duplicates
    same_scene_overlap: float = 0.4  # ...or sharing more than this share of the shorter clip
    max_per_episode_top: int = 2  # diversity cap inside the top `top_n`
    top_n: int = 20
    post_first_n: int = 5


def _overlap(a: Promo, b: Promo) -> float:
    return sum(
        max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
        for sa in a.segments for sb in b.segments if sa.episode == sb.episode
    )


def _iou(a: Promo, b: Promo) -> float:
    inter = _overlap(a, b)
    union = a.duration + b.duration - inter
    return inter / union if union > 0 else 0.0


def _same_scene(a: Promo, b: Promo, cfg: RankConfig) -> bool:
    inter = _overlap(a, b)
    if inter <= 0:
        return False
    return _iou(a, b) > cfg.overlap_iou or inter / min(a.duration, b.duration) > cfg.same_scene_overlap


def base_score(p: Promo, n_episodes: int, cfg: RankConfig) -> float:
    first_ep = p.segments[0].episode
    early = 1.0 - (first_ep - 1) / max(1, n_episodes - 1)  # 1.0 for ep 1 -> 0.0 for the last
    s = (
        cfg.w_hook * p.scores.hook / 5
        + cfg.w_cliffhanger * p.scores.cliffhanger / 5
        + cfg.w_standalone * p.scores.standalone / 5
        + cfg.w_early_arc * early
        - cfg.spoiler_penalty.get(p.spoiler_level, 0.0)
        - cfg.extra_segment_penalty * (len(p.segments) - 1)
        - cfg.gap_penalty * min(1.0, p.max_gap / 60.0)
        - (cfg.extended_end_penalty if "extended_end_past_cliffhanger" in p.repairs else 0.0)
    )
    return s


def score(p: Promo, n_episodes: int, cfg: RankConfig) -> float:
    b = base_score(p, n_episodes, cfg)
    if p.judge_score is not None and cfg.w_judge > 0:
        b = (1 - cfg.w_judge) * b + cfg.w_judge * p.judge_score
    return round(b, 4)


def _sort_key(p: Promo):
    # Ties: the miner's own best-first order, then id. (Not episode number: that silently
    # re-creates an early-episode preference and confounds the early-arc weight.)
    return (-p.score, p.miner_rank, p.promo_id)


def dedupe(promos: list[Promo], n_episodes: int, cfg: RankConfig) -> tuple[list[Promo], list[tuple[Promo, Promo]]]:
    """Keep the best cut of each scene. Returns (kept, [(dropped, kept)])."""
    for p in promos:
        p.score = score(p, n_episodes, cfg)
    kept: list[Promo] = []
    dupes: list[tuple[Promo, Promo]] = []
    for p in sorted(promos, key=_sort_key):
        twin = next((k for k in kept if _same_scene(p, k, cfg)), None)
        if twin:
            dupes.append((p, twin))
            twin.alt_cuts.append(p.promo_id)
        else:
            kept.append(p)
    return kept, dupes


def _flags(p: Promo) -> list[str]:
    f = []
    if "extended_end_past_cliffhanger" in p.repairs:
        f.append("ending extended past the cliffhanger to reach 20 s: check the cut")
    if len(p.segments) > 1 and p.max_gap > 20:
        f.append(f"jump cut skips {p.max_gap:.0f}s: check it reads as one moment")
    return f


def post_first_reason(p: Promo) -> str:
    ep = p.segments[0].episode
    facts = [p.hook_type.replace("_", " ") + " hook"]
    if p.context_needed:
        facts.append("needs a little context: " + p.context_needed.strip().rstrip(".;"))
    else:
        facts.append("needs no backstory")
    if p.spoiler_level == "none":
        facts.append(f"no spoilers, funnels into episode {ep}" if ep <= 5 else "no spoilers")
    else:
        facts.append(f"{p.spoiler_level} spoiler risk")
    tags = "[" + " · ".join(facts) + "]"
    return f"{p.judge_reason.strip().rstrip('.')}. {tags}" if p.judge_reason else tags


def order(kept: list[Promo], n_episodes: int, cfg: RankConfig) -> list[Promo]:
    """Final ranking. Ranks 1..post_first_n are the posting order: the best promos with
    distinct episodes and hook types (variety for the first week of ads)."""
    for p in kept:
        p.score = score(p, n_episodes, cfg)
        p.flags = _flags(p)
    ordered = sorted(kept, key=_sort_key)

    # Diversity: at most N per episode inside the top slice; the rest slide down.
    top: list[Promo] = []
    rest: list[Promo] = []
    per_ep: dict[int, int] = {}
    for p in ordered:
        ep = p.segments[0].episode
        if len(top) < cfg.top_n and per_ep.get(ep, 0) < cfg.max_per_episode_top:
            top.append(p)
            per_ep[ep] = per_ep.get(ep, 0) + 1
        else:
            rest.append(p)

    # Post-first set: greedy over the top slice, one per episode and per hook type when possible.
    first: list[Promo] = []
    eps: set[int] = set()
    types: set[str] = set()
    for p in top:
        if len(first) == cfg.post_first_n:
            break
        if p.segments[0].episode not in eps and p.hook_type not in types:
            first.append(p)
            eps.add(p.segments[0].episode)
            types.add(p.hook_type)
    for p in top:  # top up if variety ran out
        if len(first) < cfg.post_first_n and p not in first:
            first.append(p)

    final = first + [p for p in top if p not in first] + rest
    for i, p in enumerate(final, start=1):
        p.rank = i
        p.post_first_reason = post_first_reason(p) if i <= len(first) else None
    return final


def rank(promos: list[Promo], n_episodes: int, cfg: RankConfig | None = None) -> tuple[list[Promo], list[tuple[Promo, Promo]]]:
    """dedupe + order, without the judge. Returns (ranked promos, [(dropped_duplicate, kept)])."""
    cfg = cfg or RankConfig()
    kept, dupes = dedupe(promos, n_episodes, cfg)
    return order(kept, n_episodes, cfg), dupes
