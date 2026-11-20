"""Quality checks for a run: recall vs a hand-labelled gold set, the
no-LLM baseline, validator stats, and (if filled in) human ratings."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from .parser import fmt_ts, parse_ts
from .rank import RankConfig, order
from .schemas import MinedScores, Promo, Segment

HOOK_TOL = 5.0
MIN_IOU = 0.3


def load_promos(run_dir: Path) -> list[Promo]:
    return [Promo.model_validate(p) for p in json.loads((run_dir / "promos.json").read_text())]


def load_gold(path: Path) -> list[dict]:
    """Gold moments. Plot notes live in a gitignored sibling file (gold_notes.private.yaml)
    because they paraphrase the confidential script; the public file holds only times."""
    moments = yaml.safe_load(path.read_text()).get("moments") or []
    notes_path = path.with_name("gold_notes.private.yaml")
    notes = (yaml.safe_load(notes_path.read_text()) or {}) if notes_path.exists() else {}
    for m in moments:
        m["hook_s"], m["end_s"] = parse_ts(m["hook_at"]), parse_ts(m["end_at"])
        m.setdefault("note", notes.get(m["id"], ""))
    return moments


def matches(p: Promo, g: dict) -> bool:
    """Hook within +-5 s of the gold hook, or time-IoU >= 0.3 with the gold span."""
    first = p.segments[0]
    if first.episode == g["episode"] and abs(first.start + p.hook_offset - g["hook_s"]) <= HOOK_TOL:
        return True
    g0, g1 = min(g["hook_s"], g["end_s"]), max(g["hook_s"], g["end_s"])
    inter = sum(max(0.0, min(s.end, g1) - max(s.start, g0)) for s in p.segments if s.episode == g["episode"])
    union = p.duration + (g1 - g0) - inter
    return union > 0 and inter / union >= MIN_IOU


def contains_hook(p: Promo, g: dict) -> bool:
    """Looser, moment-level match: the clip shows the gold hook moment, however it is cut.
    Added after the first full run (see report), so it is reported second, never instead."""
    return any(s.episode == g["episode"] and s.start - 1.0 <= g["hook_s"] <= s.end + 1.0 for s in p.segments)


def recall(promos: list[Promo], gold: list[dict], k: int | None = None, match=matches) -> tuple[float, list[str], dict[str, str]]:
    pool = promos if k is None else promos[:k]
    hit: dict[str, str] = {}
    for g in gold:
        for p in pool:
            if match(p, g):
                hit[g["id"]] = p.promo_id
                break
    missed = [g["id"] for g in gold if g["id"] not in hit]
    return (len(hit) / len(gold) if gold else 0.0), missed, hit


def mrr(promos: list[Promo], gold: list[dict], match=contains_hook) -> tuple[float, float]:
    """Rank-sensitive check: mean reciprocal rank of each gold moment's first match
    (0 if missed), and the median rank of the moments that were found."""
    ranks = []
    for g in gold:
        r = next((i for i, p in enumerate(promos, start=1) if match(p, g)), None)
        ranks.append(r)
    rr = sum(1 / r for r in ranks if r) / len(gold) if gold else 0.0
    found = sorted(r for r in ranks if r)
    med = float(found[len(found) // 2]) if found else float("nan")
    return rr, med


def random_floor(durations: dict[int, float], gold: list[dict], n_clips: int, clip_len: float = 35.0,
                 draws: int = 500, seed: int = 0) -> tuple[float, float]:
    """Mean recall (strict, moment-level) of n random clips spread over the show."""
    rng = random.Random(seed)
    eps = sorted(durations)
    totals = [0.0, 0.0]
    for _ in range(draws):
        clips = []
        for i in range(n_clips):
            ep = rng.choice(eps)
            start = rng.uniform(0, max(0.0, durations[ep] - clip_len))
            clips.append(Promo(
                promo_id=f"R{i}", title="", segments=[Segment(episode=ep, start=start, end=start + clip_len, start_id="", end_id="")],
                duration=clip_len, hook="", hook_id="", hook_offset=0.0, cliffhanger="", cliffhanger_id="", hook_type="other",
                why_it_works="", context_needed="", spoiler_level="none", scores=MinedScores(hook=0, cliffhanger=0, standalone=0)))
        totals[0] += recall(clips, gold)[0]
        totals[1] += recall(clips, gold, match=contains_hook)[0]
    return totals[0] / draws, totals[1] / draws


def review_sheet(promos: list[Promo], run_dir: Path, ratings_path: Path, baseline: list[Promo] | None = None,
                 seed: int = 7) -> None:
    """Blind human review: the top 10, 5 from rank 40+, and 5 baseline clips (last 45 s of an
    episode), shuffled. Rating baseline clips on the same sheet tests whether the LLM adds value
    at all; top 10 vs rank 40+ tests the ranking.

    The sheet quotes the script, so it goes in the (gitignored) run folder; the
    ratings CSV holds only promo IDs and scores."""
    if ratings_path.exists():
        return
    rng = random.Random(seed)
    tail = [p for p in promos if (p.rank or 0) >= 40]
    picked = promos[:10] + rng.sample(tail, min(5, len(tail)))
    if baseline:
        picked += rng.sample(baseline, min(5, len(baseline)))
    rng.shuffle(picked)
    md = ["# Blind review sheet", "",
          "Rate each clip 1-5 on hook (first 3 s), cliffhanger (ending), standalone (makes sense to a stranger),",
          f"from the transcript only. Ranks and sources are hidden on purpose. Enter scores in `{ratings_path}`.", ""]
    with ratings_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "promo_id", "hook_1to5", "cliffhanger_1to5", "standalone_1to5", "comment"])
        for i, p in enumerate(picked, start=1):
            w.writerow([i, p.promo_id, "", "", "", ""])
            seg = " + ".join(f"{fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in p.segments)
            md += [f"## Item {i} ({seg}, {p.duration:.0f}s)", "", "```", *p.transcript, "```", ""]
    (run_dir / "review_sheet.md").write_text("\n".join(md), encoding="utf-8")


def human_scores(path: Path, promos: list[Promo]) -> dict | None:
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r.get("hook_1to5")]
    if not rows:
        return None
    rank_of = {p.promo_id: p.rank or 0 for p in promos}

    def summarise(rs: list[dict]) -> dict:
        if not rs:
            return {}
        vals = [(int(r["hook_1to5"]), int(r["cliffhanger_1to5"]), int(r["standalone_1to5"])) for r in rs]
        n = len(vals)
        return {"rated": n, "mean_hook": round(sum(v[0] for v in vals) / n, 2),
                "mean_cliffhanger": round(sum(v[1] for v in vals) / n, 2),
                "mean_standalone": round(sum(v[2] for v in vals) / n, 2),
                "share_all_ge3": round(sum(all(x >= 3 for x in v) for v in vals) / n, 2)}

    return {"top10": summarise([r for r in rows if rank_of.get(r["promo_id"], 99) <= 10]),
            "rank40plus": summarise([r for r in rows if rank_of.get(r["promo_id"], 0) >= 40]),
            "baseline": summarise([r for r in rows if r["promo_id"].endswith("-BASE")])}


def evaluate(run_dir: Path, gold_path: Path, baseline_dir: Path | None, ratings_path: Path) -> str:
    promos = load_promos(run_dir)
    gold = load_gold(gold_path)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    rejected = json.loads((run_dir / "rejected.json").read_text())

    lines = [f"# Eval report: {run_dir.name}", "",
             f"Model: {meta['provider']}/{meta['model']} (effort={meta['effort']}), "
             f"{len(meta['episodes_mined'])} episodes. Cost of a fresh (uncached) run: ${meta.get('estimated_cost_uncached_usd')}, "
             f"{meta['usage']['input_tokens'] + meta['usage'].get('cached_input_tokens', 0)} in / "
             f"{meta['usage']['output_tokens'] + meta['usage'].get('cached_output_tokens', 0)} out tokens.", ""]

    # 1. Validator
    reasons = Counter(r["reason"] for r in rejected)
    repairs = Counter(r for p in promos for r in p.repairs)
    lines += ["## 1. Validator (hard rules in code)", "",
              f"- Candidates from the LLM: {meta['candidates']}; passed: {len(promos) + meta['duplicates']} "
              f"({meta['validator_pass_rate']:.0%}); near-duplicates merged: {meta['duplicates']}; final promos: {len(promos)}",
              f"- Rejections by reason: {dict(reasons) or 'none'}",
              f"- Repairs applied to passing promos: {dict(repairs) or 'none'} "
              f"({sum(1 for p in promos if p.repairs)} of {len(promos)} promos changed)",
              f"- Every final promo: 20-60 s ({min(p.duration for p in promos):.1f}-{max(p.duration for p in promos):.1f} s), "
              f"hook offset <= 3 s (max {max(p.hook_offset for p in promos):.1f} s), "
              f"multi-segment: {sum(len(p.segments) > 1 for p in promos)}", ""]

    # 2. Recall vs gold
    lines += ["## 2. Recall against the gold set", "",
              f"Gold: {len(gold)} moments ({gold_path}). Two match rules, defined under the table.", "",
              "| Ranking slice | Strict recall (pre-registered) | Missed | Moment-level recall (post-hoc) | Missed |",
              "|---|---|---|---|---|"]
    for k in (10, 20, None):
        r, missed, _ = recall(promos, gold, k)
        r2, missed2, _ = recall(promos, gold, k, match=contains_hook)
        lines.append(f"| top {k or 'all'} ({min(k or len(promos), len(promos))} promos) | {r:.0%} | {', '.join(missed) or '-'} "
                     f"| {r2:.0%} | {', '.join(missed2) or '-'} |")
    lines += ["", "Strict = hook within +-5 s of the gold hook, or time-IoU >= 0.3 with the gold span (defined before any run). "
                  "Moment-level = some segment of the clip contains the gold hook moment (+-1 s); added after run 1 showed the strict rule "
                  "scoring a different cut of the same scene (ep 1 altar) as a miss."]
    _, _, hits = recall(promos, gold)
    if baseline_dir and (baseline_dir / "promos.json").exists():
        base = load_promos(baseline_dir)
        br, bmissed, _ = recall(base, gold)
        br2, bmissed2, _ = recall(base, gold, match=contains_hook)
        lines += ["", f"**Baseline (last 45 s of every episode, {len(base)} clips, no LLM):** strict recall {br:.0%} "
                      f"(missed {', '.join(bmissed) or '-'}); moment-level {br2:.0%} (missed {', '.join(bmissed2) or '-'})."]
    # Fair comparisons: equal clip budget, random floor, ranking-weight sensitivity.
    lines += ["", "### Fair comparisons", "",
              "Recall over all promos is mostly coverage: more clips cover more of the show. These hold the clip budget fixed.", "",
              "| Comparison | Strict | Moment-level |", "|---|---|---|"]
    if baseline_dir and (baseline_dir / "promos.json").exists():
        n = len(base)
        lines.append(f"| Pipeline top {n} (same budget as baseline) | {recall(promos, gold, n)[0]:.0%} | "
                     f"{recall(promos, gold, n, match=contains_hook)[0]:.0%} |")
        lines.append(f"| Baseline: last 45 s of each episode ({n} clips) | {br:.0%} | {br2:.0%} |")
        durations = {p.segments[0].episode: p.segments[0].end for p in base}
        for k in (20, n):
            f1, f2 = random_floor(durations, gold, k)
            lines.append(f"| Random 35 s clips, {k} of them (mean of 500 draws) | {f1:.0%} | {f2:.0%} |")
    lines.append(f"| Pipeline top 20 (shipped ranking) | {recall(promos, gold, 20)[0]:.0%} | "
                 f"{recall(promos, gold, 20, match=contains_hook)[0]:.0%} |")

    # Ranking variants on the same promo set (rank-sensitive, added after the EGM review):
    # if the judge never ran, the judge rows equal the self-score rows.
    n_eps = len(meta["episodes_mined"])
    shipped_cfg = RankConfig(**{k: v for k, v in meta.get("rank_config", {}).items() if hasattr(RankConfig, k)})

    def variant(**kw) -> list[Promo]:
        cfg = RankConfig(**{**shipped_cfg.__dict__, **kw})
        return order([p.model_copy(deep=True) for p in promos], n_eps, cfg)

    variants = [("Shipped (self-scores blended with judge)", promos),
                ("Self-scores only (judge weight 0)", variant(w_judge=0.0)),
                ("Judge only (judge weight 1)", variant(w_judge=1.0)),
                ("Shipped, early-episode weight 0", variant(w_early_arc=0.0))]
    lines += ["", "### Ranking variants (same promos, different order)", "",
              "Recall@20 moves in steps of about 8% (13 gold moments), so MRR (moment-level) is the rank-sensitive check.", "",
              "| Ranking | Strict R@20 | Moment R@20 | MRR | Median rank of found gold |", "|---|---|---|---|---|"]
    for name, lst in variants:
        rr, med = mrr(lst, gold)
        lines.append(f"| {name} | {recall(lst, gold, 20)[0]:.0%} | {recall(lst, gold, 20, match=contains_hook)[0]:.0%} "
                     f"| {rr:.2f} | {med:.0f} |")
    j = meta.get("judge", {})
    if j.get("ran"):
        top_key = next((k for k in j if k.startswith("pass_top")), None)
        lines += ["", f"Judge: {j.get('provider')}/{j.get('model')}, {j.get('passes')} shuffled passes; "
                      f"agreement between passes: Spearman {j.get('pass_agreement_spearman')}, "
                      f"top-20 overlap {j.get(top_key) if top_key else 'n/a'}."]
    else:
        lines += ["", f"Judge did not run ({j.get('reason') or j.get('error') or 'not recorded'})."]
    top_eps = Counter(p.segments[0].episode <= 10 for p in promos[:20])
    lines += ["", f"Shipped top 20: {top_eps[True]} of 20 promos come from episodes 1-10."]

    lines += ["", "Gold moment -> matching promo (rank):"]
    rank_of = {p.promo_id: p.rank for p in promos}
    for g in gold:
        pid = hits.get(g["id"])
        lines.append(f"- {g['id']} (ep {g['episode']} {g['hook_at']}): "
                     + (f"{pid} (rank {rank_of[pid]})" if pid else "**missed**") + (f": {g['note']}" if g.get("note") else ""))
    lines.append("")

    # 3. Self-score calibration
    allhigh = sum(min(p.scores.hook, p.scores.cliffhanger, p.scores.standalone) >= 4 for p in promos)
    spoil = Counter(p.spoiler_level for p in promos)
    hooks = Counter(p.hook_type for p in promos)
    lines += ["## 3. Score calibration and mix", "",
              f"- Promos the LLM scored >= 4 on all three criteria: {allhigh}/{len(promos)} "
              f"(if this is most of them, its self-scores do not discriminate; human ratings below are the check)",
              f"- Spoiler levels: {dict(spoil)}",
              f"- Hook types: {dict(hooks.most_common())}",
              f"- Episodes with at least one promo: {len({p.segments[0].episode for p in promos})}/{len(meta['episodes_mined'])}", ""]

    # 4. Human ratings
    review_sheet(promos, run_dir, ratings_path,
                 load_promos(baseline_dir) if baseline_dir and (baseline_dir / "promos.json").exists() else None)
    hs = human_scores(ratings_path, promos)
    lines += ["## 4. Blind human ratings (top 10 vs. rank 40+ vs. baseline, shuffled)", ""]
    if hs:
        lines += [f"- Top 10: {hs['top10']}", f"- Rank 40+: {hs['rank40plus']}", f"- Baseline (episode endings): {hs['baseline']}",
                  "- If the top 10 does not clearly beat rank 40+, the ranking is not adding value;",
                  "  if it does not beat the baseline, the LLM is not adding value."]
    else:
        lines += [f"- Not filled in yet: read `{run_dir / 'review_sheet.md'}` and score the rows in `{ratings_path}`."]
    lines.append("")
    return "\n".join(lines)
