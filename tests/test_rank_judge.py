"""Ranking and judge behaviour on synthetic promos (no network, no screenplay data)."""

from __future__ import annotations

import json

from promo_finder.judge import JudgeConfig, judge_promos
from promo_finder.llm import LLMResult
from promo_finder.rank import RankConfig, dedupe, order, rank
from promo_finder.schemas import JudgeOrdering, MinedScores, Promo, Segment


def promo(pid: str, ep: int, segs: list[tuple[float, float]], h=3, c=3, s=3, hook_type="other",
          miner_rank=1, repairs=None, spoiler="none") -> Promo:
    segments = [Segment(episode=ep, start=a, end=b, start_id="B1", end_id="B2") for a, b in segs]
    return Promo(promo_id=pid, title=pid, segments=segments, duration=sum(b - a for a, b in segs), hook="h",
                 hook_id="B1", hook_offset=0.0, cliffhanger="c", cliffhanger_id="B2", hook_type=hook_type,
                 why_it_works="w", context_needed="", spoiler_level=spoiler,
                 scores=MinedScores(hook=h, cliffhanger=c, standalone=s), miner_rank=miner_rank,
                 repairs=repairs or [], transcript=[f"line of {pid}"])


def test_two_cuts_of_one_scene_become_alt_cuts():
    # IoU is about 0.35 (below 0.5), but they share more than half of the shorter clip
    a = promo("E03-P1", 3, [(58.3, 88.1)], h=4)
    b = promo("E03-P2", 3, [(43.3, 60.0), (69.9, 84.0)], h=3)
    kept, dupes = dedupe([a, b], 50, RankConfig())
    assert [p.promo_id for p in kept] == ["E03-P1"]
    assert kept[0].alt_cuts == ["E03-P2"] and dupes[0][0].promo_id == "E03-P2"


def test_ties_break_on_miner_order_not_episode():
    early = promo("E01-P2", 1, [(0, 30)], miner_rank=2)
    late = promo("E40-P1", 40, [(0, 30)], miner_rank=1)
    ranked, _ = rank([early, late], 50, RankConfig(w_early_arc=0.0))
    assert ranked[0].promo_id == "E40-P1"


def test_jump_cuts_and_extended_endings_are_penalised_and_flagged():
    clean = promo("E05-P1", 5, [(0, 30)])
    jumpy = promo("E06-P1", 6, [(0, 15), (100, 115)])
    stretched = promo("E07-P1", 7, [(0, 25)], repairs=["extended_end_past_cliffhanger"])
    ranked, _ = rank([jumpy, stretched, clean], 50, RankConfig(w_early_arc=0.0))
    assert ranked[0].promo_id == "E05-P1"
    flags = {p.promo_id: " ".join(p.flags) for p in ranked}
    assert "jump cut" in flags["E06-P1"] and "extended" in flags["E07-P1"] and flags["E05-P1"] == ""


def test_post_first_set_varies_episode_and_hook_type():
    ps = [promo("E01-P1", 1, [(0, 30)], h=5, hook_type="betrayal_reveal"),
          promo("E01-P2", 1, [(40, 70)], h=5, hook_type="humiliation", miner_rank=2),
          promo("E02-P1", 2, [(0, 30)], h=5, hook_type="betrayal_reveal"),
          promo("E09-P1", 9, [(0, 30)], h=4, hook_type="threat_or_ultimatum")]
    ranked, _ = rank(ps, 50, RankConfig(post_first_n=2))
    first = [p for p in ranked if p.post_first_reason]
    assert len(first) == 2 and len({p.segments[0].episode for p in first}) == 2
    assert len({p.hook_type for p in first}) == 2
    assert [p.rank for p in first] == [1, 2]


class FakeLLM:
    """Returns the clips in reverse label order, and drops one label to test the fallback."""

    def __init__(self):
        self.calls = 0

    def structured(self, *, task, system, user, schema):
        self.calls += 1
        labels = [line[4:7] for line in user.splitlines() if line.startswith("=== C")]
        order_ = list(reversed(labels))[:-1]  # omit one label on purpose
        data = {"ranking": order_, "top_reasons": [{"clip": order_[0], "reason": "strong opener."}]}
        return LLMResult(parsed=JudgeOrdering.model_validate(data), raw=json.dumps(data), cached=False)


def test_judge_scores_every_promo_and_reports_agreement():
    ps = [promo(f"E{i:02d}-P1", i, [(0, 30)]) for i in range(1, 8)]
    llm = FakeLLM()
    stats = judge_promos(llm, ps, JudgeConfig(group_size=4, passes=2))  # type: ignore[arg-type]
    assert stats["ran"] and stats["passes"] == 2 and "pass_agreement_spearman" in stats
    assert all(p.judge_score is not None and 0.0 <= p.judge_score <= 1.0 for p in ps)
    assert any(p.judge_reason for p in ps)
    assert llm.calls == 4  # 7 clips -> 2 groups per pass, 2 passes


def test_judge_score_changes_the_order():
    a = promo("E01-P1", 1, [(0, 30)], h=4)
    b = promo("E02-P1", 2, [(0, 30)], h=3)
    a.judge_score, b.judge_score = 0.0, 1.0
    ranked = order([a, b], 50, RankConfig(w_judge=0.6))
    assert ranked[0].promo_id == "E02-P1"
    ranked = order([a, b], 50, RankConfig(w_judge=0.0))
    assert ranked[0].promo_id == "E01-P1"
