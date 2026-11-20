import pytest

from promo_finder.assemble import Reject, assemble
from promo_finder.rank import rank
from promo_finder.schemas import CandidateSegment, Episode, Event, MinedCandidate, MinedScores, Scene


def make_episode() -> Episode:
    """Camera shot every 2 s; dialogue line i spans [3i+0.5, 3i+2.5]; 120 s long."""
    raw = [("camera", t, t + 2.0, f"shot {t}") for t in range(0, 120, 2)]
    raw += [("dialogue", 3 * i + 0.5, 3 * i + 2.5, f"line {i}") for i in range(40)]
    raw.sort(key=lambda r: (r[1], r[0] != "camera"))
    events = [
        Event(id=f"E01-S01-B{i:03d}", episode=1, scene=1, idx=i, type=t,
              speaker="A" if t == "dialogue" else None, start=s, end=e, text=txt)
        for i, (t, s, e, txt) in enumerate(raw)
    ]
    return Episode(number=1, source_file="x", logline="", continuity="",
                   scenes=[Scene(number=1, heading="INT", start=0, end=120)], events=events, duration=120.0)


EP = make_episode()


def bid(line: int) -> str:
    """Short ID of dialogue line N."""
    ev = next(e for e in EP.events if e.text == f"line {line}")
    return f"B{ev.idx:03d}"


def cand(segs, hook, cliff) -> MinedCandidate:
    return MinedCandidate(
        title="t", segments=[CandidateSegment(start_id=bid(a), end_id=bid(b)) for a, b in segs],
        hook_id=bid(hook), hook="h", cliffhanger_id=bid(cliff), cliffhanger="c", hook_type="shocking_line",
        why_it_works="w", context_needed="", spoiler_level="none", scores=MinedScores(hook=9, cliffhanger=3, standalone=0),
    )


def test_clean_candidate_snaps_to_shots_and_clamps_scores():
    p = assemble(EP, cand([(2, 13)], 2, 13), 1)
    seg = p.segments[0]
    assert seg.start == 6.0  # line 2 starts 6.5 -> snapped back to shot at 6.0
    assert seg.end == 42.0  # line 13 ends 41.5 -> shot ends 0.5 s later, snapped
    assert p.hook_offset == 0.5 and 20 <= p.duration <= 60 and p.repairs == []
    assert (p.scores.hook, p.scores.standalone) == (5, 1)


def test_lead_in_is_trimmed_to_hook():
    p = assemble(EP, cand([(0, 12)], 4, 12), 1)
    assert "trimmed_lead_in_to_hook" in p.repairs and p.hook_offset <= 3.0


def test_too_long_single_segment_is_split():
    p = assemble(EP, cand([(0, 30)], 0, 30), 1)
    assert "split_long_clip_into_head_and_tail" in p.repairs
    assert len(p.segments) == 2 and p.duration <= 60
    assert p.segments[-1].end == 92.5  # still ends on the cliffhanger line (shot end is 1.5 s away: no snap)


def test_too_short_is_rejected():
    with pytest.raises(Reject) as r:
        assemble(EP, cand([(0, 2)], 0, 2), 1)
    assert r.value.reason == "too_short"


def test_slightly_short_is_extended_and_flagged():
    p = assemble(EP, cand([(0, 5)], 0, 5), 1)
    assert "extended_end_past_cliffhanger" in p.repairs and p.duration >= 20


def test_runs_past_cliffhanger_is_trimmed():
    p = assemble(EP, cand([(2, 16)], 2, 12), 1)
    assert "trimmed_after_cliffhanger" in p.repairs and p.segments[-1].end == 38.5


def test_unknown_id_and_hook_outside_are_rejected():
    c = cand([(2, 12)], 2, 12)
    c.hook_id = "B999"
    with pytest.raises(Reject) as r:
        assemble(EP, c, 1)
    assert r.value.reason == "unknown_id"
    with pytest.raises(Reject) as r:
        assemble(EP, cand([(2, 12)], 20, 12), 1)
    assert r.value.reason == "hook_outside_first_segment"


def test_multi_segment_overlap_rejected():
    with pytest.raises(Reject) as r:
        assemble(EP, cand([(2, 8), (6, 14)], 2, 14), 1)
    assert r.value.reason == "overlapping_segments"


def test_back_to_back_segments_are_merged():
    # lines 2-6 and 7-12 are separated by a 1 s gap: one stretch, not a jump cut
    p = assemble(EP, cand([(2, 6), (7, 12)], 2, 12), 1)
    assert len(p.segments) == 1 and "merged_adjacent_segments" in p.repairs
    # four segments that collapse to two are no longer rejected for being > 3
    p = assemble(EP, cand([(2, 4), (5, 6), (12, 14), (15, 16)], 2, 16), 1)
    assert len(p.segments) == 2


def test_rank_drops_overlapping_duplicates():
    a = assemble(EP, cand([(2, 12)], 2, 12), 1)
    b = assemble(EP, cand([(3, 12)], 3, 12), 2)
    c = assemble(EP, cand([(20, 30)], 20, 30), 3)
    ranked, dupes = rank([a, b, c], n_episodes=50)
    assert len(ranked) == 2 and len(dupes) == 1
    assert [p.rank for p in ranked] == [1, 2] and ranked[0].post_first_reason
