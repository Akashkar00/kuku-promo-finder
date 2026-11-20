"""No-LLM baseline: the last ~45 s of every episode.

Microdrama episodes are written to end on a cliffhanger, so "the episode's
own ending" is the obvious thing a busy editor would cut. The LLM pipeline
has to beat this to justify its cost.
"""

from __future__ import annotations

from .assemble import Rules, Timeline
from .condense import event_line
from .schemas import Episode, MinedScores, Promo, Segment


def baseline_promos(episodes: list[Episode], length: float = 45.0) -> list[Promo]:
    out: list[Promo] = []
    for ep in episodes:
        tl = Timeline(ep, Rules())
        end = ep.duration
        start = tl.snap_start(max(0.0, end - length))
        inside = tl.visible_between(start, end)
        if not inside:
            continue
        first, last = inside[0], inside[-1]
        out.append(Promo(
            promo_id=f"E{ep.number:02d}-BASE", title=f"Episode {ep.number} ending",
            segments=[Segment(episode=ep.number, start=round(start, 2), end=round(end, 2), start_id=first.id, end_id=last.id)],
            duration=round(end - start, 2), hook=first.text, hook_id=first.id, hook_offset=round(first.start - start, 2),
            cliffhanger=last.text, cliffhanger_id=last.id, hook_type="other", why_it_works="baseline: episode ending",
            context_needed="unknown", spoiler_level="unknown", scores=MinedScores(hook=0, cliffhanger=0, standalone=0),
            transcript=[event_line(v) for v in inside], rank=ep.number,
        ))
    return out
