"""LLM candidate (event IDs) -> validated Promo with exact timestamps.

All hard rules live here, in code, not in the prompt:
  * every cut lands on a shot boundary when one is close, never mid-line
  * the hook starts <= 3 s after the clip starts
  * total length 20-60 s, 1-3 segments, segments do not overlap
Fixable problems are repaired and recorded in `Promo.repairs`; anything
else becomes a `Rejection` with a reason, so the reject rate is measurable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .condense import event_line
from .parser import fmt_ts
from .schemas import Episode, Event, MinedCandidate, Promo, Rejection, Segment


@dataclass
class Rules:
    min_len: float = 20.0
    max_len: float = 60.0
    hook_window: float = 3.0
    max_segments: int = 3
    min_segment: float = 2.0
    merge_gap: float = 1.5  # segments closer than this are merged into one stretch
    snap_tolerance: float = 1.0  # how far a cut may move to reach a shot boundary
    extend_short_from: float = 14.0  # clips shorter than this are rejected, not extended
    split_head: float = 25.0  # when splitting an over-long clip
    split_tail: float = 30.0


class Reject(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


_ID = re.compile(r"B(\d+)", re.I)


class Timeline:
    """Boundary lookups over one episode."""

    def __init__(self, ep: Episode, rules: Rules):
        self.ep, self.rules = ep, rules
        self.events = ep.events
        self.cams = [e for e in ep.events if e.type == "camera"]
        self.lines = [e for e in ep.events if e.type == "dialogue"]
        self.visible = [e for e in ep.events if e.type != "camera"]

    def resolve(self, ref: str) -> Event:
        m = _ID.search(ref or "")
        if not m or int(m.group(1)) >= len(self.events):
            raise Reject("unknown_id", f"{ref!r} is not an event in episode {self.ep.number}")
        return self.events[int(m.group(1))]

    def snap_start(self, t: float) -> float:
        for d in self.lines:  # never start mid-line
            if d.start < t < d.end:
                t = d.start
        for c in self.cams:  # prefer starting on a shot change
            if c.start <= t < c.end and t - c.start <= self.rules.snap_tolerance:
                return c.start
        return t

    def snap_end(self, t: float) -> float:
        for d in self.lines:  # never cut mid-line
            if d.start < t < d.end:
                t = d.end
        for c in self.cams:
            if c.start < t < c.end and c.end - t <= self.rules.snap_tolerance:
                t = c.end
                break
        return min(t, self.ep.duration)

    def visible_between(self, start: float, end: float) -> list[Event]:
        return [e for e in self.visible if e.start >= start - 0.05 and e.start < end]


def assemble(ep: Episode, cand: MinedCandidate, n: int, rules: Rules | None = None) -> Promo:
    rules = rules or Rules()
    tl = Timeline(ep, rules)
    repairs: list[str] = []

    if not cand.segments:
        raise Reject("no_segments")

    hook = tl.resolve(cand.hook_id)
    cliff = tl.resolve(cand.cliffhanger_id)

    # Resolve segments to event ranges (fix reversed ranges).
    ranges: list[tuple[Event, Event]] = []
    for seg in cand.segments:
        a, b = tl.resolve(seg.start_id), tl.resolve(seg.end_id)
        if a.idx > b.idx:
            a, b = b, a
            repairs.append("swapped_reversed_segment")
        ranges.append((a, b))

    # Segments that are back to back in the episode are one stretch, not a jump cut.
    merged = [ranges[0]]
    for a, b in ranges[1:]:
        pa, pb = merged[-1]
        if a.idx > pb.idx and a.start - pb.end <= rules.merge_gap:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    if len(merged) < len(ranges):
        repairs.append("merged_adjacent_segments")
        ranges = merged
    if len(ranges) > rules.max_segments:
        raise Reject("too_many_segments", f"{len(ranges)} > {rules.max_segments}")

    first_a, first_b = ranges[0]
    last_a, last_b = ranges[-1]
    if not (first_a.idx <= hook.idx <= first_b.idx):
        raise Reject("hook_outside_first_segment", f"hook {cand.hook_id} not in {first_a.idx}-{first_b.idx}")
    if not (last_a.idx <= cliff.idx <= last_b.idx):
        raise Reject("cliffhanger_outside_last_segment", f"cliffhanger {cand.cliffhanger_id} not in last segment")

    # Convert to times, snapping each cut.
    spans: list[list[float]] = []
    for i, (a, b) in enumerate(ranges):
        start = tl.snap_start(a.start)
        # A line inside the range may run past the end event (overlapping speech).
        end = max([e.end for e in tl.events[a.idx : b.idx + 1] if e.type != "camera"] or [b.end])
        if i == len(ranges) - 1:
            if cliff.idx < b.idx:  # LLM ran past its own cliffhanger: cut at it
                end = cliff.end
                repairs.append("trimmed_after_cliffhanger")
        end = tl.snap_end(end)
        spans.append([start, end])

    # Hook must land in the first 3 s: if the LLM padded a lead-in, cut it.
    if hook.start - spans[0][0] > rules.hook_window:
        spans[0][0] = tl.snap_start(hook.start)
        if hook.start - spans[0][0] > rules.hook_window:  # snap moved it too far back
            spans[0][0] = hook.start
        repairs.append("trimmed_lead_in_to_hook")

    total = sum(e - s for s, e in spans)

    # Too long: a single stretch becomes hook-head + cliffhanger-tail.
    if total > rules.max_len:
        if len(spans) != 1:
            raise Reject("too_long", f"{total:.1f}s across {len(spans)} segments")
        s, e = spans[0]
        head_events = [v for v in tl.visible_between(s, e) if v.end <= s + rules.split_head]
        tail_events = [v for v in tl.visible_between(s, e) if v.start >= e - rules.split_tail]
        if not head_events or not tail_events:
            raise Reject("too_long", f"{total:.1f}s and no clean split point")
        head_end = tl.snap_end(max(v.end for v in head_events))
        tail_start = tl.snap_start(min(v.start for v in tail_events))
        if tail_start <= head_end + 1.0:
            raise Reject("too_long", f"{total:.1f}s and split points overlap")
        spans = [[s, head_end], [tail_start, e]]
        repairs.append("split_long_clip_into_head_and_tail")
        total = sum(b - a for a, b in spans)

    # Too short: extend past the cliffhanger only a little, and say so.
    if total < rules.min_len:
        if total < rules.extend_short_from:
            raise Reject("too_short", f"{total:.1f}s")
        need = rules.min_len - total
        nxt = [v for v in tl.visible if v.start >= spans[-1][1] - 0.05]
        new_end = spans[-1][1]
        for v in nxt:
            new_end = v.end
            if new_end - spans[-1][1] >= need:
                break
        new_end = tl.snap_end(new_end)
        if sum(b - a for a, b in spans) + (new_end - spans[-1][1]) < rules.min_len:
            raise Reject("too_short", f"{total:.1f}s and episode ends")
        spans[-1][1] = new_end
        repairs.append("extended_end_past_cliffhanger")
        total = sum(b - a for a, b in spans)

    # Final hard checks.
    for i, (s, e) in enumerate(spans):
        if e - s < rules.min_segment:
            raise Reject("segment_too_short", f"segment {i + 1} is {e - s:.1f}s")
    ordered = sorted(spans)
    for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
        if s2 < e1 - 0.05:
            raise Reject("overlapping_segments", f"{fmt_ts(s2)} < {fmt_ts(e1)}")
    hook_offset = round(hook.start - spans[0][0], 2)
    if not (0 <= hook_offset <= rules.hook_window):
        raise Reject("hook_not_in_first_3s", f"offset {hook_offset}s")
    if not (rules.min_len <= total <= rules.max_len):
        raise Reject("bad_length", f"{total:.1f}s")

    segments: list[Segment] = []
    transcript: list[str] = []
    for s, e in spans:
        inside = tl.visible_between(s, e)
        start_ev = inside[0] if inside else hook
        end_ev = inside[-1] if inside else cliff
        segments.append(Segment(episode=ep.number, start=round(s, 2), end=round(e, 2), start_id=start_ev.id, end_id=end_ev.id))
        transcript += [event_line(v) for v in inside] + ["--- CUT ---"]
    transcript = transcript[:-1]

    return Promo(
        promo_id=f"E{ep.number:02d}-P{n}",
        miner_rank=n,
        title=cand.title,
        segments=segments,
        duration=round(total, 2),
        hook=cand.hook,
        hook_id=hook.id,
        hook_offset=hook_offset,
        cliffhanger=cand.cliffhanger,
        cliffhanger_id=cliff.id,
        hook_type=cand.hook_type,
        why_it_works=cand.why_it_works,
        context_needed=cand.context_needed,
        spoiler_level=cand.spoiler_level,
        scores=_clamp_scores(cand),
        repairs=repairs,
        transcript=transcript,
    )


def _clamp_scores(cand: MinedCandidate):
    s = cand.scores.model_copy()
    s.hook, s.cliffhanger, s.standalone = (max(1, min(5, v)) for v in (s.hook, s.cliffhanger, s.standalone))
    return s


def assemble_all(
    episodes: list[Episode], mined: dict[int, list[MinedCandidate]], rules: Rules | None = None
) -> tuple[list[Promo], list[Rejection]]:
    by_num = {e.number: e for e in episodes}
    promos: list[Promo] = []
    rejected: list[Rejection] = []
    for num, cands in sorted(mined.items()):
        for i, cand in enumerate(cands, start=1):
            try:
                promos.append(assemble(by_num[num], cand, i, rules))
            except Reject as r:
                rejected.append(Rejection(episode=num, title=cand.title, reason=r.reason, detail=r.detail, candidate=cand.model_dump()))
    return promos, rejected
