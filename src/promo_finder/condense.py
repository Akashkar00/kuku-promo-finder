"""Episode -> compact, ID-tagged text timeline for the LLM.

CAMERA blocks are ~47% of all blocks and mostly restate the action, so
they are dropped from the LLM view by default (they are still used by
the assembler to snap cuts onto shot changes). Known cost: a hook that
exists *only* in a camera description is invisible to the miner.
"""

from __future__ import annotations

from .parser import fmt_ts
from .schemas import Episode, Event


def short_id(event: Event) -> str:
    return f"B{event.idx:03d}"


def event_line(event: Event) -> str:
    span = f"[{fmt_ts(event.start)}-{fmt_ts(event.end)}]"
    if event.type == "dialogue":
        tags = ", ".join(event.modifiers + event.emotions)
        who = f"{event.speaker} ({tags})" if tags else event.speaker
        return f'{short_id(event)} {span} {who}: "{event.text}"'
    label = {"action": "ACTION", "sfx": "SFX", "camera": "CAMERA"}[event.type]
    return f"{short_id(event)} {span} {label}: {event.text}"


def condense_episode(ep: Episode, include_camera: bool = False) -> str:
    out = [
        f"EPISODE {ep.number} (duration {fmt_ts(ep.duration)})",
        f"LOGLINE: {ep.logline}",
        f"CONTINUITY: {ep.continuity}",
    ]
    scenes = {s.number: s for s in ep.scenes}
    current = None
    for ev in ep.events:
        if ev.scene != current:
            current = ev.scene
            s = scenes.get(current)
            if s:
                span = f"[{fmt_ts(s.start or 0)}-{fmt_ts(s.end or ep.duration)}]"
                out.append(f"== SCENE {s.number}: {s.heading} {span} ==")
        if ev.type == "camera" and not include_camera:
            continue
        out.append(event_line(ev))
    return "\n".join(out)
