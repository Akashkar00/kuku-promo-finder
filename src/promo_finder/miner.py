"""LLM stages: one show-brief call, then one mining call per episode."""

from __future__ import annotations

import json
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Callable

from rich.progress import track

from .condense import condense_episode
from .llm import LLM, LLMError
from .schemas import Episode, EpisodeMining, MinedCandidate, ShowBrief


def _prompt(name: str) -> str:
    return resources.files("promo_finder").joinpath(f"prompts/{name}.md").read_text(encoding="utf-8")


def speaker_labels(episodes: list[Episode]) -> str:
    """'LABEL: first seen ep N, M lines' for every speaker label in the script."""
    first: dict[str, int] = {}
    count: Counter[str] = Counter()
    for ep in episodes:
        for ev in ep.events:
            if ev.type == "dialogue" and ev.speaker:
                first.setdefault(ev.speaker, ep.number)
                count[ev.speaker] += 1
    return "\n".join(f"{s}: first seen ep {first[s]}, {n} lines" for s, n in count.most_common())


def build_show_brief(llm: LLM, episodes: list[Episode]) -> ShowBrief:
    loglines = "\n".join(f"Episode {e.number}: {e.logline}" for e in episodes)
    user = (
        f"EPISODE LOGLINES:\n{loglines}\n\n"
        f"SPEAKER LABELS USED IN THE SCREENPLAY (map each one to a character where you can):\n{speaker_labels(episodes)}"
    )
    res = llm.structured(task="show_brief", system=_prompt("show_brief"), user=user, schema=ShowBrief)
    return res.parsed  # type: ignore[return-value]


def brief_text(brief: ShowBrief) -> str:
    chars = "\n".join(
        f"- {c.name} ({c.role}); also labelled: {', '.join(c.aliases) or 'n/a'}" for c in brief.main_characters
    )
    return f"PREMISE: {brief.premise}\nCHARACTERS:\n{chars}\nTROPES: {', '.join(brief.core_tropes)}"


def mine_episodes(
    llm: LLM,
    episodes: list[Episode],
    brief: ShowBrief,
    raw_dir: Path,
    include_camera: bool = False,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[dict[int, list[MinedCandidate]], dict[int, str]]:
    """Return candidates per episode and a map of episode -> error for failures."""
    system = _prompt("mine")
    show = brief_text(brief)
    raw_dir.mkdir(parents=True, exist_ok=True)
    found: dict[int, list[MinedCandidate]] = {}
    errors: dict[int, str] = {}
    loop = episodes if on_progress else track(episodes, description="Mining episodes")
    for i, ep in enumerate(loop, start=1):
        if on_progress:
            on_progress(i, len(episodes), ep.number)
        user = (
            f"SHOW BRIEF (for context only; a promo viewer will NOT know any of this):\n{show}\n\n"
            f"EPISODE TIMELINE:\n{condense_episode(ep, include_camera=include_camera)}"
        )
        try:
            res = llm.structured(task="mine", system=system, user=user, schema=EpisodeMining)
        except LLMError as exc:
            errors[ep.number] = str(exc)
            continue
        (raw_dir / f"mine_E{ep.number:02d}.json").write_text(json.dumps(json.loads(res.raw), indent=2))
        found[ep.number] = res.parsed.candidates  # type: ignore[attr-defined]
    return found, errors
