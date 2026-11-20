"""Typed data model shared by every stage of the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal["camera", "action", "dialogue", "sfx"]


# ---------------------------------------------------------------- parsed input


class Event(BaseModel):
    """One timed block of the screenplay (a shot, an action, a line, a sound)."""

    id: str  # stable, e.g. "E01-S01-B023"
    episode: int
    scene: int
    idx: int  # position within the episode, 0-based
    type: EventType
    speaker: str | None = None
    modifiers: list[str] = Field(default_factory=list)  # O.S., V.O.
    emotions: list[str] = Field(default_factory=list)  # "Outraged", "Accusing"
    start: float  # seconds, episode-relative
    end: float
    text: str


class Scene(BaseModel):
    number: int
    heading: str
    start: float | None
    end: float | None


class Episode(BaseModel):
    number: int
    source_file: str
    logline: str
    continuity: str
    scenes: list[Scene]
    events: list[Event]
    duration: float  # seconds


class ParseAnomaly(BaseModel):
    file: str
    line: int
    kind: str
    detail: str


# ---------------------------------------------------------------- LLM-facing
# Kept flat and constraint-free so they work as structured-output schemas on
# every provider; range checks (scores 1-5, 1-3 segments) are enforced in code.

HookType = Literal[
    "slap_or_violence", "betrayal_reveal", "shocking_line", "identity_reveal",
    "humiliation", "threat_or_ultimatum", "rescue_or_power_move", "romance_beat", "other",
]


class Character(BaseModel):
    name: str
    aliases: list[str]  # other labels the script uses for the same person
    role: str


class ShowBrief(BaseModel):
    premise: str
    main_characters: list[Character]
    core_tropes: list[str]


class CandidateSegment(BaseModel):
    start_id: str  # e.g. "B012", inclusive
    end_id: str  # inclusive


class MinedScores(BaseModel):
    hook: int
    cliffhanger: int
    standalone: int


class MinedCandidate(BaseModel):
    title: str
    segments: list[CandidateSegment]
    hook_id: str
    hook: str
    cliffhanger_id: str
    cliffhanger: str
    hook_type: HookType
    why_it_works: str
    context_needed: str  # what a cold viewer would be missing; "" if nothing
    spoiler_level: Literal["none", "low", "high"]
    scores: MinedScores


class EpisodeMining(BaseModel):
    candidates: list[MinedCandidate]


# ---------------------------------------------------------------- outputs


class Segment(BaseModel):
    episode: int
    start: float
    end: float
    start_id: str
    end_id: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class Promo(BaseModel):
    promo_id: str
    title: str
    segments: list[Segment]
    duration: float
    hook: str
    hook_id: str
    hook_offset: float  # seconds from clip start to the hook event
    cliffhanger: str
    cliffhanger_id: str
    hook_type: str
    why_it_works: str
    context_needed: str
    spoiler_level: str
    scores: MinedScores
    repairs: list[str] = Field(default_factory=list)  # what the validator changed
    transcript: list[str] = Field(default_factory=list)  # lines inside the clip
    miner_rank: int = 1  # the miner's own best-first position within its episode
    judge_score: float | None = None  # 0-1 percentile from the cold-viewer judge (None = judge not run)
    judge_reason: str | None = None
    alt_cuts: list[str] = Field(default_factory=list)  # other promos that cut the same scene
    flags: list[str] = Field(default_factory=list)  # editor warnings, e.g. "ending extended"
    score: float = 0.0
    rank: int | None = None
    post_first_reason: str | None = None

    @property
    def max_gap(self) -> float:
        """Largest source-time gap skipped between consecutive segments (0 for one stretch)."""
        return max((b.start - a.end for a, b in zip(self.segments, self.segments[1:])), default=0.0)

    @property
    def cut_style(self) -> str:
        if len(self.segments) == 1:
            return "single stretch"
        return f"{len(self.segments)} segments, max gap {self.max_gap:.0f}s"


class Rejection(BaseModel):
    episode: int
    title: str
    reason: str
    detail: str
    candidate: dict


class JudgeReason(BaseModel):
    clip: str
    reason: str


class JudgeOrdering(BaseModel):
    ranking: list[str]  # clip labels, best first
    top_reasons: list[JudgeReason]  # one line for each of the top 3
