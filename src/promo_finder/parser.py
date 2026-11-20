"""Screenplay files -> typed Episode/Event objects.

Pure code, no LLM. Tolerant of header-case drift (episode 6 uses
uppercase SCENE HEADING/START/END) and logs anything it cannot parse
instead of crashing, so a new show with slightly different formatting
degrades visibly rather than silently.
"""

from __future__ import annotations

import re
from pathlib import Path

from .schemas import Episode, Event, ParseAnomaly, Scene

_FILE_EP = re.compile(r"episode-(\d+)", re.I)
_HEADER = {
    "number": re.compile(r"^EPISODE NUMBER:\s*(.*)$", re.I | re.M),
    "logline": re.compile(r"^EPISODE LOGLINE:\s*(.*)$", re.I | re.M),
    "continuity": re.compile(r"^CONTINUITY NOTES:\s*(.*)$", re.I | re.M),
}
_SCRIPT_START = re.compile(r"^-+\s*SCRIPT START\s*-+\s*$", re.I | re.M)
_SCRIPT_END = re.compile(r"^-+\s*SCRIPT END\s*-+\s*$", re.I | re.M)
_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
_SCENE_FIELD = re.compile(r"^Scene (heading|start|end):\s*(.*)$", re.I)
_ACTIONS_HDR = re.compile(r"^Actions and dialogue:\s*$", re.I)
_LABEL = re.compile(r"^\*\*(.+?):\*\*\s*$")
_TIMING = re.compile(r"^Start:\s*\(([\d:.]+)\)\s*\|\s*End:\s*\(([\d:.]+)\)\s*$", re.I)
_PAREN = re.compile(r"\(([^)]*)\)")
_MODIFIERS = {"O.S.", "V.O.", "O.C.", "CONT'D", "CONTINUES"}

_NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def parse_ts(value: str) -> float:
    """'01:05.60' -> 65.6 ; also accepts 'h:mm:ss.cc' and bare seconds."""
    parts = value.strip().split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return round(seconds, 3)


def fmt_ts(seconds: float) -> str:
    """65.6 -> '01:05.60' (same format as the screenplay)."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _parse_label(label: str) -> tuple[str, str | None, list[str], list[str]]:
    """Return (type, speaker, modifiers, emotions) for a block label."""
    upper = label.strip().upper()
    if upper in _NON_DIALOGUE:
        return _NON_DIALOGUE[upper], None, [], []
    speaker = _PAREN.split(label)[0].strip()
    modifiers: list[str] = []
    emotions: list[str] = []
    for group in _PAREN.findall(label):
        for token in re.split(r"[,/]", group):
            token = token.strip()
            if not token:
                continue
            if token.upper() in _MODIFIERS:
                modifiers.append(token.upper())
            else:
                emotions.append(token)
    return "dialogue", speaker, modifiers, emotions


def episode_number(path: Path, text: str) -> int:
    m = _FILE_EP.search(path.name)
    if m:
        return int(m.group(1))
    m = _HEADER["number"].search(text)
    digits = re.search(r"\d+", m.group(1)) if m else None
    if digits:
        return int(digits.group())
    raise ValueError(f"cannot determine episode number for {path.name}")


def parse_episode(path: Path) -> tuple[Episode, list[ParseAnomaly]]:
    text = path.read_text(encoding="utf-8")
    anomalies: list[ParseAnomaly] = []

    def flag(line: int, kind: str, detail: str) -> None:
        anomalies.append(ParseAnomaly(file=path.name, line=line, kind=kind, detail=detail))

    ep = episode_number(path, text)
    header = {k: (m.group(1).strip() if (m := rx.search(text)) else "") for k, rx in _HEADER.items()}

    lines = text.splitlines()
    start_m, end_m = _SCRIPT_START.search(text), _SCRIPT_END.search(text)
    first = text[: start_m.end()].count("\n") + 1 if start_m else 0
    last = text[: end_m.start()].count("\n") if end_m else len(lines)
    if not start_m:
        flag(0, "missing_marker", "no '--- SCRIPT START ---'; parsing whole file")

    scenes: list[Scene] = []
    events: list[Event] = []
    pending: dict | None = None  # block currently being read

    def close_block(lineno: int) -> None:
        nonlocal pending
        if pending is None:
            return
        body = " ".join(pending["text"]).strip()
        if pending.get("start") is None:
            flag(pending["line"], "missing_timing", f"block '{pending['label']}' has no Start/End line")
        elif not body:
            flag(pending["line"], "empty_block", f"block '{pending['label']}' has no text")
        else:
            etype, speaker, mods, emos = _parse_label(pending["label"])
            if etype == "dialogue":
                body = body.strip().strip('"').strip()
            start, end = pending["start"], pending["end"]
            if end < start:
                flag(pending["line"], "negative_duration", f"{start} > {end}; swapped")
                start, end = end, start
            scene_no = scenes[-1].number if scenes else 1
            events.append(
                Event(
                    id=f"E{ep:02d}-S{scene_no:02d}-B{len(events):03d}",
                    episode=ep,
                    scene=scene_no,
                    idx=len(events),
                    type=etype,
                    speaker=speaker,
                    modifiers=mods,
                    emotions=emos,
                    start=start,
                    end=end,
                    text=body,
                )
            )
        pending = None

    for i in range(first, last):
        lineno, raw = i + 1, lines[i]
        line = raw.strip()
        if not line:
            continue
        if m := _SCENE.match(line):
            close_block(lineno)
            scenes.append(Scene(number=int(m.group(1)), heading="", start=None, end=None))
            continue
        if m := _SCENE_FIELD.match(line):
            close_block(lineno)
            if not scenes:
                scenes.append(Scene(number=1, heading="", start=None, end=None))
            field, value = m.group(1).lower(), m.group(2).strip()
            if field == "heading":
                scenes[-1].heading = value
            else:
                try:
                    setattr(scenes[-1], field, parse_ts(value))
                except ValueError:
                    flag(lineno, "bad_timestamp", f"scene {field}: {value!r}")
            continue
        if _ACTIONS_HDR.match(line):
            continue
        if m := _LABEL.match(line):
            close_block(lineno)
            pending = {"label": m.group(1), "line": lineno, "start": None, "end": None, "text": []}
            continue
        if pending is not None and pending["start"] is None and (m := _TIMING.match(line)):
            try:
                pending["start"], pending["end"] = parse_ts(m.group(1)), parse_ts(m.group(2))
            except ValueError:
                flag(lineno, "bad_timestamp", line)
            continue
        if pending is not None and pending["start"] is not None:
            pending["text"].append(line)
            continue
        flag(lineno, "unparsed_line", line[:120])
    close_block(last)

    # Sanity checks that matter for cutting promos.
    for prev, cur in zip(events, events[1:]):
        if cur.start + 1.0 < prev.start:
            flag(0, "non_monotonic", f"{cur.id} starts at {cur.start}s, before {prev.id} at {prev.start}s")

    scene_ends = [s.end for s in scenes if s.end is not None]
    duration = max(scene_ends + [e.end for e in events] + [0.0])

    return (
        Episode(
            number=ep,
            source_file=path.name,
            logline=header["logline"],
            continuity=header["continuity"],
            scenes=scenes,
            events=events,
            duration=duration,
        ),
        anomalies,
    )


def parse_show(directory: Path, episodes: set[int] | None = None) -> tuple[list[Episode], list[ParseAnomaly]]:
    """Parse every episode file in a directory, ordered by episode number."""
    files = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if not files:
        raise FileNotFoundError(f"no .txt screenplay files in {directory}")
    parsed: list[Episode] = []
    anomalies: list[ParseAnomaly] = []
    for path in files:
        try:
            ep, issues = parse_episode(path)
        except ValueError as exc:
            anomalies.append(ParseAnomaly(file=path.name, line=0, kind="skipped_file", detail=str(exc)))
            continue
        if episodes is None or ep.number in episodes:
            parsed.append(ep)
            anomalies.extend(issues)
    parsed.sort(key=lambda e: e.number)
    return parsed, anomalies
