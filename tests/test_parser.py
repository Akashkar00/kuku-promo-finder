import os
from pathlib import Path

import pytest

from promo_finder.parser import _parse_label, fmt_ts, parse_episode, parse_show, parse_ts

PACK = Path(os.environ.get("PROMO_PACK_DIR", "/nonexistent"))  # confidential pack, never in the repo
needs_pack = pytest.mark.skipif(not PACK.exists(), reason="confidential input pack not present")

SAMPLE = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
SCENE HEADING: INT. ROOM – DAY
SCENE START: 00:00.00
SCENE END: 00:10.00

ACTIONS AND DIALOGUE:

**CAMERA:**
Start: (00:00.00) | End: (00:02.00)
Close-up on a door.

**MATRIARCH (O.S.) (Cold, Commanding):**
Start: (00:01.00) | End: (00:03.50)
"The ferry leaves at noon."

**ACTION:**
Start: (00:04.00) | End: (00:03.00)
She slaps him.

**ACTION:**
This block has no timing line.

stray text outside any block

--- SCRIPT END ---
"""


def test_timestamps_roundtrip():
    assert parse_ts("01:05.60") == 65.6
    assert parse_ts("1:02:03.5") == 3723.5
    assert fmt_ts(65.6) == "01:05.60"


def test_label_parsing():
    assert _parse_label("CAMERA") == ("camera", None, [], [])
    assert _parse_label("SOUND FX") == ("sfx", None, [], [])
    t, speaker, mods, emos = _parse_label("PRUDENCE (Solemn, O.S.)")
    assert (t, speaker, mods, emos) == ("dialogue", "PRUDENCE", ["O.S."], ["Solemn"])
    t, speaker, mods, emos = _parse_label("HARBOUR MASTER (O.S. / Continues)")
    assert speaker == "HARBOUR MASTER" and mods == ["O.S.", "CONTINUES"] and emos == []


def test_synthetic_uppercase_and_malformed(tmp_path):
    f = tmp_path / "episode-007-Episode 99.txt"
    f.write_text(SAMPLE)
    ep, anomalies = parse_episode(f)
    assert ep.number == 7 and ep.logline == "A test logline."
    assert ep.scenes[0].heading == "INT. ROOM – DAY" and ep.scenes[0].end == 10.0
    assert [e.type for e in ep.events] == ["camera", "dialogue", "action"]
    line = ep.events[1]
    assert line.text == "The ferry leaves at noon." and line.modifiers == ["O.S."]
    assert line.emotions == ["Cold", "Commanding"] and line.id == "E07-S01-B001"
    kinds = sorted(a.kind for a in anomalies)
    # the untimed block's body and the stray line are both reported
    assert kinds == ["missing_timing", "negative_duration", "unparsed_line", "unparsed_line"]
    assert (ep.events[2].start, ep.events[2].end) == (3.0, 4.0)  # swapped


@needs_pack
def test_full_pack_parses_cleanly():
    episodes, anomalies = parse_show(PACK)
    assert len(episodes) == 50
    assert [e.number for e in episodes] == list(range(1, 51))
    assert anomalies == []
    assert sum(len(e.events) for e in episodes) == 3727


@needs_pack
def test_episode_6_header_drift():
    episodes, _ = parse_show(PACK, episodes={6})
    ep = episodes[0]
    assert len(ep.scenes) == 2
    assert ep.scenes[1].heading.startswith("INT. WOODEN CABIN")
    assert ep.scenes[1].start == 33.7 and ep.duration == 115.0
