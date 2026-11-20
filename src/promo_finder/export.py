"""Write run outputs: JSON for machines, Markdown + CSV for editors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .parser import fmt_ts
from .schemas import Promo, Rejection


def seg_str(p: Promo) -> str:
    return " + ".join(f"E{s.episode:02d} {fmt_ts(s.start)}–{fmt_ts(s.end)}" for s in p.segments)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_promos(out: Path, promos: list[Promo], rejected: list[Rejection], dupes, title: str, top_n: int = 20) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "promos.json", [p.model_dump() for p in promos])
    write_json(out / "rejected.json", [r.model_dump() for r in rejected])
    write_json(out / "duplicates.json", [{"dropped": d.promo_id, "kept": k.promo_id, "dropped_title": d.title} for d, k in dupes])

    with (out / "promos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "promo_id", "segments", "duration_s", "hook_offset_s", "hook_type", "hook", "cliffhanger",
                    "why_it_works", "spoiler", "context_needed", "hook_score", "cliff_score", "standalone_score",
                    "cut", "judge_score", "judge_reason", "score", "flags", "alt_cuts", "repairs", "post_first_reason"])
        for p in promos:
            w.writerow([p.rank, p.promo_id, seg_str(p), p.duration, p.hook_offset, p.hook_type, p.hook, p.cliffhanger,
                        p.why_it_works, p.spoiler_level, p.context_needed, p.scores.hook, p.scores.cliffhanger,
                        p.scores.standalone, p.cut_style, p.judge_score, p.judge_reason or "", p.score,
                        " | ".join(p.flags), ";".join(p.alt_cuts), ";".join(p.repairs), p.post_first_reason or ""])

    md = [f"# Promo shortlist: {title}", "",
          f"{len(promos)} promos passed validation; {len(rejected)} candidates rejected; {len(dupes)} near-duplicates merged.",
          "Timestamps are episode-relative (mm:ss.cc), as in the screenplay.", "", "## Post these first", ""]
    for p in promos:
        if p.post_first_reason:
            md += [f"**#{p.rank}. {p.title}** ({seg_str(p)}, {p.duration:.0f}s)", f"- Hook: {p.hook}",
                   f"- Cliffhanger: {p.cliffhanger}", f"- Why: {p.why_it_works}", f"- Post first because: {p.post_first_reason}"]
            md += [f"- Check: {f}" for f in p.flags]
            md += ([f"- Other cuts of this scene: {', '.join(p.alt_cuts)}"] if p.alt_cuts else []) + [""]
    md += ["Ranks 1-5 are the posting order: the strongest promos with distinct episodes and hook types.", ""]
    md += [f"## Top {top_n}", "", "| # | Promo | Segments | Len | Hook | Cliffhanger | Why it works | Spoiler | H/C/S | Judge | Check |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in promos[:top_n]:
        cells = [str(p.rank), p.title, seg_str(p), f"{p.duration:.0f}s", p.hook, p.cliffhanger, p.why_it_works,
                 p.spoiler_level, f"{p.scores.hook}/{p.scores.cliffhanger}/{p.scores.standalone}",
                 "-" if p.judge_score is None else f"{p.judge_score:.2f}", "; ".join(p.flags) or "-"]
        md.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    md += ["", f"All {len(promos)} promos are in promos.csv / promos.json. Rejected candidates and reasons: rejected.json."]
    (out / "promos.md").write_text("\n".join(md), encoding="utf-8")
