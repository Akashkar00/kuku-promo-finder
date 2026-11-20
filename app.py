"""Streamlit front end for promo-finder.

    uv run streamlit run app.py

Upload a show's episode .txt files (or one .zip), pick a model, get a ranked
promo shortlist. No screenplay data ships with the app; it only sees what is
uploaded. Runs the exact same pipeline as the CLI (promo_finder.pipeline).
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from promo_finder.cli import parse_range
from promo_finder.export import seg_str
from promo_finder.llm import LLMError
from promo_finder.parser import parse_show
from promo_finder.pipeline import PRICES, load_settings, run_pipeline
from promo_finder.schemas import Promo

load_dotenv()
st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")

ROOT = Path(__file__).parent
RUNS = ROOT / "outputs"
# Hosted mode (Streamlit Community Cloud): set APP_HOSTED = "1" (and ANTHROPIC_API_KEY) as app secrets.
# Root-level Streamlit secrets are exposed as environment variables.
# - Each visitor's runs go to a private temp folder, so no one sees another visitor's (confidential) show.
# - Optional APP_PASSCODE: if set, the server's key is used only after the passcode is entered
#   (a visitor's own key needs no passcode).
PASSCODE = os.environ.get("APP_PASSCODE", "")
HOSTED = bool(PASSCODE) or os.environ.get("APP_HOSTED") == "1"
DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5", "gemini": "gemini-flash-latest", "openai_compat": ""}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai_compat": "OPENAI_COMPAT_API_KEY"}
# Measured on the 50-episode pack: ~5.6k input / ~0.8k output tokens per episode call.
TOKENS_PER_EP = (5600, 830)


# ------------------------------------------------------------------ helpers


def save_uploads(files) -> Path:
    """Write uploaded .txt files (or .txt files inside .zip uploads) to a fresh temp dir."""
    d = Path(tempfile.mkdtemp(prefix="promo_upload_"))
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                for info in z.infolist():
                    name = Path(info.filename).name  # flatten; never trust paths inside the zip
                    if info.is_dir() or not name.lower().endswith(".txt") or name.startswith("._"):
                        continue
                    (d / name).write_bytes(z.read(info))
        elif f.name.lower().endswith(".txt"):
            (d / Path(f.name).name).write_bytes(f.getvalue())
    return d


def load_run(out: Path) -> tuple[list[Promo], list[dict], dict, dict]:
    promos = [Promo.model_validate(p) for p in json.loads((out / "promos.json").read_text())]
    rejected = json.loads((out / "rejected.json").read_text())
    meta = json.loads((out / "run_meta.json").read_text())
    brief = json.loads((out / "show_brief.json").read_text()) if (out / "show_brief.json").exists() else {}
    return promos, rejected, meta, brief


def zip_dir(out: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out))
    return buf.getvalue()


def show_results(out: Path, k: str) -> None:
    """Render one run folder. `k` prefixes widget keys so two tabs can render at once."""
    promos, rejected, meta, brief = load_run(out)
    if not promos:
        st.warning("This run produced no promos.")
        return

    cost = meta.get("estimated_cost_uncached_usd")
    judge_cost = meta.get("estimated_judge_cost_usd")
    if cost is not None and judge_cost is not None:
        cost = cost + judge_cost
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Promos", len(promos))
    c2.metric("LLM candidates", meta.get("candidates"))
    c3.metric("Passed validator", f"{(meta.get('validator_pass_rate') or 0):.0%}")
    if cost is not None:
        cost_label = f"${cost:.2f}"
    elif meta.get("provider") == "gemini":
        cost_label = "not priced"
    else:
        cost_label = "not priced"
    c4.metric("Cost (fresh run)", cost_label,
              help=f"Mining + judge, estimated from token counts at list price"
                   + (f" (judge ${judge_cost:.2f})" if judge_cost is not None else "")
                   + ". Gemini and other OpenAI-compatible models have no price table here.")
    c5.metric("Episodes", len(meta.get("episodes_mined", [])))
    j = meta.get("judge") or {}
    judge_note = (f" Ranked with a cold-viewer judge ({j.get('provider')}/{j.get('model')}, {j.get('passes')} passes, "
                  f"agreement {j.get('pass_agreement_spearman')})." if j.get("ran") else " Ranked on the miner's self-scores (no judge).")
    st.caption(f"Model: {meta.get('provider')}/{meta.get('model')} (effort={meta.get('effort')}). "
               f"Timestamps are episode-relative (mm:ss.cc), as in the screenplay.{judge_note} "
               "Ranks 1-5 are the posting order.")
    if meta.get("episode_errors"):
        st.error(f"Episodes that failed: {meta['episode_errors']}")

    # Post-first picks
    st.subheader("Post these first")
    for p in [p for p in promos if p.post_first_reason]:
        with st.container(border=True):
            st.markdown(f"**#{p.rank} · {p.title}** · `{seg_str(p)}` · {p.duration:.0f}s")
            st.markdown(f"**Hook:** {p.hook}  \n**Cliffhanger:** {p.cliffhanger}  \n**Why:** {p.why_it_works}")
            st.caption(f"Post first because: {p.post_first_reason}")
            for f in p.flags:
                st.warning(f"Check: {f}", icon="⚠️")
            if p.alt_cuts:
                st.caption(f"Other cuts of this scene (merged into this one): {', '.join(p.alt_cuts)}")

    # Filterable table
    st.subheader("All promos")
    f1, f2, f3 = st.columns([2, 2, 1])
    eps = sorted({p.segments[0].episode for p in promos})
    if len(eps) > 1:
        ep_lo, ep_hi = f1.select_slider("Episodes", options=eps, value=(eps[0], eps[-1]), key=f"{k}_eps")
    else:
        ep_lo = ep_hi = eps[0]
        f1.markdown(f"**Episodes**  \nAll promos are from episode {eps[0]}.")
    spoilers = f2.pills("Spoiler level", ["none", "low", "high"], selection_mode="multi", default=["none", "low", "high"], key=f"{k}_spoil")
    min_hook = f3.number_input("Min hook score", 1, 5, 1, key=f"{k}_minhook")
    shown = [p for p in promos if ep_lo <= p.segments[0].episode <= ep_hi
             and p.spoiler_level in (spoilers or []) and p.scores.hook >= min_hook]
    rows = [{
        "rank": p.rank, "promo": p.title, "segments": seg_str(p), "secs": round(p.duration),
        "score": p.score, "judge": p.judge_score, "check": "; ".join(p.flags) or "-",
        "alt cuts": ", ".join(p.alt_cuts),
        "type": p.hook_type.replace("_", " "), "spoiler": p.spoiler_level,
        "hook": p.scores.hook, "cliff": p.scores.cliffhanger, "standalone": p.scores.standalone, "cut": p.cut_style,
    } for p in shown]
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.2f"),
        "judge": st.column_config.NumberColumn("judge", format="%.2f", help="Cold-viewer judge percentile (0-1)"),
    })

    # Editor feedback: the human decision on each promo, saved next to the run.
    st.subheader("Editor review")
    st.caption("Approve or reject promos and leave notes. Saved as feedback.csv in the run folder; "
               "over time these decisions are the data for tuning the rank weights and prompts.")
    fb_path = out / "feedback.csv"
    prior: dict[str, dict] = {}
    if fb_path.exists():
        with fb_path.open(newline="", encoding="utf-8") as f:
            prior = {r["promo_id"]: r for r in csv.DictReader(f)}
    fb_rows = [{"rank": p.rank, "promo_id": p.promo_id, "promo": p.title, "segments": seg_str(p),
                "decision": prior.get(p.promo_id, {}).get("decision") or "pending",
                "notes": prior.get(p.promo_id, {}).get("notes", "")} for p in promos[:30]]
    edited = st.data_editor(fb_rows, hide_index=True, width="stretch", key=f"{k}_fb",
                            disabled=["rank", "promo_id", "promo", "segments"],
                            column_config={"decision": st.column_config.SelectboxColumn(
                                "decision", options=["pending", "approve", "reject", "maybe"], required=True),
                                "notes": st.column_config.TextColumn("notes")})
    if st.button("Save feedback", key=f"{k}_fbsave"):
        with fb_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "promo_id", "promo", "segments", "decision", "notes"])
            w.writeheader()
            w.writerows({**r, "decision": "" if r.get("decision") in (None, "pending") else r["decision"],
                         "notes": r.get("notes") or ""} for r in edited)
        n_done = sum(1 for r in edited if r.get("decision") not in (None, "pending"))
        st.success(f"Saved {n_done} decisions to {fb_path.name}.")

    st.subheader("Details")
    for p in shown[:30]:
        with st.expander(f"#{p.rank} · {p.title} · {seg_str(p)} · {p.duration:.0f}s"):
            a, b = st.columns(2)
            a.markdown(f"**Hook** ({p.hook_type.replace('_', ' ')}, starts {p.hook_offset:.1f}s in): {p.hook}")
            a.markdown(f"**Cliffhanger:** {p.cliffhanger}")
            a.markdown(f"**Why it works:** {p.why_it_works}")
            b.markdown(f"**Scores** H/C/S: {p.scores.hook}/{p.scores.cliffhanger}/{p.scores.standalone} · "
                       f"**spoiler:** {p.spoiler_level}")
            if p.context_needed:
                b.markdown(f"**A cold viewer may miss:** {p.context_needed}")
            if p.judge_reason:
                b.markdown(f"**Cold-viewer judge:** {p.judge_reason}")
            b.markdown(f"**Cut:** {p.cut_style}")
            if p.repairs:
                b.markdown(f"**Validator changed:** {', '.join(p.repairs)}")
            for f in p.flags:
                b.markdown(f"⚠️ {f}")
            st.code("\n".join(p.transcript), language=None)
    if len(shown) > 30:
        st.caption(f"Showing details for the first 30 of {len(shown)}; the downloads have all of them.")

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("promos.csv", (out / "promos.csv").read_bytes(), "promos.csv", "text/csv", key=f"{k}_csv")
    d2.download_button("promos.json", (out / "promos.json").read_bytes(), "promos.json", "application/json", key=f"{k}_json")
    d3.download_button("promos.md", (out / "promos.md").read_bytes(), "promos.md", "text/markdown", key=f"{k}_md")
    d4.download_button("Whole run (.zip)", zip_dir(out), f"{out.name}.zip", "application/zip", key=f"{k}_zip")

    with st.expander(f"Rejected candidates ({len(rejected)})"):
        st.dataframe([{"episode": r["episode"], "title": r["title"], "reason": r["reason"], "detail": r["detail"]}
                      for r in rejected], hide_index=True, width="stretch")
    if brief:
        with st.expander("Generated show brief (premise, characters, aliases)"):
            st.markdown(brief.get("premise", ""))
            st.dataframe([{"name": c["name"], "role": c["role"], "aliases": ", ".join(c["aliases"])}
                          for c in brief.get("main_characters", [])], hide_index=True, width="stretch")


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", ["anthropic", "gemini", "openai_compat"],
                            help="gemini = free tier via Google's OpenAI-compatible endpoint; "
                                 "openai_compat = any OpenAI-compatible API")
    model = st.text_input("Model id", DEFAULT_MODEL[provider])
    effort = st.selectbox("Effort", ["low", "medium", "high"], disabled=provider != "anthropic",
                          help="Anthropic only. Higher = more thinking tokens = more cost.")
    base_url = st.text_input("Base URL", disabled=provider != "openai_compat") or None
    env_key = bool(os.environ.get(KEY_ENV[provider]))
    api_key = st.text_input("API key", type="password",
                            placeholder=("leave empty to use the server key" if HOSTED and env_key else
                                         "using key from server env" if env_key else f"paste {KEY_ENV[provider]}"),
                            help="Used for this session only; never written to disk.") or None
    passcode_ok = True
    if PASSCODE and not api_key:
        entered = st.text_input("Passcode", type="password",
                                help="Needed to run on the server's key. It is in the solution doc; "
                                     "or paste your own API key above instead.")
        passcode_ok = bool(entered) and hmac.compare_digest(entered.encode(), PASSCODE.encode())
    st.header("Run")
    ep_spec = st.text_input("Episodes", "", placeholder="all  (or e.g. 1-5, 10)")
    use_cache = st.toggle("Reuse cached LLM answers", True, help="Re-runs of the same input cost nothing.")
    use_judge = st.toggle("Cold-viewer judge (Claude Opus 5.5)", True,
                          help="A second, stronger model re-ranks the validated promos from the clip text alone "
                               "(about $0.50 per 50-episode show). Uses ANTHROPIC_API_KEY, or the key above "
                               "when the provider is anthropic. Model set in config.yaml (judge section).")
    st.caption("Same pipeline as the CLI: `promo-finder run <folder>`.")


# ------------------------------------------------------------------ main

st.title("🎬 Promo Finder")
st.markdown("Finds 20–60 second promo clips in a microdrama screenplay pack. Each clip opens on a hook, "
            "ends on a cliffhanger and should make sense to a stranger. The LLM proposes moments; "
            "code enforces timestamps, the 3-second hook rule and length, then ranks them.")

tab_run, tab_saved = st.tabs(["Run on a show", "Open a saved run"])

with tab_run:
    files = st.file_uploader("Episode screenplays: the .txt files, or one .zip of them",
                             type=["txt", "zip"], accept_multiple_files=True)
    if files:
        sig = tuple((f.name, f.size) for f in files)
        if st.session_state.get("upload_sig") != sig:
            old = st.session_state.get("upload_dir")
            if old:
                shutil.rmtree(old, ignore_errors=True)
            st.session_state.upload_dir = str(save_uploads(files))
            st.session_state.upload_sig = sig
            st.session_state.pop("last_run", None)
        upload_dir = Path(st.session_state.upload_dir)
        try:
            episodes, anomalies = parse_show(upload_dir)
        except FileNotFoundError:
            st.error("No .txt screenplay files found in the upload.")
            st.stop()

        try:
            wanted = parse_range(ep_spec.strip()) if ep_spec.strip() and ep_spec.strip() != "all" else None
        except ValueError:
            st.error("Episodes must look like `1-5` or `1,3,10-12`.")
            st.stop()
        n_mine = len([e for e in episodes if wanted is None or e.number in wanted])
        price = PRICES.get(model)
        est = (n_mine * (TOKENS_PER_EP[0] * price[0] + TOKENS_PER_EP[1] * price[1]) / 1e6) if price else None

        st.success(f"Parsed **{len(episodes)} episodes**, {sum(len(e.events) for e in episodes):,} timed events, "
                   f"{sum(e.duration for e in episodes) / 60:.1f} min of show. "
                   f"Will mine **{n_mine}** episode(s)"
                   + (f", about **${est:.2f}** if nothing is cached." if est is not None else "."))
        if anomalies:
            with st.expander(f"⚠️ {len(anomalies)} parse anomalies"):
                st.dataframe([a.model_dump() for a in anomalies], hide_index=True, width="stretch")

        if st.button("Find promos", type="primary", disabled=n_mine == 0):
            if not (api_key or env_key):
                st.error(f"Add an API key in the sidebar (or set {KEY_ENV[provider]}).")
                st.stop()
            if not passcode_ok:
                st.error("Enter the passcode in the sidebar to run on the server's key, or paste your own API key.")
                st.stop()
            if HOSTED and api_key and provider != "anthropic":
                use_judge = False  # the judge would otherwise fall back to the server's Anthropic key
            settings = load_settings(ROOT / "config.yaml", judge=use_judge, provider=provider, model=model,
                                     effort=effort if provider == "anthropic" else None,
                                     base_url=base_url, api_key=api_key, use_cache=use_cache)
            if HOSTED:
                if "private_runs" not in st.session_state:
                    st.session_state.private_runs = tempfile.mkdtemp(prefix="promo_runs_")
                out = Path(st.session_state.private_runs) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
            else:
                out = RUNS / "app_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
            bar = st.progress(0.0, text="Building show brief...")

            def progress(i: int, n: int, ep: int) -> None:
                bar.progress(i / n, text=f"Mining episode {ep} ({i}/{n})")

            try:
                run_pipeline(upload_dir, settings, out, episodes=wanted, title="uploaded show", on_progress=progress)
            except LLMError as exc:
                bar.empty()
                st.error(f"LLM call failed: {exc}")
                st.stop()
            bar.empty()
            st.session_state.last_run = str(out)

        if st.session_state.get("last_run"):
            st.divider()
            show_results(Path(st.session_state.last_run), "run")
    else:
        st.info("Upload a show to start. Nothing is bundled with the app: screenplays stay private to your session.")

with tab_saved:
    def run_info(d: Path) -> dict:
        try:
            return json.loads((d / "run_meta.json").read_text())
        except (OSError, ValueError):
            return {}

    found = [p.parent for p in RUNS.rglob("promos.json") if (p.parent / "run_meta.json").exists()] if RUNS.exists() else []
    infos = {d: run_info(d) for d in found}
    # Full-show runs first (most episodes), newest first within that; archived runs last.
    runs = sorted(found, key=lambda d: ("archive" in d.parts, -len(infos[d].get("episodes_mined", [])),
                                        -(d / "promos.json").stat().st_mtime))

    def run_label(d: Path) -> str:
        m = infos[d]
        return (f"{d.relative_to(RUNS)}  ·  {m.get('model', '?')}  ·  "
                f"{len(m.get('episodes_mined', []))} ep  ·  {m.get('promos', '?')} promos")
    if not runs:
        st.info("No saved runs here yet. Runs from the CLI or this app appear in `outputs/`.")
    else:
        pick = st.selectbox("Run folder", runs, format_func=run_label)
        show_results(pick, "saved")
