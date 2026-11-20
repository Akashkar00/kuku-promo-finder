# promo-finder

Reads a microdrama screenplay pack (one timestamped `.txt` file per episode) and returns a ranked shortlist of 20–60s promo clips. Each clip comes with exact segment timestamps, its hook, its cliffhanger, one line on why it works, and a "post first because" note.

Built for the Kuku AI Generalist assignment (Problem 1). The input pack is confidential and **is not in this repo**.

## Quickstart

```bash
uv sync
cp .env.example .env            # add ANTHROPIC_API_KEY
uv run promo-finder run /path/to/show_folder
```

The default is Claude Sonnet 5.5 for mining and Claude Opus 5.5 as the ranking judge. A 50-episode show costs about $0.90 to mine and about $0.50 to judge, in roughly 10 minutes. If the judge can't run, ranking falls back to the miner's own scores, and `run_meta.json` records which happened. For a cheaper or free run, swap the model (below).

Outputs go to `outputs/run_<timestamp>/`:

| File | What |
|---|---|
| `promos.md` | Editor view: the "post these first" list (ranks 1–5 = posting order), then the top 20 with "check" flags |
| `promos.csv` / `promos.json` | Every validated promo, ranked |
| `rejected.json` | Every LLM candidate the validator refused, with the reason |
| `duplicates.json` | Near-identical promos that were merged |
| `show_brief.json` | Generated premise, characters and aliases |
| `raw/mine_EXX.json` | The unedited LLM response for each episode |
| `run_meta.json` | Model, token usage, cost, validator pass rate, timings, judge agreement between passes |
| `feedback.csv` | Written by the web app: the editor's approve/reject/notes for each promo |

## Web app

```bash
uv run streamlit run app.py
```

The app works like this:
- **Run on a show:** upload the episode `.txt` files (or one `.zip` of them), pick a provider and model, and click *Find promos*.
- **Results:** the "post these first" list, a filterable table, transcripts for each promo, rejected candidates, the show brief, and CSV/JSON/MD/zip downloads.
- **Editor review:** mark promos approve/reject/maybe and add notes. This is saved as `feedback.csv` in the run folder.
- **Open a saved run:** browse any run folder under `outputs/`, including CLI runs.

Nothing is bundled with the app. It only sees what you upload. It runs the same pipeline as the CLI (`promo_finder/pipeline.py`).

**Deploying (Streamlit Community Cloud):**
- Point it at this repo and `app.py` (Python 3.12). Dependencies come from `requirements.txt`.
- Add these app secrets:
  ```toml
  ANTHROPIC_API_KEY = "sk-ant-..."
  APP_PASSCODE = "choose-a-passcode"
  ```
- With `APP_PASSCODE` set, the app runs in hosted mode:
  - The server's key is used only after a visitor enters the passcode. A visitor who pastes their own key needs no passcode.
  - Each visitor's runs go to a private temporary folder, so nobody sees another visitor's show.
  - Set a monthly spend limit on the key in the Anthropic console.

## Swap the model

```bash
uv run promo-finder run DIR --model claude-haiku-5-5                    # cheap
uv run promo-finder run DIR --model claude-opus-5-5 --effort medium     # stronger
uv run promo-finder run DIR --provider gemini --model gemini-flash-latest   # Gemini (has a free tier)
uv run promo-finder run DIR --provider openai_compat --base-url https://... --model <id>   # any OpenAI-compatible API
uv run promo-finder run DIR --judge-provider openai_compat --judge-model <id>               # swap the judge too
uv run promo-finder run DIR --no-judge                                                      # self-scores only
```

For an OpenAI-compatible API, set `OPENAI_COMPAT_API_KEY` (and `OPENAI_COMPAT_BASE_URL`, or pass `--base-url`).

Defaults live in `config.yaml`. LLM responses are cached in `.cache/` by prompt hash, so a re-run is free. Pass `--no-cache` to force fresh calls.

## Other commands

```bash
uv run promo-finder run DIR --episodes 1-5      # quick partial run
uv run promo-finder parse DIR                   # format check for a new show
uv run promo-finder baseline DIR                # no-LLM baseline: last 45s of each episode
uv run promo-finder eval outputs/run_x          # recall vs eval/gold.yaml, baseline, validator stats
uv run pytest                                   # PROMO_PACK_DIR=/path/to/pack also runs the pack tests
```

## How it works

```
parse (code) -> show brief (1 LLM call) -> condense (code) -> mine (1 LLM call per episode)
             -> assemble + validate (code) -> group alt cuts (code)
             -> cold-viewer judge (2 shuffled listwise passes, a stronger model) -> rank (code) -> export
```

- **The LLM picks event IDs, and code turns them into timestamps.** The model never writes a timestamp, so it can't make one up.
- **Hard rules live in `assemble.py`:**
  - cuts snap to shot boundaries and never land mid-line
  - the hook starts within 3s of the clip start
  - total length is 20–60s, in at most 3 segments
  - fixable problems are repaired and logged, everything else is rejected with a reason
- **The judge (`judge.py`) re-ranks without seeing the miner's view.**
  - It is a second, stronger model (Opus 5.5 judging Sonnet's picks) that sees only each clip's lines: no show brief, no episode numbers, no scores.
  - It orders groups of about 30 clips, which forces it to discriminate between them.
  - Two shuffled passes reduce position bias, and how well they agree is reported.
  - Why it exists: the miner scores clips one episode at a time, so its 1–5 scores bunch together.
- **Ranking (`rank.py`) is plain code** with weights in `config.yaml`:
  - the miner's hook, cliffhanger and standalone scores, blended with the judge's score
  - a small bonus for early episodes
  - penalties for spoilers, jump cuts and clips stretched past their cliffhanger
  - cuts of the same scene are grouped, keeping the best and listing the others as alt cuts
  - at most 2 promos per episode in the top 20
  - ranks 1–5 are the posting order: the best promos with distinct episodes and hook types
- **Nothing is show-specific.** Character aliases are generated from each pack's own loglines and speaker labels.
- **Limits:**
  - **Single-episode promos only.** Cross-episode montages aren't built.
  - **Text only.** Visual hooks are inferred from the action lines.
  - **The ranking is a shortlist for an editor, not an autopilot.**
