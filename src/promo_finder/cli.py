"""Command line entry point: `promo-finder run <screenplay_dir>`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .baseline import baseline_promos
from .export import seg_str, write_promos
from .llm import LLMError
from .parser import parse_show
from .pipeline import load_settings, run_pipeline

app = typer.Typer(add_completion=False, help="Find ranked promo clips in a microdrama screenplay pack.")
console = Console()

def parse_range(spec: str | None) -> set[int] | None:
    """'1-5,8' -> {1,2,3,4,5,8}"""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


@app.command()
def run(
    screenplays: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of episode .txt files"),
    out: Optional[Path] = typer.Option(None, help="Output folder (default outputs/run_<timestamp>)"),
    episodes: Optional[str] = typer.Option(None, help="Subset to mine, e.g. '1-5' or '1,3,10-12'"),
    provider: Optional[str] = typer.Option(None, help="anthropic | gemini | openai_compat"),
    model: Optional[str] = typer.Option(None, help="Model id, e.g. claude-sonnet-5-5"),
    effort: Optional[str] = typer.Option(None, help="Anthropic effort: low | medium | high | xhigh | max"),
    base_url: Optional[str] = typer.Option(None, help="OpenAI-compatible base URL"),
    config: Path = typer.Option(Path("config.yaml"), help="YAML config"),
    no_cache: bool = typer.Option(False, help="Ignore cached LLM responses"),
    judge: Optional[bool] = typer.Option(None, "--judge/--no-judge", help="Cold-viewer re-ranking (default from config)"),
    judge_provider: Optional[str] = typer.Option(None, help="Judge provider, e.g. gemini"),
    judge_model: Optional[str] = typer.Option(None, help="Judge model id"),
):
    """Parse, mine with an LLM, validate, rank and export promos."""
    load_dotenv()
    settings = load_settings(config, judge=judge, judge_provider=judge_provider, judge_model=judge_model,
                             provider=provider, model=model, effort=effort, base_url=base_url,
                             use_cache=False if no_cache else None)
    out = out or Path("outputs") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    c = settings.llm
    console.print(f"Mining with [bold]{c.provider}/{c.model}[/bold] (effort={c.effort}) -> {out}")
    try:
        res = run_pipeline(screenplays, settings, out, episodes=parse_range(episodes))
    except LLMError as exc:
        console.print(f"[red]Show brief failed:[/red] {exc}")
        raise typer.Exit(1)
    ranked, rejected, errors, meta = res.promos, res.rejected, res.errors, res.meta
    u = meta["usage"]
    console.print(f"Parsed {len(res.episodes)} episodes, {len(res.anomalies)} anomalies.")

    table = Table(title=f"Top promos ({len(ranked)} total, {len(rejected)} rejected)")
    for col in ("#", "Promo", "Segments", "Len", "H/C/S", "Score"):
        table.add_column(col)
    for p in ranked[:10]:
        table.add_row(str(p.rank), p.title[:40], seg_str(p), f"{p.duration:.0f}s",
                      f"{p.scores.hook}/{p.scores.cliffhanger}/{p.scores.standalone}", f"{p.score:.2f}")
    console.print(table)
    if errors:
        console.print(f"[yellow]{len(errors)} episode(s) failed:[/yellow] {sorted(errors)}")
    console.print(f"Tokens in/out: {u['input_tokens']}/{u['output_tokens']}, cache hits {u['cache_hits']}/{u['calls']}, "
                  f"est. cost ${meta['estimated_cost_usd']} (uncached run: ${meta['estimated_cost_uncached_usd']}). Outputs: {out}")


@app.command()
def baseline(
    screenplays: Path = typer.Argument(..., exists=True, file_okay=False),
    out: Path = typer.Option(Path("outputs/baseline"), help="Output folder"),
    length: float = typer.Option(45.0, help="Seconds taken from the end of each episode"),
):
    """No-LLM baseline: the last N seconds of every episode."""
    eps, _ = parse_show(screenplays)
    promos = baseline_promos(eps, length)
    write_promos(out, promos, [], [], title=f"{screenplays.name} (baseline: last {length:.0f}s)", top_n=len(promos))
    console.print(f"Wrote {len(promos)} baseline clips to {out}")


@app.command("eval")
def eval_cmd(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="A run output folder"),
    gold: Path = typer.Option(Path("eval/gold.yaml"), exists=True),
    baseline_dir: Optional[Path] = typer.Option(Path("outputs/baseline"), "--baseline"),
    ratings: Path = typer.Option(Path("eval/human_ratings.csv"), help="Filled-in human ratings (template created if missing)"),
):
    """Recall vs gold set, baseline comparison, validator stats, human ratings."""
    from .evaluate import evaluate

    report = evaluate(run_dir, gold, baseline_dir, ratings)
    (run_dir / "eval_report.md").write_text(report, encoding="utf-8")
    console.print(report)
    console.print(f"\nWrote {run_dir / 'eval_report.md'}")


@app.command()
def parse(screenplays: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Parse only, and report anomalies (quick format check for a new show)."""
    eps, anomalies = parse_show(screenplays)
    console.print(f"{len(eps)} episodes, {sum(len(e.events) for e in eps)} events, "
                  f"{sum(e.duration for e in eps) / 60:.1f} min total, {len(anomalies)} anomalies")
    for a in anomalies[:30]:
        console.print(f"  {a.file}:{a.line} {a.kind}: {a.detail}")


def main() -> None:
    app()
