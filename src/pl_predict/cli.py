"""CLI entry point for pl-predict."""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="predict-pl", help="Premier League match prediction engine")
console = Console()


@app.command()
def scrape(
    seasons: list[str] = typer.Option(
        None, "--seasons", "-s", help="Seasons to scrape (e.g. 2024-25)"
    ),
    skip_scrape: bool = typer.Option(False, "--skip-scrape", help="Use cached data"),
):
    """Run the data pipeline: scrape -> clean -> merge."""
    from pl_predict.pipeline import run_pipeline

    run_pipeline(seasons=seasons or None, skip_scrape=skip_scrape)


@app.command()
def predict(
    home: str = typer.Argument(..., help="Home team name"),
    away: str = typer.Argument(..., help="Away team name"),
    date: str = typer.Option(None, "--date", "-d", help="Match date (YYYY-MM-DD)"),
):
    """Predict outcome for a single match."""
    from pl_predict.models.ensemble import EnsemblePredictor
    from pl_predict.pipeline.utils import resolve_path

    console.print(f"Predicting [bold]{home}[/bold] vs [bold]{away}[/bold] ...")

    model_path = Path(resolve_path("data/processed/ensemble_model.pkl"))
    if not model_path.exists():
        console.print(
            "[red]No trained model found. Run `predict-pl train` first.[/red]"
        )
        raise typer.Exit(1)

    predictor = EnsemblePredictor.load(str(model_path))
    result = predictor.predict(home, away, date_str=date)
    console.print(result)


@app.command()
def train(
    seasons: list[str] = typer.Option(
        None, "--seasons", "-s", help="Seasons to train on"
    ),
    model_type: str = typer.Option(
        "ensemble", "--model", "-m", help="Deployable model type (ensemble only)"
    ),
):
    """Train prediction models on processed data."""
    if model_type != "ensemble":
        raise typer.BadParameter("Only the deployable ensemble is supported.")
    console.print(f"Training {model_type} model ...")

    from pl_predict.models.train import train_model

    train_model(model_type=model_type, seasons=seasons)
    console.print("[green]Training complete.[/green]")


@app.command()
def evaluate(
    seasons: list[str] = typer.Option(
        None, "--seasons", "-s", help="Seasons to evaluate on"
    ),
    ensemble: bool = typer.Option(
        False,
        "--ensemble",
        help="Evaluate the full ensemble on the latest chronological holdout",
    ),
):
    """Evaluate the baseline or the full ensemble."""
    if ensemble:
        from pl_predict.evaluation.backtest import run_ensemble_backtest

        run_ensemble_backtest(test_seasons=seasons)
    else:
        from pl_predict.evaluation.backtest import run_backtest

        run_backtest(seasons=seasons)


@app.command()
def season(
    n_simulations: int = typer.Option(
        10000, "--simulations", "-n", help="Number of simulations"
    ),
    season: str = typer.Option("2025-26", "--season", "-s", help="Season to simulate"),
    source: str = typer.Option(
        "auto",
        "--source",
        help="Fixture/probability source: auto | parquet | fixtures_2627 (auto = fixtures_2627 for 2026-27)",
    ),
):
    """Simulate the remaining season."""
    from pl_predict.simulation.season_sim import SeasonSimulator

    sim = SeasonSimulator(season=season, source=source)
    results = sim.run(n_simulations=n_simulations)
    table = Table(title=f"{season} Season Simulation")
    table.add_column("Team", style="cyan")
    table.add_column("Mean Pts", style="green")
    table.add_column("Title %", style="yellow")
    for row in results.to_dicts():
        table.add_row(row["team"], f"{row['mean_pts']:.1f}", f"{row['title_pct']:.1f}%")
    console.print(table)


@app.command("refresh-fixtures")
def refresh_fixtures():
    """Regenerate the 2026/27 fixture-prediction cache from the saved model."""
    from pl_predict.data.fixtures_2627 import get_fixtures, normalize_team
    from pl_predict.models.ensemble import EnsemblePredictor
    from pl_predict.pipeline.utils import resolve_path

    model_path = Path(resolve_path("data/processed/ensemble_model.pkl"))
    if not model_path.exists():
        raise typer.BadParameter(
            "No trained model found. Run `predict-pl train` first."
        )

    predictor = EnsemblePredictor.load(str(model_path))
    output = {"season": "2026-27", "fixtures": []}
    for fixture in get_fixtures():
        prediction = predictor.predict(
            normalize_team(fixture["home_team"]),
            normalize_team(fixture["away_team"]),
            date_str=fixture["date"],
        )
        if prediction and "error" not in prediction:
            output["fixtures"].append({**fixture, "prediction": prediction})

    output_path = Path(resolve_path("data/processed/fixtures_2627_pred.json"))
    output_path.write_text(json.dumps(output), encoding="utf-8")
    console.print(
        f"[green]Wrote {len(output['fixtures'])} fixture predictions to {output_path}[/green]"
    )


@app.command()
def dashboard(
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
):
    """Refresh FPL data, calculate Hybrid xP, and launch the dashboard."""
    launch_dashboard(open_browser=open_browser)


@app.command("refresh-fpl")
def refresh_fpl():
    """Refresh the official FPL snapshot and Hybrid xP cache without serving."""
    from pl_predict.dashboard.app import prime_fpl_picks

    data = prime_fpl_picks()
    console.print(
        "[green]Refreshed official FPL data and Hybrid xP for "
        f"Gameweek {data['gameweek']}.[/green]"
    )


def launch_dashboard(open_browser: bool = True):
    """Default local-app path: fresh FPL data, hybrid projections, dashboard."""
    from pl_predict.dashboard.app import prime_fpl_picks, run_dashboard

    console.print("[cyan]Refreshing official FPL data and Hybrid xP...[/cyan]")
    try:
        data = prime_fpl_picks()
        console.print(
            f"[green]Gameweek {data['gameweek']} is ready "
            f"(refreshed {data.get('refreshed_at', 'just now')}).[/green]"
        )
    except Exception as exc:
        console.print(
            f"[yellow]Could not refresh FPL data; starting with local data: {exc}[/yellow]"
        )
    run_dashboard(open_browser=open_browser)


def main():
    if len(sys.argv) == 1:
        launch_dashboard()
    else:
        app()


if __name__ == "__main__":
    main()
