# PL Predict

PL Predict is a comprehensive, local Premier League prediction and analytics platform built with free/open-source data. It covers fixture forecasts, expected goals, likely scorelines, season simulations, model evaluation, team and player analysis, shots, referees, and an optional FPL Picks view with Hybrid xP.

## Windows install

In PowerShell, run:

```powershell
irm https://raw.githubusercontent.com/kianjindal2010/pl-predict/main/scripts/install.ps1 | iex
```

Open a **new** PowerShell or Command Prompt window and run:

```powershell
pl-predict
```

That opens the complete local Premier League dashboard at `http://127.0.0.1:8000`. It also refreshes the official FPL player, availability, event and fixture feeds for the optional FPL Picks view, selects the nearest upcoming gameweek (or the earliest unfinished one), and recalculates Hybrid xP. The installer uses Winget to install Git and Python 3.12 if necessary, installs under `%LOCALAPPDATA%\PLPredict`, and never deletes an existing non-repository directory or local runtime snapshots.

Run the same installer command later to safely fetch app updates. Launches fetch current FPL data but do not silently retrain the historical match model.

The static install page is deployed separately on Vercel; the Python dashboard is intentionally local.

## Quick Start

```bash
# Install from a source checkout
pip install -e .

# Run the data pipeline (scrape -> clean -> merge)
predict-pl scrape            # or: python -m pl_predict.cli scrape

# Train models
predict-pl train             # or: python -m pl_predict.cli train

# Predict a match
predict-pl predict "Liverpool" "Manchester City"

# Run a Monte Carlo season simulation
predict-pl season --simulations 5000 --season 2025-26

# Refresh FPL data and launch the dashboard (opens your browser)
pl-predict

# Explicit commands remain available
pl-predict dashboard
pl-predict refresh-fpl
```

## Architecture

See [docs/plan.md](docs/plan.md) for the full architecture and methodology.

## Dashboard

The web dashboard (`pl-predict`) provides nine tabs:

- **Overview** — 2026/27 fixture table with probabilities, most-likely score, O/U 2.5 and conformal-set badges
- **Predict** — H/D/A probabilities, expected goals, score-matrix heatmap, 6-model breakdown, H2H history
- **Season Sim** — Monte Carlo projected final table with P10-P90 bands, title/top-4/relegation probabilities
- **Model** — walk-forward accuracy/RPS, calibration curve, worst-predictions diagnostics
- **Players / FPL Picks** — official FPL xP alongside the PL Predict model contribution and 55/45 Hybrid xP, plus historical player leaderboards
- **Shot Maps** — xG shot locations on a pitch, filtered by season/team/player
- **Teams** — form trend, home/away splits, goals histogram, attendance
- **Referees** — per-referee goals & home-win-bias analytics
- **Features** — XGBoost feature importance

API endpoints: `/api/health`, `/api/matches`, `/api/upcoming`, `/api/predict`, `/api/season`,
`/api/features`, `/api/teams`, `/api/seasons`, `/api/performance`, `/api/players`,
`/api/players/{id}`, `/api/shots`, `/api/team/{name}`, `/api/h2h`, `/api/referees`,
`/api/fixtures_2627`, `/api/fpl/picks`.

`/api/fpl/picks` returns the selected gameweek, the official-data refresh timestamp, official FPL xP, the model contribution, and their Hybrid xP separately. Runtime official snapshots and recalculated projection caches remain local and are not committed.

## Project Structure

```
pl-predict/
├── config/            # YAML config (leagues, paths)
├── data/
│   ├── raw/           # scraped sources (football_data, transfermarkt, understat)
│   ├── processed/     # cleaned features, trained models, prediction artifacts
│   └── external/      # (reserved for external datasets)
├── docs/              # plan.md (methodology), PROGRESS.md (session log)
├── logs/              # training / scrape logs
├── notebooks/         # (reserved for exploration)
├── paper/             # (reserved for write-up)
├── scripts/           # one-off runners & diagnostics (analyze_worst_predictions.py, run_*.py)
├── src/pl_predict/    # package: pipeline, features, models, evaluation, simulation, dashboard
└── tests/             # (reserved for test suite)
```

## Data Sources

- FBref (match/player stats)
- Understat (expected goals)
- Football-Data.co.uk (results + betting odds, 1993–present)
- Club Elo ratings
- Open-Meteo (weather)
- SoFIFA (player ratings)

## License

MIT
