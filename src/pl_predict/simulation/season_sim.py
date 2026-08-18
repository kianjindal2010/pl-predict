"""Season simulation engine: Monte Carlo simulation of a season remainder.

Two fixture sources:
- ``parquet``: unplayed rows from ``matches.parquet`` plus the points already
  earned in the selected season.
- ``fixtures_2627``: the real 2026/27 fixture list (uses the trained
  ensemble model for match probabilities).
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

from pl_predict.pipeline.utils import resolve_path


class SeasonSimulator:
    """Monte Carlo season simulation."""

    def __init__(
        self,
        season: str = "2025-26",
        source: str = "auto",
        model_path: Optional[str] = None,
        seed: Optional[int] = 42,
    ):
        self.season = season
        self.source = source
        self.model_path = resolve_path(
            model_path or "data/processed/ensemble_model.pkl"
        )
        self.seed = seed
        if self.source == "auto":
            self.source = "fixtures_2627" if season == "2026-27" else "parquet"

    def _load_fixtures_2627(self):
        """Load 2026/27 fixtures and predict each match with the ensemble."""
        from pl_predict.data.fixtures_2627 import get_fixtures, normalize_team
        from pl_predict.models.ensemble import EnsemblePredictor

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `predict-pl train` first."
            )

        predictor = EnsemblePredictor.load(str(model_path))
        home, away, hp, dp, ap = [], [], [], [], []
        for f in get_fixtures():
            p = predictor.predict(
                normalize_team(f["home_team"]), normalize_team(f["away_team"])
            )
            if p and "error" not in p:
                home.append(f["home_team"])
                away.append(f["away_team"])
                hp.append(p["home_win"])
                dp.append(p["draw"])
                ap.append(p["away_win"])
        return home, away, np.array(hp), np.array(dp), np.array(ap), {}

    def _load_fixtures_parquet(self):
        """Load unplayed fixtures and the current table from matches.parquet."""
        path = Path(resolve_path("data/processed/matches.parquet"))
        if not path.exists():
            raise FileNotFoundError("No match data found. Run the pipeline first.")
        df = pl.read_parquet(str(path)).filter(pl.col("season") == self.season)
        if df.is_empty():
            raise ValueError(f"No fixtures found for season {self.season}.")

        played = df.filter(
            pl.col("home_goals").is_not_null() & pl.col("away_goals").is_not_null()
        )
        remaining = df.filter(
            pl.col("home_goals").is_null() | pl.col("away_goals").is_null()
        )
        if remaining.is_empty():
            raise ValueError(
                f"No unplayed fixtures found for {self.season}; use a future fixture source."
            )

        points: dict[str, float] = {}
        for row in played.iter_rows(named=True):
            home, away = row["home_team"], row["away_team"]
            home_goals, away_goals = row["home_goals"], row["away_goals"]
            points.setdefault(home, 0.0)
            points.setdefault(away, 0.0)
            if home_goals > away_goals:
                points[home] += 3.0
            elif home_goals < away_goals:
                points[away] += 3.0
            else:
                points[home] += 1.0
                points[away] += 1.0

        if "implied_home_norm" in remaining.columns:
            home_probs = remaining["implied_home_norm"].to_numpy().astype(float)
            draw_probs = remaining["implied_draw_norm"].to_numpy().astype(float)
            away_probs = remaining["implied_away_norm"].to_numpy().astype(float)
        else:
            home_probs = np.full(len(remaining), 0.45)
            draw_probs = np.full(len(remaining), 0.25)
            away_probs = np.full(len(remaining), 0.30)

        probs = np.column_stack([home_probs, draw_probs, away_probs])
        valid = np.isfinite(probs).all(axis=1) & (probs.sum(axis=1) > 0)
        probs[~valid] = np.array([0.45, 0.25, 0.30])
        probs /= probs.sum(axis=1, keepdims=True)

        return (
            remaining["home_team"].to_list(),
            remaining["away_team"].to_list(),
            probs[:, 0],
            probs[:, 1],
            probs[:, 2],
            points,
        )

    def _load_fixtures(self):
        if self.source == "fixtures_2627":
            return self._load_fixtures_2627()
        return self._load_fixtures_parquet()

    def run(self, n_simulations: int = 5000) -> pl.DataFrame:
        """Run Monte Carlo season simulation (vectorized)."""
        print(
            f"Running {n_simulations} simulations for {self.season} [{self.source}] ..."
        )

        home, away, home_probs, draw_probs, away_probs, initial_points = (
            self._load_fixtures()
        )
        n_fixtures = len(home)
        teams = sorted(set(home) | set(away) | set(initial_points))
        team_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        # Vectorized result sampling for all fixtures and simulations
        rng = np.random.default_rng(self.seed)
        r = rng.random((n_simulations, n_fixtures))
        cum_draw = home_probs[None, :] + draw_probs[None, :]
        res = np.full_like(r, 2, dtype=np.int8)
        res[r < home_probs[None, :]] = 0
        res[(r >= home_probs[None, :]) & (r < cum_draw)] = 1

        # Accumulate points: rows = teams, cols = simulations
        pts = np.repeat(
            np.array([initial_points.get(team, 0.0) for team in teams])[:, None],
            n_simulations,
            axis=1,
        )
        for f in range(n_fixtures):
            h, a = team_idx[home[f]], team_idx[away[f]]
            r_team = res[:, f]
            pts[h][r_team == 0] += 3
            pts[h][r_team == 1] += 1
            pts[a][r_team == 1] += 1
            pts[a][r_team == 2] += 3

        # Split title probability among tied teams, so total title probability
        # remains exactly 100%. Random tie breaks make top-four/relegation
        # probabilities unbiased when points alone cannot separate teams.
        max_pts = pts.max(axis=0, keepdims=True)
        title_mask = pts == max_pts
        title_wins = (title_mask / title_mask.sum(axis=0, keepdims=True)).sum(axis=1)

        tie_break = rng.random(pts.shape) * 1e-6
        ranks = np.argsort(np.argsort(-(pts + tie_break), axis=0), axis=0)
        top4_count = ((ranks < 4).sum(axis=1)).astype(float)
        releg_count = ((ranks >= n_teams - 3).sum(axis=1)).astype(float)

        rows = []
        for i, team in enumerate(teams):
            p = pts[i]
            rows.append(
                {
                    "team": team,
                    "mean_pts": float(p.mean()),
                    "std_pts": float(p.std()),
                    "median_pts": float(np.median(p)),
                    "p10_pts": float(np.percentile(p, 10)),
                    "p90_pts": float(np.percentile(p, 90)),
                    "title_pct": float(title_wins[i] / n_simulations * 100),
                    "top4_pct": float(top4_count[i] / n_simulations * 100),
                    "relegation_pct": float(releg_count[i] / n_simulations * 100),
                }
            )

        df = pl.DataFrame(rows).sort("mean_pts", descending=True)
        out_path = Path(resolve_path("data/processed/season_sim.parquet"))
        df.write_parquet(str(out_path))
        print(
            f"\nSeason simulation ({n_fixtures} fixtures, {n_teams} teams) -> {out_path}"
        )
        return df
