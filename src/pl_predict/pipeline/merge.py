"""Merge cleaned data from all sources into a single unified match table."""

from pathlib import Path
from typing import Optional

import polars as pl

from .utils import load_config, resolve_path, team_name_normalise


def _implied_prob(odds: pl.Series) -> pl.Series:
    """Convert decimal odds to normalised implied probabilities."""
    inv = 1.0 / odds
    return inv / inv.sum()


def merge_matches(
    cleaned: dict[str, pl.DataFrame],
    out_dir: Optional[str] = None,
) -> pl.DataFrame:
    """Merge cleaned data into a single match table.

    Priority for overlapping columns: football_data > fbref > understat.
    """
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["processed_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frames = []

    # --- Football-Data (primary for results + odds) ---
    fd = cleaned.get("football_data")
    if fd is not None and not fd.is_empty():
        fd = fd.drop("source") if "source" in fd.columns else fd
        frames.append(fd)

    # --- FBref (adds xG, possession) ---
    fb = cleaned.get("fbref")
    if fb is not None and not fb.is_empty():
        drop_cols = [
            c
            for c in ("home_goals", "away_goals", "season", "source")
            if c in fb.columns
        ]
        fb = fb.drop(drop_cols)
        frames.append(fb)

    if not frames:
        print("  WARN: no data to merge")
        return pl.DataFrame()

    # Vertical concatenation
    combined = pl.concat(frames, how="diagonal")

    # Sort by date
    if "date" in combined.columns:
        combined = combined.sort("date")

    # Deduplicate: for same date+teams, keep first row (football_data priority).
    # Polars doesn't have groupby first directly across all cols, so we use unique.
    key_cols = ["date", "home_team", "away_team"]
    if all(c in combined.columns for c in key_cols):
        # Tag football_data rows with priority 1 so they stay
        combined = combined.with_columns(
            pl.when(pl.col("odds_home").is_not_null())
            .then(1)
            .otherwise(0)
            .alias("_priority")
        )
        combined = combined.sort(
            key_cols + ["_priority"], descending=[False, False, False, True]
        )
        combined = combined.unique(subset=key_cols, keep="first")
        combined = combined.drop("_priority")

    # --- Derive target variables ---
    if all(c in combined.columns for c in ("home_goals", "away_goals")):
        combined = combined.with_columns(
            pl.when(pl.col("home_goals") > pl.col("away_goals"))
            .then(pl.lit("H"))
            .when(pl.col("home_goals") < pl.col("away_goals"))
            .then(pl.lit("A"))
            .otherwise(pl.lit("D"))
            .alias("result"),
            (pl.col("home_goals") + pl.col("away_goals")).alias("total_goals"),
            (pl.col("home_goals") - pl.col("away_goals")).alias("goal_diff"),
            (pl.col("home_goals") + pl.col("away_goals") > 2.5)
            .cast(pl.Int8)
            .alias("over_2_5"),
            ((pl.col("home_goals") > 0) & (pl.col("away_goals") > 0))
            .cast(pl.Int8)
            .alias("btts"),
        )

    # --- Attach Understat xG + forecast (left join, keep football-data rows) ---
    us_path = (
        Path(resolve_path(paths["raw_dir"])) / "understat" / "understat_matches.parquet"
    )
    if us_path.exists():
        us = pl.read_parquet(str(us_path))
        if not us.is_empty():
            us = us.with_columns(
                pl.col("date")
                .str.slice(0, 10)
                .str.to_datetime(strict=False)
                .alias("date"),
                pl.col("home_team").map_elements(
                    team_name_normalise, return_dtype=pl.String
                ),
                pl.col("away_team").map_elements(
                    team_name_normalise, return_dtype=pl.String
                ),
            )
            us_cols = [
                c
                for c in (
                    "xG_home",
                    "xG_away",
                    "us_forecast_home",
                    "us_forecast_draw",
                    "us_forecast_away",
                )
                if c in us.columns
            ]
            us = us.select(["date", "home_team", "away_team"] + us_cols)
            for c in us_cols:
                us = us.with_columns(pl.col(c).cast(pl.Float64, strict=False))
            combined = combined.join(
                us, on=["date", "home_team", "away_team"], how="left"
            )
            print(
                f"  Understat xG/forecast attached to {combined['xG_home'].is_not_null().sum()} matches"
            )

    # --- Implied probabilities from odds ---
    for prefix in ["", "_close", "_max", "_avg"]:
        hc = f"odds{prefix}_home"
        dc = f"odds{prefix}_draw"
        ac = f"odds{prefix}_away"
        if all(c in combined.columns for c in (hc, dc, ac)):
            combined = combined.with_columns(
                (1.0 / pl.col(hc)).alias(f"implied_home{prefix}"),
                (1.0 / pl.col(dc)).alias(f"implied_draw{prefix}"),
                (1.0 / pl.col(ac)).alias(f"implied_away{prefix}"),
            )
            total = (
                pl.col(f"implied_home{prefix}")
                + pl.col(f"implied_draw{prefix}")
                + pl.col(f"implied_away{prefix}")
            )
            combined = combined.with_columns(
                (pl.col(f"implied_home{prefix}") / total).alias(
                    f"implied_home{prefix}_norm"
                ),
                (pl.col(f"implied_draw{prefix}") / total).alias(
                    f"implied_draw{prefix}_norm"
                ),
                (pl.col(f"implied_away{prefix}") / total).alias(
                    f"implied_away{prefix}_norm"
                ),
            )

    # Save
    out_path = Path(out_dir) / "matches.parquet"
    combined.write_parquet(str(out_path))
    print(
        f"  Merged matches -> {out_path}  ({len(combined)} rows, {len(combined.columns)} cols)"
    )

    return combined
