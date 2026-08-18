"""Data cleaning: standardise schemas, normalise teams, handle missing values."""

from pathlib import Path
from typing import Optional

import polars as pl

from .utils import (
    load_config,
    team_name_normalise,
    parse_season_from_date,
    resolve_path,
)


def _normalise_team_col(series: pl.Series) -> pl.Series:
    return series.map_elements(team_name_normalise, return_dtype=pl.String)


def clean_fbref(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    keep = {
        "home_team": pl.String,
        "away_team": pl.String,
        "home_goals": pl.Float64,
        "away_goals": pl.Float64,
        "xG_home": pl.Float64,
        "xG_away": pl.Float64,
        "possession_home": pl.Float64,
        "possession_away": pl.Float64,
        "season": pl.String,
        "date": pl.String,
    }
    existing = {c for c in keep if c in df.columns}
    df = df.select(list(existing))

    for col, dtype in keep.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))

    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date").str.to_datetime(format="%Y-%m-%d", strict=False)
        )

    df = df.with_columns(
        _normalise_team_col(pl.col("home_team")).alias("home_team"),
        _normalise_team_col(pl.col("away_team")).alias("away_team"),
        pl.lit("fbref").alias("source"),
    )
    return df


def clean_understat(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    keep = {
        "home_team": pl.String,
        "away_team": pl.String,
        "xG_home": pl.Float64,
        "xG_away": pl.Float64,
        "xG_against_home": pl.Float64,
        "xG_against_away": pl.Float64,
        "us_forecast_home": pl.Float64,
        "us_forecast_draw": pl.Float64,
        "us_forecast_away": pl.Float64,
        "season": pl.String,
        "date": pl.String,
    }
    existing = {c for c in keep if c in df.columns}
    df = df.select(list(existing))

    for col, dtype in keep.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))

    # Rename xGA columns
    rename = {}
    if "xG_against_home" in df.columns:
        rename["xG_against_home"] = "xGA_home"
    if "xG_against_away" in df.columns:
        rename["xG_against_away"] = "xGA_away"
    if rename:
        df = df.rename(rename)

    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date")
            .str.to_datetime(strict=False)
            .dt.date()
            .cast(pl.Datetime("us"))
            .alias("date")
        )

    df = df.with_columns(
        _normalise_team_col(pl.col("home_team")).alias("home_team"),
        _normalise_team_col(pl.col("away_team")).alias("away_team"),
        pl.lit("understat").alias("source"),
    )
    return df


def clean_football_data(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    numeric = [
        "home_goals",
        "away_goals",
        "home_ht_goals",
        "away_ht_goals",
        "home_shots",
        "away_shots",
        "home_shots_ontarget",
        "away_shots_ontarget",
        "home_corners",
        "away_corners",
        "home_fouls",
        "away_fouls",
        "home_yellow",
        "away_yellow",
        "home_red",
        "away_red",
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_close_home",
        "odds_close_draw",
        "odds_close_away",
        "odds_max_home",
        "odds_max_draw",
        "odds_max_away",
        "odds_avg_home",
        "odds_avg_draw",
        "odds_avg_away",
    ]
    for col in numeric:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    if "date" in df.columns:
        # Handle both DD/MM/YY (pre-2017) and DD/MM/YYYY (2017+)
        df = df.with_columns(
            pl.when(pl.col("date").str.len_chars() > 8)
            .then(pl.col("date").str.to_datetime(format="%d/%m/%Y", strict=False))
            .otherwise(pl.col("date").str.to_datetime(format="%d/%m/%y", strict=False))
            .alias("date")
        )

    df = df.with_columns(
        _normalise_team_col(pl.col("home_team")).alias("home_team"),
        _normalise_team_col(pl.col("away_team")).alias("away_team"),
        pl.lit("football_data").alias("source"),
    )

    # Derive season from date
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date")
            .map_elements(
                lambda d: parse_season_from_date(d) if d is not None else None,
                return_dtype=pl.String,
            )
            .alias("season")
        )

    return df


def clean_elo(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    if "team" in df.columns:
        df = df.with_columns(_normalise_team_col(pl.col("team")).alias("team"))
    df = df.with_columns(pl.lit("elo").alias("source"))
    return df


def clean_all(
    data: dict[str, pl.DataFrame],
    out_dir: Optional[str] = None,
) -> dict[str, pl.DataFrame]:
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["processed_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cleaners = {
        "fbref": clean_fbref,
        "understat": clean_understat,
        "football_data": clean_football_data,
        "elo": clean_elo,
    }

    cleaned = {}
    for key, df in data.items():
        cleaner = cleaners.get(key)
        if cleaner:
            print(f"  Cleaning {key} ...")
            cleaned[key] = cleaner(df)
            if not cleaned[key].is_empty():
                out_path = Path(out_dir) / f"{key}_clean.parquet"
                cleaned[key].write_parquet(str(out_path))
                print(f"    -> {out_path}  ({len(cleaned[key])} rows)")
        else:
            cleaned[key] = df

    return cleaned
