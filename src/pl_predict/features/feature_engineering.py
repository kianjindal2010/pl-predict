"""Feature engineering: build all feature groups from merged match data."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from pl_predict.pipeline.utils import load_config, resolve_path, team_name_normalise


def _exp_weight(series: np.ndarray, halflife: float = 5) -> np.ndarray:
    alpha = 1 - np.exp(-np.log(2) / halflife)
    out = np.zeros_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def _prior_season_id(season_label: str) -> str:
    """PL '2024-25' -> prior season id '2023' (matches Understat/Transfermarkt)."""
    try:
        return str(int(season_label.split("-")[0]) - 1)
    except (ValueError, IndexError):
        return ""


def _attach_prior_squad_features(features: pl.DataFrame) -> pl.DataFrame:
    """Add lagged prior-season squad-strength features (no lookahead).

    For each match in season S, the squad features are the previous season's
    (S-1) per-team player aggregates from Understat, so they are known before
    the season begins. Returns ``features`` with home_/away_ columns added.
    """
    paths = load_config("paths")
    us_path = (
        Path(resolve_path(paths["raw_dir"])) / "understat" / "understat_players.parquet"
    )
    if not us_path.exists():
        return features

    plr = pl.read_parquet(str(us_path))
    if plr.is_empty() or "xG" not in plr.columns:
        return features

    squad = (
        plr.group_by(["season", "team"])
        .agg(
            pl.col("xG").cast(pl.Float64, strict=False).sum().alias("squad_xg"),
            pl.col("xA").cast(pl.Float64, strict=False).sum().alias("squad_xa"),
            pl.col("npxG").cast(pl.Float64, strict=False).sum().alias("squad_npxg"),
            pl.col("time").cast(pl.Float64, strict=False).sum().alias("squad_minutes"),
            pl.len().alias("squad_players"),
        )
        .with_columns(
            pl.col("team").map_elements(team_name_normalise, return_dtype=pl.String)
        )
    )

    # Map each match season -> prior Understat season id
    season_map = {
        s: _prior_season_id(s)
        for s in features["season"].drop_nulls().unique().to_list()
    }
    prior_ids = features["season"].map_elements(
        lambda s: season_map.get(s, ""), return_dtype=pl.String
    )

    sq_cols = ["squad_xg", "squad_xa", "squad_npxg", "squad_minutes", "squad_players"]
    out = features
    for team_side, team_col in [("home", "home_team"), ("away", "away_team")]:
        tmp = pl.DataFrame(
            {
                "season": prior_ids.to_list(),
                "team": out[team_col].to_list(),
            }
        ).join(
            squad,
            on=["season", "team"],
            how="left",
        )
        for c in sq_cols:
            out = out.with_columns(pl.Series(f"{team_side}_{c}", tmp[c].to_list()))
    return out


def _attach_player_features(features: pl.DataFrame) -> pl.DataFrame:
    """Leakage-free in-season player features from per-match rosters.

    For a match at date d, every feature aggregates ONLY the team's player
    rows with date < d within the same season (strictly prior). Adds
    ``home_/away_{szn_player_xg, szn_player_xa, szn_minutes, star_share,
    star_xg_form, players_used_8}``.
    """
    paths = load_config("paths")
    p_path = (
        Path(resolve_path(paths["raw_dir"]))
        / "understat"
        / "understat_player_matches.parquet"
    )
    if not p_path.exists():
        return features
    plr = pl.read_parquet(str(p_path))
    if plr.is_empty() or "xG" not in plr.columns:
        return features
    plr = plr.with_columns(
        pl.col("date").str.slice(0, 10).str.to_datetime(strict=False),
        pl.col("xG").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("xA").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("minutes").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("player_id").cast(pl.String, strict=False).fill_null(""),
    ).filter(pl.col("team").is_not_null() & pl.col("date").is_not_null())

    # Per (team, date) match-level aggregates from player rows
    pm = (
        plr.group_by(["season", "team", "date"])
        .agg(
            pl.col("xG").sum().alias("m_xg"),
            pl.col("xA").sum().alias("m_xa"),
            pl.col("minutes").sum().alias("m_min"),
            pl.col("xG").max().alias("m_star_xg"),
            pl.col("player_id").n_unique().alias("m_players"),
        )
        .sort(["team", "date"])
    )

    rows = []
    for (season, team), grp in pm.group_by(["season", "team"]):
        grp = grp.sort("date")
        m_xg = grp["m_xg"].to_numpy()
        m_xa = grp["m_xa"].to_numpy()
        m_min = grp["m_min"].to_numpy()
        m_star = grp["m_star_xg"].to_numpy()
        m_players = grp["m_players"].to_numpy()
        n = len(grp)
        if n == 0:
            continue
        cum_xg = np.concatenate([[0.0], np.cumsum(np.nan_to_num(m_xg))[:-1]])
        cum_xa = np.concatenate([[0.0], np.cumsum(np.nan_to_num(m_xa))[:-1]])
        cum_min = np.concatenate([[0.0], np.cumsum(np.nan_to_num(m_min))[:-1]])
        star_ew = _exp_weight(np.maximum(np.nan_to_num(m_star), 0.0), 8.0)
        team_ew = _exp_weight(np.maximum(np.nan_to_num(m_xg), 0.0), 8.0)
        pl_ew = _exp_weight(np.maximum(np.nan_to_num(m_players), 0.0), 8.0)
        star_prev = np.concatenate([[np.nan], star_ew[:-1]])
        team_prev = np.concatenate([[np.nan], team_ew[:-1]])
        pl_prev = np.concatenate([[np.nan], pl_ew[:-1]])
        share = np.where(team_prev > 1e-9, np.nan_to_num(star_prev) / team_prev, np.nan)
        for i in range(n):
            rows.append(
                {
                    "season": season,
                    "team": team,
                    "date": grp["date"][i],
                    "szn_player_xg": float(cum_xg[i]),
                    "szn_player_xa": float(cum_xa[i]),
                    "szn_minutes": float(cum_min[i]),
                    "star_share": float(share[i]) if np.isfinite(share[i]) else None,
                    "star_xg_form": float(star_prev[i])
                    if np.isfinite(star_prev[i])
                    else None,
                    "players_used_8": float(pl_prev[i])
                    if np.isfinite(pl_prev[i])
                    else None,
                }
            )
    if not rows:
        return features
    pf = pl.DataFrame(rows)

    p_cols = [
        "szn_player_xg",
        "szn_player_xa",
        "szn_minutes",
        "star_share",
        "star_xg_form",
        "players_used_8",
    ]
    pf = pf.with_columns(pl.col("date").cast(pl.Date, strict=False))
    out = features
    for team_side, team_col in [("home", "home_team"), ("away", "away_team")]:
        tmp = pl.DataFrame(
            {
                "season": out["season"].str.split("-").list.get(0).to_list(),
                "team": out[team_col].to_list(),
                "date": out["date"].cast(pl.Date, strict=False).to_list(),
            }
        ).join(pf, on=["season", "team", "date"], how="left")
        for c in p_cols:
            out = out.with_columns(pl.Series(f"{team_side}_{c}", tmp[c].to_list()))
    return out


def _attach_transfermarkt_features(features: pl.DataFrame) -> pl.DataFrame:
    """Lagged prior-season squad market value / average age / size."""
    paths = load_config("paths")
    t_path = (
        Path(resolve_path(paths["raw_dir"]))
        / "transfermarkt"
        / "transfermarkt_squads.parquet"
    )
    if not t_path.exists():
        return features
    tm = pl.read_parquet(str(t_path))
    if tm.is_empty():
        return features
    tm = (
        tm.with_columns(
            pl.col("team").map_elements(team_name_normalise, return_dtype=pl.String)
        )
        .with_columns(
            pl.col("squad_value").cast(pl.Float64, strict=False),
            pl.col("avg_age").cast(pl.Float64, strict=False),
            pl.col("squad_size").cast(pl.Float64, strict=False),
        )
        .select(["season", "team", "squad_value", "avg_age", "squad_size"])
    )

    season_map = {
        s: _prior_season_id(s)
        for s in features["season"].drop_nulls().unique().to_list()
    }
    prior_ids = features["season"].map_elements(
        lambda s: season_map.get(s, ""), return_dtype=pl.String
    )

    cols = ["squad_value", "avg_age", "squad_size"]
    out = features
    for team_side, team_col in [("home", "home_team"), ("away", "away_team")]:
        tmp = pl.DataFrame(
            {
                "season": prior_ids.to_list(),
                "team": out[team_col].to_list(),
            }
        ).join(tm, on=["season", "team"], how="left")
        for c in cols:
            out = out.with_columns(pl.Series(f"{team_side}_{c}", tmp[c].to_list()))
    return out


def _compute_team_form(
    team_df: pl.DataFrame, team_list: list, n: int, has_xg: bool
) -> dict:
    """Compute strictly prior-match EWMA form for each team and match side."""
    del team_list  # The team dimension is read directly from tagged records.
    home_results = {
        "form_pts_avg": np.full(n, np.nan),
        "form_gd_avg": np.full(n, np.nan),
        "form_gf_avg": np.full(n, np.nan),
        "form_ga_avg": np.full(n, np.nan),
    }
    away_results = {
        "form_pts_avg": np.full(n, np.nan),
        "form_gd_avg": np.full(n, np.nan),
        "form_gf_avg": np.full(n, np.nan),
        "form_ga_avg": np.full(n, np.nan),
    }
    if has_xg:
        home_results["form_xg_for_avg"] = np.full(n, np.nan)
        home_results["form_xg_against_avg"] = np.full(n, np.nan)
        away_results["form_xg_for_avg"] = np.full(n, np.nan)
        away_results["form_xg_against_avg"] = np.full(n, np.nan)
    for _, group in team_df.group_by("team", maintain_order=True):
        group = group.sort("match_id")
        metrics = {
            "form_pts_avg": group["pts"].to_numpy(),
            "form_gd_avg": group["gd"].to_numpy(),
            "form_gf_avg": group["gf"].to_numpy(),
            "form_ga_avg": group["ga"].to_numpy(),
        }
        if has_xg:
            metrics["form_xg_for_avg"] = np.nan_to_num(
                group["xg_for"].to_numpy(), nan=0.0
            )
            metrics["form_xg_against_avg"] = np.nan_to_num(
                group["xg_against"].to_numpy(), nan=0.0
            )

        for name, values in metrics.items():
            prior = np.roll(_exp_weight(values, halflife=5), 1)
            prior[0] = np.nan
            for match_id, side, value in zip(
                group["match_id"].to_numpy(), group["side"].to_list(), prior
            ):
                target = home_results if side == "home" else away_results
                target[name][int(match_id)] = value

    return home_results, away_results


def build_single_features(
    home: str,
    away: str,
    matches: Optional[pl.DataFrame] = None,
    date: Optional[str] = None,
) -> pl.DataFrame:
    """Build a one-row feature frame for a hypothetical future match.

    Reuses the same EWMA form logic as :func:`build_features`, but evaluates
    each team's CURRENT form (i.e. through their last played match). Market
    probabilities and congestion are left null (unknown for future fixtures).
    """
    if matches is None:
        matches = pl.read_parquet(resolve_path("data/processed/matches.parquet"))
    has_xg = "xG_home" in matches.columns and "xG_away" in matches.columns

    def team_form(team: str) -> Optional[dict]:
        sub = matches.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        ).sort("date")
        if sub.is_empty():
            return None
        pts, gd, gf, ga, xgf, xga = [], [], [], [], [], []
        for row in sub.iter_rows(named=True):
            if row["home_team"] == team:
                hg = float(row.get("home_goals") or 0.0)
                ag = float(row.get("away_goals") or 0.0)
            else:
                hg = float(row.get("away_goals") or 0.0)
                ag = float(row.get("home_goals") or 0.0)
            pts.append(3 if hg > ag else 1 if hg == ag else 0)
            gd.append(hg - ag)
            gf.append(hg)
            ga.append(ag)
            if has_xg:
                xgf.append(
                    float(row.get("xG_home") or 0.0)
                    if row["home_team"] == team
                    else float(row.get("xG_away") or 0.0)
                )
                xga.append(
                    float(row.get("xG_away") or 0.0)
                    if row["home_team"] == team
                    else float(row.get("xG_home") or 0.0)
                )

        def ewma(vals):
            arr = np.asarray(vals, dtype=float)
            return float(_exp_weight(arr, halflife=5)[-1]) if len(arr) else None

        out = {
            "form_pts_avg": ewma(pts),
            "form_gd_avg": ewma(gd),
            "form_gf_avg": ewma(gf),
            "form_ga_avg": ewma(ga),
            "match_count": len(sub),
        }
        if has_xg:
            out["form_xg_for_avg"] = ewma(xgf)
            out["form_xg_against_avg"] = ewma(xga)
        return out

    hf = team_form(home)
    af = team_form(away)

    row = {
        "date": date or datetime.now(),
        "home_team": home,
        "away_team": away,
        "season": "2026-27",
        "match_id": 0,
    }

    for k in ["form_pts_avg", "form_gd_avg", "form_gf_avg", "form_ga_avg"]:
        row[f"home_{k}"] = hf[k] if hf else None
        row[f"away_{k}"] = af[k] if af else None
    if has_xg:
        for k in ["form_xg_for_avg", "form_xg_against_avg"]:
            row[f"home_{k}"] = hf[k] if hf else None
            row[f"away_{k}"] = af[k] if af else None

    row["home_match_count"] = hf["match_count"] if hf else 0
    row["away_match_count"] = af["match_count"] if af else 0
    row["home_days_rest"] = None
    row["away_days_rest"] = None

    for prefix in ["", "_close", "_max", "_avg"]:
        row[f"market_home_prob{prefix}"] = None
        row[f"market_draw_prob{prefix}"] = None
        row[f"market_away_prob{prefix}"] = None

    if hf and af:
        row["form_gd_avg_diff"] = hf["form_gd_avg"] - af["form_gd_avg"]
        row["form_pts_avg_diff"] = hf["form_pts_avg"] - af["form_pts_avg"]
        row["is_home_team_stronger"] = (
            1 if hf["form_pts_avg"] > af["form_pts_avg"] else 0
        )
    else:
        row["form_gd_avg_diff"] = None
        row["form_pts_avg_diff"] = None
        row["is_home_team_stronger"] = None
    row["rest_advantage"] = None

    for t in [
        "target_result",
        "target_home_goals",
        "target_away_goals",
        "target_over_2_5",
        "target_btts",
        "target_total_goals",
        "target_goal_diff",
    ]:
        row[t] = None

    return pl.DataFrame([row])


def build_features(
    matches: pl.DataFrame, out_dir: Optional[str] = None
) -> pl.DataFrame:
    """Build all feature groups from merged match data."""
    if matches.is_empty():
        return matches

    df = matches.sort(["season", "date"])
    n = len(df)

    features = df.select(["date", "home_team", "away_team", "season"])
    features = features.with_columns(pl.Series("match_id", range(n)))

    has_goals = "home_goals" in df.columns and "away_goals" in df.columns
    has_xg = "xG_home" in df.columns and "xG_away" in df.columns

    # ------------------------------------------------------------------
    # A. Team Strength - rolling form metrics
    # ------------------------------------------------------------------
    print("  Building rolling form features ...")

    home_teams = df["home_team"].to_list()
    away_teams = df["away_team"].to_list()
    all_teams = sorted(set(home_teams) | set(away_teams))

    # Build team-level records
    team_records = []
    xg_home_vals = df["xG_home"].to_list() if has_xg else None
    xg_away_vals = df["xG_away"].to_list() if has_xg else None
    for i in range(n):
        ht = home_teams[i]
        at = away_teams[i]
        if has_goals:
            hg = float(df["home_goals"][i])
            ag = float(df["away_goals"][i])
            if hg > ag:
                hp, ap = 3, 0
            elif hg < ag:
                hp, ap = 0, 3
            else:
                hp, ap = 1, 1
        else:
            hg, ag, hp, ap = 0, 0, 0, 0

        if has_xg:
            xh = xg_home_vals[i]
            xa = xg_away_vals[i]
            xh_v = float(xh) if xh is not None and xh == xh else np.nan
            xa_v = float(xa) if xa is not None and xa == xa else np.nan
        else:
            xh_v = xa_v = np.nan

        team_records.append(
            {
                "match_id": i,
                "side": "home",
                "team": ht,
                "pts": hp,
                "gd": hg - ag,
                "gf": hg,
                "ga": ag,
                "xg_for": xh_v,
                "xg_against": xa_v,
            }
        )
        team_records.append(
            {
                "match_id": i,
                "side": "away",
                "team": at,
                "pts": ap,
                "gd": ag - hg,
                "gf": ag,
                "ga": hg,
                "xg_for": xa_v,
                "xg_against": xh_v,
            }
        )

    team_df = pl.DataFrame(team_records)

    home_form, away_form = _compute_team_form(team_df, all_teams, n, has_xg)

    for key in home_form:
        features = features.with_columns(
            pl.Series(f"home_{key}", home_form[key].tolist())
        )
        features = features.with_columns(
            pl.Series(f"away_{key}", away_form[key].tolist())
        )

    # ------------------------------------------------------------------
    # B. Days rest & match count
    # ------------------------------------------------------------------
    print("  Building congestion features ...")

    home_rest = np.full(n, np.nan)
    away_rest = np.full(n, np.nan)
    home_count = np.zeros(n, dtype=float)
    away_count = np.zeros(n, dtype=float)

    team_last = {}
    team_mc = {}
    for t in all_teams:
        team_last[t] = None
        team_mc[t] = 0

    for i in range(n):
        ht = home_teams[i]
        at = away_teams[i]
        d = df["date"][i]
        if team_last[ht] is not None:
            home_rest[i] = (d - team_last[ht]).days if hasattr(d, "days") else np.nan
        if team_last[at] is not None:
            away_rest[i] = (d - team_last[at]).days if hasattr(d, "days") else np.nan
        team_mc[ht] += 1
        team_mc[at] += 1
        home_count[i] = team_mc[ht]
        away_count[i] = team_mc[at]
        team_last[ht] = d
        team_last[at] = d

    features = features.with_columns(
        pl.Series("home_match_count", home_count),
        pl.Series("away_match_count", away_count),
        pl.Series("home_days_rest", home_rest),
        pl.Series("away_days_rest", away_rest),
    )

    # ------------------------------------------------------------------
    # C. Market-implied features
    # ------------------------------------------------------------------
    print("  Building market features ...")

    for prefix in ["", "_close", "_max", "_avg"]:
        hc = f"implied_home{prefix}_norm"
        if hc in df.columns:
            features = features.with_columns(
                pl.Series(f"market_home_prob{prefix}", df[hc].to_list()),
                pl.Series(
                    f"market_draw_prob{prefix}",
                    df[f"implied_draw{prefix}_norm"].to_list(),
                ),
                pl.Series(
                    f"market_away_prob{prefix}",
                    df[f"implied_away{prefix}_norm"].to_list(),
                ),
            )

    # Understat pre-match forecast (external model probabilities, 2014+)
    if "us_forecast_home" in df.columns:
        features = features.with_columns(
            pl.Series("us_home_prob", df["us_forecast_home"].to_list()),
            pl.Series("us_draw_prob", df["us_forecast_draw"].to_list()),
            pl.Series("us_away_prob", df["us_forecast_away"].to_list()),
        )
    # Per-match expected goals (2014+)
    for c in ("xG_home", "xG_away"):
        if c in df.columns:
            features = features.with_columns(pl.Series(c, df[c].to_list()))

    # ------------------------------------------------------------------
    # C2. Prior-season squad strength (Understat player aggregates, lagged)
    # ------------------------------------------------------------------
    print("  Building squad-strength features ...")
    features = _attach_prior_squad_features(features)

    # C3. In-season player features (per-match rosters, strictly prior)
    print("  Building player-level features ...")
    features = _attach_player_features(features)

    # C4. Prior-season Transfermarkt squad value / age / size (lagged)
    print("  Building transfermarkt features ...")
    features = _attach_transfermarkt_features(features)

    # ------------------------------------------------------------------
    # D. Derived match features
    # ------------------------------------------------------------------
    print("  Building derived match features ...")

    for col in ["form_gd_avg", "form_pts_avg", "form_xg_for_avg"]:
        h = f"home_{col}"
        a = f"away_{col}"
        if h in features.columns and a in features.columns:
            features = features.with_columns(
                (pl.col(h) - pl.col(a)).alias(f"{col}_diff")
            )

    if (
        "home_form_pts_avg" in features.columns
        and "away_form_pts_avg" in features.columns
    ):
        features = features.with_columns(
            (pl.col("home_form_pts_avg") > pl.col("away_form_pts_avg"))
            .cast(pl.Int8)
            .alias("is_home_team_stronger")
        )

    if "home_days_rest" in features.columns and "away_days_rest" in features.columns:
        features = features.with_columns(
            (pl.col("home_days_rest") - pl.col("away_days_rest")).alias(
                "rest_advantage"
            )
        )

    # ------------------------------------------------------------------
    # E. Target variables
    # ------------------------------------------------------------------
    if "result" in df.columns:
        features = features.with_columns(
            pl.Series("target_result", df["result"].to_list()),
            pl.Series(
                "target_home_goals",
                df.get_column("home_goals").to_numpy()
                if "home_goals" in df.columns
                else np.zeros(n),
            ),
            pl.Series(
                "target_away_goals",
                df.get_column("away_goals").to_numpy()
                if "away_goals" in df.columns
                else np.zeros(n),
            ),
            pl.Series(
                "target_over_2_5",
                df.get_column("over_2_5").to_numpy()
                if "over_2_5" in df.columns
                else np.zeros(n),
            ),
            pl.Series(
                "target_btts",
                df.get_column("btts").to_numpy()
                if "btts" in df.columns
                else np.zeros(n),
            ),
            pl.Series(
                "target_total_goals",
                df.get_column("total_goals").to_numpy()
                if "total_goals" in df.columns
                else np.zeros(n),
            ),
            pl.Series(
                "target_goal_diff",
                df.get_column("goal_diff").to_numpy()
                if "goal_diff" in df.columns
                else np.zeros(n),
            ),
        )

    # Save
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["processed_dir"])
    features_path = Path(out_dir) / "features.parquet"
    features.write_parquet(str(features_path))
    print(
        f"  Features -> {features_path}  ({len(features)} rows, {len(features.columns)} cols)"
    )

    return features
