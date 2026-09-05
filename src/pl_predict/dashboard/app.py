"""FastAPI + Plotly web dashboard for PL Predict.

Run with: python -m pl_predict.dashboard.app   (or `predict-pl dashboard`)
"""

import json as _json
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pl_predict.pipeline.utils import resolve_path, team_name_normalise


def _clean(obj):
    """Recursively replace inf/nan with None for JSON serialization."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


@asynccontextmanager
async def _lifespan(app):
    threading.Thread(target=_warm_cache, daemon=True).start()
    yield


app = FastAPI(title="PL Predict Dashboard", version="0.2.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)

DASH_DIR = Path(__file__).parent
STATIC_DIR = DASH_DIR / "static"
PROCESSED = Path(resolve_path("data/processed"))
RAW_UNDERSTAT = Path(resolve_path("data/raw/understat"))
MODEL_PATH = PROCESSED / "ensemble_model.pkl"
FIXTURES_CACHE_PATH = PROCESSED / "fixtures_2627_pred.json"
FPL_PICKS_CACHE_PATH = PROCESSED / "fpl_hybrid_picks.json"

# ---------------------------------------------------------------------------
# Startup artifact cache: everything is loaded once into memory.
# ---------------------------------------------------------------------------

CACHE = {}
FPL_CACHE_TTL_SECONDS = 15 * 60

WARM_DONE = threading.Event()


def _warm_cache():
    """Eagerly load the heavy artifacts so the first request is fast."""
    _predictor()
    _read_parquet("matches.parquet")
    _read_parquet("season_sim.parquet")
    _read_parquet("walkforward_preds.parquet")
    _teams()
    _feature_importances()
    _player_agg()
    _shots()
    _fixtures_result()
    WARM_DONE.set()


def _read_parquet(name: str):
    """Cached parquet reader keyed by file mtime."""
    p = PROCESSED / name
    key = f"parquet:{name}"
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    entry = CACHE.get(key)
    if entry and entry[0] == mtime:
        return entry[1]
    import polars as pl

    df = pl.read_parquet(str(p))
    CACHE[key] = (mtime, df)
    return df


def _predictor():
    """Cached ensemble predictor (reloads only when the pickle changes)."""
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    entry = CACHE.get("predictor")
    if entry and entry[0] == mtime:
        return entry[1]
    from pl_predict.models.ensemble import EnsemblePredictor

    predictor = EnsemblePredictor.load(str(MODEL_PATH))
    CACHE["predictor"] = (mtime, predictor)
    return predictor


_PRED_KEYS = (
    "home_win",
    "draw",
    "away_win",
    "expected_home_goals",
    "expected_away_goals",
    "most_likely_score",
    "over_1_5",
    "over_2_5",
    "over_3_5",
    "btts",
    "clean_sheet_home",
    "clean_sheet_away",
    "conformal_80",
    "conformal_90",
    "score_matrix",
    "top_scores",
    "model_breakdown",
)


def _cached_prediction(home: str, away: str) -> Optional[dict]:
    """Prediction for (home, away), reused from the 380-fixture JSON when possible."""
    try:
        from pl_predict.data.fixtures_2627 import normalize_team

        nh, na = normalize_team(home), normalize_team(away)
    except Exception:
        nh, na = home, away

    try:
        fj = _fixtures_result()
        if fj:
            for f in fj.get("fixtures", []):
                if f.get("home_team") == nh and f.get("away_team") == na:
                    p = f.get("prediction") or {}
                    return {k: p[k] for k in _PRED_KEYS if k in p}
    except Exception:
        pass

    key = ("h2h_pred", MODEL_PATH.stat().st_mtime)
    store = CACHE.get(key)
    if store is None:
        store = {}
        CACHE[key] = store
    pair = (nh, na)
    if pair in store:
        return store[pair]

    predictor = _predictor()
    p = None
    if predictor is not None:
        try:
            r = predictor.predict(nh, na)
            if r and "error" not in r:
                p = {k: r[k] for k in _PRED_KEYS if k in r}
        except Exception:
            p = None
    store[pair] = p
    return p


def _fpl_snapshot(force: bool = False) -> dict:
    """Live official FPL data, refreshed at most once every 15 minutes."""
    entry = CACHE.get("fpl_snapshot")
    if not force and entry and time.time() - entry[0] < FPL_CACHE_TTL_SECONDS:
        return entry[1]

    from pl_predict.pipeline.fpl import fetch_fpl_snapshot

    snapshot = fetch_fpl_snapshot()
    CACHE["fpl_snapshot"] = (time.time(), snapshot)
    return snapshot


def _player_agg():
    """Per (player, season) aggregates, computed once at startup."""
    entry = CACHE.get("player_agg")
    if entry is not None:
        return entry
    import polars as pl

    pm = _player_matches_clean()
    if pm is None:
        CACHE["player_agg"] = None
        return None
    npx = (
        pl.read_parquet(str(RAW_UNDERSTAT / "understat_players.parquet"))
        .select(["player_id", "season", pl.col("npxG").cast(pl.Float64, strict=False)])
        .unique(subset=["player_id", "season"])
    )
    agg = (
        pm.group_by(["player_id", "player_name", "team", "season", "position"])
        .agg(
            pl.col("minutes").sum().alias("minutes"),
            pl.col("xG").sum().alias("xG"),
            pl.col("xA").sum().alias("xA"),
            pl.col("goals").sum().alias("goals"),
            pl.col("assists").sum().alias("assists"),
            pl.col("shots").sum().alias("shots"),
            pl.col("key_passes").sum().alias("key_passes"),
            pl.len().alias("matches"),
        )
        .join(npx, on=["player_id", "season"], how="left")
        .with_columns((pl.col("minutes") / 90).round(1).alias("games_90"))
    )
    CACHE["player_agg"] = agg
    return agg


def _player_matches_clean():
    """Cached, numeric-cast per-match player dataframe."""
    entry = CACHE.get("player_matches")
    if entry is not None:
        return entry
    import polars as pl

    p = RAW_UNDERSTAT / "understat_player_matches.parquet"
    if not p.exists():
        CACHE["player_matches"] = None
        return None
    df = pl.read_parquet(str(p)).with_columns(
        pl.col("minutes").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("xG").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("xA").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("goals").cast(pl.Int64, strict=False).fill_null(0),
        pl.col("assists").cast(pl.Int64, strict=False).fill_null(0),
        pl.col("shots").cast(pl.Int64, strict=False).fill_null(0),
        pl.col("key_passes").cast(pl.Int64, strict=False).fill_null(0),
        pl.col("yellow_card").cast(pl.Int64, strict=False).fill_null(0),
        pl.col("red_card").cast(pl.Int64, strict=False).fill_null(0),
    )
    CACHE["player_matches"] = df
    return df


def _shots():
    """Cached shot-level dataframe with team names joined in."""
    entry = CACHE.get("shots")
    if entry is not None:
        return entry
    import polars as pl

    p = RAW_UNDERSTAT / "understat_shots.parquet"
    if not p.exists():
        CACHE["shots"] = None
        return None
    pm = _player_matches_clean()
    team_map = pm.select(["match_id", "h_a", "team"]).unique()
    df = (
        pl.read_parquet(str(p))
        .join(team_map, on=["match_id", "h_a"], how="left")
        .with_columns(
            pl.col("X").cast(pl.Float64, strict=False),
            pl.col("Y").cast(pl.Float64, strict=False),
            pl.col("xG").cast(pl.Float64, strict=False),
            pl.col("minute").cast(pl.Int32, strict=False),
        )
    )
    CACHE["shots"] = df
    return df


def _teams() -> list:
    entry = CACHE.get("teams")
    if entry is not None:
        return entry
    df = _read_parquet("matches.parquet")
    teams = []
    if df is not None:
        teams = sorted(
            set(df["home_team"].drop_nulls().to_list())
            | set(df["away_team"].drop_nulls().to_list())
        )
        teams = [t for t in teams if t]
    CACHE["teams"] = teams
    return teams


def _feature_importances() -> list:
    entry = CACHE.get("feature_importances")
    if entry is not None:
        return entry
    out = []
    predictor = _predictor()
    if predictor is not None:
        model = predictor.xgb_model
        names = predictor.xgb_features or []
        if model is not None:
            importances = model.feature_importances_
            pairs = sorted(zip(names, importances), key=lambda x: -x[1])[:30]
            out = [{"name": n, "importance": float(v)} for n, v in pairs]
    CACHE["feature_importances"] = out
    return out


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (DASH_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return html


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# API: data availability
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    df = _read_parquet("matches.parquet")
    return {
        "status": "ok",
        "ready": WARM_DONE.is_set(),
        "total_matches": len(df) if df is not None else 0,
        "has_matches": df is not None,
        "has_features": (PROCESSED / "features.parquet").exists(),
        "has_model": MODEL_PATH.exists(),
        "has_season_sim": (PROCESSED / "season_sim.parquet").exists(),
        "has_players": (RAW_UNDERSTAT / "understat_player_matches.parquet").exists(),
        "has_shots": (RAW_UNDERSTAT / "understat_shots.parquet").exists(),
        "has_performance": (PROCESSED / "walkforward_preds.parquet").exists(),
        "teams": len(_teams()),
    }


# ---------------------------------------------------------------------------
# API: matches & predictions
# ---------------------------------------------------------------------------

MATCH_COLUMNS = [
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "implied_home_norm",
    "implied_draw_norm",
    "implied_away_norm",
]


@app.get("/api/matches")
async def matches(
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """Paginated matches, most recent first."""
    df = _read_parquet("matches.parquet")
    if df is None:
        return {"matches": [], "total": 0, "offset": 0}
    total = len(df)
    df = df.sort("date", descending=True).slice(offset, limit)
    rows = df.select(MATCH_COLUMNS).to_dicts()
    return _clean(
        {"matches": rows, "total": total, "offset": offset, "limit": len(rows)}
    )


@app.get("/api/upcoming")
async def upcoming(limit: int = Query(20, ge=1, le=50)):
    """Upcoming fixtures (matches without a result / in the future)."""
    df = _read_parquet("matches.parquet")
    if df is None:
        return {"fixtures": [], "message": "No match data yet."}
    import polars as pl

    upcoming_df = df.filter(pl.col("home_goals").is_null())
    if upcoming_df.is_empty():
        return {"fixtures": [], "message": "No upcoming fixtures found."}
    rows = upcoming_df.head(limit).to_dicts()
    return _clean({"fixtures": rows})


def _fixtures_result():
    """2026/27 fixtures + predictions; loads JSON file into memory (mtime-keyed),
    or regenerates + writes the cache file if missing."""
    cur_mtime = (
        FIXTURES_CACHE_PATH.stat().st_mtime if FIXTURES_CACHE_PATH.exists() else 0
    )
    entry = CACHE.get("fixtures")
    if entry and entry[0] == cur_mtime:
        return entry[1]

    if FIXTURES_CACHE_PATH.exists():
        try:
            result = _json.loads(FIXTURES_CACHE_PATH.read_text(encoding="utf-8"))
            CACHE["fixtures"] = (cur_mtime, result)
            return result
        except Exception:
            pass

    from pl_predict.data.fixtures_2627 import get_upcoming_fixtures, normalize_team

    fixtures = get_upcoming_fixtures()
    predictions = {}
    predictor = _predictor()
    if predictor is not None:
        for f in fixtures:
            try:
                home = normalize_team(f["home_team"])
                away = normalize_team(f["away_team"])
                p = predictor.predict(home, away)
                if p and "error" not in p:
                    predictions[(home, away)] = p
            except Exception:
                continue

    for f in fixtures:
        key = (normalize_team(f["home_team"]), normalize_team(f["away_team"]))
        f["prediction"] = predictions.get(key)

    result = _clean({"fixtures": fixtures, "season": "2026/27"})
    CACHE["fixtures"] = (cur_mtime, result)
    try:
        FIXTURES_CACHE_PATH.write_text(_json.dumps(result), encoding="utf-8")
    except Exception:
        pass
    return result


@app.get("/api/fixtures_2627")
async def fixtures_2627():
    """Return 2026/27 fixtures with predictions (cached to disk + memory)."""
    return _fixtures_result()


def _build_fpl_picks(snapshot: dict, position: Optional[str] = None) -> dict:
    """Calculate Hybrid xP from one official FPL snapshot and our match model."""
    from pl_predict.pipeline.fpl import select_upcoming_gameweek

    bootstrap = snapshot["bootstrap"]
    next_event = select_upcoming_gameweek(bootstrap["events"])

    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    fixtures_by_team: dict[int, list] = {}
    for fixture in snapshot["fixtures"]:
        if fixture.get("event") != next_event["id"]:
            continue
        fixtures_by_team.setdefault(fixture["team_h"], []).append((fixture, True))
        fixtures_by_team.setdefault(fixture["team_a"], []).append((fixture, False))

    players = []
    for player in bootstrap["elements"]:
        player_position = positions.get(player["element_type"], "")
        if position and player_position != position:
            continue
        team_name = teams[player["team"]]
        fixture_outlook = []
        model_team_points = 0.0
        for fixture, is_home in fixtures_by_team.get(player["team"], []):
            opponent_id = fixture["team_a"] if is_home else fixture["team_h"]
            opponent = teams[opponent_id]
            home, away = (team_name, opponent) if is_home else (opponent, team_name)
            prediction = _cached_prediction(
                team_name_normalise(home), team_name_normalise(away)
            )
            win_probability = None
            clean_sheet_probability = None
            if prediction:
                win_probability = (
                    prediction["home_win"] if is_home else prediction["away_win"]
                )
                clean_sheet_probability = (
                    prediction["clean_sheet_home"]
                    if is_home
                    else prediction["clean_sheet_away"]
                )
                model_team_points += 3 * win_probability + prediction["draw"]
                team_expected_goals = (
                    prediction["expected_home_goals"]
                    if is_home
                    else prediction["expected_away_goals"]
                )
            else:
                team_expected_goals = None
            fixture_outlook.append(
                {
                    "opponent": opponent,
                    "home": is_home,
                    "difficulty": fixture.get(
                        "team_h_difficulty" if is_home else "team_a_difficulty"
                    ),
                    "kickoff": fixture.get("kickoff_time"),
                    "win_probability": win_probability,
                    "clean_sheet_probability": clean_sheet_probability,
                    "team_expected_goals": team_expected_goals,
                }
            )

        availability = player.get("chance_of_playing_next_round")
        availability_factor = 1.0 if availability is None else float(availability) / 100
        expected_minutes = min(90.0, 25.0 + 65.0 * min(1.0, player["starts"] / 38))
        expected_minutes *= availability_factor
        minutes_factor = expected_minutes / 90.0
        goal_points = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}[player_position]
        clean_sheet_points = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}[player_position]
        player_xg90 = float(player.get("expected_goals_per_90") or 0.0)
        player_xa90 = float(player.get("expected_assists_per_90") or 0.0)
        model_xp = 0.0
        for fixture in fixture_outlook:
            attack_factor = (fixture["team_expected_goals"] or 1.35) / 1.35
            model_xp += 2 * min(1.0, expected_minutes / 60.0)
            model_xp += goal_points * player_xg90 * minutes_factor * attack_factor
            model_xp += 3 * player_xa90 * minutes_factor * attack_factor
            model_xp += (
                clean_sheet_points
                * (fixture["clean_sheet_probability"] or 0.0)
                * minutes_factor
            )

        official_xp = float(player.get("ep_next") or 0.0)
        hybrid_xp = 0.55 * official_xp + 0.45 * model_xp

        players.append(
            {
                "player_id": player["id"],
                "name": player["web_name"],
                "team": team_name,
                "position": player_position,
                "price": player["now_cost"] / 10,
                "official_xp": official_xp,
                "model_xp": model_xp,
                "hybrid_xp": hybrid_xp,
                "form": float(player.get("form") or 0.0),
                "total_points": player["total_points"],
                "ownership": float(player.get("selected_by_percent") or 0.0),
                "availability": availability,
                "fixtures": fixture_outlook,
                "model_team_points": model_team_points,
            }
        )

    players.sort(
        key=lambda item: (item["hybrid_xp"], item["official_xp"]),
        reverse=True,
    )
    return _clean(
        {
            "gameweek": next_event["id"],
            "deadline": next_event.get("deadline_time"),
            "refreshed_at": snapshot.get("fetched_at"),
            "players": players,
            "source": "Official Fantasy Premier League data + PL Predict match model",
        }
    )


def _fpl_projection_catalog(snapshot: dict, horizon: int = 6) -> dict:
    """Build reusable multi-gameweek projections from one official snapshot."""
    from pl_predict.pipeline.fpl import select_upcoming_gameweek

    bootstrap = snapshot["bootstrap"]
    events = bootstrap.get("events", [])
    next_event = select_upcoming_gameweek(events)
    upcoming = [
        event for event in sorted(events, key=lambda item: item.get("id", 0))
        if event.get("id", 0) >= next_event["id"] and not event.get("finished")
    ][:max(1, min(horizon, 10))]
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    fixtures_by_event_team: dict[tuple[int, int], list] = {}
    for fixture in snapshot["fixtures"]:
        event_id = fixture.get("event")
        if event_id is None:
            continue
        fixtures_by_event_team.setdefault((event_id, fixture["team_h"]), []).append(
            (fixture, True)
        )
        fixtures_by_event_team.setdefault((event_id, fixture["team_a"]), []).append(
            (fixture, False)
        )

    def project(player: dict, event: dict) -> dict:
        position = positions.get(player["element_type"], "")
        team_name = teams[player["team"]]
        availability = player.get("chance_of_playing_next_round")
        availability_factor = 1.0 if availability is None else float(availability) / 100
        expected_minutes = min(90.0, 25.0 + 65.0 * min(1.0, player["starts"] / 38))
        expected_minutes *= availability_factor
        minutes_factor = expected_minutes / 90.0
        goal_points = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}[position]
        clean_sheet_points = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}[position]
        player_xg90 = float(player.get("expected_goals_per_90") or 0.0)
        player_xa90 = float(player.get("expected_assists_per_90") or 0.0)
        fixtures = []
        xp = 0.0
        for fixture, is_home in fixtures_by_event_team.get(
            (event["id"], player["team"]), []
        ):
            opponent_id = fixture["team_a"] if is_home else fixture["team_h"]
            opponent = teams[opponent_id]
            home, away = (team_name, opponent) if is_home else (opponent, team_name)
            prediction = _cached_prediction(
                team_name_normalise(home), team_name_normalise(away)
            )
            win_probability = None
            clean_sheet_probability = None
            team_xg = None
            if prediction:
                win_probability = (
                    prediction["home_win"] if is_home else prediction["away_win"]
                )
                clean_sheet_probability = (
                    prediction["clean_sheet_home"]
                    if is_home else prediction["clean_sheet_away"]
                )
                team_xg = (
                    prediction["expected_home_goals"]
                    if is_home else prediction["expected_away_goals"]
                )
            attack_factor = (team_xg or 1.35) / 1.35
            fixture_xp = 2 * min(1.0, expected_minutes / 60.0)
            fixture_xp += goal_points * player_xg90 * minutes_factor * attack_factor
            fixture_xp += 3 * player_xa90 * minutes_factor * attack_factor
            fixture_xp += clean_sheet_points * (clean_sheet_probability or 0.0) * minutes_factor
            xp += fixture_xp
            fixtures.append({
                "opponent": opponent,
                "home": is_home,
                "difficulty": fixture.get(
                    "team_h_difficulty" if is_home else "team_a_difficulty"
                ),
                "kickoff": fixture.get("kickoff_time"),
                "xP": round(fixture_xp, 2),
                "win_probability": win_probability,
                "clean_sheet_probability": clean_sheet_probability,
                "team_expected_goals": team_xg,
            })
        if event["id"] == next_event["id"]:
            official_xp = float(player.get("ep_next") or 0.0)
            xp = 0.55 * official_xp + 0.45 * xp
        return {
            "xP": round(xp, 2),
            "fixtures": fixtures,
            "official_xp": float(player.get("ep_next") or 0.0),
        }

    players = []
    for player in bootstrap["elements"]:
        position = positions.get(player["element_type"], "")
        projections = {
            str(event["id"]): project(player, event) for event in upcoming
        }
        players.append({
            "player_id": player["id"],
            "name": player["web_name"],
            "team": teams[player["team"]],
            "position": position,
            "price": player["now_cost"] / 10,
            "price_tenths": player["now_cost"],
            "form": float(player.get("form") or 0.0),
            "total_points": player["total_points"],
            "ownership": float(player.get("selected_by_percent") or 0.0),
            "availability": player.get("chance_of_playing_next_round"),
            "projections": projections,
        })
    return {
        "gameweeks": [
            {
                "id": event["id"],
                "name": event.get("name"),
                "deadline": event.get("deadline_time"),
            }
            for event in upcoming
        ],
        "players": players,
        "next_gameweek": next_event["id"],
        "refreshed_at": snapshot.get("fetched_at"),
    }


def _select_legal_fpl_xi(players: list[dict], gameweek: str) -> list[dict]:
    """Select an XI while respecting the minimum FPL formation rules."""
    by_position = {
        position: sorted(
            [p for p in players if p["position"] == position],
            key=lambda p: p["projections"].get(gameweek, {}).get("xP", 0),
            reverse=True,
        )
        for position in ("GK", "DEF", "MID", "FWD")
    }
    minimums = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    if any(len(by_position[position]) < minimum for position, minimum in minimums.items()):
        raise ValueError("Squad does not contain enough players for a legal FPL formation.")
    selected = (
        by_position["GK"][:1] + by_position["DEF"][:3]
        + by_position["MID"][:2] + by_position["FWD"][:1]
    )
    counts = {position: minimums[position] for position in minimums}
    maximums = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
    remaining = [p for p in players if p not in selected]
    for player in sorted(
        remaining,
        key=lambda p: p["projections"].get(gameweek, {}).get("xP", 0),
        reverse=True,
    ):
        if len(selected) == 11:
            break
        if counts[player["position"]] < maximums[player["position"]]:
            selected.append(player)
            counts[player["position"]] += 1
    if len(selected) != 11:
        raise ValueError("Squad cannot produce a legal starting XI.")
    return selected


def _fpl_squad_analysis(catalog: dict, player_ids: list[int], gameweek: str,
                        bank: float = 0.0) -> dict:
    """Return lineup, captaincy, and budget-aware replacement suggestions."""
    players = catalog["players"]
    selected_ids = set(player_ids)
    selected = [p for p in players if p["player_id"] in selected_ids]
    if not selected:
        raise ValueError("No valid FPL player IDs were supplied.")
    if len(selected) != 15:
        raise ValueError("Enter exactly 15 valid FPL player IDs.")
    xi = _select_legal_fpl_xi(selected, gameweek)
    bench = [p for p in selected if p not in xi]
    ranked_xi = sorted(
        xi, key=lambda p: p["projections"].get(gameweek, {}).get("xP", 0),
        reverse=True,
    )
    captain = ranked_xi[0] if ranked_xi else None
    vice = ranked_xi[1] if len(ranked_xi) > 1 else None
    suggestions = []
    team_counts = {}
    for player in selected:
        team_counts[player["team"]] = team_counts.get(player["team"], 0) + 1
    for current in selected:
        current_xp = current["projections"].get(gameweek, {}).get("xP", 0)
        replacements = [
            candidate for candidate in players
            if candidate["player_id"] not in selected_ids
            and candidate["position"] == current["position"]
            and candidate["price_tenths"] <= current["price_tenths"] + round(bank * 10)
            and (
                candidate["team"] == current["team"]
                or team_counts.get(candidate["team"], 0) < 3
            )
        ]
        replacements.sort(
            key=lambda p: p["projections"].get(gameweek, {}).get("xP", 0) - current_xp,
            reverse=True,
        )
        if replacements:
            target = replacements[0]
            gain = target["projections"].get(gameweek, {}).get("xP", 0) - current_xp
            if gain > 0.2:
                suggestions.append({
                    "sell": current,
                    "buy": target,
                    "gain": round(gain, 2),
                    "cost_delta": round(target["price"] - current["price"], 1),
                })
    suggestions.sort(key=lambda item: item["gain"], reverse=True)
    return _clean({
        "gameweek": int(gameweek),
        "squad": selected,
        "starting_xi": xi,
        "bench": bench,
        "captain": captain,
        "vice_captain": vice,
        "transfers": suggestions[:8],
    })


def prime_fpl_picks() -> dict:
    """Refresh official data and persist fresh Hybrid xP before dashboard launch."""
    snapshot = _fpl_snapshot(force=True)
    data = _build_fpl_picks(snapshot)
    FPL_PICKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FPL_PICKS_CACHE_PATH.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    CACHE["fpl_picks"] = data
    return data


@app.get("/api/fpl/planner")
async def fpl_planner(horizon: int = Query(6, ge=1, le=10)):
    """Return player projections for the next several gameweeks."""
    try:
        cache_key = f"fpl_catalog:{horizon}"
        data = CACHE.get(cache_key)
        if data is None:
            data = _fpl_projection_catalog(_fpl_snapshot(), horizon)
            CACHE[cache_key] = data
        return _clean(data)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"FPL planner unavailable: {exc}") from exc


@app.post("/api/fpl/analyze")
async def fpl_analyze(body: dict):
    """Analyze a manually entered squad or a public FPL manager squad."""
    try:
        catalog = CACHE.get("fpl_catalog:6")
        if catalog is None:
            catalog = _fpl_projection_catalog(_fpl_snapshot(), 6)
            CACHE["fpl_catalog:6"] = catalog
        player_ids = [int(value) for value in body.get("player_ids", [])]
        manager_id = str(body.get("manager_id") or "").strip()
        if manager_id:
            import requests
            event_id = catalog["next_gameweek"]
            response = requests.get(
                f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{event_id}/picks/",
                timeout=15,
            )
            response.raise_for_status()
            player_ids = [pick["element"] for pick in response.json().get("picks", [])]
        if not player_ids:
            raise ValueError("Enter player IDs or a public FPL manager ID.")
        gameweek = str(body.get("gameweek") or catalog["next_gameweek"])
        if gameweek not in {str(item["id"]) for item in catalog["gameweeks"]}:
            raise ValueError("The selected gameweek is outside the projection horizon.")
        return _fpl_squad_analysis(
            catalog, player_ids, gameweek, float(body.get("bank") or 0.0)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not analyze squad: {exc}") from exc


@app.get("/api/fpl/picks")
async def fpl_picks(
    position: Optional[str] = Query(None, pattern="^(GK|DEF|MID|FWD)?$"),
    top: int = Query(50, ge=1, le=590),
):
    """Next-gameweek FPL picks from the official feed plus match-model context."""
    try:
        data = CACHE.get("fpl_picks")
        if data is None:
            data = _build_fpl_picks(_fpl_snapshot())
            CACHE["fpl_picks"] = data
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Official FPL data unavailable: {exc}"
        ) from exc

    players = data["players"]
    if position:
        players = [player for player in players if player["position"] == position]
    return _clean({**data, "players": players[:top]})


@app.post("/api/predict")
async def predict_match(body: dict):
    """Predict a single match: {home_team, away_team}."""
    home = (body.get("home_team") or "").strip()
    away = (body.get("away_team") or "").strip()
    if not home or not away:
        raise HTTPException(
            status_code=400, detail="home_team and away_team are required"
        )
    if home == away:
        raise HTTPException(status_code=400, detail="Teams must differ")

    predictor = _predictor()
    if predictor is None:
        raise HTTPException(
            status_code=404,
            detail="No trained model found. Run `predict-pl train` first.",
        )

    result = predictor.predict(home, away)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return _clean(result)


# ---------------------------------------------------------------------------
# API: season simulation
# ---------------------------------------------------------------------------


@app.get("/api/season")
async def season():
    df = _read_parquet("season_sim.parquet")
    if df is None:
        return {
            "teams": [],
            "message": "No season simulation yet. Run `predict-pl season`.",
        }
    df = df.sort("mean_pts", descending=True)
    return _clean({"teams": df.to_dicts()})


# ---------------------------------------------------------------------------
# API: feature importance
# ---------------------------------------------------------------------------


@app.get("/api/features")
async def features():
    """Feature importance from the trained XGBoost model (cached)."""
    out = _feature_importances()
    if not out:
        return {"features": [], "message": "No trained model."}
    return _clean({"features": out})


@app.get("/api/teams")
async def teams():
    return {"teams": _teams()}


@app.get("/api/seasons")
async def seasons():
    """Available seasons for match and player/shot data."""
    df = _read_parquet("matches.parquet")
    match_seasons = (
        sorted(df["season"].drop_nulls().unique().to_list()) if df is not None else []
    )
    player_seasons = []
    agg = _player_agg()
    if agg is not None:
        player_seasons = sorted(agg["season"].drop_nulls().unique().to_list())
    return {"match_seasons": match_seasons, "player_seasons": player_seasons}


# ---------------------------------------------------------------------------
# API: model performance (walk-forward diagnostics)
# ---------------------------------------------------------------------------


@app.get("/api/performance")
async def performance():
    df = _read_parquet("walkforward_preds.parquet")
    if df is None or df.is_empty():
        return {
            "message": "No walk-forward predictions yet. Run `predict-pl backtest`."
        }

    import numpy as np

    def rps_vec(probs, y):
        out = []
        for p, yy in zip(probs, y):
            cp = np.cumsum(p)[:-1]
            ct = np.cumsum(np.eye(3)[yy])[:-1]
            out.append(float(np.sum((cp - ct) ** 2) / 2))
        return np.array(out)

    labels = {"H": 0, "D": 1, "A": 2}
    y = np.array([labels[r] for r in df["result"].to_list()])
    ens = np.stack([df["ens_h"], df["ens_d"], df["ens_a"]], axis=1).astype(float)
    dc = np.stack([df["dc_h"], df["dc_d"], df["dc_a"]], axis=1).astype(float)

    ens_rps = rps_vec(ens, y)
    dc_rps = rps_vec(dc, y)
    ens_pred = ens.argmax(1)
    dc_pred = dc.argmax(1)

    by_season = []
    seasons = sorted(df["season"].unique().to_list())
    season_list = df["season"].to_list()
    for season in seasons:
        idx = [i for i, s in enumerate(season_list) if s == season]
        gy = y[idx]
        g_ens = ens[idx]
        g_dc = dc[idx]
        if len(idx) == 0:
            continue
        by_season.append(
            {
                "season": season,
                "n": len(idx),
                "ens_acc": float(np.mean(ens_pred[idx] == gy)),
                "dc_acc": float(np.mean(dc_pred[idx] == gy)),
                "ens_rps": float(np.mean(ens_rps[idx])),
                "dc_rps": float(np.mean(dc_rps[idx])),
                "ens_brier": float(
                    np.mean(np.sum((g_ens - np.eye(3)[gy]) ** 2, axis=1))
                ),
                "dc_brier": float(np.mean(np.sum((g_dc - np.eye(3)[gy]) ** 2, axis=1))),
            }
        )

    # Calibration curve on max-probability
    pmax = ens.max(axis=1)
    edges = np.linspace(0, 1, 11)
    cal = []
    for i in range(10):
        mask = (pmax > edges[i]) & (pmax <= edges[i + 1])
        if mask.sum() > 0:
            cal.append(
                [float(pmax[mask].mean()), float((ens_pred[mask] == y[mask]).mean())]
            )

    # Worst predictions
    wrong = []
    for i in range(len(y)):
        if ens_pred[i] == y[i]:
            continue
        prob = {0: "H", 1: "D", 2: "A"}[ens_pred[i]]
        act = {0: "H", 1: "D", 2: "A"}[y[i]]
        wrong.append(
            {
                "date": str(df["date"][i])[:10],
                "season": df["season"][i],
                "home_team": df["home_team"][i],
                "away_team": df["away_team"][i],
                "result": act,
                "ph": round(float(ens[i, 0]), 3),
                "pd": round(float(ens[i, 1]), 3),
                "pa": round(float(ens[i, 2]), 3),
                "predicted": prob,
                "conf": round(float(ens[i, ens_pred[i]]), 3),
                "p_actual": round(float(ens[i, y[i]]), 3),
                "rps": round(float(ens_rps[i]), 4),
            }
        )
    wrong.sort(key=lambda r: -r["conf"])

    hist = np.histogram(ens_rps, bins=20, range=(0, np.quantile(ens_rps, 0.99)))
    response = {
            "n_matches": len(y),
            "ens_acc": float(np.mean(ens_pred == y)),
            "dc_acc": float(np.mean(dc_pred == y)),
            "ens_rps": float(np.mean(ens_rps)),
            "dc_rps": float(np.mean(dc_rps)),
            "by_season": by_season,
            "calibration": cal,
            "rps_hist": [[float(v) for v in hist[1][:-1]], [float(v) for v in hist[0]]],
            "worst": wrong[:20],
        }
    rolling = _read_parquet("rolling_evaluation.parquet")
    if rolling is not None and not rolling.is_empty():
        response["rolling_evaluation"] = {
            "folds": rolling.to_dicts(),
            "model": "Dixon-Coles rolling",
            "n_matches": int(rolling["n_matches"].sum()),
            "rps": float(
                np.average(rolling["dc_rps"].to_numpy(), weights=rolling["n_matches"].to_numpy())
            ),
            "accuracy": float(
                np.average(
                    rolling["dc_accuracy"].to_numpy(),
                    weights=rolling["n_matches"].to_numpy(),
                )
            ),
        }
    return _clean(response)


@app.get("/api/model-comparison")
async def model_comparison():
    """Return rolling model comparison metrics when the report exists."""
    df = _read_parquet("rolling_model_comparison.parquet")
    if df is None or df.is_empty():
        return {
            "models": [],
            "message": "No rolling model comparison yet. Run `pl-predict compare-models`.",
        }
    import numpy as np

    labels = {"H": 0, "D": 1, "A": 2}
    y = np.asarray([labels[result] for result in df["result"].to_list()])
    from pl_predict.evaluation.metrics import evaluate_predictions, expected_calibration_error

    models = []
    for model in ("dc", "elo", "xgb"):
        columns = [f"{model}_{outcome}" for outcome in ("h", "d", "a")]
        if not all(column in df.columns for column in columns):
            continue
        probs = df.select(columns).to_numpy().astype(float)
        probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        metrics = evaluate_predictions(y, probs)
        metrics["ece"] = expected_calibration_error(
            (probs.argmax(axis=1) == y).astype(int), probs.max(axis=1)
        )
        models.append({"model": model, **metrics})
    return _clean({"n_matches": len(y), "models": models})


# ---------------------------------------------------------------------------
# API: player stats
# ---------------------------------------------------------------------------


@app.get("/api/players")
async def players(
    season: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    stat: str = Query(
        "xG", pattern="^(xG|xA|npxG|minutes|goals|assists|shots|fantasy_pts)$"
    ),
    top: int = Query(50, ge=1, le=200),
    all: bool = Query(False),
):
    """Player leaderboard from per-match aggregates, with FPL-style points."""
    agg = _player_agg()
    if agg is None:
        return {"players": [], "message": "No player data yet."}
    import polars as pl

    pm = _player_matches_clean()
    ycols = {}
    if pm is not None:
        yr = pm.group_by(["player_id"]).agg(
            pl.col("yellow_card").sum().alias("yellow_cards"),
            pl.col("red_card").sum().alias("red_cards"),
        )
        ycols = {r["player_id"]: r for r in yr.to_dicts()}

    df = agg
    if season:
        df = df.filter(pl.col("season") == season)
    if team:
        df = df.filter(pl.col("team") == team)
    if position:
        pos_upper = position.upper()
        df = df.filter(
            pl.col("position").str.to_uppercase().str.contains(pos_upper, literal=True)
        )
    if q:
        df = df.filter(pl.col("player_name").str.contains(q, literal=False, case=False))
    df = df.filter(pl.col("minutes") >= 90)

    rows = df.select(
        [
            "player_id",
            "player_name",
            "team",
            "season",
            "position",
            "minutes",
            "games_90",
            "xG",
            "xA",
            "npxG",
            "goals",
            "assists",
            "shots",
            "key_passes",
            "matches",
        ]
    ).to_dicts()

    for r in rows:
        pos = (r.get("position") or "").upper()
        goals = r.get("goals") or 0
        assists = r.get("assists") or 0
        minutes = r.get("minutes") or 0
        matches = r.get("matches") or 1
        yc = ycols.get(r["player_id"], {}).get("yellow_cards", 0) if ycols else 0
        rc = ycols.get(r["player_id"], {}).get("red_cards", 0) if ycols else 0

        if "F" in pos:
            pts = (
                goals * 4
                + assists * 3
                + (minutes >= 60 and matches > 0) * 0
                - yc * 1
                - rc * 3
            )
        elif "M" in pos:
            pts = goals * 5 + assists * 3 - yc * 1 - rc * 3
        elif "D" in pos or "B" in pos:
            pts = goals * 6 + assists * 3 - yc * 1 - rc * 3
        else:
            pts = goals * 6 + assists * 3 - yc * 1 - rc * 3

        r["fantasy_pts"] = round(pts, 1)

    if stat == "fantasy_pts":
        rows.sort(key=lambda x: x.get("fantasy_pts", 0) or 0, reverse=True)
    else:
        rows.sort(key=lambda x: x.get(stat, 0) or 0, reverse=True)

    if not all:
        rows = rows[:top]

    return _clean({"players": rows})


@app.get("/api/players/{player_id}")
async def player_detail(player_id: str):
    """Per-match history for one player."""
    import polars as pl

    df = _player_matches_clean()
    if df is None:
        raise HTTPException(status_code=404, detail="No player data.")
    df = (
        df.filter(pl.col("player_id").cast(pl.String, strict=False) == player_id)
        .select(
            [
                "season",
                "team",
                "date",
                "minutes",
                "goals",
                "assists",
                "xG",
                "xA",
                "shots",
                "key_passes",
                "yellow_card",
                "red_card",
            ]
        )
        .sort("date")
    )
    if df.is_empty():
        raise HTTPException(status_code=404, detail="Player not found.")
    return _clean({"player_id": player_id, "matches": df.to_dicts()})


# ---------------------------------------------------------------------------
# API: shot maps
# ---------------------------------------------------------------------------


@app.get("/api/shots")
async def shots(
    season: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    player: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
):
    """Shot locations (X/Y 0-100) with result + xG, filtered."""
    df = _shots()
    if df is None:
        return {"shots": [], "message": "No shot data yet."}
    import polars as pl

    out = df
    if season:
        out = out.filter(pl.col("season") == season)
    if team:
        out = out.filter(pl.col("team") == team)
    if player:
        out = out.filter(pl.col("player_name").str.contains(player, literal=True))
    out = out.head(limit).select(
        [
            "season",
            "date",
            "team",
            "player_name",
            "minute",
            "result",
            "situation",
            "shotType",
            "assisted_by",
            "xG",
            "X",
            "Y",
            "h_a",
        ]
    )
    return _clean({"shots": out.to_dicts()})


# ---------------------------------------------------------------------------
# API: team analysis
# ---------------------------------------------------------------------------


@app.get("/api/team/{name}")
async def team_analysis(name: str):
    """Form trend, home/away splits, goals + attendance stats for a team."""
    import polars as pl
    from urllib.parse import unquote

    team = unquote(name)
    df = _read_parquet("matches.parquet")
    if df is None:
        raise HTTPException(status_code=404, detail="No match data.")
    home = (
        df.filter(pl.col("home_team") == team)
        .select(
            [
                pl.col("date"),
                pl.col("season"),
                pl.lit(team).alias("home_team"),
                pl.col("away_team"),
                pl.col("home_goals").alias("gf"),
                pl.col("away_goals").alias("ga"),
                pl.col("result"),
                pl.col("xG_home").alias("xg_f"),
                pl.col("xG_away").alias("xg_a"),
                pl.col("Attendance"),
            ]
        )
        .with_columns(pl.lit("home").alias("venue"))
    )
    away = (
        df.filter(pl.col("away_team") == team)
        .select(
            [
                pl.col("date"),
                pl.col("season"),
                pl.col("home_team"),
                pl.lit(team).alias("away_team"),
                pl.col("away_goals").alias("gf"),
                pl.col("home_goals").alias("ga"),
                pl.col("result"),
                pl.col("xG_away").alias("xg_f"),
                pl.col("xG_home").alias("xg_a"),
                pl.col("Attendance"),
            ]
        )
        .with_columns(pl.lit("away").alias("venue"))
    )
    team_matches = pl.concat([home, away]).sort("date")
    if team_matches.is_empty():
        raise HTTPException(status_code=404, detail="Team not found.")
    with_venue = team_matches
    rec = with_venue.filter(pl.col("gf").is_finite())
    if rec.is_empty():
        return _clean(
            {
                "name": team,
                "recent": [],
                "splits": {},
                "goals_hist": [],
                "attendance": [],
            }
        )

    recent = (
        rec.tail(20)
        .select(
            [
                "date",
                "season",
                "home_team",
                "away_team",
                "gf",
                "ga",
                "venue",
                "result",
                "xg_f",
                "xg_a",
            ]
        )
        .to_dicts()
    )
    for r in recent:
        r["date"] = str(r["date"])[:10]

    splits = {}
    for venue in ("home", "away"):
        v = rec.filter(pl.col("venue") == venue)
        if v.is_empty():
            continue
        splits[venue] = {
            "n": len(v),
            "w": int((v["result"] == "H").sum())
            if venue == "home"
            else int((v["result"] == "A").sum()),
            "d": int((v["result"] == "D").sum()),
            "l": len(v)
            - int(
                (v["result"] == "H").sum()
                if venue == "home"
                else (v["result"] == "A").sum()
            )
            - int((v["result"] == "D").sum()),
            "avg_gf": float(v["gf"].mean()),
            "avg_ga": float(v["ga"].mean()),
        }

    goals_hist = (
        rec.with_columns((pl.col("gf") + pl.col("ga")).alias("total_goals"))
        .group_by("total_goals")
        .len()
        .sort("total_goals")
        .to_dicts()
    )

    attendance = []
    att = (
        rec.select(["season", "Attendance"])
        .with_columns(
            pl.when(pl.col("Attendance").str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col("Attendance").cast(pl.Float64, strict=False))
            .alias("att_num")
        )
        .group_by("season")
        .agg(pl.col("att_num").mean().alias("avg_attendance"))
        .sort("season")
    )
    for r in att.to_dicts():
        attendance.append(
            {"season": r["season"], "avg_attendance": r["avg_attendance"]}
        )

    for r in recent:
        r["pts"] = {"H": 3, "D": 1, "A": 0}.get(r["result"], 0)

    return _clean(
        {
            "name": team,
            "recent": recent,
            "splits": splits,
            "goals_hist": goals_hist,
            "attendance": attendance,
        }
    )


# ---------------------------------------------------------------------------
# API: head-to-head
# ---------------------------------------------------------------------------


@app.get("/api/h2h")
async def h2h(
    home: str = Query(...),
    away: str = Query(...),
    limit: int = Query(10, ge=1, le=30),
):
    """Head-to-head history + current model prediction."""
    import polars as pl

    df = _read_parquet("matches.parquet")
    if df is None:
        raise HTTPException(status_code=404, detail="No match data.")
    hist = (
        df.filter(
            ((pl.col("home_team") == home) & (pl.col("away_team") == away))
            | ((pl.col("home_team") == away) & (pl.col("away_team") == home))
        )
        .filter(pl.col("result").is_not_null())
        .sort("date", descending=True)
        .select(
            [
                "date",
                "season",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
            ]
        )
        .head(limit)
    )
    if hist.is_empty():
        raise HTTPException(status_code=404, detail="No head-to-head history found.")

    prediction = None
    predictor = _predictor()
    if predictor is not None:
        prediction = _cached_prediction(home, away)
    return _clean({"history": hist.to_dicts(), "prediction": prediction})


# ---------------------------------------------------------------------------
# API: referee analytics
# ---------------------------------------------------------------------------


@app.get("/api/referees")
async def referees(min_matches: int = Query(30, ge=1, le=500)):
    """Per-referee aggregate stats (goals, home bias, attendance)."""
    import polars as pl

    df = _read_parquet("matches.parquet")
    if df is None or "Referee" not in df.columns:
        return {"referees": [], "message": "No referee data."}
    played = df.filter(
        pl.col("home_goals").is_finite()
        & pl.col("result").is_not_null()
        & pl.col("Referee").is_not_null()
        & (pl.col("Referee").str.strip_chars() != "")
    )
    out = (
        played.with_columns(
            pl.when(pl.col("Attendance").str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col("Attendance").cast(pl.Float64, strict=False))
            .alias("att_num")
        )
        .group_by("Referee")
        .agg(
            pl.len().alias("n"),
            ((pl.col("home_goals") + pl.col("away_goals")).mean()).alias("avg_goals"),
            (pl.col("home_goals").mean()).alias("avg_home_goals"),
            (pl.col("away_goals").mean()).alias("avg_away_goals"),
            (pl.col("result") == "H").mean().alias("home_win_rate"),
            (pl.col("result") == "D").mean().alias("draw_rate"),
            (pl.col("att_num").mean()).alias("avg_attendance"),
        )
        .filter(pl.col("n") >= min_matches)
        .sort("n", descending=True)
        .head(40)
        .to_dicts()
    )
    return _clean({"referees": out})


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    open_browser: bool = False,
):
    print(f"\n  PL-Predict Dashboard: http://{host}:{port}")
    print("  Ctrl+C to stop\n")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="info", reload=reload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    run_dashboard(args.host, args.port, args.reload)
