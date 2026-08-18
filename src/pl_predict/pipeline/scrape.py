"""Data ingestion: scrape/load from all free data sources directly (no pandas/soccerdata dependency)."""

from pathlib import Path
from typing import Optional
import re
import csv
import time
from io import StringIO
from datetime import datetime

import polars as pl
import requests
from bs4 import BeautifulSoup

from .utils import load_config, team_name_normalise, resolve_path


# ---------------------------------------------------------------------------
# 1. FBref — scrape match results and basic stats from HTML tables
# ---------------------------------------------------------------------------

FBREF_URL = (
    "https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Schedule"
)


def _parse_fbref_row(row) -> dict | None:
    """Parse a single row from the FBref schedule table."""
    cells = row.find_all("td")
    if len(cells) < 10:
        return None

    # Find column indices by header
    cols = {
        th.get("data-stat", ""): i
        for i, th in enumerate(row.parent.parent.find("thead").find_all("th"))
        if th.get("data-stat")
    }

    def get(key):
        i = cols.get(key)
        if i is not None and i < len(cells):
            return cells[i].get_text(strip=True) or None
        return None

    date_str = get("date")
    if not date_str:
        return None

    home = get("home_team")
    away = get("away_team")
    score_str = get("score")
    if not home or not away or not score_str or score_str == "":
        return None

    score_parts = score_str.split("–")
    if len(score_parts) != 2:
        score_parts = score_str.split("-")
    if len(score_parts) != 2:
        return None

    try:
        hg = float(score_parts[0].strip())
        ag = float(score_parts[1].strip())
    except (ValueError, TypeError):
        return None

    xg_h = get("xg_home")
    xg_a = get("xg_away")

    return {
        "date": date_str,
        "home_team": team_name_normalise(home),
        "away_team": team_name_normalise(away),
        "home_goals": hg,
        "away_goals": ag,
        "xG_home": float(xg_h) if xg_h and xg_h != "" else None,
        "xG_away": float(xg_a) if xg_a and xg_a != "" else None,
    }


def scrape_fbref(
    seasons: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
) -> pl.DataFrame:
    """Scrape match data from FBref HTML tables."""
    cfg = load_config("leagues")
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["fbref_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    seasons = seasons or cfg["fbref"]["seasons"]
    all_rows = []

    for season in seasons:
        url = FBREF_URL.format(season=season)
        print(f"  FBref {season} ...")
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            table = soup.find("table", id=lambda x: x and "sched" in x.lower())
            if not table:
                print("    No schedule table found")
                continue
            for tr in table.find("tbody").find_all("tr"):
                parsed = _parse_fbref_row(tr)
                if parsed:
                    parsed["season"] = season
                    all_rows.append(parsed)
        except Exception as e:
            print(f"    WARN: {e}")

    if not all_rows:
        return pl.DataFrame()

    df = pl.DataFrame(all_rows)
    out_path = Path(out_dir) / "fbref_matches.parquet"
    df.write_parquet(str(out_path))
    print(f"  -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 2. Understat — xG + player data via the JSON API
# ---------------------------------------------------------------------------

UNDERSTAT_URL = "https://understat.com/league/EPL/{season}"
UNDERSTAT_API = "https://understat.com/main/getLeagueData/EPL/{season}"

_UNDERSTAT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}


def _understat_team_name(name: str) -> str:
    """Understat full names -> model canonical names (e.g. 'Sheffield United')."""
    return team_name_normalise((name or "").strip())


def _fetch_understat_season(season: str) -> dict:
    """Return parsed JSON for one Understat season via the JSON API."""
    import json

    resp = requests.get(
        UNDERSTAT_API.format(season=season), headers=_UNDERSTAT_HEADERS, timeout=30
    )
    resp.raise_for_status()
    return json.loads(resp.text)


def scrape_understat(
    seasons: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
) -> pl.DataFrame:
    """Scrape per-match xG + team history + player season stats from Understat.

    Understat no longer embeds the data in the league page; the JSON API at
    ``/main/getLeagueData/EPL/{season}`` returns teams (per-match history),
    players (season stats) and dates (the 380 fixtures with xG + their own
    pre-match forecast). Writes three parquet files under the raw dir.
    """
    cfg = load_config("leagues")
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["understat_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    seasons = seasons or cfg["understat"]["seasons"]

    match_rows, team_rows, player_rows = [], [], []
    for season in seasons:
        print(f"  Understat {season} ...")
        try:
            data = _fetch_understat_season(season)
        except Exception as e:
            print(f"    WARN: {e}")
            continue

        # --- dates: 380 fixtures, per-match xG + forecast ---
        for m in data.get("dates", []):
            h = m.get("h", {})
            a = m.get("a", {})
            match_rows.append(
                {
                    "date": m.get("datetime"),
                    "home_team": _understat_team_name(h.get("title")),
                    "away_team": _understat_team_name(a.get("title")),
                    "home_goals": m.get("goals", {}).get("h"),
                    "away_goals": m.get("goals", {}).get("a"),
                    "xG_home": m.get("xG", {}).get("h"),
                    "xG_away": m.get("xG", {}).get("a"),
                    "us_forecast_home": m.get("forecast", {}).get("w"),
                    "us_forecast_draw": m.get("forecast", {}).get("d"),
                    "us_forecast_away": m.get("forecast", {}).get("l"),
                    "season": season,
                }
            )

        # --- teams: per-team per-match advanced history ---
        for tid, team in data.get("teams", {}).items():
            title = _understat_team_name(team.get("title"))
            for hist in team.get("history", []):
                ppda = hist.get("ppda") or {}
                ppda_a = hist.get("ppda_allowed") or {}
                team_rows.append(
                    {
                        "date": hist.get("date"),
                        "team": title,
                        "h_a": hist.get("h_a"),
                        "xG": hist.get("xG"),
                        "xGA": hist.get("xGA"),
                        "npxG": hist.get("npxG"),
                        "npxGA": hist.get("npxGA"),
                        "ppda_att": ppda.get("att"),
                        "ppda_def": ppda.get("def"),
                        "ppda_a_att": ppda_a.get("att"),
                        "ppda_a_def": ppda_a.get("def"),
                        "deep": hist.get("deep"),
                        "deep_allowed": hist.get("deep_allowed"),
                        "xpts": hist.get("xpts"),
                        "result": hist.get("result"),
                        "scored": hist.get("scored"),
                        "missed": hist.get("missed"),
                        "pts": hist.get("pts"),
                        "npxGD": hist.get("npxGD"),
                        "season": season,
                    }
                )

        # --- players: season totals per player ---
        for p in data.get("players", []):
            player_rows.append(
                {
                    "player_id": p.get("id"),
                    "player_name": p.get("player_name"),
                    "team": _understat_team_name(p.get("team_title")),
                    "position": p.get("position"),
                    "games": p.get("games"),
                    "time": p.get("time"),
                    "goals": p.get("goals"),
                    "xG": p.get("xG"),
                    "assists": p.get("assists"),
                    "xA": p.get("xA"),
                    "shots": p.get("shots"),
                    "key_passes": p.get("key_passes"),
                    "npg": p.get("npg"),
                    "npxG": p.get("npxG"),
                    "xGChain": p.get("xGChain"),
                    "xGBuildup": p.get("xGBuildup"),
                    "season": season,
                }
            )
        print(
            f"    {len(data.get('dates', []))} matches, "
            f"{len(data.get('players', []))} players"
        )

    if not match_rows:
        return pl.DataFrame()

    pl.DataFrame(match_rows).write_parquet(
        str(Path(out_dir) / "understat_matches.parquet")
    )
    if team_rows:
        pl.DataFrame(team_rows).write_parquet(
            str(Path(out_dir) / "understat_team_history.parquet")
        )
    if player_rows:
        pl.DataFrame(player_rows).write_parquet(
            str(Path(out_dir) / "understat_players.parquet")
        )

    df = pl.DataFrame(match_rows)
    print(f"  -> {out_dir}/understat_matches.parquet  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 2b. Understat — per-match player rosters + shots (getMatchData, resumable)
# ---------------------------------------------------------------------------

UNDERSTAT_MATCH_API = "https://understat.com/main/getMatchData/{mid}"


def _load_understat_match_ids(seasons: Optional[list[str]] = None) -> list[dict]:
    """Collect {match_id, date, home_team, away_team, season} for every match."""
    cfg = load_config("leagues")
    seasons = seasons or cfg["understat"]["seasons"]
    ids = []
    for season in seasons:
        try:
            data = _fetch_understat_season(season)
        except Exception as e:
            print(f"    WARN: {e}")
            continue
        for m in data.get("dates", []):
            ids.append(
                {
                    "match_id": str(m.get("id")),
                    "date": m.get("datetime"),
                    "home_team": _understat_team_name(m.get("h", {}).get("title")),
                    "away_team": _understat_team_name(m.get("a", {}).get("title")),
                    "season": season,
                }
            )
    return ids


def _fetch_match_data(mid: str, max_retries: int = 3) -> Optional[dict]:
    """Fetch one Understat match payload with retries (thread-safe)."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                UNDERSTAT_MATCH_API.format(mid=mid),
                headers=_UNDERSTAT_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def scrape_understat_player_matches(
    seasons: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
    delay: float = 0.15,
    max_retries: int = 3,
    workers: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch per-match player rosters + shots for all matches (resumable).

    Uses a small thread pool (Understat serves match JSON without hard rate
    limits); resumes from already-scraped ``match_id``s so an interrupted run
    can simply be re-launched. Returns (player_frame, shots_frame).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["understat_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    player_path = Path(out_dir) / "understat_player_matches.parquet"
    shots_path = Path(out_dir) / "understat_shots.parquet"

    match_ids = _load_understat_match_ids(seasons)
    done = set()
    if player_path.exists():
        try:
            done = set(pl.read_parquet(str(player_path))["match_id"].to_list())
        except Exception:
            pass

    todo = [m for m in match_ids if m["match_id"] not in done]
    print(
        f"  Understat per-match: {len(match_ids)} total, {len(todo)} to fetch "
        f"({len(done)} already done)"
    )

    player_rows, shot_rows = [], []
    fetched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_match_data, m["match_id"]): m for m in todo}
        for fut in as_completed(futures):
            m = futures[fut]
            data = fut.result()
            if data is None:
                print(
                    f"    WARN: failed match {m['match_id']} {m['home_team']} v {m['away_team']}"
                )
                continue

            h_name = m["home_team"]
            a_name = m["away_team"]
            for side, team_name in (("h", h_name), ("a", a_name)):
                rosters = data.get("rosters", {}).get(side, {})
                if not isinstance(rosters, dict):
                    continue
                for pid, ps in rosters.items():
                    player_rows.append(
                        {
                            "match_id": m["match_id"],
                            "season": m["season"],
                            "date": m["date"],
                            "h_a": side,
                            "team": team_name,
                            "player_id": ps.get("player_id"),
                            "player_name": ps.get("player"),
                            "position": ps.get("position"),
                            "minutes": ps.get("time"),
                            "goals": ps.get("goals"),
                            "own_goals": ps.get("own_goals"),
                            "shots": ps.get("shots"),
                            "xG": ps.get("xG"),
                            "assists": ps.get("assists"),
                            "xA": ps.get("xA"),
                            "key_passes": ps.get("key_passes"),
                            "xGChain": ps.get("xGChain"),
                            "xGBuildup": ps.get("xGBuildup"),
                            "yellow_card": ps.get("yellow_card"),
                            "red_card": ps.get("red_card"),
                        }
                    )

            shots = data.get("shots", {}).get("h", []) + data.get("shots", {}).get(
                "a", []
            )
            for s in shots:
                shot_rows.append(
                    {
                        "match_id": m["match_id"],
                        "season": m["season"],
                        "date": m["date"],
                        "player_id": s.get("player_id"),
                        "player_name": s.get("player"),
                        "h_a": s.get("h_a"),
                        "minute": s.get("minute"),
                        "result": s.get("result"),
                        "xG": s.get("xG"),
                        "situation": s.get("situation"),
                        "shotType": s.get("shotType"),
                        "lastAction": s.get("lastAction"),
                        "assisted_by": s.get("player_assisted"),
                        "X": s.get("X"),
                        "Y": s.get("Y"),
                    }
                )

            fetched += 1
            if fetched % 300 == 0:
                print(f"    {fetched}/{len(todo)} fetched")
            time.sleep(delay)

    print(
        f"  fetched {fetched} matches, {len(player_rows)} player rows, "
        f"{len(shot_rows)} shots"
    )

    _FLOAT_COLS = [
        "minutes",
        "goals",
        "own_goals",
        "shots",
        "xG",
        "assists",
        "xA",
        "key_passes",
        "xGChain",
        "xGBuildup",
    ]
    player_df = pl.DataFrame(player_rows) if player_rows else pl.DataFrame()
    shot_df = pl.DataFrame(shot_rows) if shot_rows else pl.DataFrame()
    for df, cols in ((player_df, _FLOAT_COLS), (shot_df, ["minute", "xG", "X", "Y"])):
        for c in cols:
            if c in df.columns:
                df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))

    # Append to existing cache if present (resume-safe)
    if player_path.exists() and len(player_df):
        player_df = pl.concat(
            [pl.read_parquet(str(player_path)), player_df], how="diagonal_relaxed"
        )
    if shots_path.exists() and len(shot_df):
        shot_df = pl.concat(
            [pl.read_parquet(str(shots_path)), shot_df], how="diagonal_relaxed"
        )
    if len(player_df):
        player_df.write_parquet(str(player_path))
    if len(shot_df):
        shot_df.write_parquet(str(shots_path))
    print(
        f"  -> {player_path} ({len(player_df)} rows), {shots_path} ({len(shot_df)} rows)"
    )
    return player_df, shot_df


# ---------------------------------------------------------------------------
# 2c. Transfermarkt — squad market values (retries, resumable)
# ---------------------------------------------------------------------------

TM_HOME = "https://www.transfermarkt.us"
TM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Transfermarkt team ids for clubs that appear in the modern PL (fallback map)
TM_TEAM_IDS = {
    "Arsenal": 11,
    "Aston Villa": 405,
    "Bournemouth": 989,
    "Brentford": 1148,
    "Brighton": 1237,
    "Burnley": 1000,
    "Chelsea": 631,
    "Crystal Palace": 873,
    "Everton": 29,
    "Fulham": 931,
    "Huddersfield": 2077,
    "Hull": 300,
    "Leeds": 399,
    "Leicester": 1003,
    "Liverpool": 31,
    "Luton": 1062,
    "Man City": 281,
    "Man United": 985,
    "Middlesbrough": 489,
    "Newcastle": 762,
    "Norwich": 1122,
    "Nott'm Forest": 703,
    "QPR": 1138,
    "Sheffield Utd": 1028,
    "Southampton": 180,
    "Stoke": 1116,
    "Sunderland": 289,
    "Swansea": 1082,
    "Tottenham": 148,
    "Watford": 1010,
    "West Brom": 984,
    "West Ham": 379,
    "Wolves": 543,
}


def _tm_get(url: str, max_retries: int = 5) -> Optional[str]:
    """GET with retries/backoff for Transfermarkt's flaky 503/timeouts."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=TM_HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        time.sleep(2 + attempt * 2)  # polite backoff
    return None


def _tm_team_name(raw: str) -> str:
    """Transfermarkt club name -> canonical team name."""
    name = re.sub(r"\s*FC\s*$", "", (raw or "").strip())
    return team_name_normalise(name)


def _discover_tm_clubs(seasons: list[str]) -> dict:
    """Discover {team: (verein_id, slug)} from PL league pages across seasons."""
    clubs: dict = {}
    for season in seasons:
        url = f"{TM_HOME}/premier-league/startseite/wettbewerb/GB1/saison_id/{season}"
        text = _tm_get(url)
        if not text:
            print(f"    WARN: league page {season} failed")
            continue
        for slug, vid, raw_name in re.findall(
            r'href="(/[^"]+?)/startseite/verein/(\d+)/[^"]*"[^>]*>([^<]{2,40})<',
            text,
        ):
            team = _tm_team_name(raw_name)
            if team:
                clubs[team] = (vid, slug)
        time.sleep(1.0)
    print(f"  Transfermarkt: discovered {len(clubs)} clubs from league pages")
    return clubs


def _normalise_mv(text: str) -> Optional[float]:
    """'€200.5m' / '€1.5bn' / '€45.6k' -> euros (float)."""
    if not text:
        return None
    t = text.replace("€", "").replace("\u20ac", "").strip()
    t = t.replace(",", ".")
    multiplier = 1.0
    if "bn" in t.lower():
        multiplier = 1e9
        t = t.lower().replace("bn", "").strip()
    elif "m" in t.lower():
        multiplier = 1e6
        t = t.lower().replace("m", "").strip()
    elif "k" in t.lower():
        multiplier = 1e3
        t = t.lower().replace("k", "").strip()
    t = t.replace(" ", "").strip()
    try:
        return float(t) * multiplier
    except (ValueError, TypeError):
        return None


def scrape_transfermarkt(
    seasons: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
    delay: float = 1.0,
) -> pl.DataFrame:
    """Scrape squad market value / size / average age per team per season.

    Club ids are discovered from the per-season PL league pages (robust) with
    a curated map as fallback. Resumable: (team, season) pairs already cached
    are skipped. Squad value reflects the squad as of that season and is used
    as a LAGGED feature.
    """
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or Path(paths["raw_dir"]) / "transfermarkt")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "transfermarkt_squads.parquet"

    seasons = seasons or [str(y) for y in range(2015, 2026)]
    clubs = _discover_tm_clubs(seasons)
    # Fall back to the curated id map for clubs not on any league page in window
    for team, vid in TM_TEAM_IDS.items():
        clubs.setdefault(team, (vid, f"/{team.lower().replace(' ', '-')}"))

    rows = []
    if out_path.exists():
        try:
            rows = pl.read_parquet(str(out_path)).to_dicts()
        except Exception:
            rows = []

    done_keys = {(r["team"], r["season"]) for r in rows}
    todo = []
    for season in seasons:
        for team in clubs:
            if (team, season) not in done_keys:
                todo.append((team, season))
    print(f"  Transfermarkt: {len(todo)} (team, season) pages to fetch")

    for i, (team, season) in enumerate(todo):
        vid, slug = clubs.get(team, (TM_TEAM_IDS.get(team), ""))
        record = {
            "team": team,
            "season": season,
            "squad_value": None,
            "squad_size": None,
            "avg_age": None,
        }
        if vid:
            url = (
                f"{TM_HOME}{slug or '/' + team.lower().replace(' ', '-')}"
                f"/kader/verein/{vid}/saison_id/{season}"
            )
            text = _tm_get(url)
            if text:
                record.update(_parse_tm_squad(text))
        rows.append(record)
        if (i + 1) % 30 == 0:
            pl.DataFrame(rows).write_parquet(str(out_path))
            print(f"    {i + 1}/{len(todo)} done (cached)")
        time.sleep(delay)

    df = pl.DataFrame(rows)
    df.write_parquet(str(out_path))
    print(f"  -> {out_path} ({len(df)} rows)")
    return df


def _parse_tm_squad(text: str) -> dict:
    """Extract total market value, squad size, avg age from a Transfermarkt page."""
    out = {"squad_value": None, "squad_size": None, "avg_age": None}

    # Squad size & average age live in the header details list
    m = re.search(
        r"Squad size:\s*<span[^>]*>\s*(\d+)\s*</span>.*?"
        r"Average age:\s*<span[^>]*>\s*([\d.]+)\s*</span>",
        text,
        re.DOTALL,
    )
    if m:
        out["squad_size"] = int(m.group(1))
        out["avg_age"] = float(m.group(2))
    else:
        m = re.search(r"Squad size:\s*<span[^>]*>\s*(\d+)\s*</span>", text)
        if m:
            out["squad_size"] = int(m.group(1))

    # Total market value: <span class="waehrung">€</span>1.41<span class="waehrung">bn</span>
    m = re.search(
        r'<span class="waehrung">\u20ac?</span>([\d.]+)\s*'
        r'<span class="waehrung">([a-zA-Z]+)?</span>',
        text,
    )
    if m:
        unit = m.group(2) or ""
        out["squad_value"] = _normalise_mv(f"{m.group(1)}{unit}")
    return out


# ---------------------------------------------------------------------------
# 3. Football-Data.co.uk — CSV downloads
# ---------------------------------------------------------------------------

_COL_MAP = {
    "Div": "division",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_ht_goals",
    "HTAG": "away_ht_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_ontarget",
    "AST": "away_shots_ontarget",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
    "PSCAH": "odds_close_home",
    "PSCAD": "odds_close_draw",
    "PSCAA": "odds_close_away",
    "MaxH": "odds_max_home",
    "MaxD": "odds_max_draw",
    "MaxA": "odds_max_away",
    "AvgH": "odds_avg_home",
    "AvgD": "odds_avg_draw",
    "AvgA": "odds_avg_away",
}


def scrape_football_data(out_dir: Optional[str] = None) -> pl.DataFrame:
    cfg = load_config("leagues")
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["football_data_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    season_ids = cfg["football_data"]["season_ids"]
    url_template = cfg["football_data"]["url_template"]

    all_records = []
    for sid in season_ids:
        url = url_template.format(season_id=sid)
        print(f"  Football-Data E0 {sid} ...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.text.lstrip("\ufeff")
            reader = csv.DictReader(StringIO(text))
            for row in reader:
                mapped = {
                    _COL_MAP.get(k, k): v for k, v in row.items() if k is not None
                }
                mapped["season_id"] = sid
                # Normalise teams
                mapped["home_team"] = team_name_normalise(mapped.get("home_team", ""))
                mapped["away_team"] = team_name_normalise(mapped.get("away_team", ""))
                # Convert all numeric fields
                numeric_fields = [
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
                for col in numeric_fields:
                    if col in mapped and mapped[col] is not None and mapped[col] != "":
                        try:
                            mapped[col] = float(mapped[col])
                        except (ValueError, TypeError):
                            mapped[col] = None
                    else:
                        mapped[col] = None
                all_records.append(mapped)
        except Exception as e:
            print(f"    WARN: {e}")

    if not all_records:
        return pl.DataFrame()

    # Build unified schema: union of all keys across all rows
    all_keys = set()
    for r in all_records:
        all_keys.update(r.keys())

    # Determine column types by sampling first non-null value per key
    str_keys = set()
    numeric_keys = set()
    for k in all_keys:
        for r in all_records:
            v = r.get(k)
            if v is not None and v != "":
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)
                else:
                    str_keys.add(k)
                break
        else:
            str_keys.add(k)

    # Fill missing keys and normalize types
    for r in all_records:
        for k in all_keys:
            if k not in r or r[k] is None:
                if k in numeric_keys:
                    r[k] = float("nan")
                else:
                    r[k] = ""
            elif k in numeric_keys:
                if not isinstance(r[k], (int, float)):
                    try:
                        r[k] = float(r[k])
                    except (ValueError, TypeError):
                        r[k] = float("nan")
            elif not isinstance(r[k], str):
                r[k] = str(r[k]) if r[k] is not None else ""

    # Build polars DataFrame via dict of columns (avoids per-row type issues)
    cols = {k: [] for k in all_keys}
    for r in all_records:
        for k in all_keys:
            cols[k].append(r.get(k, "" if k not in numeric_keys else float("nan")))

    df = pl.DataFrame(cols)
    out_path = Path(out_dir) / "football_data.parquet"
    df.write_parquet(str(out_path))
    print(f"  -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 4. Club Elo — scrape from clubelo.com
# ---------------------------------------------------------------------------

CLUBELO_URL = "https://api.clubelo.com/{date}"


def scrape_elo(
    seasons: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
) -> pl.DataFrame:
    """Download Elo ratings for all PL teams from clubelo.com API."""
    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["elo_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    url = "https://api.clubelo.com/england"
    print("  ClubElo England ...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        reader = csv.DictReader(lines)
        rows = []
        for row in reader:
            rows.append(
                {
                    "team": team_name_normalise(row.get("Club", "")),
                    "elo": float(row.get("Elo", 0)),
                    "from_date": row.get("From", ""),
                    "to_date": row.get("To", ""),
                }
            )
        df = pl.DataFrame(rows)
    except Exception as e:
        print(f"    WARN: {e}")
        return pl.DataFrame()

    out_path = Path(out_dir) / "elo_ratings.parquet"
    df.write_parquet(str(out_path))
    print(f"  -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 5. Weather (Open-Meteo)
# ---------------------------------------------------------------------------


def fetch_weather(
    lat: float = 51.5,
    lon: float = -0.1,
    start_date: str = "2020-08-01",
    end_date: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> pl.DataFrame:
    endpoints = {
        "London": (51.5, -0.1),
        "Manchester": (53.5, -2.2),
        "Liverpool": (53.4, -3.0),
        "Birmingham": (52.5, -1.9),
        "Newcastle": (55.0, -1.6),
        "Brighton": (50.8, -0.1),
        "Southampton": (50.9, -1.4),
        "Leicester": (52.6, -1.1),
        "Wolverhampton": (52.6, -2.1),
        "Leeds": (53.8, -1.5),
        "Nottingham": (53.0, -1.1),
        "Brentford": (51.5, -0.3),
        "Sheffield": (53.4, -1.5),
        "Bournemouth": (50.7, -1.9),
        "Ipswich": (52.1, -1.2),
    }

    paths = load_config("paths")
    out_dir = resolve_path(out_dir or paths["weather_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    all_frames = []
    for city, (lat, lon) in endpoints.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"windspeed_10m_max,rain_sum"
            f"&timezone=Europe/London"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            if daily.get("time"):
                records = []
                for i in range(len(daily["time"])):
                    records.append(
                        {
                            "city": city,
                            "date": daily["time"][i],
                            "temp_max": daily.get("temperature_2m_max", [None])[i],
                            "temp_min": daily.get("temperature_2m_min", [None])[i],
                            "precipitation": daily.get("precipitation_sum", [None])[i],
                            "wind_speed": daily.get("windspeed_10m_max", [None])[i],
                            "rain": daily.get("rain_sum", [None])[i],
                        }
                    )
                all_frames.append(pl.DataFrame(records))
        except Exception as e:
            print(f"    WARN: {city} weather: {e}")

    if not all_frames:
        return pl.DataFrame()

    df = pl.concat(all_frames)
    out_path = Path(out_dir) / "weather.parquet"
    df.write_parquet(str(out_path))
    print(f"  -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 6. Run all
# ---------------------------------------------------------------------------


def scrape_all(seasons: Optional[list[str]] = None) -> dict[str, pl.DataFrame]:
    print("=== Scraping all data sources ===\n")
    results = {}

    print("[1/3] FBref ...")
    results["fbref"] = scrape_fbref(seasons=seasons)
    print()

    print("[2/3] Understat ...")
    results["understat"] = scrape_understat(seasons=seasons)
    print()

    print("[3/3] Football-Data.co.uk ...")
    results["football_data"] = scrape_football_data()
    print()

    print("=== Done ===")
    return results
