"""Fixture schedule for the 2026/27 Premier League season.

All 380 fixtures as released 19 June 2026 (source: Wikipedia).
"""

from datetime import datetime
from typing import Dict, List

# Map display team names to the model's canonical team names.
# The model was trained on football-data.co.uk names (short forms).
TEAM_NAME_MAP = {
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
}


def normalize_team(name: str) -> str:
    """Map a display team name to the model's canonical name."""
    return TEAM_NAME_MAP.get(name, name)


# Full 2026/27 Premier League fixture list (380 matches)
FIXTURES_2627: List[Dict] = [
    # ---- Matchweek 1 (21-24 Aug 2026) ----
    {
        "date": "2026-08-21",
        "time": "20:00",
        "home_team": "Arsenal",
        "away_team": "Coventry City",
    },
    {
        "date": "2026-08-22",
        "time": "12:30",
        "home_team": "Hull City",
        "away_team": "Man United",
    },
    {
        "date": "2026-08-22",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2026-08-22",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-08-22",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-08-22",
        "time": "17:30",
        "home_team": "Brentford",
        "away_team": "Tottenham",
    },
    {
        "date": "2026-08-23",
        "time": "14:00",
        "home_team": "Brighton",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-08-23",
        "time": "14:00",
        "home_team": "Man City",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-08-23",
        "time": "16:30",
        "home_team": "Newcastle United",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-08-24",
        "time": "20:00",
        "home_team": "Fulham",
        "away_team": "Chelsea",
    },
    # ---- Matchweek 2 (28-31 Aug 2026) ----
    {
        "date": "2026-08-28",
        "time": "20:00",
        "home_team": "Crystal Palace",
        "away_team": "Man City",
    },
    {
        "date": "2026-08-29",
        "time": "12:30",
        "home_team": "Liverpool",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-08-29",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Everton",
    },
    {
        "date": "2026-08-29",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Hull City",
    },
    {
        "date": "2026-08-29",
        "time": "17:30",
        "home_team": "Tottenham",
        "away_team": "Newcastle United",
    },
    {
        "date": "2026-08-30",
        "time": "14:00",
        "home_team": "Chelsea",
        "away_team": "Brighton",
    },
    {
        "date": "2026-08-30",
        "time": "14:00",
        "home_team": "Leeds United",
        "away_team": "Brentford",
    },
    {
        "date": "2026-08-30",
        "time": "14:00",
        "home_team": "Sunderland",
        "away_team": "Fulham",
    },
    {
        "date": "2026-08-30",
        "time": "16:30",
        "home_team": "Man United",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-08-31",
        "time": "20:00",
        "home_team": "Aston Villa",
        "away_team": "Arsenal",
    },
    # ---- Matchweek 3 (4-6 Sep 2026) ----
    {
        "date": "2026-09-04",
        "time": "20:00",
        "home_team": "Ipswich Town",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-09-05",
        "time": "12:30",
        "home_team": "Newcastle United",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-09-05",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-09-05",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-09-05",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2026-09-05",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Coventry City",
    },
    {
        "date": "2026-09-05",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Tottenham",
    },
    {
        "date": "2026-09-05",
        "time": "17:30",
        "home_team": "Hull City",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-09-06",
        "time": "14:00",
        "home_team": "Everton",
        "away_team": "Man United",
    },
    {
        "date": "2026-09-06",
        "time": "16:30",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
    },
    # ---- Matchweek 4 (12-14 Sep 2026) ----
    {
        "date": "2026-09-12",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Brentford",
    },
    {
        "date": "2026-09-12",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-09-12",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Hull City",
    },
    {
        "date": "2026-09-12",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-09-12",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Fulham",
    },
    {
        "date": "2026-09-12",
        "time": "17:30",
        "home_team": "Tottenham",
        "away_team": "Everton",
    },
    {
        "date": "2026-09-12",
        "time": "20:00",
        "home_team": "Sunderland",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-09-13",
        "time": "14:00",
        "home_team": "Coventry City",
        "away_team": "Brighton",
    },
    {
        "date": "2026-09-13",
        "time": "16:30",
        "home_team": "Man United",
        "away_team": "Man City",
    },
    {
        "date": "2026-09-14",
        "time": "20:00",
        "home_team": "Leeds United",
        "away_team": "Newcastle United",
    },
    # ---- Matchweek 5 (18-20 Sep 2026) ----
    {
        "date": "2026-09-18",
        "time": "20:00",
        "home_team": "Brentford",
        "away_team": "Chelsea",
    },
    {
        "date": "2026-09-19",
        "time": "12:30",
        "home_team": "Tottenham",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-09-19",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Hull City",
    },
    {
        "date": "2026-09-19",
        "time": "17:30",
        "home_team": "Nottingham Forest",
        "away_team": "Coventry City",
    },
    {
        "date": "2026-09-20",
        "time": "14:00",
        "home_team": "Bournemouth",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-09-20",
        "time": "16:30",
        "home_team": "Fulham",
        "away_team": "Man United",
    },
    # ---- Matchweek 6 (10 Oct 2026) ----
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Brentford",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Newcastle United",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Everton",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Fulham",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Man City",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Tottenham",
    },
    {
        "date": "2026-10-10",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Brighton",
    },
    # ---- Matchweek 7 (17 Oct 2026) ----
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Chelsea",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Hull City",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Man United",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-10-17",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Coventry City",
    },
    # ---- Matchweek 8 (24 Oct 2026) ----
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Everton",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Man City",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Tottenham",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Fulham",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Newcastle United",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Brentford",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Brighton",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-10-24",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Leeds United",
    },
    # ---- Matchweek 9 (31 Oct 2026) ----
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Fulham",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Man United",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Brighton",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Everton",
    },
    {
        "date": "2026-10-31",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Crystal Palace",
    },
    # ---- Matchweek 10 (7 Nov 2026) ----
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Hull City",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Brentford",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Coventry City",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Newcastle United",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Tottenham",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Man City",
    },
    {
        "date": "2026-11-07",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Chelsea",
    },
    # ---- Matchweek 11 (21 Nov 2026) ----
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Everton",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Brighton",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Man United",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Fulham",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Ipswich Town",
    },
    # ---- Matchweek 12 (28 Nov 2026) ----
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Man City",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Newcastle United",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Hull City",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Coventry City",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Brentford",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Chelsea",
    },
    {
        "date": "2026-11-28",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Tottenham",
    },
    # ---- Matchweek 13 (2 Dec 2026) ----
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Bournemouth",
        "away_team": "Brighton",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Aston Villa",
        "away_team": "Everton",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Brentford",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Chelsea",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Coventry City",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Hull City",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Liverpool",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Man City",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Newcastle United",
        "away_team": "Man United",
    },
    {
        "date": "2026-12-02",
        "time": "20:00",
        "home_team": "Tottenham",
        "away_team": "Fulham",
    },
    # ---- Matchweek 14 (5 Dec 2026) ----
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Hull City",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Man City",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Fulham",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Coventry City",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Brighton",
    },
    {
        "date": "2026-12-05",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Arsenal",
    },
    # ---- Matchweek 15 (12 Dec 2026) ----
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Everton",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Man United",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Brentford",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Tottenham",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Newcastle United",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Chelsea",
    },
    {
        "date": "2026-12-12",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Nottingham Forest",
    },
    # ---- Matchweek 16 (19 Dec 2026) ----
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Coventry City",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Man United",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Newcastle United",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Aston Villa",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Fulham",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Tottenham",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Hull City",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Everton",
    },
    {
        "date": "2026-12-19",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Crystal Palace",
    },
    # ---- Matchweek 17 (26 Dec 2026) ----
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Chelsea",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Brighton",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Brentford",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Man City",
    },
    {
        "date": "2026-12-26",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Bournemouth",
    },
    # ---- Matchweek 18 (30 Dec 2026) ----
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Aston Villa",
        "away_team": "Liverpool",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Coventry City",
        "away_team": "Brentford",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Crystal Palace",
        "away_team": "Bournemouth",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Everton",
        "away_team": "Man City",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Fulham",
        "away_team": "Arsenal",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Hull City",
        "away_team": "Leeds United",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Ipswich Town",
        "away_team": "Chelsea",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Man United",
        "away_team": "Sunderland",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Newcastle United",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2026-12-30",
        "time": "20:00",
        "home_team": "Tottenham",
        "away_team": "Brighton",
    },
    # ---- Matchweek 19 (2 Jan 2027) ----
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Man United",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Everton",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Fulham",
    },
    {
        "date": "2027-01-02",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Hull City",
    },
    # ---- Matchweek 20 (6 Jan 2027) ----
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Arsenal",
        "away_team": "Brentford",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Brighton",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Crystal Palace",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Everton",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Fulham",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Ipswich Town",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Leeds United",
        "away_team": "Man City",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Man United",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Nottingham Forest",
        "away_team": "Hull City",
    },
    {
        "date": "2027-01-06",
        "time": "20:00",
        "home_team": "Sunderland",
        "away_team": "Liverpool",
    },
    # ---- Matchweek 21 (16 Jan 2027) ----
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Man United",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Brighton",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Everton",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Fulham",
    },
    {
        "date": "2027-01-16",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Leeds United",
    },
    # ---- Matchweek 22 (23 Jan 2027) ----
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Man City",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Brentford",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Hull City",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Liverpool",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-01-23",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Coventry City",
    },
    # ---- Matchweek 23 (30 Jan 2027) ----
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Fulham",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Man United",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Everton",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Brighton",
    },
    {
        "date": "2027-01-30",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Sunderland",
    },
    # ---- Matchweek 24 (6 Feb 2027) ----
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Liverpool",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Hull City",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Man City",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Brentford",
    },
    {
        "date": "2027-02-06",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Aston Villa",
    },
    # ---- Matchweek 25 (10 Feb 2027) ----
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Aston Villa",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Coventry City",
        "away_team": "Liverpool",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Crystal Palace",
        "away_team": "Brentford",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Everton",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Fulham",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Hull City",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Ipswich Town",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Man United",
        "away_team": "Brighton",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Newcastle United",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-02-10",
        "time": "20:00",
        "home_team": "Tottenham",
        "away_team": "Man City",
    },
    # ---- Matchweek 26 (20 Feb 2027) ----
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Fulham",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Hull City",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Man United",
    },
    {
        "date": "2027-02-20",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Everton",
    },
    # ---- Matchweek 27 (27 Feb 2027) ----
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Man City",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Brighton",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Brentford",
    },
    {
        "date": "2027-02-27",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Liverpool",
    },
    # ---- Matchweek 28 (3 Mar 2027) ----
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Bournemouth",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Arsenal",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Brentford",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Brighton",
        "away_team": "Fulham",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Chelsea",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Leeds United",
        "away_team": "Hull City",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Liverpool",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Man City",
        "away_team": "Everton",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Nottingham Forest",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-03-03",
        "time": "20:00",
        "home_team": "Sunderland",
        "away_team": "Man United",
    },
    # ---- Matchweek 29 (13 Mar 2027) ----
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Hull City",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Man City",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Fulham",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Brighton",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Everton",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Brentford",
    },
    {
        "date": "2027-03-13",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Nottingham Forest",
    },
    # ---- Matchweek 30 (20 Mar 2027) ----
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Liverpool",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Man United",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-03-20",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Aston Villa",
    },
    # ---- Matchweek 31 (10 Apr 2027) ----
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Man City",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Brighton",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Fulham",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Everton",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Hull City",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-04-10",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Brentford",
    },
    # ---- Matchweek 32 (17 Apr 2027) ----
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Man United",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-04-17",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Liverpool",
    },
    # ---- Matchweek 33 (24 Apr 2027) ----
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Fulham",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Man City",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Brighton",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Liverpool",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-04-24",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Hull City",
    },
    # ---- Matchweek 34 (1 May 2027) ----
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Man United",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Everton",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Brentford",
    },
    {
        "date": "2027-05-01",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Newcastle United",
    },
    # ---- Matchweek 35 (8 May 2027) ----
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Bournemouth",
        "away_team": "Man United",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Brentford",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Brighton",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Everton",
        "away_team": "Hull City",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Fulham",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Leeds United",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Man City",
        "away_team": "Liverpool",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Newcastle United",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Nottingham Forest",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-05-08",
        "time": "15:00",
        "home_team": "Tottenham",
        "away_team": "Chelsea",
    },
    # ---- Matchweek 36 (15 May 2027) ----
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Arsenal",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Aston Villa",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Chelsea",
        "away_team": "Everton",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Coventry City",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Crystal Palace",
        "away_team": "Brighton",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Hull City",
        "away_team": "Fulham",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Ipswich Town",
        "away_team": "Man City",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Liverpool",
        "away_team": "Brentford",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Man United",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-05-15",
        "time": "15:00",
        "home_team": "Sunderland",
        "away_team": "Bournemouth",
    },
    # ---- Matchweek 37 (23 May 2027) ----
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Bournemouth",
        "away_team": "Chelsea",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Brentford",
        "away_team": "Hull City",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Brighton",
        "away_team": "Liverpool",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Everton",
        "away_team": "Arsenal",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Fulham",
        "away_team": "Coventry City",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Leeds United",
        "away_team": "Sunderland",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Man City",
        "away_team": "Aston Villa",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Newcastle United",
        "away_team": "Crystal Palace",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Nottingham Forest",
        "away_team": "Ipswich Town",
    },
    {
        "date": "2027-05-23",
        "time": "16:00",
        "home_team": "Tottenham",
        "away_team": "Man United",
    },
    # ---- Matchweek 38 (30 May 2027) ----
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Arsenal",
        "away_team": "Brighton",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Aston Villa",
        "away_team": "Tottenham",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Chelsea",
        "away_team": "Brentford",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Coventry City",
        "away_team": "Nottingham Forest",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Crystal Palace",
        "away_team": "Leeds United",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Hull City",
        "away_team": "Newcastle United",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Ipswich Town",
        "away_team": "Everton",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Liverpool",
        "away_team": "Bournemouth",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Man United",
        "away_team": "Fulham",
    },
    {
        "date": "2027-05-30",
        "time": "16:00",
        "home_team": "Sunderland",
        "away_team": "Man City",
    },
]


def get_fixtures() -> List[Dict]:
    """Return the full 2026/27 fixture list."""
    return FIXTURES_2627


def get_teams() -> List[str]:
    """Return the 20 teams in the 2026/27 Premier League."""
    teams = set()
    for f in FIXTURES_2627:
        teams.add(f["home_team"])
        teams.add(f["away_team"])
    return sorted(teams)


def get_upcoming_fixtures() -> List[Dict]:
    """Return upcoming fixtures (today = 2026-08-03)."""
    today = datetime(2026, 8, 3)
    upcoming = [
        f for f in FIXTURES_2627 if datetime.strptime(f["date"], "%Y-%m-%d") >= today
    ]
    return sorted(upcoming, key=lambda x: x["date"])


if __name__ == "__main__":
    fixtures = get_fixtures()
    print(f"Total fixtures: {len(fixtures)}")
    teams = get_teams()
    print(f"Teams ({len(teams)}): {teams}")
    print("First 5 fixtures:")
    for f in fixtures[:5]:
        print(f"  {f['date']} {f['home_team']} vs {f['away_team']}")
    print("Last 2 fixtures:")
    for f in fixtures[-2:]:
        print(f"  {f['date']} {f['home_team']} vs {f['away_team']}")
