"""Shared utilities for the data pipeline."""

import yaml
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parents[3]


def load_config(name: str) -> dict[str, Any]:
    path = Path(__file__).parents[3] / "config" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(rel_path: str) -> str:
    """Resolve a relative path from the project root."""
    return str((PROJECT_ROOT / rel_path).resolve())


def team_name_normalise(name: str) -> str:
    mapping = {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Man Utd": "Man United",
        "Newcastle United": "Newcastle",
        "Tottenham Hotspur": "Tottenham",
        "Tottenham": "Tottenham",
        "Spurs": "Tottenham",
        "Wolverhampton Wanderers": "Wolves",
        "Wolverhampton": "Wolves",
        "Brighton & Hove Albion": "Brighton",
        "Brighton and Hove Albion": "Brighton",
        "Leicester City": "Leicester",
        "West Ham United": "West Ham",
        "Norwich City": "Norwich",
        "AFC Bournemouth": "Bournemouth",
        "Bournemouth": "Bournemouth",
        "Nottingham Forest": "Nott'm Forest",
        "Nott'm Forest": "Nott'm Forest",
        "Sheffield United": "Sheffield Utd",
        "Sheffield Utd": "Sheffield Utd",
        "Leeds United": "Leeds",
        "Huddersfield Town": "Huddersfield",
        "Cardiff City": "Cardiff",
        "Swansea City": "Swansea",
        "Stoke City": "Stoke",
        "West Bromwich Albion": "West Brom",
        "Queens Park Rangers": "QPR",
        "Coventry City": "Coventry",
        "Ipswich Town": "Ipswich",
        "Hull City": "Hull",
    }
    return mapping.get(name, name)


def parse_season_from_date(dt: any) -> str:
    year = dt.year
    month = dt.month
    if month >= 8:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"
