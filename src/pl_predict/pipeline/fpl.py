"""Official Fantasy Premier League data access and gameweek selection."""

import json
from datetime import UTC, datetime
from pathlib import Path

import requests

from .utils import resolve_path


FPL_BASE_URL = "https://fantasy.premierleague.com/api"


def select_upcoming_gameweek(events: list[dict]) -> dict:
    """Return FPL's next event, or the earliest event which is not finished.

    The official feed occasionally has a short gap before it marks an event as
    ``is_next``. Falling back to an unfinished event keeps projections useful
    through that transition.
    """
    next_events = [event for event in events if event.get("is_next")]
    candidates = next_events or [event for event in events if not event.get("finished")]
    if not candidates:
        raise ValueError("The official FPL feed has no upcoming gameweek.")
    return min(candidates, key=lambda event: event.get("id", float("inf")))


def fetch_fpl_snapshot() -> dict:
    """Fetch and persist the official player, team, gameweek and fixture feed."""
    bootstrap = requests.get(f"{FPL_BASE_URL}/bootstrap-static/", timeout=30)
    fixtures = requests.get(f"{FPL_BASE_URL}/fixtures/", timeout=30)
    bootstrap.raise_for_status()
    fixtures.raise_for_status()
    snapshot = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "bootstrap": bootstrap.json(),
        "fixtures": fixtures.json(),
    }

    output = Path(resolve_path("data/raw/fpl/fpl_snapshot.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot), encoding="utf-8")
    return snapshot
