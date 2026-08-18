from pl_predict import cli
from pl_predict.pipeline.fpl import select_upcoming_gameweek


def test_select_upcoming_gameweek_prefers_official_next():
    events = [
        {"id": 4, "finished": False},
        {"id": 5, "finished": False, "is_next": True},
    ]
    assert select_upcoming_gameweek(events)["id"] == 5


def test_select_upcoming_gameweek_falls_back_to_earliest_unfinished():
    events = [
        {"id": 3, "finished": True},
        {"id": 7, "finished": False},
        {"id": 4, "finished": False},
    ]
    assert select_upcoming_gameweek(events)["id"] == 4


def test_launch_dashboard_refreshes_before_running(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "pl_predict.dashboard.app.prime_fpl_picks",
        lambda: {"gameweek": 2, "refreshed_at": "2026-08-18T00:00:00+00:00"},
    )
    monkeypatch.setattr(
        "pl_predict.dashboard.app.run_dashboard",
        lambda **kwargs: calls.append(kwargs),
    )

    cli.launch_dashboard(open_browser=False)

    assert calls == [{"open_browser": False}]
