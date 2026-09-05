import polars as pl

from pl_predict import cli
from pl_predict.dashboard.app import _fpl_squad_analysis
from pl_predict.evaluation.backtest import _elo_probabilities, make_rolling_folds
from pl_predict.evaluation.metrics import ranked_probability_score
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


def test_fpl_squad_analysis_picks_captain_and_budgeted_transfer():
    def player(player_id, name, position, price, xp):
        return {
            "player_id": player_id,
            "name": name,
            "team": "Team",
            "position": position,
            "price": price,
            "price_tenths": round(price * 10),
            "projections": {"2": {"xP": xp, "fixtures": []}},
        }

    squad = [
        player(1, "Keeper", "GK", 5.0, 4.0),
        *[player(i, f"Defender {i}", "DEF", 5.0, 4.0) for i in range(2, 5)],
        *[player(i, f"Midfielder {i}", "MID", 6.0, 5.0) for i in range(5, 7)],
        *[player(i, f"Forward {i}", "FWD", 7.0, 5.0) for i in range(7, 9)],
        player(9, "Captain", "MID", 10.0, 9.0),
        *[player(i, f"Bench {i}", "DEF", 4.0, 1.0) for i in range(10, 16)],
    ]
    replacement = player(99, "Upgrade", "MID", 6.5, 7.0)
    result = _fpl_squad_analysis(
        {"players": squad + [replacement], "next_gameweek": 2, "gameweeks": [{"id": 2}]},
        list(range(1, 16)),
        "2",
        bank=1.0,
    )

    assert result["captain"]["name"] == "Captain"
    assert result["vice_captain"]["name"] != result["captain"]["name"]
    assert len(result["starting_xi"]) == 11
    assert result["transfers"][0]["buy"]["name"] == "Upgrade"


def test_rolling_folds_never_train_on_the_test_season():
    matches = pl.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2021-22", "2021-22", "2022-23", "2022-23"],
            "date": pl.date_range(
                pl.date(2020, 8, 1), pl.date(2020, 8, 6), interval="1d", eager=True
            ),
        }
    )

    folds = make_rolling_folds(matches, min_train_seasons=1)

    assert folds == [
        (["2020-21"], "2021-22"),
        (["2020-21", "2021-22"], "2022-23"),
    ]


def test_ranked_probability_score_excludes_constant_final_cumulative_term():
    assert ranked_probability_score(
        pl.Series([0, 1, 2]).to_numpy(),
        pl.DataFrame(
            {
                "h": [1.0, 0.0, 0.0],
                "d": [0.0, 1.0, 0.0],
                "a": [0.0, 0.0, 1.0],
            }
        ).to_numpy(),
    ) == 0.0
    assert abs(ranked_probability_score(
        pl.Series([0]).to_numpy(),
        pl.DataFrame({"h": [1 / 3], "d": [1 / 3], "a": [1 / 3]}).to_numpy(),
    ) - 5 / 18) < 1e-12


def test_elo_probabilities_are_normalized_and_favor_stronger_home_team():
    probabilities = _elo_probabilities({"Strong": 1700.0, "Weak": 1300.0}, "Strong", "Weak")

    assert abs(probabilities.sum() - 1.0) < 1e-12
    assert probabilities[0] > probabilities[2]
    assert probabilities[1] > 0
