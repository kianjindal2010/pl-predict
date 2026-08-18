from datetime import datetime

import numpy as np
import polars as pl

from pl_predict.features import feature_engineering


def test_team_form_is_attached_to_the_correct_match_side(monkeypatch, tmp_path):
    """A team's form must include all of its earlier home and away matches."""
    monkeypatch.setattr(feature_engineering, "resolve_path", lambda _: str(tmp_path))
    monkeypatch.setattr(
        feature_engineering,
        "load_config",
        lambda _: {"processed_dir": "ignored", "raw_dir": "ignored"},
    )
    matches = pl.DataFrame(
        {
            "date": [
                datetime(2025, 8, 1),
                datetime(2025, 8, 8),
                datetime(2025, 8, 15),
            ],
            "season": ["2025-26"] * 3,
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "A", "C"],
            "home_goals": [2.0, 0.0, 1.0],
            "away_goals": [0.0, 1.0, 1.0],
            "result": ["H", "A", "D"],
        }
    )

    features = feature_engineering.build_features(matches)

    # Team A won both prior fixtures, once home and once away.
    assert np.isclose(features["home_form_pts_avg"][2], 3.0)
