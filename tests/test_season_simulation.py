import numpy as np

from pl_predict.simulation import season_sim


def test_title_probability_is_shared_when_points_are_tied(monkeypatch, tmp_path):
    output_dir = tmp_path / "data" / "processed"
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        season_sim, "resolve_path", lambda _: str(output_dir / "season_sim.parquet")
    )
    simulator = season_sim.SeasonSimulator(season="test", source="parquet", seed=1)
    simulator._load_fixtures = lambda: (
        ["A"],
        ["B"],
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
        {},
    )

    results = simulator.run(n_simulations=100)

    assert results["title_pct"].sum() == 100.0
    assert set(results["title_pct"].to_list()) == {50.0}
