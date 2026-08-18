"""Model training orchestration."""

from typing import Optional

import polars as pl

from .ensemble import EnsemblePredictor
from ..features.feature_engineering import build_features
from ..pipeline.utils import resolve_path


def train_model(
    model_type: str = "ensemble",
    seasons: Optional[list[str]] = None,
    data_path: Optional[str] = None,
) -> EnsemblePredictor:
    """Train the deployable ensemble model on processed data."""
    if model_type != "ensemble":
        raise ValueError(
            "Only the ensemble is deployable through this interface. "
            "Train component models through their dedicated research APIs."
        )
    if data_path is None:
        data_path = resolve_path("data/processed/matches.parquet")

    print(f"Loading matches from {data_path}")
    matches = pl.read_parquet(data_path)

    if seasons:
        season_col = "season" if "season" in matches.columns else "season_id"
        matches = matches.filter(pl.col(season_col).is_in(seasons))

    print(f"  {len(matches)} matches loaded")

    predictor = EnsemblePredictor()

    print("Building features ...")
    features = build_features(matches)
    predictor.train(matches, features, use_deep=True)

    predictor.save()
    return predictor
