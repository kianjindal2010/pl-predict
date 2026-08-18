"""Layer 2: XGBoost gradient boosted trees — works with Polars DataFrames."""

import numpy as np
import polars as pl
import xgboost as xgb


def _select_features(df: pl.DataFrame) -> tuple:
    exclude = {
        "date",
        "home_team",
        "away_team",
        "season",
        "match_id",
        "target_result",
        "target_home_goals",
        "target_away_goals",
        "target_over_2_5",
        "target_btts",
        "target_total_goals",
        "target_goal_diff",
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude
        and df[c].dtype in (pl.Float64, pl.Int64, pl.Int32, pl.Float32)
    ]
    X = df.select(feature_cols).to_numpy()
    return X, feature_cols


def train_xgb(
    features: pl.DataFrame,
    target_col: str = "target_result",
    params: dict | None = None,
) -> tuple:
    X, feature_cols = _select_features(features)
    y = features[target_col].to_numpy()

    if y.dtype.kind == "U" or y.dtype.kind == "O":
        vals = set(y)
        if vals <= {"H", "D", "A"}:
            # Explicit order so predict_proba columns are [H, D, A]
            y = np.array([{"H": 0, "D": 1, "A": 2}[v] for v in y])
        else:
            from sklearn.preprocessing import LabelEncoder

            y = LabelEncoder().fit_transform(y)

    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    n_classes = len(np.unique(y))
    if n_classes == 2:
        default_params["objective"] = "binary:logistic"
    else:
        default_params["objective"] = "multi:softprob"
        default_params["num_class"] = n_classes

    model = xgb.XGBClassifier(**default_params)
    X = np.nan_to_num(X, nan=0.0)
    model.fit(X, y, verbose=False)
    return model, feature_cols


def predict_xgb(
    model, features: pl.DataFrame, feature_cols: list | None = None
) -> np.ndarray:
    if feature_cols is None:
        X, feature_cols = _select_features(features)
    else:
        X = features.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    return model.predict_proba(X)
