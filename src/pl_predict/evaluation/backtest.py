"""Walk-forward backtesting framework."""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from .metrics import (
    evaluate_predictions,
    calibration_curve,
    expected_calibration_error,
)
from ..models.poisson import fit_dixon_coles, predict_dixon_coles


def _season_order(matches: pl.DataFrame) -> list[str]:
    """Return seasons in chronological order using their latest match date."""
    return (
        matches.sort("date")
        .group_by("season")
        .agg(pl.col("date").max().alias("last_date"))
        .sort("last_date")["season"]
        .to_list()
    )


def make_rolling_folds(
    matches: pl.DataFrame,
    min_train_seasons: int = 3,
    test_seasons: Optional[list[str]] = None,
) -> list[tuple[list[str], str]]:
    """Create chronological folds with no future season in a training set."""
    ordered = _season_order(matches)
    if test_seasons is not None:
        wanted = set(test_seasons)
        ordered = [season for season in ordered if season in wanted]
    all_ordered = _season_order(matches)
    folds = []
    for season in ordered:
        index = all_ordered.index(season)
        if index >= min_train_seasons:
            folds.append((all_ordered[:index], season))
    return folds


def run_rolling_evaluation(
    seasons: Optional[list[str]] = None,
    data_path: Optional[str] = None,
    min_train_seasons: int = 3,
) -> dict:
    """Evaluate Dixon-Coles chronologically and persist fold-level evidence.

    Every fold trains only on seasons strictly before its test season. The
    returned summary includes a uniform baseline, fold metrics, calibration,
    and provenance columns for reproducing each prediction.
    """
    if data_path is None:
        from ..pipeline.utils import resolve_path

        data_path = resolve_path("data/processed/matches.parquet")
    matches = pl.read_parquet(data_path).filter(
        pl.col("result").is_in(["H", "D", "A"])
    )
    folds = make_rolling_folds(matches, min_train_seasons, seasons)
    prediction_rows = []
    fold_rows = []
    labels = {"H": 0, "D": 1, "A": 2}

    for train_seasons, test_season in folds:
        train = matches.filter(pl.col("season").is_in(train_seasons))
        test = matches.filter(pl.col("season") == test_season).sort("date")
        if train.is_empty() or test.is_empty():
            continue
        params = fit_dixon_coles(train)
        y, dc_probs, uniform_probs = [], [], []
        for row in test.iter_rows(named=True):
            dc = predict_dixon_coles(params, row["home_team"], row["away_team"])
            probs = np.asarray([dc["home_win"], dc["draw"], dc["away_win"]], dtype=float)
            actual = labels[row["result"]]
            y.append(actual)
            dc_probs.append(probs)
            uniform_probs.append([1 / 3, 1 / 3, 1 / 3])
            prediction_rows.append(
                {
                    "date": row["date"],
                    "season": test_season,
                    "train_through": train_seasons[-1],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "result": row["result"],
                    "dc_h": float(probs[0]),
                    "dc_d": float(probs[1]),
                    "dc_a": float(probs[2]),
                    "model_version": "dixon-coles-rolling-v1",
                }
            )
        y = np.asarray(y)
        dc_probs = np.asarray(dc_probs)
        uniform_probs = np.asarray(uniform_probs)
        metrics = evaluate_predictions(y, dc_probs)
        baseline = evaluate_predictions(y, uniform_probs)
        confidence = dc_probs.max(axis=1)
        correct = (dc_probs.argmax(axis=1) == y).astype(int)
        metrics["ece"] = expected_calibration_error(correct, confidence)
        fold_rows.append(
            {
                "season": test_season,
                "train_through": train_seasons[-1],
                "n_matches": len(y),
                **{f"dc_{key}": value for key, value in metrics.items()},
                **{f"baseline_{key}": value for key, value in baseline.items()},
            }
        )

    from ..pipeline.utils import resolve_path

    output_dir = Path(resolve_path("data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pl.DataFrame(prediction_rows)
    folds_df = pl.DataFrame(fold_rows)
    predictions.write_parquet(str(output_dir / "rolling_predictions.parquet"))
    folds_df.write_parquet(str(output_dir / "rolling_evaluation.parquet"))
    summary = {
        "model_version": "dixon-coles-rolling-v1",
        "min_train_seasons": min_train_seasons,
        "folds": folds_df.to_dicts(),
        "predictions_path": str(output_dir / "rolling_predictions.parquet"),
        "evaluation_path": str(output_dir / "rolling_evaluation.parquet"),
        "n_matches": len(prediction_rows),
    }
    if prediction_rows:
        y = np.asarray([labels[row["result"]] for row in prediction_rows])
        probs = np.asarray(
            [[row["dc_h"], row["dc_d"], row["dc_a"]] for row in prediction_rows]
        )
        summary["overall"] = evaluate_predictions(y, probs)
        summary["overall"]["ece"] = expected_calibration_error(
            (probs.argmax(axis=1) == y).astype(int), probs.max(axis=1)
        )
    else:
        summary["overall"] = {}
    return summary


def _elo_probabilities(ratings: dict[str, float], home: str, away: str) -> np.ndarray:
    """Convert Elo ratings into calibrated-shaped H/D/A probabilities."""
    home_rating = ratings.get(home, 1500.0) + 60.0
    away_rating = ratings.get(away, 1500.0)
    win = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))
    draw = 0.27 * np.exp(-abs(home_rating - away_rating) / 500.0)
    remaining = max(1e-9, 1.0 - draw)
    return np.asarray([win * remaining, draw, (1.0 - win) * remaining])


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Keep model outputs valid when upstream estimators have tiny drift."""
    totals = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.clip(totals, 1e-12, None)


def _update_elo(
    ratings: dict[str, float], home: str, away: str, result: str, k_factor: float = 20.0
) -> None:
    """Update ratings after a completed match, including an unseen-team prior."""
    home_rating = ratings.setdefault(home, 1500.0)
    away_rating = ratings.setdefault(away, 1500.0)
    expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - 60.0) / 400.0))
    actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
    change = k_factor * (actual_home - expected_home)
    ratings[home] += change
    ratings[away] -= change


def run_rolling_model_comparison(
    seasons: Optional[list[str]] = None,
    data_path: Optional[str] = None,
    min_train_seasons: int = 3,
    include_xgb: bool = True,
    include_ensemble: bool = False,
) -> dict:
    """Compare models on identical chronological folds.

    The ensemble is opt-in because each fold retrains its OOF stacker.  It
    uses the production ensemble without the optional deep model, making the
    comparison reproducible on machines without PyTorch.
    """
    if data_path is None:
        from ..pipeline.utils import resolve_path

        data_path = resolve_path("data/processed/matches.parquet")
    matches = pl.read_parquet(data_path).filter(
        pl.col("result").is_in(["H", "D", "A"])
    )
    folds = make_rolling_folds(matches, min_train_seasons, seasons)
    features = None
    if include_xgb or include_ensemble:
        from ..features.feature_engineering import build_features

        features = build_features(matches)
    labels = {"H": 0, "D": 1, "A": 2}
    rows = []

    for train_seasons, test_season in folds:
        train = matches.filter(pl.col("season").is_in(train_seasons)).sort("date")
        test = matches.filter(pl.col("season") == test_season).sort("date")
        dc_params = fit_dixon_coles(train)
        ratings: dict[str, float] = {}
        for row in train.iter_rows(named=True):
            _update_elo(ratings, row["home_team"], row["away_team"], row["result"])

        xgb_model = xgb_columns = None
        if include_xgb:
            from ..models.xgb_model import train_xgb

            train_features = features.filter(pl.col("season").is_in(train_seasons))
            xgb_model, xgb_columns = train_xgb(
                train_features,
                params={"n_estimators": 150, "max_depth": 4},
            )
            test_features = features.filter(pl.col("season") == test_season).sort("date")
            xgb_probs = xgb_model.predict_proba(
                np.nan_to_num(test_features.select(xgb_columns).to_numpy(), nan=0.0)
            )
        ensemble_probs = None
        if include_ensemble:
            from ..models.ensemble import EnsemblePredictor

            predictor = EnsemblePredictor()
            train_features = features.filter(pl.col("season").is_in(train_seasons))
            test_features = features.filter(pl.col("season") == test_season).sort("date")
            predictor.train(
                train,
                features=train_features,
                use_deep=False,
                n_folds=3,
            )
            ensemble_probs = predictor.predict_batch(test_features)
        for index, row in enumerate(test.iter_rows(named=True)):
            y = labels[row["result"]]
            dc = predict_dixon_coles(dc_params, row["home_team"], row["away_team"])
            dc_probs = np.asarray([dc["home_win"], dc["draw"], dc["away_win"]])
            elo_probs = _elo_probabilities(ratings, row["home_team"], row["away_team"])
            record = {
                "season": test_season,
                "train_through": train_seasons[-1],
                "date": row["date"],
                "result": row["result"],
                "dc_h": dc_probs[0],
                "dc_d": dc_probs[1],
                "dc_a": dc_probs[2],
                "elo_h": elo_probs[0],
                "elo_d": elo_probs[1],
                "elo_a": elo_probs[2],
            }
            if include_xgb and xgb_probs is not None:
                record.update(
                    {
                        "xgb_h": float(xgb_probs[index, 0]),
                        "xgb_d": float(xgb_probs[index, 1]),
                        "xgb_a": float(xgb_probs[index, 2]),
                    }
                )
            if ensemble_probs is not None:
                record.update(
                    {
                        "ensemble_h": float(ensemble_probs[index, 0]),
                        "ensemble_d": float(ensemble_probs[index, 1]),
                        "ensemble_a": float(ensemble_probs[index, 2]),
                    }
                )
            rows.append(record)
            _update_elo(ratings, row["home_team"], row["away_team"], row["result"])

    predictions = pl.DataFrame(rows)
    from ..pipeline.utils import resolve_path

    output_path = Path(resolve_path("data/processed/rolling_model_comparison.parquet"))
    predictions.write_parquet(str(output_path))
    summary = {"n_matches": len(predictions), "models": {}, "path": str(output_path)}
    y = np.asarray([labels[result] for result in predictions["result"].to_list()])
    for model in ("dc", "elo", "xgb", "ensemble"):
        columns = [f"{model}_{outcome}" for outcome in ("h", "d", "a")]
        if not all(column in predictions.columns for column in columns):
            continue
        probs = _normalize_probabilities(predictions.select(columns).to_numpy().astype(float))
        confidence = probs.max(axis=1)
        summary["models"][model] = evaluate_predictions(y, probs)
        summary["models"][model]["ece"] = expected_calibration_error(
            (probs.argmax(axis=1) == y).astype(int), confidence
        )
    return summary


def run_backtest(
    seasons: Optional[list[str]] = None,
    data_path: Optional[str] = None,
) -> pl.DataFrame:
    """Run walk-forward backtest: train on S-1, test on S for each season."""
    if data_path is None:
        data_path = "data/processed/matches.parquet"

    matches = pl.read_parquet(data_path)
    seasons_list = sorted(matches["season"].unique().to_list())
    if seasons:
        seasons_list = [s for s in seasons_list if s in seasons]

    print(f"Running walk-forward backtest over {len(seasons_list)} seasons ...")

    results = []
    for i, test_season in enumerate(seasons_list):
        train_seasons = seasons_list[:i]
        if not train_seasons:
            print(f"  Skip {test_season}: no train data")
            continue

        train = matches.filter(pl.col("season").is_in(train_seasons))
        test = matches.filter(pl.col("season") == test_season)

        print(
            f"  Train: {train_seasons[0]}..{train_seasons[-1]} "
            f"({len(train)} matches) -> Test: {test_season} ({len(test)} matches)"
        )

        # Fit Dixon-Coles
        try:
            dc_params = fit_dixon_coles(train)
        except Exception as e:
            print(f"    DC fit failed: {e}")
            continue

        # Predict test set
        preds = []
        actuals = []
        for row in test.iter_rows(named=True):
            try:
                dc = predict_dixon_coles(dc_params, row["home_team"], row["away_team"])
                preds.append([dc["home_win"], dc["draw"], dc["away_win"]])
                actuals.append({"H": 0, "D": 1, "A": 2}.get(row.get("result"), -1))
            except Exception:
                preds.append([1 / 3, 1 / 3, 1 / 3])
                actuals.append(-1)

        actuals = np.array(actuals)
        preds = np.array(preds)
        valid = actuals >= 0

        if valid.sum() == 0:
            continue

        metrics = evaluate_predictions(actuals[valid], preds[valid])
        metrics["season"] = test_season
        metrics["n_matches"] = int(valid.sum())

        results.append(metrics)

        print(
            f"    RPS={metrics.get('rps', 0):.4f}  "
            f"Acc={metrics['accuracy']:.3f}  "
            f"LogLoss={metrics.get('log_loss', 0):.4f}"
        )

    result_df = pl.DataFrame(results) if results else pl.DataFrame()
    out_path = Path("data/processed/backtest_results.parquet")
    result_df.write_parquet(str(out_path))
    print(f"\nBacktest results -> {out_path}")
    return result_df


def run_ensemble_backtest(
    test_seasons: Optional[list[str]] = None,
    data_path: Optional[str] = None,
    use_deep: bool = True,
    n_folds: int = 5,
) -> dict:
    """Chronological holdout evaluation of the full ensemble vs Dixon-Coles.

    Trains once on all matches before ``test_seasons[0]`` and evaluates on the
    held-out test period. The ensemble's form/sequence caches are pinned to the
    training period so no future data leaks into predictions.
    """
    if data_path is None:
        data_path = "data/processed/matches.parquet"

    matches = pl.read_parquet(data_path)
    if "target_result" in matches.columns:
        matches = matches.filter(pl.col("target_result").is_not_null())
    else:
        matches = matches.filter(pl.col("result").is_not_null())

    if test_seasons is None:
        order = (
            matches.sort("date")
            .group_by("season")
            .agg(pl.col("date").max())
            .sort("date")["season"]
            .to_list()
        )
        test_seasons = order[-3:]
    test_seasons = sorted(set(test_seasons))

    train = matches.filter(~pl.col("season").is_in(test_seasons))
    test = matches.filter(pl.col("season").is_in(test_seasons))
    print(
        f"Train: {len(train)} matches (<= {train['season'].max()})  "
        f"-> Test: {len(test)} matches {test_seasons}"
    )

    # Rebuild from this exact match frame. Persisted feature artifacts can be
    # stale or correspond to a different source snapshot.
    from pl_predict.features.feature_engineering import build_features

    features = build_features(matches)
    features_train = features.filter(~pl.col("season").is_in(test_seasons))

    # --- Dixon-Coles baseline (train only) ---
    dc_params = fit_dixon_coles(train)

    # --- Full ensemble (train only) ---
    from pl_predict.models.ensemble import EnsemblePredictor

    ens = EnsemblePredictor()
    ens.train(train, features=features_train, use_deep=use_deep, n_folds=n_folds)
    # Pin caches to the training period to prevent future-data leakage
    ens._matches_cache = train
    ens._feat_cache = features_train
    ens._state = None

    rows = []
    for row in test.iter_rows(named=True):
        h, a, res = row["home_team"], row["away_team"], row.get("result")
        if res not in ("H", "D", "A"):
            continue
        try:
            dc = predict_dixon_coles(dc_params, h, a)
            dc_p = [dc["home_win"], dc["draw"], dc["away_win"]]
        except Exception:
            continue
        p = ens.predict(h, a, date_str=str(row["date"]))
        if "error" in p:
            continue
        rows.append(
            {
                "date": row["date"],
                "season": row["season"],
                "home_team": h,
                "away_team": a,
                "result": res,
                "dc_h": dc_p[0],
                "dc_d": dc_p[1],
                "dc_a": dc_p[2],
                "ens_h": p["home_win"],
                "ens_d": p["draw"],
                "ens_a": p["away_win"],
                "breakdown": p.get("model_breakdown", {}),
            }
        )

    preds = pl.DataFrame(rows)
    y = np.array([{"H": 0, "D": 1, "A": 2}[r] for r in preds["result"].to_list()])
    dc_probs = np.stack([preds["dc_h"], preds["dc_d"], preds["dc_a"]], axis=1).astype(
        float
    )
    ens_probs = np.stack(
        [preds["ens_h"], preds["ens_d"], preds["ens_a"]], axis=1
    ).astype(float)

    dc_confidence = dc_probs.max(axis=1)
    ens_confidence = ens_probs.max(axis=1)
    dc_correct = (dc_probs.argmax(axis=1) == y).astype(int)
    ens_correct = (ens_probs.argmax(axis=1) == y).astype(int)

    out = {
        "n_matches": int(len(preds)),
        "dc": evaluate_predictions(y, dc_probs),
        "ensemble": evaluate_predictions(y, ens_probs),
        "dc_calibration": list(
            map(tuple, calibration_curve(dc_correct, dc_confidence))
        ),
        "ens_calibration": list(
            map(tuple, calibration_curve(ens_correct, ens_confidence))
        ),
    }
    out["dc"]["ece"] = expected_calibration_error(dc_correct, dc_confidence)
    out["ensemble"]["ece"] = expected_calibration_error(ens_correct, ens_confidence)
    out["dc"]["ece_ovr"] = float(
        np.mean(
            [
                expected_calibration_error((y == i).astype(int), dc_probs[:, i])
                for i in range(3)
            ]
        )
    )
    out["ensemble"]["ece_ovr"] = float(
        np.mean(
            [
                expected_calibration_error((y == i).astype(int), ens_probs[:, i])
                for i in range(3)
            ]
        )
    )

    # Per-outcome calibration of the ensemble
    for i, name in enumerate(["H", "D", "A"]):
        y_bin = (y == i).astype(int)
        bp, bf = calibration_curve(y_bin, ens_probs[:, i])
        out[f"ens_cal_{name}"] = list(zip(bp.tolist(), bf.tolist()))

    print("\n=== Walk-forward backtest (held-out) ===")
    for label, m in [("DC-only", out["dc"]), ("Ensemble", out["ensemble"])]:
        print(
            f"  {label:9s} RPS={m['rps']:.4f}  Acc={m['accuracy']:.3f}  "
            f"Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  "
            f"ECE={m.get('ece', float('nan')):.4f}"
        )
    for name in ["H", "D", "A"]:
        bp, bf = zip(*out[f"ens_cal_{name}"])
        print(
            f"  Ensemble P({name}) calibration (pred -> actual): "
            + ", ".join(f"{p:.2f}->{f:.2f}" for p, f in zip(bp, bf))
        )

    preds_path = Path("data/processed/walkforward_preds.parquet")
    preds.write_parquet(str(preds_path))
    print(f"\nPer-match predictions -> {preds_path}")
    return out
