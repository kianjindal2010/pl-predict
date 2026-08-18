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
