"""Layer 4: Ensemble — stacking of Dixon-Coles + XGBoost (+ optional deep)."""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from .poisson import fit_dixon_coles, predict_dixon_coles
from .xgb_model import train_xgb, predict_xgb

try:
    from .deep import build_tensors, _TORCH_OK
except Exception:
    _TORCH_OK = False


_SQUAD_CACHE: Optional[dict] = None


def _squad_lookup() -> dict:
    """{(understat_season_id, team): {squad_xg, squad_xa, squad_npxg, ...}}."""
    global _SQUAD_CACHE
    if _SQUAD_CACHE is not None:
        return _SQUAD_CACHE
    _SQUAD_CACHE = {}
    try:
        from pl_predict.pipeline.utils import team_name_normalise

        path = Path("data/raw/understat/understat_players.parquet")
        if path.exists():
            plr = pl.read_parquet(str(path))
            squad = (
                plr.group_by(["season", "team"])
                .agg(
                    pl.col("xG").cast(pl.Float64, strict=False).sum().alias("squad_xg"),
                    pl.col("xA").cast(pl.Float64, strict=False).sum().alias("squad_xa"),
                    pl.col("npxG")
                    .cast(pl.Float64, strict=False)
                    .sum()
                    .alias("squad_npxg"),
                    pl.col("time")
                    .cast(pl.Float64, strict=False)
                    .sum()
                    .alias("squad_minutes"),
                    pl.len().alias("squad_players"),
                )
                .with_columns(
                    pl.col("team").map_elements(
                        team_name_normalise, return_dtype=pl.String
                    )
                )
            )
            for r in squad.iter_rows(named=True):
                _SQUAD_CACHE[(r["season"], r["team"])] = {
                    "squad_xg": r["squad_xg"],
                    "squad_xa": r["squad_xa"],
                    "squad_npxg": r["squad_npxg"],
                    "squad_minutes": r["squad_minutes"],
                    "squad_players": r["squad_players"],
                }
    except Exception:
        pass
    return _SQUAD_CACHE


def _prior_understat_season(date) -> Optional[str]:
    """Understat season id for the season BEFORE the season of ``date``.

    A match on 2026-08-21 belongs to 2026-27; its prior season is Understat
    id '2025' (2025-26). Returns None when the date can't be parsed.
    """
    try:
        if isinstance(date, str):
            from datetime import datetime

            date = datetime.fromisoformat(date[:10])
        year = date.year
        month = date.month
        season_start = year if month >= 8 else year - 1  # PL season start year
        return str(season_start - 1)
    except Exception:
        return None


def _squad_features(team: str, prior_season: Optional[str]) -> Optional[dict]:
    if not prior_season:
        return None
    return _squad_lookup().get((prior_season, team))


_TM_CACHE = None


def _tm_squad_lookup() -> dict:
    """{(season_id, team): {squad_value, avg_age, squad_size}} from Transfermarkt."""
    global _TM_CACHE
    if _TM_CACHE is not None:
        return _TM_CACHE
    _TM_CACHE = {}
    try:
        from pl_predict.pipeline.utils import team_name_normalise

        path = Path("data/raw/transfermarkt/transfermarkt_squads.parquet")
        if path.exists():
            tm = pl.read_parquet(str(path))
            for r in tm.iter_rows(named=True):
                team = team_name_normalise(str(r.get("team") or ""))
                _TM_CACHE[(str(r.get("season")), team)] = {
                    "squad_value": r.get("squad_value"),
                    "avg_age": r.get("avg_age"),
                    "squad_size": r.get("squad_size"),
                }
    except Exception:
        pass
    return _TM_CACHE


def _tm_squad_features(team: str, prior_season: Optional[str]) -> Optional[dict]:
    if not prior_season:
        return None
    return _tm_squad_lookup().get((prior_season, team))


class EnsemblePredictor:
    """Ensemble of Dixon-Coles + XGBoost (+ optional deep LSTM)."""

    def __init__(self):
        self.dc_params = None
        self.xgb_model = None
        self.xgb_features = None
        self.xgb_nomarket_model = None
        self.xgb_nomarket_features = None
        self.stacker = None
        self.stacker_cal = None
        self.stacker_nomarket = None
        self.stacker_nomarket_cal = None
        self.rps_weights = None
        self.rps_weights_nomarket = None
        self.deep_model = None
        self.deep_meta = None
        self.deep_weights = None
        self._feat_cache = None
        self._matches_cache = None
        self._state = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, matches, features=None, use_deep: bool = True, n_folds: int = 5):
        """Train all component models with out-of-fold stacking + calibration.

        Two stackers are fitted:
        - ``stacker``        (market available):  dc + market-XGB + deep
        - ``stacker_nomarket`` (future fixtures): dc + no-market-XGB + deep

        The no-market XGB is trained without betting-odds columns so it stays
        reliable for 2026/27 fixtures (which have no market data).
        """
        print("  Training Dixon-Coles (Poisson) ...")
        self.dc_params = fit_dixon_coles(matches)

        print("  Building features ...")
        if features is None:
            from pl_predict.features.feature_engineering import build_features

            features = build_features(matches)

        market_cols = [c for c in features.columns if c.startswith("market_")]
        features_nm = features.select(
            [c for c in features.columns if c not in market_cols]
        )

        print(f"  Building out-of-fold predictions ({n_folds} folds) ...")
        dc_oof, xgb_oof, xgb_nm_oof, deep_oof, oof_mask = self._out_of_fold_predictions(
            features, features_nm, n_folds=n_folds, use_deep=use_deep
        )

        y = features["target_result"].to_numpy()
        if y.dtype.kind in ("U", "O"):
            if set(y) <= {"H", "D", "A"}:
                y = np.array([{"H": 0, "D": 1, "A": 2}[v] for v in y])
            else:
                from sklearn.preprocessing import LabelEncoder

                y = LabelEncoder().fit_transform(y)

        # The initial training window has no strictly-prior OOF model.  It is
        # deliberately excluded from stacking and calibration rather than
        # represented by uniform placeholder probabilities.
        y = y[oof_mask]
        dc_oof = dc_oof[oof_mask]
        xgb_oof = xgb_oof[oof_mask]
        xgb_nm_oof = xgb_nm_oof[oof_mask]
        deep_oof = deep_oof[oof_mask]

        # --- Market stacker (dc + market-xgb + deep) ---
        X_stack = np.hstack([dc_oof, xgb_oof, deep_oof])
        self.stacker = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        self.stacker.fit(X_stack, y)
        try:
            self.rps_weights = self.fit_rps_weights([dc_oof, xgb_oof, deep_oof], y)
        except Exception as e:
            print(f"    RPS weight fit (market) failed: {e}")
            self.rps_weights = None
        if self.rps_weights is not None:
            blend_m = (
                self.rps_weights[0] * dc_oof
                + self.rps_weights[1] * xgb_oof
                + self.rps_weights[2] * deep_oof
            )
            blend_m /= blend_m.sum(axis=1, keepdims=True)
            self.stacker_cal = self._fit_calibrator(blend_m, y)
        else:
            self.stacker_cal = self._fit_calibrator(
                self.stacker.predict_proba(X_stack), y
            )
        _cal_desc = (
            "isotonic (CV)"
            if self.stacker_cal and self.stacker_cal.get("method") == "isotonic"
            else f"T={self.stacker_cal.get('T', 1.0):.3f}"
            if self.stacker_cal
            else ""
        )
        print(
            f"  Stacker (market) trained on {X_stack.shape[1]} features; "
            f"calibration={_cal_desc}"
            if _cal_desc
            else f"  Stacker (market) trained on {X_stack.shape[1]} features."
        )

        # --- No-market stacker (dc + no-market-xgb + deep) ---
        X_nm = np.hstack([dc_oof, xgb_nm_oof, deep_oof])
        self.stacker_nomarket = LogisticRegression(
            C=0.5, max_iter=1000, random_state=42
        )
        self.stacker_nomarket.fit(X_nm, y)
        try:
            self.rps_weights_nomarket = self.fit_rps_weights(
                [dc_oof, xgb_nm_oof, deep_oof], y
            )
        except Exception as e:
            print(f"    RPS weight fit (no-market) failed: {e}")
            self.rps_weights_nomarket = None
        if self.rps_weights_nomarket is not None:
            blend_nm = (
                self.rps_weights_nomarket[0] * dc_oof
                + self.rps_weights_nomarket[1] * xgb_nm_oof
                + self.rps_weights_nomarket[2] * deep_oof
            )
            blend_nm /= blend_nm.sum(axis=1, keepdims=True)
            self.stacker_nomarket_cal = self._fit_calibrator(blend_nm, y)
        else:
            self.stacker_nomarket_cal = self._fit_calibrator(
                self.stacker_nomarket.predict_proba(X_nm), y
            )
        print(
            f"  Stacker (no-market) trained on {X_nm.shape[1]} features; "
            f"T={self.stacker_nomarket_cal.get('T', 1.0):.3f}"
            if self.stacker_nomarket_cal
            else f"  Stacker (no-market) trained on {X_nm.shape[1]} features."
        )

        # Refit final models on all data for deployment
        print("  Refitting final XGBoost (market) on full data ...")
        self.xgb_model, self.xgb_features = train_xgb(features)
        print("  Refitting final XGBoost (no-market) on full data ...")
        self.xgb_nomarket_model, self.xgb_nomarket_features = train_xgb(features_nm)

        if use_deep and _TORCH_OK:
            print("  Refitting final deep LSTM on full data ...")
            try:
                from .deep import train_deep

                result = train_deep(
                    features, epochs=15, save_path="data/processed/deep_model.pt"
                )
                self.deep_model = result["model"]
                self.deep_meta = result["meta"]
                self.deep_weights = "data/processed/deep_model.pt"
            except Exception as e:
                print(f"    Deep training skipped: {e}")
                self.deep_model = None

    @staticmethod
    def _fit_temperature(probs: np.ndarray, y: np.ndarray) -> Optional[dict]:
        """Temperature-scale a probability array on the OOF data.

        Returns {"method": "temperature", "T": float} or None on failure.
        """
        try:
            from scipy.optimize import minimize_scalar

            probs = np.asarray(probs, dtype=float)
            p = np.clip(probs, 1e-7, 1 - 1e-7)
            logits = np.log(p / (1 - p))
            n = len(y)

            def nll(T):
                z = logits / T
                z -= z.max(axis=1, keepdims=True)
                e = np.exp(z)
                soft = e / e.sum(axis=1, keepdims=True)
                return -np.log(np.clip(soft[np.arange(n), y], 1e-15, 1.0)).mean()

            res = minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded")
            T = float(res.x)
            return {"method": "temperature", "T": T}
        except Exception:
            return None

    @staticmethod
    def fit_rps_weights(
        components: list, y: np.ndarray, n_restarts: int = 12, seed: int = 42
    ) -> np.ndarray:
        """Convex weights over component (n,3) probability vectors that minimize
        Ranked Probability Score (the metric we actually report).

        Fits on out-of-fold predictions so the components' OOF errors are
        honestly combined. Returns non-negative weights summing to 1.
        """
        from scipy.optimize import minimize
        from pl_predict.evaluation.metrics import ranked_probability_score

        C = np.stack(
            [np.asarray(c, dtype=float) for c in components], axis=1
        )  # (n, m, 3)
        m = C.shape[1]
        y = np.asarray(y)
        rng = np.random.RandomState(seed)

        def blend_probs(w: np.ndarray) -> np.ndarray:
            p = np.tensordot(w, C, axes=(0, 1))  # (n, 3)
            s = p.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return p / s

        def obj(w: np.ndarray) -> float:
            return float(ranked_probability_score(y, blend_probs(w)))

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * m

        starts = [np.ones(m) / m]
        starts += [rng.dirichlet(np.ones(m)) for _ in range(n_restarts)]

        best_w, best_rps = np.ones(m) / m, np.inf
        for w0 in starts:
            try:
                res = minimize(
                    obj,
                    np.asarray(w0, dtype=float),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-12},
                )
            except Exception:
                continue
            if np.isfinite(res.fun) and res.fun < best_rps:
                best_rps = float(res.fun)
                best_w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
                s = best_w.sum()
                best_w = best_w / s if s > 0 else np.ones(m) / m
        print(f"  RPS-blend weights={np.round(best_w, 3)}  OOF RPS={best_rps:.4f}")
        return best_w

    @staticmethod
    def _rps_blend(components: list, weights: np.ndarray) -> np.ndarray:
        """Weighted, normalized 3-class blend of component probability vectors."""
        comps = [np.asarray(c, dtype=float) for c in components]
        w = np.asarray(weights, dtype=float)[: len(comps)]
        s = float(w.sum())
        if s <= 0 or not comps:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        w = w / s
        out = np.zeros_like(comps[0])
        for wi, ci in zip(w, comps):
            out = out + wi * ci
        total = out.sum() if out.ndim == 1 else out.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            fallback = np.full_like(out, 1 / 3)
            if out.ndim == 2:
                fallback[:, :] = 1 / 3
            return fallback
        return out / total

    @staticmethod
    def _fit_conformal(p_cal: np.ndarray, y: np.ndarray) -> dict:
        """Conformal quantiles from calibration predictions.

        Nonconformity score s = 1 - p[true]. The quantile q_alpha =
        ceil((n+1)(1-alpha))/n of the scores gives marginal coverage >= 1-alpha.
        """
        n = len(y)
        scores = 1.0 - p_cal[np.arange(n), y]

        def quantile(alpha: float) -> float:
            k = int(np.ceil((n + 1) * (1 - alpha)))
            return (
                float(np.partition(scores, k - 1)[k - 1])
                if k >= 1
                else float(scores.max())
            )

        q80, q90 = quantile(0.20), quantile(0.10)
        cov80 = float(np.mean(1.0 - p_cal[np.arange(n), y] <= q80))
        cov90 = float(np.mean(1.0 - p_cal[np.arange(n), y] <= q90))
        return {
            "alpha_80": q80,
            "alpha_90": q90,
            "n": n,
            "coverage_80": cov80,
            "coverage_90": cov90,
        }

    @staticmethod
    def _fit_calibrator(probs: np.ndarray, y: np.ndarray) -> Optional[dict]:
        """Fit CV-averaged per-class isotonic + conformal quantiles.

        The isotonic mapping is fit with 5-fold internal CV on the OOF
        probabilities (so it never sees its own predictions). Folds are
        chronological (non-shuffled) because the OOF rows arrive in time
        order — random folds leak future structure into the calibration fit.
        Isotonic is a monotone transform, so it subsumes temperature scaling
        on its own.
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.model_selection import KFold

            probs = np.asarray(probs, dtype=float)
            n = len(y)
            iso_models = []  # per class: [5 IsotonicRegression]
            iso_pred = np.zeros_like(probs)
            kf = KFold(n_splits=5, shuffle=False)
            for c in range(probs.shape[1]):
                y_bin = (y == c).astype(int)
                pred_c = np.zeros(n)
                models = []
                for tr, va in kf.split(probs):
                    m = IsotonicRegression(out_of_bounds="clip")
                    m.fit(probs[tr, c], y_bin[tr])
                    pred_c[va] = m.predict(probs[va, c])
                    models.append(m)
                iso_models.append(models)
                iso_pred[:, c] = pred_c
            iso_pred = np.clip(iso_pred, 1e-6, 1 - 1e-6)
            iso_pred = iso_pred / iso_pred.sum(axis=1, keepdims=True)
            cal = {"method": "isotonic", "isotonic": iso_models}
            cal["conformal"] = EnsemblePredictor._fit_conformal(iso_pred, y)
            return cal
        except Exception as e:
            print(f"  isotonic calibrator failed ({e}); falling back to temperature")
            return EnsemblePredictor._fit_temperature(probs, y)

    @staticmethod
    def _apply_calibration(probs: np.ndarray, calibrator: Optional[dict]) -> np.ndarray:
        if not calibrator:
            return probs
        method = calibrator.get("method")
        if method == "isotonic" and calibrator.get("isotonic"):
            try:
                p = np.clip(probs, 1e-7, 1 - 1e-7)
                out = np.zeros_like(p)
                for c, models in enumerate(calibrator["isotonic"]):
                    vals = [m.predict([float(p[c])])[0] for m in models]
                    out[c] = float(np.mean(vals))
                out = np.clip(out, 1e-6, 1 - 1e-6)
                return out / out.sum()
            except Exception:
                pass
        if method == "temperature":
            try:
                T = float(calibrator["T"])
                p = np.clip(probs, 1e-7, 1 - 1e-7)
                logits = np.log(p / (1 - p)) / T
                z = logits - logits.max()
                e = np.exp(z)
                return e / e.sum()
            except Exception:
                return probs
        return probs

    @staticmethod
    def conformal_sets(probs: np.ndarray, calibrator: Optional[dict]) -> Optional[dict]:
        """{label: set of outcome labels} for 90% and 80% conformal levels."""
        if not calibrator:
            return None
        conf = calibrator.get("conformal")
        if not conf:
            return None
        labels = ["H", "D", "A"]
        out = {}
        for level, key in (("90", "alpha_90"), ("80", "alpha_80")):
            q = conf.get(key)
            if q is None:
                continue
            out[f"conformal_{level}"] = [
                labels[i] for i in range(3) if 1.0 - probs[i] <= q
            ]
        return out

    @staticmethod
    def _to_matches(features: pl.DataFrame) -> pl.DataFrame:
        """Build a mini match frame (goals + teams) from a feature frame."""
        if "target_home_goals" in features.columns:
            return features.select(
                [
                    pl.col("home_team"),
                    pl.col("away_team"),
                    pl.col("target_home_goals").alias("home_goals"),
                    pl.col("target_away_goals").alias("away_goals"),
                ]
            )
        return features

    def _out_of_fold_predictions(self, features, features_nm, n_folds=5, use_deep=True):
        """Chronologically-ordered OOF predictions for DC, market-XGB,
        no-market-XGB and deep LSTM."""
        n = len(features)
        dc_oof = np.full((n, 3), 1 / 3)
        xgb_oof = np.full((n, 3), 1 / 3)
        xgb_nm_oof = np.full((n, 3), 1 / 3)
        deep_oof = np.full((n, 3), 1 / 3)

        if n_folds < 1 or n <= n_folds + 1:
            raise ValueError("Need more feature rows than chronological OOF folds")

        # Reserve the first block as an initial training window.  Every
        # validation block is predicted by a model trained only on the past.
        bounds = np.linspace(0, n, n_folds + 2, dtype=int)
        oof_mask = np.zeros(n, dtype=bool)

        for k in range(n_folds):
            vk0, vk1 = int(bounds[k + 1]), int(bounds[k + 2])
            val_feat = features.slice(vk0, vk1 - vk0)
            val_nm = features_nm.slice(vk0, vk1 - vk0)
            trn_feat = features.slice(0, vk0)
            trn_nm = features_nm.slice(0, vk0)
            oof_mask[vk0:vk1] = True
            print(
                f"  Fold {k + 1}/{n_folds}: train={len(trn_feat)} val={len(val_feat)}"
            )

            # --- Dixon-Coles (per-fold fit on train matches) ---
            try:
                trn_mini = self._to_matches(trn_feat)
                dc_params = fit_dixon_coles(trn_mini)
                for i in range(vk0, vk1):
                    h = features["home_team"][i]
                    a = features["away_team"][i]
                    if h in dc_params["teams"] and a in dc_params["teams"]:
                        dc = predict_dixon_coles(dc_params, h, a)
                        dc_oof[i] = [dc["home_win"], dc["draw"], dc["away_win"]]
            except Exception as e:
                print(f"    DC fold failed: {e}")

            # --- XGBoost (market features) ---
            try:
                xgb_model, xgb_cols = train_xgb(trn_feat)
                xgb_oof[vk0:vk1] = predict_xgb(xgb_model, val_feat, xgb_cols)
            except Exception as e:
                print(f"    XGB(market) fold failed: {e}")

            # --- XGBoost (no market features) ---
            try:
                xgb_nm_model, xgb_nm_cols = train_xgb(trn_nm)
                xgb_nm_oof[vk0:vk1] = predict_xgb(xgb_nm_model, val_nm, xgb_nm_cols)
            except Exception as e:
                print(f"    XGB(no-market) fold failed: {e}")

            # --- Deep LSTM ---
            if use_deep and _TORCH_OK:
                try:
                    from .deep import train_deep, build_tensors

                    res = train_deep(trn_feat, epochs=10, save_path=None)
                    fold_model, fold_meta = res["model"], res["meta"]
                    dataset, _ = build_tensors(
                        features,
                        window=fold_meta.get("window", 8),
                        feature_cols=fold_meta.get("feature_cols"),
                        team_to_id=fold_meta.get("team_to_id"),
                    )
                    fold_model.eval()
                    import torch

                    out = []
                    with torch.no_grad():
                        for j in range(0, len(dataset), 256):
                            batch = [
                                dataset[jj]
                                for jj in range(j, min(j + 256, len(dataset)))
                            ]
                            xh = torch.stack([b[0] for b in batch])
                            xa = torch.stack([b[1] for b in batch])
                            hid = torch.stack([b[2] for b in batch])
                            aid = torch.stack([b[3] for b in batch])
                            logits = fold_model(xh, xa, hid, aid)
                            out.append(torch.softmax(logits, dim=1).numpy())
                    probs = np.vstack(out)
                    row_ids = getattr(dataset, "row_ids", [])
                    for j, rid in enumerate(row_ids):
                        if rid is not None and vk0 <= int(rid) < vk1:
                            deep_oof[int(rid)] = probs[j]
                except Exception as e:
                    print(f"    Deep fold failed: {e}")

        return dc_oof, xgb_oof, xgb_nm_oof, deep_oof, oof_mask

    def _fit_stacker(self, features, matches):
        """Legacy in-sample stacker fit (kept for compatibility)."""
        n = len(features)
        dc_preds = np.full((n, 3), 1 / 3)
        for i in range(n):
            try:
                if isinstance(features, pl.DataFrame):
                    h = features.row(i)[features.columns.index("home_team")]
                    a = features.row(i)[features.columns.index("away_team")]
                else:
                    h, a = features.iloc[i]["home_team"], features.iloc[i]["away_team"]
                if (
                    self.dc_params
                    and h in self.dc_params["teams"]
                    and a in self.dc_params["teams"]
                ):
                    dc = predict_dixon_coles(self.dc_params, h, a)
                    dc_preds[i] = [dc["home_win"], dc["draw"], dc["away_win"]]
            except Exception:
                pass

        xgb_probs = predict_xgb(self.xgb_model, features, self.xgb_features)

        # Deep batch predictions (if available)
        deep_probs = None
        if self.deep_model is not None:
            try:
                deep_probs = self._predict_deep_batch(features)
            except Exception:
                deep_probs = None

        if deep_probs is not None:
            X_stack = np.hstack([dc_preds, xgb_probs, deep_probs])
        else:
            X_stack = np.hstack([dc_preds, xgb_probs])

        if isinstance(features, pl.DataFrame):
            y = features["target_result"].to_numpy()
        else:
            y = features["target_result"].values

        if y.dtype.kind in ("U", "O"):
            if set(y) <= {"H", "D", "A"}:
                y = np.array([{"H": 0, "D": 1, "A": 2}[v] for v in y])
            else:
                from sklearn.preprocessing import LabelEncoder

                y = LabelEncoder().fit_transform(y)

        self.stacker = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        self.stacker.fit(X_stack, y)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _ensure_state(self):
        """Build per-team form + deep-sequence caches (vectorized + numba).

        Computed once per loaded model; makes repeated predictions fast.
        """
        if getattr(self, "_state", None) is not None:
            return self._state
        state = {"team_form": {}, "deep_seqs": {}}

        # --- Team form from matches (chronological EWMA, numba-accelerated) ---
        try:
            from pl_predict.models.numba_kernels import ewma_last, build_sequence

            m = self._matches_cache
            if m is None:
                m = pl.read_parquet(str(Path("data/processed/matches.parquet")))
                self._matches_cache = m
            ms = m.sort("date")
            home_arr = ms["home_team"].to_numpy()
            away_arr = ms["away_team"].to_numpy()
            hg = ms["home_goals"].to_numpy()
            ag = ms["away_goals"].to_numpy()
            valid = np.isfinite(hg) & np.isfinite(ag)
            teams = sorted(set(home_arr) | set(away_arr))

            for t in teams:
                h_idx = np.flatnonzero((home_arr == t) & valid)
                a_idx = np.flatnonzero((away_arr == t) & valid)
                idx = np.sort(np.concatenate([h_idx, a_idx]))
                if idx.size == 0:
                    continue
                is_home = np.isin(idx, h_idx)
                hgv, agv = hg[idx], ag[idx]
                pts = np.where(
                    is_home,
                    np.where(hgv > agv, 3.0, np.where(hgv == agv, 1.0, 0.0)),
                    np.where(agv > hgv, 3.0, np.where(agv == hgv, 1.0, 0.0)),
                ).astype(np.float64)
                gd = np.where(is_home, hgv - agv, agv - hgv).astype(np.float64)
                gf = np.where(is_home, hgv, agv).astype(np.float64)
                ga = np.where(is_home, agv, hgv).astype(np.float64)
                entry = {
                    "form_pts_avg": ewma_last(pts, 5.0),
                    "form_gd_avg": ewma_last(gd, 5.0),
                    "form_gf_avg": ewma_last(gf, 5.0),
                    "form_ga_avg": ewma_last(ga, 5.0),
                    "match_count": int(idx.size),
                }
                if "xG_home" in m.columns and "xG_away" in m.columns:
                    xgfor = np.where(
                        is_home, ms["xG_home"][idx], ms["xG_away"][idx]
                    ).astype(np.float64)
                    xgaga = np.where(
                        is_home, ms["xG_away"][idx], ms["xG_home"][idx]
                    ).astype(np.float64)
                    entry["form_xg_for_avg"] = ewma_last(xgfor, 5.0)
                    entry["form_xg_against_avg"] = ewma_last(xgaga, 5.0)
                state["team_form"][t] = entry
        except Exception:
            pass

        # --- Deep sequences from features frame (numpy + numba) ---
        try:
            if self.deep_model is not None and self.deep_meta is not None:
                from pl_predict.models.numba_kernels import build_sequence

                f = self._feat_cache
                if f is None:
                    f = pl.read_parquet(str(Path("data/processed/features.parquet")))
                    self._feat_cache = f
                fs = f.sort("date")
                meta = self.deep_meta
                feature_cols = meta.get("feature_cols") or []
                h_cols = [c for c in feature_cols if c.startswith("home_")]
                a_cols = [c for c in feature_cols if c.startswith("away_")]
                if not h_cols or len(h_cols) != len(a_cols):
                    a_cols = h_cols
                if h_cols:
                    window = int(meta.get("window", 8))
                    ncol = len(h_cols)
                    home_vals = np.column_stack(
                        [
                            np.nan_to_num(
                                fs[c].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
                            )
                            for c in h_cols
                        ]
                    )
                    away_vals = np.column_stack(
                        [
                            np.nan_to_num(
                                fs[c].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
                            )
                            for c in a_cols
                        ]
                    )
                    fh_arr = fs["home_team"].to_numpy()
                    fa_arr = fs["away_team"].to_numpy()
                    fteams = set(state["team_form"].keys())
                    for t in fteams:
                        h_idx = np.flatnonzero(fh_arr == t)
                        a_idx = np.flatnonzero(fa_arr == t)
                        idx = np.sort(np.concatenate([h_idx, a_idx]))
                        if idx.size == 0:
                            continue
                        is_home = np.isin(idx, h_idx)
                        own = np.where(is_home[:, None], home_vals[idx], away_vals[idx])
                        state["deep_seqs"][t] = build_sequence(own, window, ncol)
        except Exception:
            pass

        self._state = state
        return state

    def _build_feature_row(
        self, home: str, away: str, date_str: Optional[str] = None
    ) -> Optional[pl.DataFrame]:
        """One-row feature frame built from cached team form (no row iteration)."""
        state = self._ensure_state()
        tf = state.get("team_form", {})
        hf, af = tf.get(home), tf.get(away)
        if hf is None or af is None:
            return None

        row = {
            "date": date_str or datetime.now(),
            "home_team": home,
            "away_team": away,
            "season": "2026-27",
            "match_id": 0,
        }
        for k in ["form_pts_avg", "form_gd_avg", "form_gf_avg", "form_ga_avg"]:
            row[f"home_{k}"] = hf[k]
            row[f"away_{k}"] = af[k]
        for k in ["form_xg_for_avg", "form_xg_against_avg"]:
            row[f"home_{k}"] = hf.get(k)
            row[f"away_{k}"] = af.get(k)
        row["home_match_count"] = hf["match_count"]
        row["away_match_count"] = af["match_count"]
        row["home_days_rest"] = None
        row["away_days_rest"] = None
        for prefix in ["", "_close", "_max", "_avg"]:
            row[f"market_home_prob{prefix}"] = None
            row[f"market_draw_prob{prefix}"] = None
            row[f"market_away_prob{prefix}"] = None
        row["us_home_prob"] = None
        row["us_draw_prob"] = None
        row["us_away_prob"] = None
        row["xG_home"] = None
        row["xG_away"] = None
        row["form_gd_avg_diff"] = hf["form_gd_avg"] - af["form_gd_avg"]
        row["form_pts_avg_diff"] = hf["form_pts_avg"] - af["form_pts_avg"]
        row["is_home_team_stronger"] = (
            1 if hf["form_pts_avg"] > af["form_pts_avg"] else 0
        )
        row["rest_advantage"] = None

        # Prior-season squad strength (from match date, leakage-free)
        try:
            prior = _prior_understat_season(date_str or row["date"])
            if prior:
                hs = _squad_features(home, prior)
                asq = _squad_features(away, prior)
                for k in [
                    "squad_xg",
                    "squad_xa",
                    "squad_npxg",
                    "squad_minutes",
                    "squad_players",
                ]:
                    row[f"home_{k}"] = hs.get(k) if hs else None
                    row[f"away_{k}"] = asq.get(k) if asq else None
                ht = _tm_squad_features(home, prior)
                at = _tm_squad_features(away, prior)
                for k in ["squad_value", "avg_age", "squad_size"]:
                    row[f"home_{k}"] = ht.get(k) if ht else None
                    row[f"away_{k}"] = at.get(k) if at else None
        except Exception:
            pass

        # In-season player features: no matches played yet for a future fixture
        for k in [
            "szn_player_xg",
            "szn_player_xa",
            "szn_minutes",
            "star_share",
            "star_xg_form",
            "players_used_8",
        ]:
            row[f"home_{k}"] = 0.0
            row[f"away_{k}"] = 0.0

        for t in [
            "target_result",
            "target_home_goals",
            "target_away_goals",
            "target_over_2_5",
            "target_btts",
            "target_total_goals",
            "target_goal_diff",
        ]:
            row[t] = None

        # Guarantee every column the tree models expect is present (None if n/a)
        for cols in (
            getattr(self, "xgb_features", None) or [],
            getattr(self, "xgb_nomarket_features", None) or [],
        ):
            for c in cols:
                if c not in row:
                    row[c] = None
        return pl.DataFrame([row])

    def _predict_deep_batch(self, features: pl.DataFrame) -> np.ndarray:
        """Return deep model probabilities (N, 3) for a feature frame."""
        if self.deep_model is None:
            return None
        try:
            import torch

            dataset, meta = build_tensors(
                features,
                window=self.deep_meta.get("window", 8),
                feature_cols=self.deep_meta.get("feature_cols"),
            )
            if len(dataset) == 0:
                return None
            model = self.deep_model
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(dataset), 256):
                    batch = [dataset[j] for j in range(i, min(i + 256, len(dataset)))]
                    xh = torch.stack([b[0] for b in batch])
                    xa = torch.stack([b[1] for b in batch])
                    hid = torch.stack([b[2] for b in batch])
                    aid = torch.stack([b[3] for b in batch])
                    logits = model(xh, xa, hid, aid)
                    out.append(torch.softmax(logits, dim=1).numpy())
            probs = np.vstack(out)
            # Align back to the full feature frame via match_id (deep dataset may skip rows)
            full = np.full((len(features), 3), 1 / 3)
            row_ids = getattr(dataset, "row_ids", None)
            for i, rid in enumerate(row_ids):
                if rid is not None:
                    full[int(rid)] = probs[i]
            return full
        except Exception:
            return None

    def _predict_deep_single(self, home: str, away: str) -> Optional[np.ndarray]:
        """Deep LSTM probabilities (3,) for a single match using cached sequences."""
        if self.deep_model is None or self.deep_meta is None:
            return None
        try:
            import torch

            state = self._ensure_state()
            seqs = state.get("deep_seqs", {})
            if home not in seqs or away not in seqs:
                return None
            meta = self.deep_meta
            team_to_id = meta.get("team_to_id") or {}
            if home not in team_to_id or away not in team_to_id:
                return None

            xh = torch.from_numpy(seqs[home][None]).float()
            xa = torch.from_numpy(seqs[away][None]).float()
            hid = torch.tensor([team_to_id[home]], dtype=torch.long)
            aid = torch.tensor([team_to_id[away]], dtype=torch.long)
            self.deep_model.eval()
            with torch.no_grad():
                logits = self.deep_model(xh, xa, hid, aid)
            return torch.softmax(logits, dim=1).numpy()[0]
        except Exception:
            return None

    def predict_single(
        self, home: str, away: str, date_str: Optional[str] = None
    ) -> dict:
        """Full-ensemble prediction for one match: DC + XGB + Deep -> stacker.

        Also returns the DC scoreline matrix and derived markets.
        """
        if self.dc_params is None:
            return {"error": "Model not trained"}
        if home not in self.dc_params["teams"] or away not in self.dc_params["teams"]:
            known = sorted(self.dc_params["teams"])
            return {"error": f"Unknown team(s). Known teams: {', '.join(known[:8])}..."}

        dc = predict_dixon_coles(self.dc_params, home, away)

        # Single-row feature frame for the tree model (from cached team form)
        feat = None
        if self.xgb_model is not None:
            try:
                if getattr(self, "_matches_cache", None) is None:
                    self._matches_cache = pl.read_parquet(
                        str(Path("data/processed/matches.parquet"))
                    )
                feat = self._build_feature_row(home, away, date_str=date_str)
            except Exception:
                feat = None

        breakdown = {"dc": [dc["home_win"], dc["draw"], dc["away_win"]]}

        xgb_probs = None
        if self.xgb_model is not None and feat is not None:
            try:
                xgb_probs = predict_xgb(self.xgb_model, feat, self.xgb_features)[0]
                breakdown["xgboost"] = [float(x) for x in xgb_probs]
            except Exception:
                xgb_probs = None

        deep_probs = None
        if self.deep_model is not None:
            deep_probs = self._predict_deep_single(home, away)
            if deep_probs is not None:
                breakdown["deep"] = [float(x) for x in deep_probs]

        # Blend components through the appropriate stacker
        # (XGB is only trustworthy when betting-market features are present)
        has_market = False
        if feat is not None:
            has_market = feat["market_home_prob"][0] is not None

        base = np.array([dc["home_win"], dc["draw"], dc["away_win"]])
        deep_use = (
            deep_probs if deep_probs is not None else np.array([1 / 3, 1 / 3, 1 / 3])
        )

        if not has_market:
            # Future fixtures (no betting data): dc + no-market-XGB + deep
            xgb_use = None
            if (
                getattr(self, "xgb_nomarket_model", None) is not None
                and feat is not None
            ):
                try:
                    xgb_use = predict_xgb(
                        self.xgb_nomarket_model, feat, self.xgb_nomarket_features
                    )[0]
                    breakdown["xgboost_nomarket"] = [float(x) for x in xgb_use]
                except Exception:
                    xgb_use = None

            weights_nm = getattr(self, "rps_weights_nomarket", None)
            if weights_nm is not None:
                comps = [base]
                wsel = [0]
                if xgb_use is not None:
                    comps.append(xgb_use)
                    wsel.append(1)
                comps.append(deep_use)
                wsel.append(2)
                ens = self._rps_blend(comps, np.asarray(weights_nm, dtype=float)[wsel])
                ens = self._apply_calibration(
                    ens, getattr(self, "stacker_nomarket_cal", None)
                )
                base = ens
                breakdown["ensemble"] = [float(x) for x in ens]
            elif getattr(self, "stacker_nomarket", None) is not None:
                components = [base]
                if xgb_use is not None:
                    components.append(xgb_use)
                components.append(deep_use)
                try:
                    X_nm = np.hstack(components).reshape(1, -1)
                    ens = self.stacker_nomarket.predict_proba(X_nm)[0]
                    ens = self._apply_calibration(
                        ens, getattr(self, "stacker_nomarket_cal", None)
                    )
                    base = ens
                    breakdown["ensemble"] = [float(x) for x in ens]
                except Exception:
                    pass
            if "ensemble" not in breakdown:
                # Fallback: DC (+ deep modifier)
                raw = base * 0.7 + deep_use * 0.3
                base = raw / raw.sum()
                breakdown["ensemble"] = [float(x) for x in base]
        elif self.stacker is not None or getattr(self, "rps_weights", None) is not None:
            components = [
                base,
                xgb_probs if xgb_probs is not None else np.array([1 / 3] * 3),
                deep_use,
            ]
            weights_m = getattr(self, "rps_weights", None)
            if weights_m is not None:
                ens = self._rps_blend(components, np.asarray(weights_m, dtype=float))
                ens = self._apply_calibration(ens, getattr(self, "stacker_cal", None))
                base = ens
                breakdown["ensemble"] = [float(x) for x in ens]
            if "ensemble" not in breakdown and self.stacker is not None:
                try:
                    X_stack = np.hstack(components).reshape(1, -1)
                    ens = self.stacker.predict_proba(X_stack)[0]
                    ens = self._apply_calibration(
                        ens, getattr(self, "stacker_cal", None)
                    )
                    base = ens
                    breakdown["ensemble"] = [float(x) for x in ens]
                except Exception:
                    pass
        if "ensemble" not in breakdown:
            breakdown["ensemble"] = [float(x) for x in base]

        result = dict(dc)
        result["home_win"] = float(base[0])
        result["draw"] = float(base[1])
        result["away_win"] = float(base[2])
        result["model_breakdown"] = breakdown
        try:
            cal = getattr(self, "stacker_nomarket_cal", None) or getattr(
                self, "stacker_cal", None
            )
            cs = self.conformal_sets(np.asarray(base, dtype=float), cal)
            if cs:
                result.update(cs)
        except Exception:
            pass
        return result

    def predict(self, home: str, away: str, date_str: Optional[str] = None) -> dict:
        """Full-ensemble prediction for a single match."""
        return self.predict_single(home, away, date_str=date_str)

    def predict_batch(self, features: pl.DataFrame) -> dict:
        """Return blended probabilities for an entire feature frame."""
        n = len(features)
        dc_preds = np.full((n, 3), 1 / 3)
        if self.dc_params:
            for i in range(n):
                try:
                    h = features.row(i)[features.columns.index("home_team")]
                    a = features.row(i)[features.columns.index("away_team")]
                    if h in self.dc_params["teams"] and a in self.dc_params["teams"]:
                        dc = predict_dixon_coles(self.dc_params, h, a)
                        dc_preds[i] = [dc["home_win"], dc["draw"], dc["away_win"]]
                except Exception:
                    pass
        xgb_probs = predict_xgb(self.xgb_model, features, self.xgb_features)
        deep_probs = self._predict_deep_batch(features)
        if deep_probs is None:
            # The stackers are always trained with a deep-model slot.  Keep
            # batch inference shape-stable when deep training is disabled.
            deep_probs = np.full((n, 3), 1 / 3)

        X = np.hstack([dc_preds, xgb_probs, deep_probs])
        stacker = self.stacker
        calibrator = self.stacker_cal
        if stacker is None:
            stacker = self.stacker_nomarket
            calibrator = self.stacker_nomarket_cal
        if stacker is None:
            return dc_preds
        probabilities = stacker.predict_proba(X)
        return np.asarray(
            [self._apply_calibration(row, calibrator) for row in probabilities]
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = "data/processed/ensemble_model.pkl"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved to {p}")

    @classmethod
    def load(cls, path: str = "data/processed/ensemble_model.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "_feat_cache"):
            obj._feat_cache = None
        if not hasattr(obj, "_matches_cache"):
            obj._matches_cache = None
        if not hasattr(obj, "_state"):
            obj._state = None
        for attr in [
            "xgb_nomarket_model",
            "xgb_nomarket_features",
            "stacker_nomarket",
            "stacker_cal",
            "stacker_nomarket_cal",
        ]:
            if not hasattr(obj, attr):
                setattr(obj, attr, None)
        for attr in ["rps_weights", "rps_weights_nomarket"]:
            if not hasattr(obj, attr):
                setattr(obj, attr, None)
        return obj
