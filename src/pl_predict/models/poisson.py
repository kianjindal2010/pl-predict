"""Layer 1: Poisson / Dixon-Coles models — pure numpy, no pandas dependency."""

import numpy as np
import polars as pl
from scipy.stats import poisson
from scipy.optimize import minimize


def _dc_log_likelihood(params, home_goals, away_goals, home_team, away_team, n_teams):
    attack = params[:n_teams]
    defence = params[n_teams : 2 * n_teams]
    home_adv = params[-2]
    rho = params[-1]

    lambda_h = np.exp(attack[home_team] + defence[away_team] + home_adv)
    lambda_a = np.exp(attack[away_team] + defence[home_team])
    tau = np.ones_like(lambda_h)
    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)
    tau[zero_zero] = 1.0 - lambda_h[zero_zero] * lambda_a[zero_zero] * rho
    tau[zero_one] = 1.0 + lambda_h[zero_one] * rho
    tau[one_zero] = 1.0 + lambda_a[one_zero] * rho
    tau[one_one] = 1.0 - rho
    if np.any(tau <= 0) or not np.all(np.isfinite(tau)):
        return 1e12
    log_likelihood = (
        np.log(tau)
        + poisson.logpmf(home_goals, lambda_h)
        + poisson.logpmf(away_goals, lambda_a)
    )
    return float(-np.sum(log_likelihood))


def _dc_tau(h, a, lam, mu, rho):
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def fit_dixon_coles(matches) -> dict:
    """Fit Dixon-Coles model from a Polars DataFrame."""
    if isinstance(matches, pl.DataFrame):
        # Drop rows with missing goals / missing or empty team names
        matches = matches.filter(
            pl.col("home_goals").is_not_null()
            & pl.col("away_goals").is_not_null()
            & ~pl.col("home_goals").is_nan()
            & ~pl.col("away_goals").is_nan()
            & (pl.col("home_team") != "")
            & (pl.col("away_team") != "")
        )
        home_teams = matches["home_team"].to_list()
        away_teams = matches["away_team"].to_list()
        home_goals = matches["home_goals"].to_numpy().astype(float)
        away_goals = matches["away_goals"].to_numpy().astype(float)
    else:
        raise TypeError("Expected Polars DataFrame")

    teams = sorted(set(home_teams) | set(away_teams))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    h_idx = np.array([team_to_idx[t] for t in home_teams], dtype=int)
    a_idx = np.array([team_to_idx[t] for t in away_teams], dtype=int)

    n = len(teams)
    x0 = np.zeros(2 * n + 2)
    x0[-2] = 0.3
    x0[-1] = 0.0

    bounds = [(-5.0, 5.0)] * (2 * n) + [(-2.0, 2.0), (-0.5, 0.5)]
    result = minimize(
        _dc_log_likelihood,
        x0,
        args=(home_goals, away_goals, h_idx, a_idx, n),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 250, "ftol": 1e-8, "maxls": 20},
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(
            "Dixon-Coles optimization failed to produce a finite likelihood"
        )

    attack = dict(zip(teams, result.x[:n]))
    defence = dict(zip(teams, result.x[n : 2 * n]))

    return {
        "teams": teams,
        "attack": attack,
        "defence": defence,
        "home_adv": float(result.x[-2]),
        "rho": float(result.x[-1]),
        "nll": float(result.fun),
    }


def scoreline_matrix(params: dict, home: str, away: str) -> tuple:
    """Return (probs, lam, mu) — 7x7 scoreline matrix (i=home goals, j=away goals)."""
    # New or promoted teams may not exist in a historical training window.
    # Use the fitted league-average rating instead of failing the whole fold.
    attack_mean = float(np.mean(list(params["attack"].values())))
    defence_mean = float(np.mean(list(params["defence"].values())))
    home_attack = params["attack"].get(home, attack_mean)
    away_attack = params["attack"].get(away, attack_mean)
    home_defence = params["defence"].get(home, defence_mean)
    away_defence = params["defence"].get(away, defence_mean)
    lam = np.exp(home_attack + away_defence + params["home_adv"])
    mu = np.exp(away_attack + home_defence)

    max_goals = 6
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            tau = _dc_tau(i, j, lam, mu, params["rho"])
            probs[i, j] = tau * poisson.pmf(i, lam) * poisson.pmf(j, mu)

    probs /= probs.sum()
    return probs, lam, mu


def derived_markets(probs: np.ndarray, max_goals: int = 6) -> dict:
    """Derive market probabilities from a scoreline matrix."""
    home_win = float(probs[np.tril_indices(max_goals + 1, -1)].sum())  # i > j
    draw = float(np.trace(probs))
    away_win = float(probs[np.triu_indices(max_goals + 1, 1)].sum())  # j > i

    idx = np.arange(max_goals + 1)
    totals = idx[:, None] + idx[None, :]
    over_2_5 = float(probs[totals > 2.5].sum())
    over_1_5 = float(probs[totals > 1.5].sum())
    over_3_5 = float(probs[totals > 3.5].sum())
    btts = float(probs[1:, 1:].sum())  # both score >= 1
    home_cs = float(probs[:, 0].sum())  # away scores 0
    away_cs = float(probs[0, :].sum())  # home scores 0

    flat = probs.flatten()
    top_idx = np.argsort(flat)[::-1]
    top_scores = [
        {
            "home": int(i // (max_goals + 1)),
            "away": int(i % (max_goals + 1)),
            "prob": float(flat[i]),
        }
        for i in top_idx[:5]
    ]

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_1_5": over_1_5,
        "over_2_5": over_2_5,
        "over_3_5": over_3_5,
        "btts": btts,
        "clean_sheet_home": home_cs,
        "clean_sheet_away": away_cs,
        "most_likely_score": top_scores[0],
        "top_scores": top_scores,
        "score_matrix": [[float(x) for x in row] for row in probs],
    }


def predict_dixon_coles(params: dict, home: str, away: str) -> dict:
    """Predict match outcome + derived markets from fitted DC model."""
    probs, lam, mu = scoreline_matrix(params, home, away)
    markets = derived_markets(probs)
    markets["expected_home_goals"] = float(lam)
    markets["expected_away_goals"] = float(mu)
    return markets
