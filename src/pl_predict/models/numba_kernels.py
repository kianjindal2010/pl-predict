"""Numba-accelerated kernels for fast team form / sequence computation."""

import numpy as np

try:
    from numba import njit

    _NUMBA_OK = True
except Exception:  # pragma: no cover
    _NUMBA_OK = False


def _ewma_last_py(vals, halflife):
    if len(vals) == 0:
        return float("nan")
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    out = float(vals[0])
    for i in range(1, len(vals)):
        out = alpha * float(vals[i]) + (1.0 - alpha) * out
    return out


if _NUMBA_OK:

    @njit(cache=True)
    def ewma_last(vals, halflife):
        """EWMA of a 1-D float array; returns the final (current) value."""
        n = vals.shape[0]
        if n == 0:
            return np.nan
        alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
        out = vals[0]
        for i in range(1, n):
            out = alpha * vals[i] + (1.0 - alpha) * out
        return out

    @njit(cache=True)
    def roll_form(vals, halflife):
        """Return shifted EWMA array (form as of BEFORE each match).

        index j holds the EWMA over matches [0, j-1]; first value is NaN.
        """
        n = vals.shape[0]
        out = np.full(n, np.nan)
        if n == 0:
            return out
        alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
        acc = vals[0]
        for j in range(1, n):
            out[j] = acc
            acc = alpha * vals[j] + (1.0 - alpha) * acc
        return out

    @njit(cache=True)
    def build_sequence(own, window, ncol):
        """Build a (window, ncol) sequence padded with zeros from a team's own
        feature vectors (most recent last)."""
        s = np.zeros((window, ncol))
        n = own.shape[0]
        if n == 0:
            return s
        take = n if n < window else window
        s[window - take :] = own[n - take :]
        return s

else:  # pragma: no cover

    def ewma_last(vals, halflife):
        return _ewma_last_py(vals, halflife)

    def roll_form(vals, halflife):
        n = len(vals)
        out = np.full(n, np.nan)
        if n == 0:
            return out
        alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
        acc = float(vals[0])
        for j in range(1, n):
            out[j] = acc
            acc = alpha * float(vals[j]) + (1.0 - alpha) * acc
        return out

    def build_sequence(own, window, ncol):
        s = np.zeros((window, ncol))
        n = own.shape[0]
        if n == 0:
            return s
        take = n if n < window else window
        s[window - take :] = own[n - take :]
        return s


def compute_team_form(team_rows: dict, halflife: float = 5.0) -> dict:
    """Compute each team's current EWMA form from a dict of chron-ordered arrays.

    team_rows: {team: {"pts": np.ndarray, "gd": ..., "gf": ..., "ga": ...}}
    Returns: {team: {"form_pts_avg": float, "form_gd_avg": float, ...}}
    """
    out = {}
    for team, arrs in team_rows.items():
        n = len(arrs["pts"])
        out[team] = {
            "form_pts_avg": ewma_last(arrs["pts"], halflife),
            "form_gd_avg": ewma_last(arrs["gd"], halflife),
            "form_gf_avg": ewma_last(arrs["gf"], halflife),
            "form_ga_avg": ewma_last(arrs["ga"], halflife),
            "match_count": int(n),
        }
    return out
