"""Full deep walk-forward with player features (v3), logs in real-time."""

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pl_predict.evaluation.backtest import run_ensemble_backtest

t0 = time.time()
out = run_ensemble_backtest(use_deep=True, n_folds=5)
print(
    "WALK-FORWARD DC:      RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f"
    % (
        out["dc"]["rps"],
        out["dc"]["accuracy"],
        out["dc"]["brier"],
        out["dc"]["log_loss"],
    ),
    flush=True,
)
print(
    "WALK-FORWARD ENSEMBLE:RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f"
    % (
        out["ensemble"]["rps"],
        out["ensemble"]["accuracy"],
        out["ensemble"]["brier"],
        out["ensemble"]["log_loss"],
    ),
    flush=True,
)
print("elapsed %.0fs" % (time.time() - t0), flush=True)
print("DONE", flush=True)
