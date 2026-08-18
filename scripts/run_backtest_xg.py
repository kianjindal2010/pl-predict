import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pl_predict.evaluation.backtest import run_ensemble_backtest

out = run_ensemble_backtest(use_deep=False, n_folds=3)
print()
print("=== SUMMARY (with Understat xG + squad features) ===")
print(
    "DC-only  : RPS=%.4f  Acc=%.3f  Brier=%.4f  LogLoss=%.4f"
    % (
        out["dc"]["rps"],
        out["dc"]["accuracy"],
        out["dc"]["brier"],
        out["dc"]["log_loss"],
    )
)
print(
    "Ensemble : RPS=%.4f  Acc=%.3f  Brier=%.4f  LogLoss=%.4f"
    % (
        out["ensemble"]["rps"],
        out["ensemble"]["accuracy"],
        out["ensemble"]["brier"],
        out["ensemble"]["log_loss"],
    )
)
