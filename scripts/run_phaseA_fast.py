"""Phase A fast ablation: 3-fold no-deep walk-forward to validate RPS blend."""

import sys
import time
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from pl_predict.evaluation.backtest import run_ensemble_backtest

t0 = time.time()
print("=== 3-FOLD NO-DEEP WALK-FORWARD ===", flush=True)
out = run_ensemble_backtest(use_deep=False, n_folds=3)

e, d = out["ensemble"], out["dc"]
print("\n=== RESULTS ===", flush=True)
print(
    "DC:       RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f ECE=%.4f"
    % (d["rps"], d["accuracy"], d["brier"], d["log_loss"], d.get("ece", 0)),
    flush=True,
)
print(
    "ENSEMBLE: RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f ECE=%.4f"
    % (e["rps"], e["accuracy"], e["brier"], e["log_loss"], e.get("ece", 0)),
    flush=True,
)

elapsed = time.time() - t0
print("elapsed %.0fs" % elapsed, flush=True)

# Save results
results = {
    "mode": "3-fold no-deep",
    "dc": {"rps": d["rps"], "accuracy": d["accuracy"]},
    "ensemble": {"rps": e["rps"], "accuracy": e["accuracy"]},
    "elapsed": elapsed,
}
with open(ROOT / "data" / "processed" / "phaseA_fast_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE", flush=True)
