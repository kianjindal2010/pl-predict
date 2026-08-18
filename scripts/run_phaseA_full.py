"""Phase A: 3-fold NO-DEEP fast ablation -> 5-fold DEEP full walk-forward.
Results written to phaseA_results.json after each stage.
"""

import sys
import time
import json
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
OUT = ROOT / "data" / "processed" / "phaseA_results.json"

results = {}
try:
    from pl_predict.evaluation.backtest import run_ensemble_backtest

    # --- Stage 1: 3-fold no-deep (fast ~3-5 min) ---
    t0 = time.time()
    print("=== STAGE 1: 3-fold NO-DEEP ===", flush=True)
    out3 = run_ensemble_backtest(use_deep=False, n_folds=3)
    results["3fold_nodeep"] = {
        "dc": {k: out3["dc"][k] for k in ["rps", "accuracy", "brier", "log_loss"]},
        "ensemble": {
            k: out3["ensemble"][k] for k in ["rps", "accuracy", "brier", "log_loss"]
        },
        "elapsed": time.time() - t0,
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(
        "3-fold DONE: DC RPS=%.4f  Ens RPS=%.4f  (%.0fs)"
        % (
            out3["dc"]["rps"],
            out3["ensemble"]["rps"],
            results["3fold_nodeep"]["elapsed"],
        ),
        flush=True,
    )

    # --- Stage 2: 5-fold deep (slow ~15-30 min) ---
    t1 = time.time()
    print("\n=== STAGE 2: 5-fold DEEP ===", flush=True)
    out5 = run_ensemble_backtest(use_deep=True, n_folds=5)
    results["5fold_deep"] = {
        "dc": {k: out5["dc"][k] for k in ["rps", "accuracy", "brier", "log_loss"]},
        "ensemble": {
            k: out5["ensemble"][k] for k in ["rps", "accuracy", "brier", "log_loss"]
        },
        "elapsed": time.time() - t1,
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(
        "5-fold DONE: DC RPS=%.4f  Ens RPS=%.4f  (%.0fs)"
        % (
            out5["dc"]["rps"],
            out5["ensemble"]["rps"],
            results["5fold_deep"]["elapsed"],
        ),
        flush=True,
    )

except Exception as e:
    traceback.print_exc()
    results["error"] = str(e)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

print("ALL DONE -> %s" % OUT, flush=True)
