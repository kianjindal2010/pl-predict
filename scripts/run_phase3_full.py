"""Phase 3 full job: definitive walk-forward + production retrain + outputs."""

import sys
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from pl_predict.data.fixtures_2627 import get_fixtures, normalize_team
from pl_predict.evaluation.backtest import run_ensemble_backtest
from pl_predict.models.train import train_model
from pl_predict.simulation.season_sim import SeasonSimulator

# 1. Definitive walk-forward backtest (deep, 5 folds)
print("=== 1/4 FULL WALK-FORWARD (deep, 5 folds) ===", flush=True)

out = run_ensemble_backtest(use_deep=True, n_folds=5)
print(
    "FULL-WF DC:      RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f"
    % (
        out["dc"]["rps"],
        out["dc"]["accuracy"],
        out["dc"]["brier"],
        out["dc"]["log_loss"],
    ),
    flush=True,
)
print(
    "FULL-WF ENSEMBLE:RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f"
    % (
        out["ensemble"]["rps"],
        out["ensemble"]["accuracy"],
        out["ensemble"]["brier"],
        out["ensemble"]["log_loss"],
    ),
    flush=True,
)

# 2. Retrain the production model with the new features
print("\n=== 2/4 RETRAIN PRODUCTION ENSEMBLE ===", flush=True)
predictor = train_model("ensemble")

# 3. Regenerate all 380 fixture predictions
print("\n=== 3/4 REGENERATE FIXTURE PREDICTIONS ===", flush=True)
fixtures_out = {"season": "2026-27", "fixtures": []}
for f in get_fixtures():
    home = normalize_team(f["home_team"])
    away = normalize_team(f["away_team"])
    try:
        p = predictor.predict(home, away, date_str=f["date"])
        if p and "error" not in p:
            fixtures_out["fixtures"].append({**f, "prediction": p})
    except Exception as e:
        print("  predict failed", home, away, e)
with open(ROOT / "data/processed/fixtures_2627_pred.json", "w", encoding="utf-8") as fh:
    json.dump(fixtures_out, fh)
print("  wrote", len(fixtures_out["fixtures"]), "fixtures", flush=True)

# 4. Regenerate the 10,000-run season simulation
print("\n=== 4/4 SEASON SIMULATION (10k) ===", flush=True)
sim = SeasonSimulator(season="2026-27", source="fixtures_2627")
sim.run(n_simulations=10000)
print("ALL DONE", flush=True)
