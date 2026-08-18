"""Phase 4: deep walk-forward (isotonic+conformal) -> retrain -> regen outputs."""

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from pl_predict.data.fixtures_2627 import get_fixtures, normalize_team
from pl_predict.evaluation.backtest import run_ensemble_backtest
from pl_predict.models.train import train_model
from pl_predict.simulation.season_sim import SeasonSimulator

print("=== 1/4 DEEP WALK-FORWARD (isotonic calibration) ===", flush=True)

out = run_ensemble_backtest(use_deep=True, n_folds=5)
e, d = out["ensemble"], out["dc"]
print(
    "WF DC:       RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f ECE(ovr)=%.4f"
    % (d["rps"], d["accuracy"], d["brier"], d["log_loss"], d["ece_ovr"]),
    flush=True,
)
print(
    "WF ENSEMBLE: RPS=%.4f Acc=%.3f Brier=%.4f LogLoss=%.4f ECE(ovr)=%.4f"
    % (e["rps"], e["accuracy"], e["brier"], e["log_loss"], e["ece_ovr"]),
    flush=True,
)

print("\n=== 2/4 RETRAIN PRODUCTION ENSEMBLE ===", flush=True)
predictor = train_model("ensemble")

print("\n=== 3/4 REGENERATE FIXTURE PREDICTIONS ===", flush=True)
fixtures_out = {"season": "2026-27", "fixtures": []}
for f in get_fixtures():
    home = normalize_team(f["home_team"])
    away = normalize_team(f["away_team"])
    try:
        p = predictor.predict(home, away, date_str=f["date"])
        if p and "error" not in p:
            fixtures_out["fixtures"].append({**f, "prediction": p})
    except Exception as ex:
        print("  predict failed", home, away, ex)
with open(ROOT / "data/processed/fixtures_2627_pred.json", "w", encoding="utf-8") as fh:
    json.dump(fixtures_out, fh)
print("  wrote", len(fixtures_out["fixtures"]), "fixtures", flush=True)

print("\n=== 4/4 SEASON SIMULATION (10k) ===", flush=True)
sim = SeasonSimulator(season="2026-27", source="fixtures_2627")
sim.run(n_simulations=10000)
print("ALL DONE", flush=True)
