"""Phase 3B production job: retrain with player features + regen outputs."""

import sys
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from pl_predict.data.fixtures_2627 import get_fixtures, normalize_team
from pl_predict.models.train import train_model
from pl_predict.simulation.season_sim import SeasonSimulator

print("=== 1/3 RETRAIN PRODUCTION ENSEMBLE (78 features) ===", flush=True)
predictor = train_model("ensemble")

print("\n=== 2/3 REGENERATE FIXTURE PREDICTIONS ===", flush=True)
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

print("\n=== 3/3 SEASON SIMULATION (10k) ===", flush=True)
sim = SeasonSimulator(season="2026-27", source="fixtures_2627")
sim.run(n_simulations=10000)
print("ALL DONE", flush=True)
