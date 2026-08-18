import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pl_predict.evaluation.backtest import run_ensemble_backtest
from pl_predict.simulation.season_sim import SeasonSimulator

# Full walk-forward backtest (5 folds, deep, 10k sims)
print("=== FULL WALK-FORWARD BACKTEST ===")
out = run_ensemble_backtest(use_deep=True, n_folds=5)
print("\n=== FULL SEASON SIMULATION (10k) ===")
sim = SeasonSimulator(season="2026-27", source="walkforward_preds")
results = sim.run(n_simulations=10000)
print("Done.")
