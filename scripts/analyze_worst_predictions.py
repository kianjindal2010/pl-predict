"""Diagnose the model's worst predictions.

Reads walkforward_preds.parquet and classifies each bad call:
- 'MODEL ERROR (confident miss)'  -> predicted winner with high confidence, wrong
- 'FREAK RESULT (model honest)'   -> the actual outcome was a low-prob event
- otherwise 'wrong, moderate'

Writes data/processed/worst_predictions.md and prints the top rows.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import polars as pl

p = pl.read_parquet(ROOT / "data/processed/walkforward_preds.parquet")
if "ens_h" not in p.columns:
    raise SystemExit("walkforward_preds.parquet missing ens_* columns")

rows = []
for r in p.iter_rows(named=True):
    res = r["result"]
    probs = {"H": r["ens_h"], "D": r["ens_d"], "A": r["ens_a"]}
    p_actual = probs[res]
    wrong = {k: v for k, v in probs.items() if k != res}
    predicted = max(probs, key=probs.get)
    conf = probs[predicted]
    # per-match RPS
    true_onehot = [1.0 if k == res else 0.0 for k in ("H", "D", "A")]
    cum_p = 0.0
    cum_t = 0.0
    rps = 0.0
    for k in ("H", "D", "A"):
        cum_p += probs[k]
        cum_t += true_onehot[("H", "D", "A").index(k)]
        rps += (cum_p - cum_t) ** 2
    rps /= 2
    correct = predicted == res
    if not correct and conf >= 0.60:
        verdict = "MODEL ERROR (confident miss)"
    elif p_actual < 0.12 and correct:
        verdict = "FREAK RESULT (model honest)"
    elif not correct:
        verdict = "wrong, moderate"
    else:
        verdict = "correct"
    rows.append(
        {
            "date": str(r["date"])[:10],
            "season": r["season"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "result": res,
            "ph": round(r["ens_h"], 3),
            "pd": round(r["ens_d"], 3),
            "pa": round(r["ens_a"], 3),
            "predicted": predicted,
            "conf": round(conf, 3),
            "p_actual": round(p_actual, 3),
            "rps": round(rps, 4),
            "correct": correct,
            "verdict": verdict,
        }
    )

df = pl.DataFrame(rows)
worst = df.filter(~pl.col("correct")).sort("conf", descending=True)

print("\n=== WORST PREDICTIONS: CONFIDENT MISSES (model error) ===")
cm = worst.filter(pl.col("verdict") == "MODEL ERROR (confident miss)")
for r in cm.head(15).to_dicts():
    print(
        f"  {r['date']} {r['season']}  {r['home_team']} v {r['away_team']}  "
        f"act={r['result']}  H/D/A={r['ph']}/{r['pd']}/{r['pa']}  pred={r['predicted']}@{r['conf']:.0%}  "
        f"p(actual)={r['p_actual']:.1%}  RPS={r['rps']}"
    )

print("\n=== SURPRISES: LOW-PROB OUTCOMES THAT HAPPENED (model honest) ===")
for r in worst.filter(pl.col("p_actual") < 0.10).sort("p_actual").head(15).to_dicts():
    print(
        f"  {r['date']} {r['season']}  {r['home_team']} v {r['away_team']}  "
        f"act={r['result']}@{r['p_actual']:.1%}  H/D/A={r['ph']}/{r['pd']}/{r['pa']}  "
        f"pred={r['predicted']}@{r['conf']:.0%}"
    )

print("\n=== SUMMARY ===")
print(" total wrong:", worst.height, "of", df.height)
for v in df["verdict"].value_counts().sort("count", descending=True).to_dicts():
    print(f"   {v['verdict']:<35} {v['count']}")
tot = df.height
cm_n = cm.height
print(f" confident-miss rate: {cm_n}/{tot} ({cm_n / tot:.1%})")

# markdown report
lines = [
    "# Worst-prediction diagnostic\n",
    f"_From `walkforward_preds.parquet`, {df.height} held-out matches_",
    "\n## Confident misses (model error)\n",
    "| Date | Season | Match | Result | H | D | A | Pred | p(actual) |",
    "|------|--------|-------|--------|---|---|---|------|-----------|",
]
for r in cm.sort("conf", descending=True).head(30).to_dicts():
    lines.append(
        f"| {r['date']} | {r['season']} | {r['home_team']} v {r['away_team']} | {r['result']} | "
        f"{r['ph']} | {r['pd']} | {r['pa']} | {r['predicted']}@{r['conf']:.0%} | {r['p_actual']:.1%} |"
    )
lines.append("\n## Freak results (model was honest)\n")
lines.append("| Date | Season | Match | Result | p(actual) | H | D | A |")
lines.append("|------|--------|-------|--------|-----------|---|---|---|")
for r in worst.filter(pl.col("p_actual") < 0.10).sort("p_actual").head(30).to_dicts():
    lines.append(
        f"| {r['date']} | {r['season']} | {r['home_team']} v {r['away_team']} | {r['result']} | "
        f"{r['p_actual']:.1%} | {r['ph']} | {r['pd']} | {r['pa']} |"
    )
out = "\n".join(lines)
with open(ROOT / "data/processed/worst_predictions.md", "w", encoding="utf-8") as fh:
    fh.write(out)
print("\n  -> data/processed/worst_predictions.md")
