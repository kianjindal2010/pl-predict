# Premier League Prediction Engine — Research-Grade Plan

## 1. Data Pipeline Layer

**Free/Open Data Sources:**

| Source | Data | Access | Coverage |
|--------|------|--------|----------|
| **FBref** (via `soccerdata`) | Match stats, player stats, shooting, passing, defense | Python package (`soccerdata`) | 2017–present (detailed), 1992+ (basic) |
| **Understat** (via `soccerdata`) | xG, xA, xG chain, xG buildup | Python package | 2014–present |
| **WhoScored** (via `soccerdata`) | Match ratings, detailed stats | Python package | 2010–present |
| **Club Elo** (via `soccerdata`) | Elo ratings (team strength curve) | Python package | 1900–present |
| **Football-Data.co.uk** | Results, HT/FT, betting odds, match stats | CSV download | 1993–present |
| **OpenFootball/football.json** | Results, schedules, lineups | GitHub JSON (CC0) | 1992–present |
| **SoFIFA** (via `soccerdata`) | Player ratings, potential, attributes | Python package | Annual |
| **StatsBomb Open Data** | Event-level (passes, shots, etc.) | GitHub (CC0) | Selected comps |
| **Open-Meteo** | Weather (temp, precip, wind) | Free API | Historical + forecast |

**Architecture:**
```
raw/            -> fbref/ understat/ whoscored/ elo/ football_data/ sofifa/
staging/        -> 01_scrape.py, 02_clean.py, 03_merge.py
features/       -> 04_feature_engineering.py
models/         -> 05_train.py, 06_evaluate.py, 07_ensemble.py
output/         -> predictions/ reports/ visualizations/
notebooks/      -> research_analysis.ipynb
paper/          -> latex/ figures/
```

Stack: **Python 3.11+, Polars/Pandas, DuckDB**, `soccerdata`, `requests`, `beautifulsoup4`.

---

## 2. Feature Engineering (15+ feature groups)

**A. Team Strength (time-decayed)**
- Rolling avg goals scored/conceded (exponential weights, halflife=5 matches)
- xG for/against (from Understat/FBref)
- Expected points (xPts) from xG
- Shot conversion rate, save %
- Possession-adjusted metrics

**B. Elo-Based**
- Club Elo rating (current + trajectory)
- Match-specific Elo delta (home adjustment)
- Form Elo (last 10 matches weighted)

**C. Advanced Metrics**
- PPDA (passes per defensive action) — pressing intensity
- Field tilt, dangerous attack %
- Pass completion in final third
- High turnovers / high press success
- Set piece xG efficiency

**D. Squad & Personnel**
- Squad market value (Transfermarkt scrape)
- Injury-weighted squad strength (minutes/games missed)
- Manager experience & tenure
- New manager bounce flag
- Fixture congestion (days since last match)

**E. Historical Context**
- Head-to-head record (last 5 meetings)
- Venue-specific performance (home/away PPG)
- Same-opponent-strength comparison

**F. External Factors**
- Weather (rain probability, wind speed)
- Travel distance (km from home to away)
- European competition midweek flag
- International break recovery flag

**G. Market-Informed Features**
- Implied probabilities from betting odds (Pinnacle closing odds = most efficient)
- odds-implied xG (from over/under lines)
- Market movement signals (opening -> closing odds drift)

---

## 3. Model Architecture (4-Layer Hierarchical Ensemble)

**Layer 1 — Statistical Benchmarks:**
- **Poisson / Bivariate Poisson** — classic scoreline model
- **Bradley-Terry / Dixon-Coles** — time-weighted rating system with low-score correction
- **Elo with margin-of-victory** (Kelley-modified Elo)

**Layer 2 — Gradient Boosted Trees:**
- **XGBoost / LightGBM** with all engineered features
- Multi-class (H/D/A) + Poisson-regression objectives
- Hyperparameter search via Optuna
- SHAP analysis for feature importance

**Layer 3 — Deep Sequential Models:**
- **Transformer encoder** over match sequence (per team)
  - Input: last 20 matches, each encoded as a feature vector
  - Multi-head self-attention to capture form & matchup patterns
- **LSTM / GRU** baseline for comparison
- **Neural Rating Model** — embedding-based team strength learned end-to-end

**Layer 4 — Ensemble & Calibration:**
- **Stacking meta-model** (Logistic Regression or shallow NN)
- **Platt scaling / isotonic regression** for probability calibration
- **Conformal prediction** for prediction intervals (uncertainty quantification)
- **Bayesian model averaging** across layers

**Target variables (multi-output):**
1. Match outcome: P(H), P(D), P(A) — primary
2. Exact scoreline distribution — probabilistic
3. Over/under 2.5 goals probability
4. Both teams to score (BTTS) probability
5. Expected goals (home & away)

---

## 4. Evaluation Framework

**Walk-Forward Validation:**
- Train on seasons 2000-2020, test on 2021
- Slide window forward: train on 2000-2021, test on 2022, etc.
- Each week: train on all data before matchweek, predict next week

**Metrics:**
- **Ranked Probability Score (RPS)** — standard for ordered outcomes
- **Brier Score** (decomposed into refinement + calibration + uncertainty)
- **Log-loss** (cross-entropy)
- **Expected return** under fixed fractional Kelly staking (betting simulation)
- **Calibration curves** (reliability diagrams)
- **AUC-ROC** (one-vs-rest for H/D/A)

**Ablation studies** — remove each feature group, measure RPS degradation.

---

## 5. Output & Research Deliverables

**CLI Tool (`predict-pl`):**
```bash
predict-pl predict --home "Liverpool" --away "Man City" --date "2026-08-15"
# -> { "home_win": 0.42, "draw": 0.28, "away_win": 0.30, "xg_home": 1.8, "xg_away": 1.3 }
predict-pl season --simulate 10000
# -> Final table probabilities, top-4 %, relegation %
predict-pl features --importance
# -> SHAP summary plot, top-20 features
```

**Web Dashboard:**
- FastAPI backend + Plotly/Dash frontend
- Live match predictions for upcoming fixtures
- Season simulation explorer
- Model performance tracker
- Feature importance explorer

**Research Paper:**
- Title: *"A Hierarchical Ensemble Framework for Premier League Match Prediction"*
- Sections: Introduction | Data & Features | Methodology | Experiments | Discussion | Conclusion
- Cite: Dixon-Coles (1997), Rue-Salvesen (2000), Karlis-Ntzoufras (2003), Tax & Joustra (2015)

---

## 6. Project Structure

```
pl-predict/
├── data/
│   ├── raw/              # untouched scraped data
│   ├── processed/        # cleaned, merged parquet
│   └── external/         # reference data (stadiums, weather)
├── src/
│   ├── pipeline/         # scrape, clean, merge
│   ├── features/         # all feature engineering
│   ├── models/           # statistical, ml, deep, ensemble
│   ├── evaluation/       # metrics, calibration, backtesting
│   └── simulation/       # season Monte Carlo
├── notebooks/            # exploratory & research
├── config/               # league configs, hyperparams
├── tests/
├── paper/                # LaTeX source
└── README.md
```

---

## Key Design Decisions

1. **Time-decayed features** — exponential decay > fixed windows; less noise from distant matches
2. **Calibrated probabilities** — raw model outputs need Platt scaling; improper calibration destroys betting value
3. **Walk-forward, not k-fold** — temporal leakage is the #1 error in sports prediction research
4. **Betting odds as features, not targets** — market odds contain information not in match data; use them as inputs, then measure whether you beat the market
5. **Conformal prediction** — point estimates without uncertainty are not research-grade
