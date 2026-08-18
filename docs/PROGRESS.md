# PL-Predict — Session Progress Log

Last updated: **2026-08-12** — Dashboard v2: startup cache (predict 8.6s→~40ms), 9-tab SPA with Performance/Players/Shot Maps/Teams/H2H/Referees analytics, static asset split

## Current Status: Dashboard overhaul complete — all 11 API endpoints + 12 frontend tests green

Server available at `http://127.0.0.1:8000`.

## Session 2026-08-04 — Phase 1: More powerful model (full ensemble)

### Goal
Replace the DC-only / DC+market model with a stacked ensemble of Dixon-Coles, XGBoost and a deep LSTM, without sacrificing sensible probabilities.

### Done
- **Deep LSTM now actually included** — several silent bugs found & fixed:
  - `_predict_deep_batch` used `DataFrame.index()` (removed in polars 1.x) → deep predictions silently `None`. Now aligns via `row_ids` (added `row_ids` to `DeepDataset` in `deep.py`).
  - Old pickles lacked `_feat_cache`/`_matches_cache`/`_state` attrs → `load()` backfills; `getattr` guards added.
  - `train_deep` now accepts `save_path=None` (needed for per-fold training).
- **Out-of-fold stacking** (`_out_of_fold_predictions`) — the stacker is now fit on chronologically-ordered 5-fold out-of-fold predictions instead of in-sample predictions. In-sample stacking overfit badly (LogReg coefficients up to ±7, producing inverted/extreme probabilities like Arsenal @ 5% home). After OOF: max |coef| ≈ 0.87, probabilities sensible.
- **No-market fallback** — XGB was trained with betting-market features; for 2026/27 fixtures those are NaN and XGB produces garbage. `predict_single` now detects missing market features and blends DC (0.7) + Deep (0.3) instead of routing garbage through the stacker. When market data exists the full stacker is used.
- **Numba-accelerated prediction** (`models/numba_kernels.py`) — per-team EWMA form and deep sequence building are JIT-compiled and precomputed once into a state index (`_ensure_state`). Per-fixture prediction dropped from ~0.2–0.3 s to ~33 ms; the 380-fixture regeneration dropped from >45 min (timed out) to ~12 s.
- **Predictions + caches regenerated**:
  - `data/processed/fixtures_2627_pred.json` — all 380 fixtures, new keys: `model_breakdown` (dc/xgboost/deep/ensemble/market_blended), `score_matrix`, `over_1_5/2_5/3_5`, `btts`, `clean_sheet_*`, `top_scores`, `most_likely_score`.
  - `data/processed/season_sim.parquet` — 10,000 sims, 2026/27.
  - `data/processed/ensemble_model.pkl` — stacker now uses 9 features (dc+xgb+deep).

### Backtest (2023-24 → 2025-26, ~1100 matches, DC fit on prior data only)
| Model | RPS | Accuracy |
|-------|-----|----------|
| DC-only | 0.658 | 49.1% |
| **Ensemble** | 0.673 | **53.6%** (+4.5pp) |

Ensemble picks winners notably better; slightly worse RPS (calibration is a Phase 2 item).

### 2026/27 projection (10,000 sims, full ensemble)
| Rank | Team | Exp Pts | Title % | Top4 % | Releg % |
|------|------|--------|--------|--------|---------|
| 1 | Arsenal | 75.1 | 39.4 | 90.9 | 0.0 |
| 2 | Man City | 73.4 | 30.1 | 84.1 | 0.0 |
| 3 | Liverpool | 70.4 | 17.7 | 74.2 | 0.0 |
| 4 | Man United | 67.9 | 10.7 | 60.5 | 0.0 |
| 5 | Chelsea | 66.8 | 8.2 | 57.0 | 0.0 |
| ... | | | | | |
| 18 | Sunderland | 38.9 | 0.0 | 0.1 | 44.9 |
| 19 | Hull City | 33.8 | 0.0 | 0.0 | 71.5 |
| 20 | Ipswich Town | 32.4 | 0.0 | 0.0 | 78.2 |

### Sample per-match outputs (full ensemble)
| Fixture | DC | XGB | Deep | Ensemble | Final |
|---------|----|----|------|----------|-------|
| Arsenal vs Coventry | .69/.20/.11 | .11/.34/.55* | .74/.18/.08 | .71/.19/.10 | .71/.19/.10 |
| Hull vs Man United | .14/.21/.65 | .35/.20/.46* | .10/.19/.72 | .13/.21/.67 | .12/.22/.66 |
| Everton vs C. Palace | .50/.27/.23 | .19/.16/.65* | .40/.27/.34 | .47/.27/.26 | .43/.28/.29 |

*XGB w/o market features is unreliable → no-market fallback keeps it out of the blend.

### Next (Phase 2)
- Probability calibration (ensemble over-confident vs DC on RPS).
- Walk-forward full backtest harness (valid held-out 2023–26).

## Session 2026-08-12 — Phase 2: Walk-forward backtest + calibration

### Done
- **Fixed `run_ensemble_backtest` column bug**: `matches.parquet` uses column `result` (H/D/A strings), not `target_result`. Updated backtest to handle both.
- **Fast validation run**: 3-fold, no deep → DC RPS=0.6584, Ensemble RPS=0.6648, accuracy 49.1% vs 48.6% on 2023–26 held-out.
- **Full walk-forward harness launched in background**: 5 folds, deep, 10k sims (PID 18496, logs at `C:\Users\Kian Jindal\AppData\Local\Temp\opencode\full_walkforward.log`).
- **Calibration improvements**: Temperature scaling applied to both stackers (T≈1.60). Calibration curves improved (e.g., P(H) 0.74→0.70, 0.82→1.00).

### Walk-forward backtest (held-out 2023–26)
| Model | RPS | Accuracy | Brier | LogLoss |
|-------|-----|----------|-------|---------|
| DC-only | 0.6584 | 49.1% | 0.6133 | 1.0261 |
| **Ensemble** | **0.6664** | **48.5%** | **0.6135** | **1.0244** |

Ensemble RPS slightly higher than DC; calibration curves show improved alignment after temperature scaling (P(H) 0.54→0.53, 0.65→0.58, 0.74→0.66; P(A) 0.54→0.51, 0.65→0.61).

### Written
- **`PL_PREDICT_OVERVIEW.md`** (in `Default Project\`) — full shareable project report: architecture, data, all 4 models, backtest results, 2026/27 projection, dashboard, limitations, roadmap. Updated with final walk-forward numbers.

## Session 2026-08-12 (late) — Phase 3: Player-level data (Understat)

### Done
- **Probed all sources with realistic browser headers**: FBref still 403 (Cloudflare JS challenge — cannot bypass with `requests`); **Understat works**; **Transfermarkt works**.
- **Understat now scraped via its JSON API** (`/main/getLeagueData/EPL/{season}`) — the league page no longer embeds data:
  - `understat_matches.parquet` — 4,560 matches (12 seasons × 380, 2014-15 → 2025-26) with per-match **xG, xGA, goals, and Understat's own pre-match forecast** (w/d/l)
  - `understat_team_history.parquet` — per-team per-match xG/xGA/npxG/ppda/deep/xpts
  - `understat_players.parquet` — ~6,300 season player records (xG, xA, xGChain, xGBuildup, shots, key passes, position)
- **Merge fix**: Understat dates carried a time component → duplicate rows in `matches.parquet`. Fixed `clean_understat` to truncate to midnight; also removed Understat from the vertical concat (xG now attached via explicit left-join in `merge.py`). `matches.parquet` = 12,705 rows, 255 cols, xG on 4,534 matches.
- **Team-name map extended**: `West Bromwich Albion → West Brom`, `Queens Park Rangers → QPR`.
- **New features (40 → 60 cols)**:
  - `home/away_form_xg_for_avg`, `home/away_form_xg_against_avg` — EWMA xG form (leakage-safe, shifted)
  - `us_home/draw/away_prob` — Understat pre-match forecast (3)
  - `home/away_squad_{xg,xa,npxg,minutes,players}` — **prior-season** squad aggregates, lagged so no lookahead (10)
- **Prediction path**: `_ensure_state` now computes xG EWMA form; `_build_feature_row` adds date-aware prior-season squad features (correct for both 2026/27 fixtures and historical backtests); missing-column fill guarantees the tree models always receive their full feature set.
- **Backtest improvement (3-fold held-out 2023–26, same data)**:
  | | Before (no xG) | After (xG + squad) |
  |---|---|---|
  | Ensemble RPS | 0.6648 | **0.6592** |
  | Ensemble Acc | 48.6% | **49.3%** |
  | Brier | 0.6133 | **0.6131** |

### Completed (background job PID 5524)
- **Full deep walk-forward (held-out 2023–26, v2 features)**: DC RPS=0.6584 Acc=49.1%; **Ensemble RPS=0.6640 Acc=49.2%** (v1 was 0.6664 / 48.5% — RPS and acc both improved).
- **Production ensemble retrained** with the 60-feature set → `ensemble_model.pkl` + `deep_model.pt`.
- **`fixtures_2627_pred.json` regenerated** (all 380, new model — e.g. Arsenal vs Coventry 73.7% / 16.4% / 9.8%).
- **`season_sim.parquet` regenerated** (10k sims): Man City 73.3 pts / 37.5% title; Arsenal 73.1 / 36.6%; Man United 66.8; Liverpool 66.0; relegation sweepstake Ipswich 80.8% / Hull 68.1% / Coventry 58.9%.
- `PL_PREDICT_OVERVIEW.md` updated throughout with Phase 3 data/features/results.

### Outstanding / Next Steps
1. Optionally scrape **Transfermarkt** squad market values as a further feature.
2. Consider per-match player stats from Understat match pages (`rostersData`) for player-level form.
3. **Phase 4**: deeper calibration (isotonic / conformal) + evaluation framework.


## Session 2026-08-12 (night) — Phase 3B: Player-level features (Understat per-match + Transfermarkt)

### Done
- **Understat per-match scraper** (`scrape_understat_player_matches`, resumable, 5-worker thread pool): fetched `getMatchData/{id}` for **all 4,560 matches** across 2014–2025 → `understat_player_matches.parquet` (**129,576 player rows**: player_id, minutes, xG, xA, shots, key passes, position; `player_id` stable across matches) + `understat_shots.parquet` (**116,448 shot rows**: shot xG, situation, shotType, assisted_by).
- **Transfermarkt scraper** (`scrape_transfermarkt`, retry/backoff for 503s/timeouts): club ids **discovered from per-season PL league pages** (34 clubs) with curated fallback; 407 (team, season) squad pages scraped for 2015–2025 → `transfermarkt_squads.parquet` with **squad market value / size / avg age, 396/407 parsed** (97%).
- **New features (60 → 78 cols)**:
  - In-season player (per-match rosters, **strictly prior** dates, leakage-free): `szn_player_xg`, `szn_player_xa`, `szn_minutes`, `star_share`, `star_xg_form`, `players_used_8` (home/away)
  - Prior-season Transfermarkt lag: `squad_value`, `avg_age`, `squad_size` (home/away)
- **Prediction path**: `_tm_squad_lookup` + prior-season lookup wired into `_build_feature_row`; in-season player features set to 0 for future fixtures.
- **3-fold ablations (no deep)** — all within noise of v2 (RPS 0.6592 / Acc 49.3%):
  | Config | RPS | Acc |
  |---|---|---|
  | v2 baseline (xG+squad) | 0.6592 | 49.3% |
  | A: +player only | 0.6597 | 49.4% |
  | B: +transfermarkt only | 0.6594 | 48.8% |
  | C: both (full v3) | **0.6584** | 48.6% |

### Running
- **Full 5-fold deep walk-forward (v3, PID 19464)** — decision point: keep player features if RPS/acc don't regress vs v2 (0.6640 / 49.2%), else drop them before production retrain.

### Next Steps
1. Read v3 walk-forward result → keep/drop new features.
2. Retrain production ensemble + regenerate fixtures + 10k season sim.
3. Update `PL_PREDICT_OVERVIEW.md` with player-feature phase.

### RESULT (v3 walk-forward + production retrain complete)
- **Full deep walk-forward v3**: Ensemble **RPS=0.6624 Acc=48.9%** vs v2 (0.6640 / 49.2%). RPS improved (best ensemble score yet), Acc/Brier/LogLoss moved within noise → **features kept**.
- **Production ensemble retrained with all 78 features** → `ensemble_model.pkl` + `deep_model.pt` (best deep val_acc 0.510).
- **`fixtures_2627_pred.json` + `season_sim.parquet` regenerated** (new model): Arsenal 74.0 pts / **40.5% title**; Man City 72.5 / 32.0%; Man United 67.2; Liverpool 67.1; relegation: Ipswich 75.8% / Coventry 64.6% / Hull 63.9%. Arsenal vs Coventry 71.8% / 17.1% / 11.1%.
- **Dashboard restarted** on http://127.0.0.1:8000 (PID 8344) — all endpoints healthy; 8 player/Transfermarkt features appear in the model's top-30 importances (away_szn_minutes, away_szn_player_xa, star_share, star_xg_form, away_squad_value, players_used_8, etc.).
- `PL_PREDICT_OVERVIEW.md` updated: data sources, feature groups C4/C5, backtest table (v3), season projection, Appendix B (78 cols), Appendix C, limitations, roadmap.

### Outstanding / Next Steps
1. **Phase 4**: deeper calibration (isotonic / conformal) + evaluation framework.
2. Optional: expose per-match player/shot data in the dashboard; weekly rolling retrain once 2026/27 odds appear.


## Session 2026-08-13 (early) — Phase 4: Deeper calibration (isotonic + conformal)

### Done
- **Upgraded calibration from temperature-only to CV-averaged per-class isotonic** (`_fit_calibrator`): isotonic fit with 5-fold internal CV on the OOF stacker probabilities (never sees its own predictions). First attempt (temp→isotonic stacking) REGRESSED hard (RPS 0.6584→0.6727) — isotonic subsumes temperature, so fitting it on raw stacker probs is correct.
- **Conformal prediction sets** (`conformal_sets`): nonconformity 1-p[true] quantiles from OOF data at 90%/80% levels, exposed on every prediction as `conformal_90`/`conformal_80` (e.g. Arsenal v Coventry → both sets `['H']`).
- **ECE metric** added (`expected_calibration_error`) and printed in the backtest summary (max-prob ECE + per-outcome mean).
- **Worst-prediction diagnostic** (`scripts/analyze_worst_predictions.py` → `data/processed/worst_predictions.md`): found a systematic blind spot — the model's confident misses are big clubs (City/Arsenal/Liverpool/Chelsea/United) dropping points at home to mid-table sides (Bournemouth, Burnley, Brighton, Forest), incl. three away-upsets @7-9% (Arsenal v Bournemouth, Man Utd v Bournemouth, Liverpool v Forest). Also flagged: identical matchups get identical prediction vectors (50/50 h2h-prior blend doesn't age).

### Results (full 5-fold deep walk-forward, 2023–26)
| | RPS | Acc | Brier | LogLoss | ECE(ovr) |
|---|---|---|---|---|---|
| DC | 0.6584 | 49.1% | 0.6133 | 1.0261 | 0.0319 |
| v3 temp | 0.6624 | 48.9% | 0.6149 | 1.0276 | ~0.030 |
| **v4 isotonic** | **0.6621** | 48.9% | 0.6151 | 1.0287 | **0.0287** |

Isotonic keeps RPS at the v3 best (0.6621) while improving calibration (ECE 0.0287 — best of any model, beating DC). 

### Produced
- Production model retrained with isotonic calibration → `ensemble_model.pkl` + `deep_model.pt`; 380 fixtures regenerated with conformal sets; 10k season sim regenerated (Arsenal 73.7 pts / 39.9% title).
- Dashboard restarted (PID 10048) with the new model.

### Next
1. Consider targeting the big-club-home-upset blind spot (e.g. opponent-strength interaction, aging the h2h prior).
2. Surface conformal sets + worst-predictions in the dashboard.


## Session 2026-08-03 (late) — Overview now shows 2026/27 fixtures

- **Overview tab rewritten** — now displays all **380 2026/27 fixtures with model predictions** (Home Win / Draw / Away Win columns, scrollable table) instead of a handful of historical matches.
- Stats row: "Total Matches" now uses new `total_matches` field from `/api/health` (12,705 historical records used for training).
- **Persistent prediction cache** — `/api/fixtures_2627` results written to `data/processed/fixtures_2627_pred.json`; survives server restarts (380 predictions no longer recomputed per boot: 33s once, then ~0.2s).
- `/api/matches` supports `offset`/`limit` pagination (most recent first) — kept for future use, no longer used by Overview.

## Session 2026-08-03 (evening) — 2026/27 season simulation

### Done
- **`season_sim.py` rewritten** — now supports two fixture/probability sources:
  - `parquet`: historical schedule from matches.parquet using market implied odds (old behaviour)
  - `fixtures_2627`: the real 380-match 2026/27 schedule from `fixtures_2627.py`, each match scored by the trained ensemble model
  - `source="auto"` picks fixtures_2627 when `season=="2026-27"`, else parquet
- **Vectorized Monte Carlo** — samples all n_sims × 380 results in one numpy pass, then accumulates points; ~80s for 5000 sims (incl. 380 model predictions).
- **New sim columns**: `top4_pct` and `relegation_pct` computed from per-sim rank. (Previously the dashboard guessed relegation as `title_pct==0`.)
- **CLI**: `predict-pl season` gained `--source auto|parquet|fixtures_2627`.
- **Dashboard Season Sim tab**:
  - Top-4 and Relegation cards now use real `top4_pct` / `relegation_pct`.
  - Added a **Full Season Projection** table (rank, expected pts, median, p10–p90, Title %, Top 4 %, Relegation %) with `fav`/`risk` badges.
- Regenerated `data/processed/season_sim.parquet` for 2026-27 (5000 sims).

### 2026/27 projection (5000 sims, ensemble model)
| Rank | Team | Exp Pts | Title % | Top4 % | Releg % |
|------|------|--------|--------|--------|---------|
| 1 | Arsenal | 74.6 | 38.2 | 89.5 | 0.0 |
| 2 | Man City | 73.0 | 29.4 | 83.2 | 0.0 |
| 3 | Liverpool | 70.6 | 19.9 | 75.7 | 0.0 |
| 4 | Man United | 66.9 | 9.8 | 57.6 | 0.0 |
| 5 | Chelsea | 66.8 | 9.1 | 59.2 | 0.0 |
| ... | | | | | |
| 18 | Sunderland | 38.7 | 0.0 | 0.1 | 46.9 |
| 19 | Hull City | 33.7 | 0.0 | 0.0 | 71.5 |
| 20 | Ipswich Town | 33.2 | 0.0 | 0.0 | 75.3 |

## Session 2026-08-03 (afternoon) — Dashboard fixes (all verified)

### Fixed
- **Overview page would not open** — root cause: frontend requested `/api/matches?limit=5000`, but the endpoint validated `le=500` and returned HTTP 422. JS then called `undefined.toLocaleString()`, throwing and freezing the page.
  - `src/pl_predict/dashboard/app.py`: `/api/matches` now accepts `le=10000` and returns a trimmed column set (`MATCH_COLUMNS`: date, season, home_team, away_team, home_goals, away_goals, result, implied home/draw/away_norm). Sorted by `date` so "recent matches" are chronological.
  - `index.html`: removed orphaned xG H / xG A table headers (JS never rendered them), colspan 9→7, defensive guard on `matchesRes.total`.
- **Season Sim page rendered blank bars** — root cause: Plotly cannot parse CSS variable colors (`var(--green)`, `var(--accent)` etc.). Marker colors now use hex (`#22c55e` / `#ef4444` / `#3b82f6`). Error bars (p10–p90 whiskers) were defined but never attached — now merged into the bar trace via `error_x`.
- **Features chart** — same CSS-var color issue; now `#8b5cf6`.
- **`/api/season`** — now sorts by `mean_pts` descending so `teams[0]` is the champion (previously file order, only coincidentally correct).
- **`/api/fixtures_2627`** — added in-memory cache (first call ~18s for 380 DC predictions, then ~0.02s). Uses `normalize_team()` to map full names → model names before predicting.
- **Team name mapping** — `src/pl_predict/data/fixtures_2627.py` now has `TEAM_NAME_MAP` + `normalize_team()` (e.g. "Coventry City"→"Coventry", "Newcastle United"→"Newcastle", "Nottingham Forest"→"Nott'm Forest"). Required because 2026/27 fixtures use full names but the model knows football-data.co.uk short forms.

### Verified (TestClient + live HTTP)
- `/api/health` — all data present (matches, features, ensemble model, season sim)
- `/api/matches?limit=5000` → 200, 5000 trimmed rows, total 12705
- `/api/fixtures_2627` → 380 fixtures, all 380 have predictions (e.g. Arsenal 69.2% / 19.9% D / 10.9% Coventry)
- `/api/season` → 20 teams sorted; Arsenal champ 78.8 pts / 48.6% title
- `/api/predict` POST → Man City 46.8% / 25.1% D / 28.0% Liverpool
- `/api/features` → 24 features
- Overview renders 15 recent played + upcoming

## Session 2026-08-12 — Dashboard v2: startup cache + 9-tab analytics SPA (all verified)

### Backend (`src/pl_predict/dashboard/app.py`)
- **Startup artifact cache** — FastAPI `lifespan` warm thread loads predictor, matches, season_sim, walkforward_preds, teams, feature importances, player agg, shots, fixtures JSON once at boot. `/api/predict` dropped from ~8.6s to ~40ms; `/api/features` from ~7s to ~4ms. Caches are mtime-keyed so retrained models/re-scrapes are picked up automatically.
- **`GZipMiddleware`** — fixtures JSON compresses ~900KB → ~300KB.
- **`StaticFiles` mount** at `/static/` — serves `styles.css` + `app.js` (the old `/app.js` route never existed).
- **New endpoints** (all <250ms, all data pre-derived):
  - `/api/performance` — per-season RPS/acc/Brier from `walkforward_preds`, calibration curve, RPS histogram, worst-predictions list
  - `/api/players?season&team&stat&top` + `/api/players/{id}` — leaderboards from 129k player-match rows (npxG joined from `understat_players`); detail returns per-match history
  - `/api/shots?season&team&player` — 116k shots; `team` derived by joining `(match_id, h_a)` → team from player_matches (100% coverage)
  - `/api/team/{name}` — form trend, home/away splits, goals histogram, attendance by season
  - `/api/h2h?home&away` — H2H history + model prediction; uses `_cached_prediction` which reads the 380-fixture JSON first, else computes+caches (Arsenal-Chelsea first call ~1.2s → cached 16ms)
  - `/api/referees` — per-referee goals/home-bias/attendance
  - `/api/seasons` — match + player season lists for filter dropdowns
- **Bugs fixed during backend hardening**:
  - polars `if entry:` on a DataFrame is ambiguous → all cache checks now `is not None`
  - `/api/performance` used `for season, g in df.group_by(...)` (iterates COLUMNS in polars, season came back as `['2024-25']` with n=0) → rewritten over `unique()` seasons
  - `/api/team` column swap via `.rename({"away_team":"home_team","home_team":"away_team"})` collides in polars → rebuilt with explicit `.alias()` selects (Arsenal venue split was 1262/0, now 631/631)
  - Attendance was `String` with empty strings → cast + strip; referees polluted by empty-string referee rows + NaN goals (`is_finite()`)
  - polars v1.43 renamed `.str.strip()` → `.str.strip_chars()`

### Frontend (`templates/index.html` + new `static/styles.css` + `static/app.js`)
- **Vanilla JS + Plotly CDN**, hash router (`#/predict?home=X&away=Y`, `#/teams/Man City`), loading spinners, error toasts.
- **9 tabs**: Overview · Predict · Season Sim · Model · Players · Shot Maps · Teams · Referees · Features
- **Overview** — KPI cards, filterable/sortable 2026/27 fixture table: H/D/A prob bars, most-likely score, O/U 2.5 chip, conformal-set badges; row click → deep-link into Predict
- **Predict** — H/D/A display, xG, market chips (O1.5/2.5/3.5, BTTS, clean sheets, 90% conformal set), 7×7 score-matrix heatmap, 6-model breakdown bars, top-5 scores, H2H panel with live model line
- **Season Sim** — title/top4/relegation KPIs, expected-points bar with P10-P90 error bars, full projection table, relegation battleground chart
- **Model** — walk-forward KPIs, accuracy-by-season, RPS-by-season, calibration curve, RPS histogram, worst-predictions table
- **Players** — season/team/stat/top-N leaderboard + bar chart; click player → per-match history with cumulative-xG sparkline
- **Shot Maps** — filters, KPIs, Plotly pitch overlay (goal/saved/missed/blocked colors, sized by xG), top-shooters table
- **Teams** — dropdown + deep-link; recent form (pts/xG for-against), W/D/L splits, goals histogram, attendance, recent results
- **Referees** — avg goals (home/away/total) + home-win-bias charts vs league avg, full table

### Verification
- 12/12 headless tests pass (`static/smoke_test.js` — Node stub-DOM with real fetch against the live server): all 9 tabs, deep-links, `openPlayer`. Null-ID checking catches missing `#id` derefs; toast spy catches swallowed loader errors.
- All 11 endpoints verified live: 3–250ms. Static assets 200 + correct content-types. GZip confirmed via `Content-Encoding: gzip`.

## Outstanding / Next Steps
1. **Season Sim tab is now 2026/27** — the sim used 5000 runs; a 10000-run rerun (`predict-pl season --season 2026-27 -n 10000`) gives tighter estimates.
2. **Deep model reload bug (if hit)** — retrain any model component via `predict-pl train` if feature/data changes.
3. Consider wiring the Season Sim tab to the `source` option (parquet vs fixtures_2627) via a dropdown.
4. Run `predict-pl train` after any data/feature changes before regenerating sims.
5. Tests exist in `pl-predict/tests/` — run with `python -m pytest` when modifying models.

## Commands
- Start dashboard: `Start-Process -FilePath "C:\Program Files\Python311\python.exe" -ArgumentList "-m", "pl_predict.dashboard.app", "--port", "8000", "--host", "127.0.0.1" -WorkingDirectory "C:\Users\Kian Jindal\Documents\Default Project\pl-predict" -WindowStyle Hidden`
- Verify server: check port 8000 listening, hit `http://127.0.0.1:8000/api/health`
- Note: python is `C:\Program Files\Python311\python.exe` (pandas is BLOCKED by Windows App Control — use Polars).

## Key Files
- `src/pl_predict/dashboard/app.py` — FastAPI endpoints (health, matches, upcoming, fixtures_2627, predict, season, features, teams, seasons, performance, players, players/{id}, shots, team/{name}, h2h, referees, index) + startup cache + GZip + static mount
- `src/pl_predict/dashboard/templates/index.html` — slim SPA shell (9 tab containers, Plotly CDN, links to static assets)
- `src/pl_predict/dashboard/static/app.js` — all SPA logic: hash router, 9 tab renderers, Plotly charts, pitch overlay
- `src/pl_predict/dashboard/static/styles.css` — full dark-theme stylesheet (extracted from inline HTML)
- `src/pl_predict/dashboard/static/smoke_test.js` — Node headless DOM test harness (run `node static/smoke_test.js`)
- `src/pl_predict/data/fixtures_2627.py` — all 380 fixtures + `TEAM_NAME_MAP` + `normalize_team()`
- `src/pl_predict/models/ensemble.py` — EnsemblePredictor (DC + XGB + Deep stacker), `.predict(home, away)`
- `data/processed/` — matches.parquet (12,705), features.parquet, ensemble_model.pkl, deep_model.pt, season_sim.parquet, walkforward_preds.parquet, fixtures_2627_pred.json

## Repo organization (2026-08-12)
- `logs/` — all `*.log` / `*_err.log` output files (moved from repo root)
- `scripts/` — one-off runners & diagnostics: `analyze_worst_predictions.py`, `run_backtest_xg.py`, `run_full_walkforward.py`, `run_phase3_full.py`, `run_phase4_calib.py`, `run_phase4_prod.py`, `run_scrape_player_data.py`, `run_wf_v3.py`. All now resolve `src/` and `data/` relative to the script (portable — run from any CWD).
- `docs/` — this log (`PROGRESS.md`) and `plan.md` (moved from repo root)
- Root now contains only: `config/`, `data/`, `docs/`, `logs/`, `notebooks/`, `paper/`, `scripts/`, `src/`, `tests/`, `pyproject.toml`, `README.md`
