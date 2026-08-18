# Worst-prediction diagnostic

_From `walkforward_preds.parquet`, 1102 held-out matches_

## Confident misses (model error)

| Date | Season | Match | Result | H | D | A | Pred | p(actual) |
|------|--------|-------|--------|---|---|---|------|-----------|
| 2026-03-04 | 2025-26 | Man City v Nott'm Forest | D | 0.82 | 0.124 | 0.056 | H@82% | 12.4% |
| 2026-01-17 | 2025-26 | Liverpool v Burnley | D | 0.807 | 0.133 | 0.06 | H@81% | 13.3% |
| 2025-03-15 | 2024-25 | Man City v Brighton | D | 0.795 | 0.141 | 0.064 | H@80% | 14.1% |
| 2026-02-21 | 2025-26 | Chelsea v Burnley | D | 0.795 | 0.142 | 0.064 | H@80% | 14.2% |
| 2026-01-07 | 2025-26 | Man City v Brighton | D | 0.795 | 0.141 | 0.064 | H@80% | 14.1% |
| 2026-04-11 | 2025-26 | Arsenal v Bournemouth | A | 0.791 | 0.141 | 0.067 | H@79% | 6.7% |
| 2025-05-03 | 2024-25 | Arsenal v Bournemouth | A | 0.791 | 0.141 | 0.067 | H@79% | 6.7% |
| 2025-04-23 | 2024-25 | Arsenal v Crystal Palace | D | 0.788 | 0.154 | 0.058 | H@79% | 15.4% |
| 2024-03-30 | 2023-24 | Chelsea v Burnley | D | 0.783 | 0.149 | 0.067 | H@78% | 14.9% |
| 2025-12-30 | 2025-26 | Chelsea v Bournemouth | D | 0.783 | 0.146 | 0.07 | H@78% | 14.6% |
| 2023-12-09 | 2023-24 | Man United v Bournemouth | A | 0.773 | 0.16 | 0.067 | H@77% | 6.7% |
| 2025-12-15 | 2025-26 | Man United v Bournemouth | D | 0.773 | 0.16 | 0.067 | H@77% | 16.0% |
| 2024-12-22 | 2024-25 | Man United v Bournemouth | A | 0.773 | 0.16 | 0.067 | H@77% | 6.7% |
| 2025-05-25 | 2024-25 | Liverpool v Crystal Palace | D | 0.769 | 0.157 | 0.074 | H@77% | 15.7% |
| 2025-01-14 | 2024-25 | Chelsea v Bournemouth | D | 0.769 | 0.16 | 0.071 | H@77% | 16.0% |
| 2025-11-22 | 2025-26 | Liverpool v Nott'm Forest | A | 0.767 | 0.162 | 0.072 | H@77% | 7.2% |
| 2024-04-27 | 2023-24 | Man United v Burnley | D | 0.766 | 0.157 | 0.077 | H@77% | 15.7% |
| 2024-09-14 | 2024-25 | Liverpool v Nott'm Forest | A | 0.766 | 0.162 | 0.072 | H@77% | 7.2% |
| 2025-04-20 | 2024-25 | Man United v Wolves | A | 0.746 | 0.164 | 0.09 | H@75% | 9.0% |
| 2024-12-07 | 2024-25 | Man United v Nott'm Forest | A | 0.746 | 0.17 | 0.083 | H@75% | 8.3% |
| 2023-12-16 | 2023-24 | Man City v Crystal Palace | D | 0.746 | 0.17 | 0.083 | H@75% | 17.0% |
| 2025-02-15 | 2024-25 | Aston Villa v Ipswich | D | 0.745 | 0.176 | 0.079 | H@74% | 17.6% |
| 2024-11-29 | 2024-25 | Brighton v Southampton | D | 0.743 | 0.154 | 0.103 | H@74% | 15.4% |
| 2024-12-26 | 2024-25 | Man City v Everton | D | 0.743 | 0.158 | 0.098 | H@74% | 15.8% |
| 2024-04-14 | 2023-24 | Liverpool v Crystal Palace | A | 0.739 | 0.174 | 0.087 | H@74% | 8.7% |
| 2024-02-04 | 2023-24 | Chelsea v Wolves | A | 0.738 | 0.176 | 0.086 | H@74% | 8.6% |
| 2025-04-13 | 2024-25 | Chelsea v Ipswich | D | 0.727 | 0.199 | 0.074 | H@73% | 19.9% |
| 2026-05-04 | 2025-26 | Chelsea v Nott'm Forest | A | 0.722 | 0.172 | 0.106 | H@72% | 10.6% |
| 2026-05-24 | 2025-26 | Man City v Aston Villa | A | 0.719 | 0.18 | 0.101 | H@72% | 10.1% |
| 2023-08-26 | 2023-24 | Arsenal v Fulham | D | 0.715 | 0.186 | 0.099 | H@72% | 18.6% |

## Freak results (model was honest)

| Date | Season | Match | Result | p(actual) | H | D | A |
|------|--------|-------|--------|-----------|---|---|---|
| 2026-04-11 | 2025-26 | Arsenal v Bournemouth | A | 6.7% | 0.791 | 0.141 | 0.067 |
| 2025-05-03 | 2024-25 | Arsenal v Bournemouth | A | 6.7% | 0.791 | 0.141 | 0.067 |
| 2023-12-09 | 2023-24 | Man United v Bournemouth | A | 6.7% | 0.773 | 0.16 | 0.067 |
| 2024-12-22 | 2024-25 | Man United v Bournemouth | A | 6.7% | 0.773 | 0.16 | 0.067 |
| 2025-11-22 | 2025-26 | Liverpool v Nott'm Forest | A | 7.2% | 0.767 | 0.162 | 0.072 |
| 2024-09-14 | 2024-25 | Liverpool v Nott'm Forest | A | 7.2% | 0.766 | 0.162 | 0.072 |
| 2024-12-07 | 2024-25 | Man United v Nott'm Forest | A | 8.3% | 0.746 | 0.17 | 0.083 |
| 2024-11-10 | 2024-25 | Tottenham v Ipswich | A | 8.3% | 0.698 | 0.219 | 0.083 |
| 2024-02-04 | 2023-24 | Chelsea v Wolves | A | 8.6% | 0.738 | 0.176 | 0.086 |
| 2024-04-14 | 2023-24 | Liverpool v Crystal Palace | A | 8.7% | 0.739 | 0.174 | 0.087 |
| 2025-04-20 | 2024-25 | Man United v Wolves | A | 9.0% | 0.746 | 0.164 | 0.09 |
| 2025-02-02 | 2024-25 | Man United v Crystal Palace | A | 9.9% | 0.713 | 0.188 | 0.099 |
| 2023-09-30 | 2023-24 | Man United v Crystal Palace | A | 9.9% | 0.713 | 0.188 | 0.099 |