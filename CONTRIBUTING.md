# Contributing to PL Predict

Thanks for helping improve PL Predict.

## Before opening a pull request

1. Describe bugs with a reproducible example, expected behaviour and actual behaviour.
2. Keep each pull request focused on one improvement.
3. Do not add credentials, generated runtime snapshots or large duplicate datasets.
4. Run the checks locally:

```bash
ruff format --check .
ruff check .
python -m pytest -q
```

## Data and model changes

When changing a data source, feature, model or projection formula, explain the reason and expected effect in the pull request. Preserve the distinction between official FPL data and PL Predict-derived values.
