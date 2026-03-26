# Linear Regression Scaffold

A small, reproducible linear regression project scaffold with training and prediction scripts.

## Project structure

- `src/model.py` — simple univariate linear regression model (gradient descent).
- `src/train.py` — trains the model from data and saves artifacts to `outputs/`.
- `src/predict.py` — loads `outputs/scaler.joblib` + `outputs/best_model.pt` and runs batch predictions from CSV.
- `src/utils.py` — shared path and dataset utilities.
- `data/` — generated dataset and generation script.
- `outputs/` — saved model artifacts (kept in repo with `.gitkeep`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset

```bash
python data/generate_data.py
```

## Train model

```bash
python src/train.py
```

Artifacts written:

- `outputs/scaler.joblib`
- `outputs/best_model.pt`
- `outputs/metrics.json`
- `outputs/training_history.json`

## Predict

Run predictions over an input CSV:

```bash
python src/predict.py --input-csv data/your_input.csv
```

Override output path:

```bash
python src/predict.py --input-csv data/your_input.csv --output-csv outputs/my_predictions.csv
```

Include optional risk banding with configurable thresholds:

```bash
python src/predict.py \
  --input-csv data/your_input.csv \
  --include-risk-band \
  --low-risk-threshold 33 \
  --high-risk-threshold 66
```

Risk band semantics:

- Low: `predicted_fire_risk_score < low_threshold`
- Medium: `low_threshold <= predicted_fire_risk_score <= high_threshold`
- High: `predicted_fire_risk_score > high_threshold`

By default, `src/utils.py` sets thresholds to 33 (low) and 66 (high).

## Notes

All scripts use `pathlib.Path` and resolve paths relative to the repository root.
