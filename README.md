# Linear Regression Scaffold

A small, reproducible linear regression project scaffold with training and prediction scripts.

## Project structure

- `src/model.py` — simple univariate linear regression model (gradient descent).
- `src/train.py` — trains the model from data and saves artifacts to `outputs/`.
- `src/predict.py` — loads the saved model and runs predictions.
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

- `outputs/model.json`
- `outputs/metrics.json`

## Predict

Single value:

```bash
python src/predict.py --x 12.5
```

Multiple values:

```bash
python src/predict.py --x-values 1 2 3 4 5
```

## Notes

All scripts use `pathlib.Path` and resolve paths relative to the repository root.
