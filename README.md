# fire-risk-predictor

`fire-risk-predictor` is a machine learning project for estimating a continuous wildfire risk score from weather, environment, and operational signals. The project trains a compact feed-forward neural network using PyTorch and packages artifacts for repeatable inference. It is designed as a practical baseline that can be extended with richer geospatial, temporal, and domain-specific data.

## Problem Statement

Wildfire response teams need early, data-driven estimates of fire risk to prioritize monitoring, staffing, and preventive interventions. Manual heuristics can be inconsistent across regions and may underweight interactions between climate, air quality, and response constraints. This project frames risk estimation as a supervised regression task that predicts a `fire_risk_score` from structured tabular inputs.

## Dataset and Selected Features

The training pipeline expects a CSV dataset under `data/` containing required columns for predictors and target. Selected predictors are:

- `temperature_c`
- `humidity_percent`
- `wind_speed_kmh`
- `air_quality_index`
- `vegetation_index`
- `distance_to_station_km`
- `response_time_min`

Target column:

- `fire_risk_score`

Excluded/non-model columns (if present in source systems) include categorical labels like `fire_risk_level` and noise fields like `random_noise`.

## Model Approach (PyTorch MLP + Scaler)

The model is a small multilayer perceptron (MLP) regressor implemented in PyTorch:

- Input layer sized to the selected feature set
- Hidden dense layers with ReLU activations
- Single-neuron output layer for continuous risk prediction

Before training, numeric inputs are standardized with a fitted scaler (mean/standard deviation normalization). The scaler is persisted and reused at inference time to ensure feature consistency.

## Repo Structure

```text
.
├── data/
│   └── generate_data.py
├── outputs/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── scaler.joblib
│   ├── loss_curve.png
│   ├── pred_vs_actual.png
│   └── residuals.png
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Install Instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train Command Example

```bash
python src/train.py --data-path data/fire_data.csv --epochs 200 --batch-size 32 --learning-rate 0.001
```

## Predict Command Example

```bash
python src/predict.py --x 42.0
```

## Metrics (MAE / RMSE / R²)

Use this section to report model quality on a held-out validation/test split:

- **MAE (Mean Absolute Error):** average absolute prediction error
- **RMSE (Root Mean Squared Error):** error metric that penalizes larger misses
- **R² (Coefficient of Determination):** proportion of variance explained by the model

Suggested reporting table format:

| Split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Validation | _TBD_ | _TBD_ | _TBD_ |
| Test | _TBD_ | _TBD_ | _TBD_ |

## Visual Diagnostics

![Loss Curve](outputs/loss_curve.png)

![Predicted vs Actual](outputs/pred_vs_actual.png)

![Residuals](outputs/residuals.png)

## Business Relevance

Accurate fire-risk scoring supports better allocation of field crews, surveillance assets, and prevention budgets. Even modest improvements in forecast precision can reduce response latency, protect infrastructure, and lower suppression costs in high-risk windows. As part of an operational workflow, this model can serve as a decision-support layer for prioritization rather than a fully autonomous decision-maker.

## Limitations

- Model performance is bounded by data quality, representativeness, and label reliability.
- A tabular MLP may miss temporal dependencies (seasonality, event sequences) without explicit time-series features.
- Geographic transferability is limited; retraining and recalibration are typically required across regions.
- Risk predictions should be interpreted with domain oversight and uncertainty-aware operational policies.
