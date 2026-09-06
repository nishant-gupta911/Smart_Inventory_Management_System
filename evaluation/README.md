# Model Evaluation

This folder contains standalone scripts to evaluate the two trained models
in the Smart Inventory Management System.

---

## Files

| File | Purpose |
|------|---------|
| `evaluate_expiry_model.py` | Evaluates the **Expiry Risk Classifier** (`expiry_predict_model.pkl`) |
| `evaluate_demand_model.py` | Evaluates the **Demand Forecast Model** (`demand_forecast_model.pkl`) |
| `run_all_evaluations.py` | Runs both scripts in sequence and prints a combined summary table |
| `results/` | Auto-created folder — text report files are saved here |

---

## What each script does

### `evaluate_expiry_model.py`

Loads the sklearn `Pipeline(StandardScaler → RandomForestClassifier)` and
evaluates it on the processed expiry data.

Metrics computed:
- **Accuracy score**
- **Precision, Recall, F1-score** per class (Safe / At Risk)
- **Full Classification Report**
- **ROC-AUC score** (binary one-vs-rest)
- **5-fold Cross-Validation** accuracy (mean ± std)

Plots saved:
- `plots/expiry_confusion_matrix.png` — seaborn heatmap
- `plots/expiry_feature_importance.png` — top-15 feature importances

Text report saved to:
- `evaluation/results/expiry_model_report.txt`

---

### `evaluate_demand_model.py`

Loads the `RandomForestRegressor` and evaluates it on the cleaned inventory
data with the same feature engineering used during training.

Metrics computed:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error
- **5-fold Cross-Validation** R² (mean ± std)

Plots saved:
- `plots/demand_actual_vs_predicted.png` — scatter plot with identity line
- `plots/demand_residuals.png` — residuals vs predicted + residual distribution
- `plots/demand_feature_importance.png` — top-15 feature importances

Text report saved to:
- `evaluation/results/demand_model_report.txt`

---

### `run_all_evaluations.py`

Orchestrates both evaluations and prints a combined summary table.
Safe to run even if one model is missing — it reports the failure and
continues with the other.

---

## How to run

Run from the **project root** (`smart_inventory/`):

```bash
# Run both evaluations at once (recommended)
python evaluation/run_all_evaluations.py

# Run expiry model evaluation only
python evaluation/evaluate_expiry_model.py

# Run demand model evaluation only
python evaluation/evaluate_demand_model.py
```

> **Note:** Both models must be trained before running evaluations.
> If the `.pkl` files are missing, run `python main.py` first.

---

## Output files

```
evaluation/
└── results/
    ├── expiry_model_report.txt
    └── demand_model_report.txt

plots/
    ├── expiry_confusion_matrix.png
    ├── expiry_feature_importance.png
    ├── demand_actual_vs_predicted.png
    ├── demand_residuals.png
    └── demand_feature_importance.png
```

---

## Dependencies

All dependencies are already listed in `requirement.txt`:

| Package | Use |
|---------|-----|
| `scikit-learn` | Metrics, cross-validation |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plot rendering |
| `seaborn` | Styled plots (confusion matrix, bar charts) |
| `joblib` | Loading `.pkl` model files |
