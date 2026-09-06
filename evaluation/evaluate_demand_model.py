#!/usr/bin/env python3
"""
Demand Forecast Model Evaluation
Model: VotingRegressor (XGBoost + LightGBM + RandomForest)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
PLOTS_DIR    = PROJECT_ROOT / "plots"
RESULTS_DIR  = PROJECT_ROOT / "evaluation" / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Must match exactly what train_demand_model.py used when training the saved .pkl
FEATURES = [
    'rolling_avg_sales_7', 'rolling_avg_3', 'rolling_avg_14',
    'sales_lag_1', 'sales_lag_7', 'sales_trend',
    'onpromotion', 'day_of_week', 'month', 'is_weekend', 'quarter',
    'store_nbr', 'cluster', 'perishable', 'dcoilwtico'
]


def load_model():
    path = MODELS_DIR / "demand_forecast_model.pkl"
    if not path.exists():
        print(f"❌ Model not found: {path}\n   Run python main.py first.")
        sys.exit(1)
    model = joblib.load(path)
    print(f"✅ Demand model loaded from: {path}")
    return model


def load_and_prepare_data():
    path = DATA_DIR / "cleaned_inventory_data.csv"
    if not path.exists():
        print(f"❌ Data not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Data loaded: {len(df):,} rows")

    # Date features
    if 'date' in df.columns:
        df['date']       = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month']       = df['date'].dt.month
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    else:
        for col, val in [('day_of_week', 0), ('month', 1), ('is_weekend', 0)]:
            if col not in df.columns:
                df[col] = val

    df['sales'] = pd.to_numeric(df.get('sales', 0), errors='coerce').fillna(0).clip(lower=0)

    # Sort for correct rolling / lag order
    sort_cols = [c for c in ['store_nbr', 'item_id', 'date'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    grp = [c for c in ['store_nbr', 'item_id'] if c in df.columns]

    def roll(w):
        if grp:
            return df.groupby(grp)['sales'].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
        return df['sales'].rolling(w, min_periods=1).mean()

    df['rolling_avg_sales_7'] = roll(7)
    df['rolling_avg_3']       = roll(3)
    df['rolling_avg_14']      = roll(14)
    df['rolling_avg_30']      = roll(30)

    if grp:
        df['sales_lag_1'] = df.groupby(grp)['sales'].shift(1)
        df['sales_lag_7'] = df.groupby(grp)['sales'].shift(7)
    else:
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_7'] = df['sales'].shift(7)

    df['sales_trend'] = df['rolling_avg_sales_7'] - df['rolling_avg_30']
    df['quarter']     = df['month'].apply(lambda x: (x - 1) // 3 + 1)

    if 'dcoilwtico' not in df.columns:
        df['dcoilwtico'] = 0.0
    else:
        df['dcoilwtico'] = pd.to_numeric(df['dcoilwtico'], errors='coerce').ffill().fillna(0)

    for col in ['onpromotion', 'store_nbr', 'cluster', 'perishable']:
        if col not in df.columns:
            df[col] = 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df = df.dropna(subset=FEATURES + ['sales'])

    df = df.sample(frac=0.1, random_state=42)
    X  = df[FEATURES]
    y  = df['sales']
    print(f"✅ Evaluation sample: {len(df):,} rows")
    print(f"   Sales stats — Mean: {y.mean():.2f} | Std: {y.std():.2f} | Min: {y.min():.2f} | Max: {y.max():.2f}")
    return X, y


def compute_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def plot_actual_vs_predicted(y_true, y_pred):
    n   = min(5000, len(y_pred))
    idx = np.random.default_rng(42).choice(len(y_pred), n, replace=False)
    yt, yp = np.array(y_true)[idx], np.array(y_pred)[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(yt, yp, alpha=0.3, s=12, color='#3498db', label='Predictions')
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit (y=x)')
    ax.set_xlabel('Actual Sales', fontsize=12)
    ax.set_ylabel('Predicted Sales', fontsize=12)
    ax.set_title('Demand Model — Actual vs Predicted Sales', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.text(0.05, 0.92, f'R² = {r2:.4f}\nMAE = {mae:.4f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    path = PLOTS_DIR / "demand_actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Actual vs Predicted saved: {path}")


def plot_residuals(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n   = min(5000, len(y_pred))
    idx = np.random.default_rng(42).choice(len(y_pred), n, replace=False)
    axes[0].scatter(np.array(y_pred)[idx], residuals[idx], alpha=0.3, s=12, color='darkorange')
    axes[0].axhline(0, color='black', linewidth=1.5, linestyle='--')
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Sales', fontsize=11)
    axes[0].set_ylabel('Residual (Actual − Predicted)', fontsize=11)

    sns.histplot(residuals, bins=60, kde=True, color='teal', ax=axes[1])
    axes[1].axvline(0, color='red', linewidth=1.5, linestyle='--')
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residual', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].text(0.05, 0.92, f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}',
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Demand Model — Residuals Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = PLOTS_DIR / "demand_residuals.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Residuals saved: {path}")


def plot_feature_importance(model, feature_names):
    try:
        rf = dict(model.named_estimators_)['rf']
        importances = rf.feature_importances_
    except Exception:
        print("⚠️  Could not extract feature importances")
        return

    fi = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi = fi.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.magma(np.linspace(0.2, 0.85, len(fi)))
    bars = ax.barh(fi['Feature'], fi['Importance'], color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_title('Demand Model — Feature Importance\n(Random Forest component)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / "demand_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Feature importance saved: {path}")


def plot_error_distribution(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    mask   = yt > 0
    pct_errors = np.abs((yt[mask] - yp[mask]) / yt[mask]) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(pct_errors[pct_errors < 200], bins=50, kde=True, color='#8e44ad', ax=ax)
    ax.axvline(np.median(pct_errors), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(pct_errors):.1f}%')
    ax.set_title('Demand Model — Percentage Error Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Absolute Percentage Error (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / "demand_error_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Error distribution saved: {path}")


def run_evaluation():
    print("\n" + "=" * 65)
    print("  DEMAND FORECAST MODEL EVALUATION")
    print("=" * 65)

    model = load_model()
    X, y  = load_and_prepare_data()

    print("\n🔍 Running predictions...")
    y_pred = model.predict(X)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)
    mape = compute_mape(y, y_pred)

    print("\n🔄 Running 5-fold cross-validation...")
    kf        = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)

    sep  = "─" * 65
    lines = [
        "=" * 65,
        "  DEMAND FORECAST MODEL — EVALUATION REPORT",
        "  Model: VotingRegressor (XGBoost + LightGBM + RandomForest)",
        "=" * 65,
        f"\nSample Size : {len(X):,} rows | {len(FEATURES)} features\n",
        sep,
        "REGRESSION METRICS",
        sep,
        f"  MAE   (Mean Absolute Error)          : {mae:.4f}",
        f"  RMSE  (Root Mean Squared Error)      : {rmse:.4f}",
        f"  R²    (Coefficient of Determination) : {r2:.4f}  ({r2*100:.2f}%)",
        f"  MAPE  (Mean Abs Percentage Error)    : {mape:.2f}%",
        "\n" + sep,
        "CROSS-VALIDATION (5-fold R²)",
        sep,
        f"  Fold scores : {np.round(cv_scores, 4).tolist()}",
        f"  Mean R²     : {cv_scores.mean():.4f}  ({cv_scores.mean()*100:.2f}%)",
        f"  Std Dev     : {cv_scores.std():.4f}",
        "\n" + sep,
        "FEATURES USED",
        sep,
    ] + [f"  {i+1}. {f}" for i, f in enumerate(FEATURES)]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = RESULTS_DIR / "demand_model_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n✅ Report saved: {report_path}")

    plot_actual_vs_predicted(y, y_pred)
    plot_residuals(y, y_pred)
    plot_feature_importance(model, FEATURES)
    plot_error_distribution(y, y_pred)

    print("\n" + "=" * 65)
    print("  DEMAND EVALUATION COMPLETE")
    print("=" * 65)

    return {
        "mae":        mae,
        "rmse":       rmse,
        "r2":         r2,
        "mape":       mape,
        "cv_mean_r2": cv_scores.mean(),
        "cv_std_r2":  cv_scores.std(),
    }


if __name__ == "__main__":
    run_evaluation()
