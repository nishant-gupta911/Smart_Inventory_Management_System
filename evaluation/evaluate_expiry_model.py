#!/usr/bin/env python3
"""
Expiry Risk Model Evaluation
Model: VotingClassifier (XGBoost + LightGBM + RandomForest)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
PLOTS_DIR    = PROJECT_ROOT / "plots"
RESULTS_DIR  = PROJECT_ROOT / "evaluation" / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Must match exactly what train_expiry_model.py used when training the saved .pkl
FEATURES = [
    'rolling_avg_sales_7', 'perishable', 'shelf_life',
    'stock_to_sales_ratio', 'sales_trend',
    'rolling_avg_3', 'rolling_avg_14', 'current_stock',
    'days_on_shelf'
]
CLASSES = ['Safe', 'Near Expiry', 'Expired']


def load_model():
    path = MODELS_DIR / "expiry_predict_model.pkl"
    if not path.exists():
        print(f"❌ Model not found: {path}\n   Run python main.py first.")
        sys.exit(1)
    model = joblib.load(path)
    print(f"✅ Expiry model loaded from: {path}")
    return model


def load_and_prepare_data():
    # Use the same file that was used for training with pre-computed Expiry_Risk labels
    path = DATA_DIR / "processed" / "expiry_risk_predictions.csv"
    if not path.exists():
        path = DATA_DIR / "cleaned_inventory_data.csv"
        if not path.exists():
            print(f"❌ Data not found: {path}")
            sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Data loaded from: {path}")
    print(f"   Rows: {len(df):,}")

    # Sort for correct rolling order
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
    df['sales_trend']         = df['rolling_avg_sales_7'] - df['rolling_avg_30']

    for col in ['current_stock', 'shelf_life', 'days_on_shelf', 'perishable']:
        if col not in df.columns:
            df[col] = 0

    df['stock_to_sales_ratio'] = df['current_stock'] / df['rolling_avg_sales_7'].replace(0, 1)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Expiry_Risk should already exist in the preprocessed file
    # If it doesn't, we can't evaluate properly — this indicates the model wasn't trained yet
    if 'Expiry_Risk' not in df.columns:
        print("❌ Expiry_Risk column not found. Run python main.py to train the model first.")
        sys.exit(1)

    df = df.dropna(subset=FEATURES + ['Expiry_Risk'])
    df = df.sample(frac=0.1, random_state=42)
    X  = df[FEATURES]
    y  = df['Expiry_Risk']
    print(f"✅ Evaluation sample: {len(df):,} rows")
    print(f"   Class distribution:\n{y.value_counts().to_string()}")
    return X, y


def plot_confusion_matrix(y_true, y_pred):
    cm     = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title('Confusion Matrix (% per class)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)

    plt.suptitle('Expiry Risk Classifier — Confusion Matrix',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = PLOTS_DIR / "expiry_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Confusion matrix saved: {path}")


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
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
    bars = ax.barh(fi['Feature'], fi['Importance'], color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_title('Expiry Risk Classifier — Feature Importance\n(Random Forest component)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / "expiry_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Feature importance saved: {path}")


def plot_class_distribution(y_true, y_pred):
    counts_true = pd.Series(y_true).value_counts().reindex(CLASSES).fillna(0)
    counts_pred = pd.Series(y_pred).value_counts().reindex(CLASSES).fillna(0)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(CLASSES, counts_true, color=colors, edgecolor='black', linewidth=0.8)
    axes[0].set_title('Actual Class Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts_true):
        axes[0].text(i, v + 50, f'{int(v):,}', ha='center', fontsize=10)

    axes[1].bar(CLASSES, counts_pred, color=colors, edgecolor='black', linewidth=0.8, alpha=0.8)
    axes[1].set_title('Predicted Class Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(counts_pred):
        axes[1].text(i, v + 50, f'{int(v):,}', ha='center', fontsize=10)

    plt.suptitle('Expiry Risk — Actual vs Predicted Distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = PLOTS_DIR / "expiry_class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Class distribution saved: {path}")


def plot_per_class_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    recall    = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    f1        = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)

    x     = np.arange(len(CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black')
    b2 = ax.bar(x,         recall,    width, label='Recall',    color='#2ecc71', edgecolor='black')
    b3 = ax.bar(x + width, f1,        width, label='F1-Score',  color='#e74c3c', edgecolor='black')

    for bars in [b1, b2, b3]:
        ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Expiry Risk Classifier — Per-Class Metrics',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = PLOTS_DIR / "expiry_per_class_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Per-class metrics saved: {path}")


def run_evaluation():
    print("\n" + "=" * 65)
    print("  EXPIRY RISK MODEL EVALUATION")
    print("=" * 65)

    model  = load_model()
    X, y   = load_and_prepare_data()

    print("\n🔍 Running predictions...")
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)

    acc       = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, labels=CLASSES, average='weighted', zero_division=0)
    recall    = recall_score(y, y_pred, labels=CLASSES, average='weighted', zero_division=0)
    f1        = f1_score(y, y_pred, labels=CLASSES, average='weighted', zero_division=0)

    try:
        y_bin   = label_binarize(y, classes=CLASSES)
        roc_auc = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = float('nan')

    print("\n🔄 Running 5-fold cross-validation...")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    cls_report = classification_report(
        y, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0
    )

    sep = "─" * 65
    lines = [
        "=" * 65,
        "  EXPIRY RISK MODEL — EVALUATION REPORT",
        "  Model: VotingClassifier (XGBoost + LightGBM + RandomForest)",
        "=" * 65,
        f"\nSample Size : {len(X):,} rows | {len(FEATURES)} features",
        f"Classes     : {CLASSES}\n",
        sep,
        "OVERALL METRICS",
        sep,
        f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)",
        f"  Weighted Precision : {precision:.4f}",
        f"  Weighted Recall    : {recall:.4f}",
        f"  Weighted F1-Score  : {f1:.4f}",
        f"  ROC-AUC (weighted) : {roc_auc:.4f}",
        "\n" + sep,
        "CROSS-VALIDATION (5-fold accuracy)",
        sep,
        f"  Fold scores : {np.round(cv_scores, 4).tolist()}",
        f"  Mean        : {cv_scores.mean():.4f}  ({cv_scores.mean()*100:.2f}%)",
        f"  Std Dev     : {cv_scores.std():.4f}",
        "\n" + sep,
        "FULL CLASSIFICATION REPORT",
        sep,
        cls_report,
        sep,
        "FEATURES USED",
        sep,
    ] + [f"  {i+1}. {f}" for i, f in enumerate(FEATURES)]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = RESULTS_DIR / "expiry_model_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n✅ Report saved: {report_path}")

    plot_confusion_matrix(y, y_pred)
    plot_feature_importance(model, FEATURES)
    plot_class_distribution(y, y_pred)
    plot_per_class_metrics(y, y_pred)

    print("\n" + "=" * 65)
    print("  EXPIRY EVALUATION COMPLETE")
    print("=" * 65)

    return {
        "accuracy":   acc,
        "roc_auc":    roc_auc,
        "f1_safe":    f1_score(y, y_pred, labels=['Safe'],    average=None, zero_division=0)[0],
        "f1_at_risk": f1_score(y, y_pred, labels=['Expired'], average=None, zero_division=0)[0],
        "cv_mean":    cv_scores.mean(),
        "cv_std":     cv_scores.std(),
    }


if __name__ == "__main__":
    run_evaluation()
