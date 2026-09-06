#!/usr/bin/env python3
"""
Run All Evaluations
====================
Runs both model evaluations in sequence and prints a combined
summary table with all key metrics at the end.

Usage
-----
    python evaluation/run_all_evaluations.py

Output files produced
---------------------
  evaluation/results/expiry_model_report.txt
  evaluation/results/demand_model_report.txt
  plots/expiry_confusion_matrix.png
  plots/expiry_feature_importance.png
  plots/expiry_class_distribution.png
  plots/expiry_per_class_metrics.png
  plots/demand_actual_vs_predicted.png
  plots/demand_residuals.png
  plots/demand_feature_importance.png
  plots/demand_error_distribution.png
"""

import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

(PROJECT_ROOT / "evaluation" / "results").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "plots").mkdir(parents=True, exist_ok=True)


def run_expiry_evaluation():
    from evaluate_expiry_model import run_evaluation
    return run_evaluation()


def run_demand_evaluation():
    from evaluate_demand_model import run_evaluation
    return run_evaluation()


def print_combined_summary(expiry_metrics: dict, demand_metrics: dict):
    sep  = "═" * 65
    thin = "─" * 65

    print("\n\n" + sep)
    print("  COMBINED MODEL EVALUATION SUMMARY")
    print("  Smart Inventory Management System — Sparkathon 2025")
    print(sep)

    # ── Expiry Risk Classifier ─────────────────────────────────────────────────
    print("\n📦  EXPIRY RISK CLASSIFIER")
    print("    Model   : VotingClassifier (XGBoost + LightGBM + RandomForest)")
    print("    Task    : Multi-class classification (Safe / Near Expiry / Expired)")
    print("    Data    : 10% random sample of cleaned_inventory_data.csv, 5-fold CV")
    print(thin)
    if expiry_metrics:
        rows = [
            ("Accuracy  (weighted)",        f"{expiry_metrics.get('accuracy',   float('nan')):.4f}  ({expiry_metrics.get('accuracy', 0)*100:.2f}%)"),
            ("ROC-AUC   (weighted OvR)",    f"{expiry_metrics.get('roc_auc',    float('nan')):.4f}"),
            ("F1 — Safe class",             f"{expiry_metrics.get('f1_safe',    float('nan')):.4f}"),
            ("F1 — Expired class",          f"{expiry_metrics.get('f1_at_risk', float('nan')):.4f}"),
            ("CV Accuracy (mean, 5-fold)",  f"{expiry_metrics.get('cv_mean',    float('nan')):.4f}  ({expiry_metrics.get('cv_mean', 0)*100:.2f}%)"),
            ("CV Accuracy (std,  5-fold)",  f"{expiry_metrics.get('cv_std',     float('nan')):.4f}"),
        ]
        for name, value in rows:
            print(f"  {name:<35} {value}")
    else:
        print("  ⚠️  Expiry evaluation did not return metrics.")

    # ── Demand Forecast Model ──────────────────────────────────────────────────
    print(f"\n📈  DEMAND FORECAST MODEL")
    print("    Model   : VotingRegressor (XGBoost + LightGBM + RandomForest)")
    print("    Task    : Regression — predict unit sales per store/item")
    print("    Data    : 10% random sample of cleaned_inventory_data.csv, 5-fold CV")
    print(thin)
    if demand_metrics:
        rows = [
            ("MAE  (Mean Absolute Error)",  f"{demand_metrics.get('mae',        float('nan')):.4f}"),
            ("RMSE (Root Mean Sq. Error)",  f"{demand_metrics.get('rmse',       float('nan')):.4f}"),
            ("R²   (Coefficient of Det.)",  f"{demand_metrics.get('r2',         float('nan')):.4f}  ({demand_metrics.get('r2', 0)*100:.2f}%)"),
            ("MAPE (Mean Abs. % Error)",    f"{demand_metrics.get('mape',       float('nan')):.2f}%"),
            ("CV R²  (mean, 5-fold)",       f"{demand_metrics.get('cv_mean_r2', float('nan')):.4f}  ({demand_metrics.get('cv_mean_r2', 0)*100:.2f}%)"),
            ("CV R²  (std,  5-fold)",       f"{demand_metrics.get('cv_std_r2',  float('nan')):.4f}"),
        ]
        for name, value in rows:
            print(f"  {name:<35} {value}")
    else:
        print("  ⚠️  Demand evaluation did not return metrics.")

    # ── Output file locations ──────────────────────────────────────────────────
    print(f"\n{thin}")
    print("  OUTPUT FILES")
    print(thin)
    outputs = [
        ("Text reports", "evaluation/results/expiry_model_report.txt"),
        ("",             "evaluation/results/demand_model_report.txt"),
        ("Expiry plots", "plots/expiry_confusion_matrix.png"),
        ("",             "plots/expiry_feature_importance.png"),
        ("",             "plots/expiry_class_distribution.png"),
        ("",             "plots/expiry_per_class_metrics.png"),
        ("Demand plots", "plots/demand_actual_vs_predicted.png"),
        ("",             "plots/demand_residuals.png"),
        ("",             "plots/demand_feature_importance.png"),
        ("",             "plots/demand_error_distribution.png"),
    ]
    for label, path in outputs:
        exists = "✅" if (PROJECT_ROOT / path).exists() else "❌"
        print(f"  {exists}  {label:<14}  {path}")

    print("\n" + sep + "\n")


def main():
    print("╔" + "═" * 63 + "╗")
    print("║  SMART INVENTORY — FULL MODEL EVALUATION SUITE          ║")
    print("║  Sparkathon 2025 — Walmart Hackathon                    ║")
    print("╚" + "═" * 63 + "╝")

    expiry_metrics = None
    demand_metrics = None
    expiry_ok      = False
    demand_ok      = False

    # ── Run expiry evaluation ──────────────────────────────────────────────────
    print("\n[1/2] Starting Expiry Risk Model evaluation …\n")
    t0 = time.time()
    try:
        expiry_metrics = run_expiry_evaluation()
        expiry_ok      = True
        print(f"\n✅ Expiry evaluation finished in {time.time() - t0:.1f}s")
    except SystemExit:
        print("\n❌ Expiry evaluation aborted (missing model or data file).")
    except Exception:
        print("\n❌ Expiry evaluation failed with an unexpected error:")
        traceback.print_exc()

    # ── Run demand evaluation ──────────────────────────────────────────────────
    print("\n[2/2] Starting Demand Forecast Model evaluation …\n")
    t0 = time.time()
    try:
        demand_metrics = run_demand_evaluation()
        demand_ok      = True
        print(f"\n✅ Demand evaluation finished in {time.time() - t0:.1f}s")
    except SystemExit:
        print("\n❌ Demand evaluation aborted (missing model or data file).")
    except Exception:
        print("\n❌ Demand evaluation failed with an unexpected error:")
        traceback.print_exc()

    # ── Combined summary ───────────────────────────────────────────────────────
    if expiry_ok or demand_ok:
        print_combined_summary(expiry_metrics, demand_metrics)
    else:
        print("\n⚠️  Both evaluations failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()