import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Possible locations for the raw/cleaned data
DATA_PATHS = [
    PROJECT_ROOT / "data" / "cleaned_inventory_data.csv",
    PROJECT_ROOT / "data" / "processed" / "inventory_analysis_results_enhanced.csv",
    PROJECT_ROOT / "src" / "data" / "interim" / "cleaned_inventory_data.csv",
    PROJECT_ROOT / "cleaned_inventory_data.csv",
]


def _load_raw_data() -> pd.DataFrame:
    """Load the best available data file."""
    for path in DATA_PATHS:
        if path.exists():
            logger.info(f"📥 Loading data from: {path}")
            return pd.read_csv(path)

    # No file found — generate synthetic data so the pipeline can still run
    logger.warning("⚠️ No data file found. Generating synthetic data for pipeline.")
    return _generate_synthetic_data()


def _generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    """Generate minimal synthetic inventory data for testing."""
    rng = np.random.default_rng(42)
    stores = rng.integers(1, 11, n)
    items = [f"ITEM_{i:05d}" for i in range(n)]
    families = rng.choice(
        ["DAIRY", "PRODUCE", "MEATS", "BREAD/BAKERY", "FROZEN FOODS", "SEAFOOD"], n
    )

    dates = pd.date_range("2023-01-01", periods=n, freq="h")

    df = pd.DataFrame(
        {
            "item_id": items,
            "product_name": [f"Product_{i:03d}" for i in range(n)],
            "store_nbr": stores,
            "family": families,
            "category": families,
            "sales": rng.exponential(5, n).round(2),
            "onpromotion": rng.integers(0, 2, n),
            "current_stock": rng.integers(0, 200, n),
            "shelf_life": rng.choice([2, 3, 7, 14, 30, 90], n),
            "days_on_shelf": rng.integers(0, 30, n),
            "days_to_expiry": rng.integers(-5, 31, n),
            "unit_price": rng.uniform(0.5, 50, n).round(2),
            "cluster": rng.integers(1, 6, n),
            "date": dates,
            "perishable": rng.integers(0, 2, n),
        }
    )
    return df


def preprocess_data() -> pd.DataFrame:
    """
    Load raw data, engineer features, and return the processed DataFrame.

    Features produced (aligned with FEATURES lists in train_demand_model.py
    and train_expiry_model.py):
        rolling_avg_sales_7, rolling_avg_3, rolling_avg_14, rolling_avg_30
        sales_lag_1, sales_lag_7
        sales_trend
        shelf_consumed_ratio
        stock_to_sales_ratio
        days_of_stock_left
        is_holiday_month
        quarter
        day_of_week, month, is_weekend
        sales_velocity, urgency_score
        Expiry_Risk  (label for expiry model)
    """
    df = _load_raw_data()

    # ── Date features ─────────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    else:
        # Fallback constants so the model still has the columns
        df["day_of_week"] = 0
        df["month"] = 1
        df["is_weekend"] = 0

    # ── Ensure 'sales' exists ──────────────────────────────────────────────────
    if "sales" not in df.columns:
        if "rolling_avg_sales_7" in df.columns:
            df["sales"] = df["rolling_avg_sales_7"]
        else:
            df["sales"] = 0.0

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).clip(lower=0)

    # ── Sort so rolling/lag ops are in chronological order ────────────────────
    sort_cols = [c for c in ["store_nbr", "item_id", "date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    group_keys = [c for c in ["store_nbr", "item_id"] if c in df.columns]

    # ── Rolling averages ──────────────────────────────────────────────────────
    def rolling(window):
        if group_keys:
            return df.groupby(group_keys)["sales"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        return df["sales"].rolling(window, min_periods=1).mean()

    df["rolling_avg_sales_7"] = rolling(7)   # used by both models + inventory_analyzer
    df["rolling_avg_3"] = rolling(3)
    df["rolling_avg_14"] = rolling(14)
    df["rolling_avg_30"] = rolling(30)

    # ── Sales trend ───────────────────────────────────────────────────────────
    df["sales_trend"] = df["rolling_avg_sales_7"] - df["rolling_avg_30"]

    # ── Lag features ──────────────────────────────────────────────────────────
    if group_keys:
        df["sales_lag_1"] = df.groupby(group_keys)["sales"].shift(1)
        df["sales_lag_7"] = df.groupby(group_keys)["sales"].shift(7)
    else:
        df["sales_lag_1"] = df["sales"].shift(1)
        df["sales_lag_7"] = df["sales"].shift(7)

    df["sales_lag_1"] = df["sales_lag_1"].fillna(0)
    df["sales_lag_7"] = df["sales_lag_7"].fillna(0)

    # ── Shelf / stock features ────────────────────────────────────────────────
    if "days_on_shelf" not in df.columns:
        df["days_on_shelf"] = 0
    if "shelf_life" not in df.columns:
        df["shelf_life"] = 30
    if "current_stock" not in df.columns:
        df["current_stock"] = 0

    df["shelf_consumed_ratio"] = df["days_on_shelf"] / df["shelf_life"].replace(0, 1)
    df["stock_to_sales_ratio"] = df["current_stock"] / df["rolling_avg_sales_7"].replace(0, 1)
    df["days_of_stock_left"] = df["current_stock"] / df["rolling_avg_sales_7"].replace(0, 1)

    # ── Calendar features ─────────────────────────────────────────────────────
    df["is_holiday_month"] = df["month"].isin([10, 11, 12]).astype(int)
    df["quarter"] = df["month"].apply(lambda x: (x - 1) // 3 + 1)

    # ── Expiry model extras ───────────────────────────────────────────────────
    if "days_to_expiry" not in df.columns:
        df["days_to_expiry"] = 30

    df["sales_velocity"] = df["rolling_avg_sales_7"]  # alias used by expiry model
    df["urgency_score"] = (
        (30 - df["days_to_expiry"].clip(lower=0)) / 30
    ).clip(0, 1)

    if "perishable" not in df.columns:
        df["perishable"] = df.get("family", pd.Series(dtype=str)).isin(
            ["DAIRY", "PRODUCE", "MEATS", "BREAD/BAKERY", "SEAFOOD"]
        ).astype(int)

    # ── Expiry_Risk label (for expiry classifier) ─────────────────────────────
    if "Expiry_Risk" not in df.columns:
        conditions = [
            df["days_to_expiry"] < -5,
            df["days_to_expiry"] <= 0,
            df["days_to_expiry"] <= 15,
        ]
        choices = ["Remove", "Expired", "Near Expiry"]
        df["Expiry_Risk"] = np.select(conditions, choices, default="Safe")

    # ── Save processed file for downstream use ────────────────────────────────
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed_inventory.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"✅ Preprocessed data saved → {out_path}  ({len(df):,} rows)")

    return df


def main():
    """Entry point when run directly."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    df = preprocess_data()
    print(f"✅ Preprocessing complete. Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
