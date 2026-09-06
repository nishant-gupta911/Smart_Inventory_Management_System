import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm

FEATURES = [
    'rolling_avg_sales_7', 'rolling_avg_3', 'rolling_avg_14',
    'sales_lag_1', 'sales_lag_7',
    'sales_trend', 'onpromotion',
    'day_of_week', 'month', 'is_weekend', 'quarter',
    'store_nbr', 'cluster', 'perishable',
    'dcoilwtico'  # oil price — economic signal
]

def train_and_evaluate():
    with tqdm(total=8, desc="📈 Demand Model", unit="step",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:

        pbar.set_postfix_str("Loading data...")
        df = pd.read_csv('data/cleaned_inventory_data.csv', low_memory=False)
        pbar.update(1)

        pbar.set_postfix_str("Computing features...")
        # Add lag features
        df['rolling_avg_3']  = df.groupby(['store_nbr', 'item_id'])['sales'].transform(lambda x: x.rolling(3,  min_periods=1).mean())
        df['rolling_avg_14'] = df.groupby(['store_nbr', 'item_id'])['sales'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df['sales_lag_1']    = df.groupby(['store_nbr', 'item_id'])['sales'].shift(1)
        df['sales_lag_7']    = df.groupby(['store_nbr', 'item_id'])['sales'].shift(7)
        df['rolling_avg_30'] = df.groupby(['store_nbr', 'item_id'])['sales'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        df['sales_trend']    = df['rolling_avg_sales_7'] - df['rolling_avg_30']
        df['quarter']        = df['month'].apply(lambda x: (x - 1) // 3 + 1)

        # Fill oil price NaN with forward fill
        df['dcoilwtico'] = df['dcoilwtico'].ffill().fillna(0)
        pbar.update(1)

        pbar.set_postfix_str("Filling missing values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df = df.dropna(subset=FEATURES + ['sales'])
        pbar.update(1)

        pbar.set_postfix_str("Building models...")
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05,
            num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
        rf = RandomForestRegressor(n_estimators=200, max_depth=10,
            random_state=42, n_jobs=-1)
        ensemble = VotingRegressor(
            [('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
            weights=[3, 2, 1])
        pbar.update(1)

        # ⚡ CV on 10% sample — fast but representative
        pbar.set_postfix_str("Cross-validating on 10% sample...")
        df_sample = df.sample(frac=0.1, random_state=42)
        X_sample = df_sample[FEATURES]
        y_sample = df_sample['sales']
        fold_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        with tqdm(total=5, desc="   └─ CV Folds", unit="fold", leave=False,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as fold_bar:
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_sample)):
                X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
                y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]
                ensemble.fit(X_train, y_train)
                score = ensemble.score(X_val, y_val)
                fold_scores.append(score)
                fold_bar.set_postfix_str(f"Fold {fold+1} R²: {score:.4f}")
                fold_bar.update(1)

        scores = np.array(fold_scores)
        pbar.update(1)

        # Final fit on ALL data
        pbar.set_postfix_str("Training on full dataset...")
        X = df[FEATURES]
        y = df['sales']
        ensemble.fit(X, y)
        pbar.update(1)

        pbar.set_postfix_str("Saving model...")
        joblib.dump(ensemble, 'models/demand_forecast_model.pkl')
        pbar.update(1)

        pbar.set_postfix_str("Done!")
        pbar.update(1)

    accuracy = max(0, scores.mean()) * 100
    print(f"\n📊 Demand Model R² Accuracy: {accuracy:.1f}% ± {scores.std()*100:.1f}%")
    print("✅ Demand model trained and saved.")
    return scores.mean()