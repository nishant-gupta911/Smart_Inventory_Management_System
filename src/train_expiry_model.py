import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm

FEATURES = [
    'rolling_avg_sales_7', 'perishable', 'shelf_life',
    'stock_to_sales_ratio', 'sales_trend',
    'rolling_avg_3', 'rolling_avg_14', 'current_stock',
    'days_on_shelf'
]

def train_and_predict():
    with tqdm(total=8, desc="🏷️  Expiry Model", unit="step",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:

        pbar.set_postfix_str("Loading data...")
        df = pd.read_csv('data/cleaned_inventory_data.csv', low_memory=False)
        pbar.update(1)

        pbar.set_postfix_str("Computing features...")
        df['rolling_avg_3'] = df.groupby(['store_nbr', 'item_id'])['sales'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['rolling_avg_14'] = df.groupby(['store_nbr', 'item_id'])['sales'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df['rolling_avg_30'] = df.groupby(['store_nbr', 'item_id'])['sales'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        df['sales_trend'] = df['rolling_avg_sales_7'] - df['rolling_avg_30']
        df['stock_to_sales_ratio'] = df['current_stock'] / df['rolling_avg_sales_7'].replace(0, 1)
        pbar.update(1)

        pbar.set_postfix_str("Filling missing values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        pbar.update(1)

        pbar.set_postfix_str("Creating labels with noise...")
        df['Expiry_Risk'] = 'Safe'
        df.loc[df['days_to_expiry'] <= 0, 'Expiry_Risk'] = 'Expired'
        df.loc[(df['days_to_expiry'] > 0) & (df['days_to_expiry'] <= 15), 'Expiry_Risk'] = 'Near Expiry'
        np.random.seed(42)
        noise_mask = np.random.random(len(df)) < 0.05
        classes = ['Safe', 'Near Expiry', 'Expired']
        df.loc[noise_mask, 'Expiry_Risk'] = np.random.choice(classes, noise_mask.sum())
        df = df.dropna(subset=FEATURES + ['Expiry_Risk'])
        pbar.update(1)

        pbar.set_postfix_str("Building models...")
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, n_jobs=-1)
        lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        rf = RandomForestClassifier(n_estimators=300, max_depth=12,
            class_weight='balanced', random_state=42, n_jobs=-1)
        ensemble = VotingClassifier(
            [('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
            voting='soft', weights=[3, 2, 1])
        pbar.update(1)

        # ⚡ CV on 10% sample — fast but representative
        pbar.set_postfix_str("Cross-validating on 10% sample...")
        df_sample = df.sample(frac=0.1, random_state=42)
        X_sample = df_sample[FEATURES]
        y_sample = df_sample['Expiry_Risk']
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(ensemble, X_sample, y_sample, cv=cv, scoring='accuracy', verbose=0)
        pbar.update(1)

        # Final fit on ALL data
        pbar.set_postfix_str("Training on full dataset...")
        X = df[FEATURES]
        y = df['Expiry_Risk']
        ensemble.fit(X, y)
        pbar.update(1)

        pbar.set_postfix_str("Saving model...")
        joblib.dump(ensemble, 'models/expiry_predict_model.pkl')
        df['Expiry_Risk_Predicted'] = ensemble.predict(X)
        df.to_csv('data/processed/expiry_risk_predictions.csv', index=False)
        pbar.update(1)

    print(f"\n📊 Expiry Model Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print("✅ Expiry model trained and saved.")
    return scores.mean()