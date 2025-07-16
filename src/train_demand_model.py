# src/train_demand_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Get project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class DemandForecastModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.imputer = None

    def load_and_validate_data(self, data_path=None):
        """Load and validate the dataset with fallback data generation"""
        try:
            # Try multiple possible data paths
            possible_paths = [
                data_path,
                os.path.join(project_root, "data", "cleaned_inventory_data.csv"),
                os.path.join(project_root, "data", "interim", "cleaned_inventory_data.csv"),
                os.path.join(project_root, "cleaned_inventory_data.csv"),
                "data/cleaned_inventory_data.csv",
                "cleaned_inventory_data.csv"
            ]
            
            df = None
            data_found = False
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    try:
                        df = pd.read_csv(path)
                        logger.info(f"✅ Data loaded from: {path}")
                        data_found = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {str(e)}")
                        continue
            
            if not data_found:
                logger.warning("⚠️ No data file found. Generating sample data for demonstration.")
                df = self._generate_sample_data()

            # Validate and clean data
            df = self._validate_and_clean_data(df)

            logger.info(f"Loaded dataset with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _validate_and_clean_data(self, df):
        """Validate and clean the loaded data"""
        # Ensure date column exists and is properly formatted
        if 'date' not in df.columns:
            df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Remove rows with invalid dates
            df = df.dropna(subset=['date'])

        # Ensure unit_sales column exists
        if 'unit_sales' not in df.columns:
            if 'rolling_avg_sales_7' in df.columns:
                df['unit_sales'] = df['rolling_avg_sales_7']
            else:
                df['unit_sales'] = np.random.exponential(2, len(df))

        # Handle negative sales values
        df['unit_sales'] = df['unit_sales'].clip(lower=0)

        # Basic data validation
        if df['unit_sales'].isna().all():
            logger.warning("All unit_sales values are missing, generating random values")
            df['unit_sales'] = np.random.exponential(2, len(df))

        # Ensure required columns exist
        required_columns = ['date', 'unit_sales']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        logger.info("Data validation passed")
        return df

    def _generate_sample_data(self):
        """Generate sample inventory data"""
        logger.info("🔧 Generating sample demand data...")
        
        np.random.seed(42)
        n_samples = 5000
        
        # Generate dates
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Generate more realistic data with patterns
        df = pd.DataFrame({
            'date': dates,
            'item_id': [f'Item_{i%100:03d}' for i in range(n_samples)],
            'store_id': np.random.randint(1, 11, n_samples),
            'shelf_life': np.random.randint(7, 31, n_samples),
            'days_to_expiry': np.random.randint(1, 21, n_samples),
            'family': np.random.choice(['Grocery', 'Electronics', 'Clothing', 'Home'], n_samples)
        })
        
        # Generate unit_sales with seasonal patterns
        base_sales = np.random.exponential(2, n_samples)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        weekend_factor = np.where(dates.day_of_week >= 5, 1.2, 1.0)
        df['unit_sales'] = base_sales * seasonal_factor * weekend_factor
        
        # Add donation-related columns for testing
        df['donation_eligible'] = np.random.choice([True, False], n_samples, p=[0.2, 0.8])
        df['donation_status'] = np.random.choice(['', 'Pending', 'Donated', 'Rejected'], n_samples, p=[0.6, 0.15, 0.15, 0.1])
        
        return df

    def engineer_features(self, df):
        """Enhanced feature engineering"""
        try:
            df_processed = df.copy()

            # Ensure date column is datetime
            df_processed['date'] = pd.to_datetime(df_processed['date'])

            # Sort by date for time-based features
            df_processed = df_processed.sort_values(['date'])

            # Date-based features
            df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
            df_processed['month'] = df_processed['date'].dt.month
            df_processed['quarter'] = df_processed['date'].dt.quarter
            df_processed['day_of_month'] = df_processed['date'].dt.day
            df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
            df_processed['is_month_start'] = (df_processed['date'].dt.day <= 7).astype(int)
            df_processed['is_month_end'] = (df_processed['date'].dt.day >= 24).astype(int)

            # Rolling statistics (if not already present)
            if 'rolling_avg_sales_7' not in df_processed.columns:
                # Use groupby with transform to maintain index alignment
                df_processed['rolling_avg_sales_7'] = df_processed.groupby('item_id')['unit_sales'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )
                df_processed['rolling_avg_sales_14'] = df_processed.groupby('item_id')['unit_sales'].transform(
                    lambda x: x.rolling(window=14, min_periods=1).mean()
                )
                df_processed['rolling_std_sales_7'] = df_processed.groupby('item_id')['unit_sales'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).std()
                )

            # Lag features
            df_processed['sales_lag_1'] = df_processed.groupby('item_id')['unit_sales'].shift(1)
            df_processed['sales_lag_7'] = df_processed.groupby('item_id')['unit_sales'].shift(7)

            # Inventory-related features
            if 'shelf_life' in df_processed.columns:
                df_processed['shelf_life'] = df_processed['shelf_life'].clip(lower=1)  # Avoid zero/negative values
                df_processed['shelf_life_log'] = np.log1p(df_processed['shelf_life'])

            if 'days_to_expiry' in df_processed.columns:
                df_processed['days_to_expiry'] = df_processed['days_to_expiry'].clip(lower=0)
                df_processed['expiry_urgency'] = np.where(
                    df_processed['days_to_expiry'] <= 3, 1, 0
                )
                if 'shelf_life' in df_processed.columns:
                    df_processed['expiry_ratio'] = df_processed['days_to_expiry'] / df_processed['shelf_life']
                else:
                    df_processed['expiry_ratio'] = df_processed['days_to_expiry'] / 30  # Normalize by 30 days

            # Seasonal patterns
            df_processed['sin_day'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
            df_processed['cos_day'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
            df_processed['sin_month'] = np.sin(2 * np.pi * df_processed['month'] / 12)
            df_processed['cos_month'] = np.cos(2 * np.pi * df_processed['month'] / 12)
            df_processed['sin_day_of_year'] = np.sin(2 * np.pi * df_processed['day_of_year'] / 365)

            # Handle categorical features
            if 'family' in df_processed.columns:
                df_processed = pd.get_dummies(df_processed, columns=['family'], prefix='family', drop_first=True)

            # Fill remaining NaN values
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)

            logger.info(f"Feature engineering completed. New shape: {df_processed.shape}")
            return df_processed

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise

    def select_features(self, df):
        """Select and validate features"""
        base_features = [
            'day_of_week', 'day_of_year', 'month', 'quarter', 'day_of_month',
            'is_weekend', 'is_month_start', 'is_month_end',
            'sin_day', 'cos_day', 'sin_month', 'cos_month', 'sin_day_of_year'
        ]

        # Add available features
        optional_features = [
            'rolling_avg_sales_7', 'rolling_avg_sales_14', 'rolling_std_sales_7',
            'sales_lag_1', 'sales_lag_7', 'shelf_life', 'shelf_life_log',
            'days_to_expiry', 'expiry_urgency', 'expiry_ratio', 'donation_flag'
        ]

        # Include only existing features
        features = [f for f in base_features if f in df.columns]
        features.extend([f for f in optional_features if f in df.columns])

        # Add family dummy variables
        family_features = [col for col in df.columns if col.startswith('family_')]
        features.extend(family_features)
        
        # Add donation status dummy variables
        donation_status_features = [col for col in df.columns if col.startswith('donation_status_')]
        features.extend(donation_status_features)

        # Remove features with too many missing values or zero variance
        final_features = []
        for feature in features:
            if feature in df.columns:
                missing_pct = df[feature].isna().mean()
                if missing_pct < 0.5:  # Less than 50% missing
                    if df[feature].var() > 0:  # Has variance
                        final_features.append(feature)
                    else:
                        logger.warning(f"Removing feature {feature} due to zero variance")
                else:
                    logger.warning(f"Removing feature {feature} due to high missing values ({missing_pct:.1%})")

        if not final_features:
            raise ValueError("No valid features found after selection")

        logger.info(f"Selected {len(final_features)} features: {final_features}")
        return final_features

    def preprocess_data(self, X, y, is_training=True):
        """Preprocess features and handle missing values"""
        try:
            # Handle missing values
            if is_training:
                self.imputer = SimpleImputer(strategy='median')
                X_imputed = self.imputer.fit_transform(X)
            else:
                if self.imputer is None:
                    raise ValueError("Model not trained. Call fit() first.")
                X_imputed = self.imputer.transform(X)

            # Convert back to DataFrame
            X_processed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

            # Remove outliers (only during training)
            if is_training and len(y) > 0:
                # Remove extreme outliers in target variable using IQR method
                Q1 = y.quantile(0.25)
                Q3 = y.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (y >= lower_bound) & (y <= upper_bound)
                X_processed = X_processed[outlier_mask]
                y = y[outlier_mask]

                outliers_removed = (~outlier_mask).sum()
                if outliers_removed > 0:
                    logger.info(f"Removed {outliers_removed} outliers ({outliers_removed/len(y)*100:.1f}%)")

            return X_processed, y

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    def train_model(self, X_train, y_train, optimize_params=False):
        """Train the model with optional hyperparameter optimization"""
        try:
            if optimize_params:
                # Hyperparameter tuning with reduced search space for faster training
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                }

                rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
                grid_search = GridSearchCV(
                    rf, param_grid, cv=3, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=1
                )

                logger.info("Starting hyperparameter optimization...")
                grid_search.fit(X_train, y_train)

                self.model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

            else:
                # Use optimized default parameters
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                self.model.fit(X_train, y_train)

            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        """Comprehensive model evaluation"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train_model() first.")

            y_pred = self.model.predict(X_test)

            # Calculate metrics
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE safely
            mape = 0
            if len(y_test) > 0:
                non_zero_mask = y_test != 0
                if non_zero_mask.sum() > 0:
                    mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100

            # Cross-validation score
            cv_rmse = None
            if X_train is not None and y_train is not None:
                try:
                    cv_scores = cross_val_score(
                        self.model, X_train, y_train, cv=3,
                        scoring='neg_mean_squared_error'
                    )
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    logger.info(f"Cross-validation RMSE: {cv_rmse:.4f}")
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {str(e)}")

            metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape,
                'CV_RMSE': cv_rmse
            }

            # Print results
            print("\n" + "=" * 50)
            print("MODEL EVALUATION RESULTS")
            print("=" * 50)
            print(f"📊 RMSE: {rmse:.4f}")
            print(f"📊 MAE: {mae:.4f}")
            print(f"📊 R² Score: {r2:.4f}")
            print(f"📊 MAPE: {mape:.2f}%")
            if cv_rmse:
                print(f"📊 CV RMSE: {cv_rmse:.4f}")

            # Feature importance
            if self.feature_importance is not None:
                print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
                print(self.feature_importance.head(10).to_string(index=False))

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def save_model(self, model_dir=None):
        """Save the trained model and preprocessing objects"""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")

            if model_dir is None:
                model_dir = os.path.join(project_root, "models")
            
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(model_dir, "demand_forecast_model.pkl")
            joblib.dump(self.model, model_path)

            # Save preprocessing objects
            if self.imputer:
                imputer_path = os.path.join(model_dir, "demand_imputer.pkl")
                joblib.dump(self.imputer, imputer_path)

            # Save feature names
            if self.feature_names:
                feature_names_path = os.path.join(model_dir, "demand_feature_names.pkl")
                joblib.dump(self.feature_names, feature_names_path)

            # Save feature importance
            if self.feature_importance is not None:
                importance_path = os.path.join(model_dir, "demand_feature_importance.csv")
                self.feature_importance.to_csv(importance_path, index=False)

            logger.info(f"Model and artifacts saved to {model_dir}")
            print(f"🎯 Model saved to: {model_path}")
            print(f"🎯 Preprocessing objects saved to: {model_dir}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_dir=None):
        """Load a trained model and preprocessing objects"""
        try:
            if model_dir is None:
                model_dir = os.path.join(project_root, "models")

            # Load model
            model_path = os.path.join(model_dir, "demand_forecast_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)

            # Load preprocessing objects
            imputer_path = os.path.join(model_dir, "demand_imputer.pkl")
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)

            # Load feature names
            feature_names_path = os.path.join(model_dir, "demand_feature_names.pkl")
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)

            # Load feature importance
            importance_path = os.path.join(model_dir, "demand_feature_importance.csv")
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path)

            logger.info(f"Model loaded from {model_dir}")
            print(f"🎯 Model loaded from: {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, X):
        """Make predictions with the trained model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train_model() or load_model() first.")

            # Preprocess the input data
            X_processed, _ = self.preprocess_data(X, pd.Series([]), is_training=False)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            # Ensure non-negative predictions
            predictions = np.maximum(predictions, 0)
            
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def filter_donation_data(self, df):
        """
        Filter out donated items from training data and add donation-related features.
        """
        try:
            logger.info("Filtering donation data from training set...")
            
            initial_count = len(df)
            
            # Add default values for donation columns if they don't exist
            if 'donation_eligible' not in df.columns:
                logger.warning("donation_eligible column not found, assuming no items are donation eligible")
                df['donation_eligible'] = False
            
            if 'donation_status' not in df.columns:
                logger.warning("donation_status column not found, assuming empty status")
                df['donation_status'] = ""
            
            # Fill NaN values in donation columns
            df['donation_eligible'] = df['donation_eligible'].fillna(False)
            df['donation_status'] = df['donation_status'].fillna("")
            
            # Count items that will be excluded (donated items)
            excluded_items = df[
                (df['donation_eligible'] == True) & 
                (df['donation_status'] == 'Donated')
            ]
            excluded_count = len(excluded_items)
            
            # Filter out donated items from training data
            # We want to predict actual sales, not donated inventory
            filtered_df = df[
                ~((df['donation_eligible'] == True) & 
                  (df['donation_status'] == 'Donated'))
            ].copy()
            
            final_count = len(filtered_df)
            
            # Add donation-related features for better prediction
            # Donation flag feature
            filtered_df['donation_flag'] = filtered_df['donation_eligible'].astype(int)
            
            # Donation status as categorical feature (excluding 'Donated' since those are filtered out)
            donation_status_dummies = pd.get_dummies(
                filtered_df['donation_status'], 
                prefix='donation_status',
                drop_first=True
            )
            filtered_df = pd.concat([filtered_df, donation_status_dummies], axis=1)
            
            logger.info(f"📊 Donation filtering results:")
            logger.info(f"   Initial training samples: {initial_count:,}")
            logger.info(f"   Excluded (donated items): {excluded_count:,}")
            logger.info(f"   Remaining for training: {final_count:,}")
            logger.info(f"   Added donation features: donation_flag + {len(donation_status_dummies.columns)} status dummies")
            
            if excluded_count > 0:
                # Log what was excluded
                logger.info(f"   Excluded donated items to focus on actual sales prediction")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering donation data: {str(e)}")
            # Return original dataframe if filtering fails
            return df


def main():
    """Main training pipeline"""
    try:
        logger.info("🚀 Starting Demand Forecasting Model Training Pipeline")
        
        # Initialize model
        demand_model = DemandForecastModel(random_state=42)

        # Load and process data
        df = demand_model.load_and_validate_data()

        # Filter donation data
        df_filtered = demand_model.filter_donation_data(df)

        # Feature engineering
        df_processed = demand_model.engineer_features(df_filtered)

        # Select features
        features = demand_model.select_features(df_processed)
        demand_model.feature_names = features

        # Prepare data
        X = df_processed[features]
        y = df_processed['unit_sales']

        # Remove rows with missing target
        valid_mask = ~y.isna()
        X, y = X[valid_mask], y[valid_mask]

        if len(X) == 0:
            raise ValueError("No valid data found after filtering")

        print(f"📈 Dataset shape: {X.shape}")
        print(f"📈 Target variable stats:")
        print(f"   Mean: {y.mean():.4f}")
        print(f"   Std: {y.std():.4f}")
        print(f"   Min: {y.min():.4f}")
        print(f"   Max: {y.max():.4f}")
        print(f"   Median: {y.median():.4f}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        # Preprocess data
        X_train_processed, y_train_processed = demand_model.preprocess_data(
            X_train, y_train, is_training=True
        )
        X_test_processed, y_test_processed = demand_model.preprocess_data(
            X_test, y_test, is_training=False
        )

        # Train model
        print("\n🤖 Training model...")
        demand_model.train_model(
            X_train_processed, y_train_processed,
            optimize_params=False  # Set to True for hyperparameter tuning
        )

        # Evaluate model
        metrics = demand_model.evaluate_model(
            X_test_processed, y_test_processed,
            X_train_processed, y_train_processed
        )

        # Save model
        demand_model.save_model()

        print("\n✅ Demand forecasting training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


def run():
    """Entry point for the training pipeline"""
    try:
        main()
        print("✅ Demand model trained successfully")
        return True
    except Exception as e:
        logger.error(f"Demand model training failed: {str(e)}")
        print(f"❌ Demand model training failed: {str(e)}")
        return False


if __name__ == "__main__":
    run()