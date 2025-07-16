import pandas as pd
import joblib
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/restock_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RestockPlanGenerator:
    """
    Smart restocking plan generator with demand forecasting and expiry risk management.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the restock plan generator with configuration."""
        self.config = config or self._get_default_config()
        self.demand_model = None
        self.expiry_model = None
        self.feature_cols = [
            'day_of_week', 'month', 'is_weekend',
            'rolling_avg_sales_7', 'shelf_life', 'days_on_shelf', 'days_to_expiry'
        ]

    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'LOW_STOCK_THRESHOLD': 5,
            'EXPIRY_RISK_THRESHOLD': 0.7,
            'DEMAND_THRESHOLD': 10,
            'SAFETY_STOCK_MULTIPLIER': 1.2,
            'MIN_DAYS_TO_EXPIRY': 2,
            'MAX_RESTOCK_MULTIPLIER': 3.0,
            'DISCOUNT_THRESHOLD': 0.5,
            'HIGH_PRIORITY_FAMILIES': ['GROCERY I', 'BEVERAGES', 'DAIRY'],
            'PATHS': {
                'input_data': 'data/interim/cleaned_inventory_data.csv',
                'demand_model': 'models/demand_forecast_model.pkl',
                'expiry_model': 'models/expiry_predict_model.pkl',
                'output_dir': 'data/processed',
                'logs_dir': 'logs'
            }
        }

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path_key, path_value in self.config['PATHS'].items():
            if path_key in ['output_dir', 'logs_dir']:
                Path(path_value).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and validate input data."""
        try:
            logger.info("Loading preprocessed data...")
            df = pd.read_csv(self.config['PATHS']['input_data'])

            # Validate required columns
            required_cols = ['date', 'store_nbr', 'item_nbr', 'unit_sales', 'family']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])

            # Handle missing values
            df['unit_sales'].fillna(0, inplace=True)
            df['days_to_expiry'].fillna(df['days_to_expiry'].median(), inplace=True)

            logger.info(f"Loaded {len(df)} records from {df['date'].min()} to {df['date'].max()}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_models(self):
        """Load trained ML models."""
        try:
            logger.info("Loading trained models...")
            self.demand_model = joblib.load(self.config['PATHS']['demand_model'])
            self.expiry_model = joblib.load(self.config['PATHS']['expiry_model'])
            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction."""
        try:
            logger.info("Preparing features for prediction...")

            # Create time-based features if they don't exist
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['date'].dt.dayofweek
            if 'month' not in df.columns:
                df['month'] = df['date'].dt.month
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Create rolling averages if they don't exist
            if 'rolling_avg_sales_7' not in df.columns:
                df['rolling_avg_sales_7'] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )

            # One-hot encode family
            df_encoded = pd.get_dummies(df, columns=['family'], drop_first=True)

            # Create feature matrix
            family_cols = [col for col in df_encoded.columns if col.startswith('family_')]
            feature_columns = self.feature_cols + family_cols

            # Ensure all feature columns exist
            missing_features = [col for col in feature_columns if col not in df_encoded.columns]
            if missing_features:
                logger.warning(f"Missing features will be filled with 0: {missing_features}")
                for col in missing_features:
                    df_encoded[col] = 0

            return df_encoded

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate demand and expiry predictions."""
        try:
            logger.info("Generating predictions...")

            # Prepare feature matrix
            family_cols = [col for col in df.columns if col.startswith('family_')]
            X = df[self.feature_cols + family_cols]

            # Handle any remaining NaN values
            X = X.fillna(0)

            # Generate predictions
            df['predicted_demand'] = self.demand_model.predict(X)
            df['expiry_risk'] = self.expiry_model.predict_proba(X)[:, 1]

            # Ensure predictions are non-negative
            df['predicted_demand'] = np.maximum(df['predicted_demand'], 0)

            logger.info("Predictions generated successfully")
            return df

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def calculate_restock_quantity(self, row: pd.Series) -> int:
        """Calculate optimal restock quantity based on multiple factors."""
        try:
            predicted_demand = row['predicted_demand']
            current_stock = row['unit_sales']  # Assuming this represents current stock
            days_to_expiry = row.get('days_to_expiry', 30)
            family = row.get('family', '')

            # Base restock calculation
            base_restock = max(0, predicted_demand - current_stock)

            # Apply safety stock multiplier
            safety_stock = base_restock * self.config['SAFETY_STOCK_MULTIPLIER']

            # Adjust for expiry risk
            if days_to_expiry <= self.config['MIN_DAYS_TO_EXPIRY']:
                return 0  # Don't restock items about to expire

            # Adjust for high-priority families
            if family in self.config['HIGH_PRIORITY_FAMILIES']:
                safety_stock *= 1.1

            # Cap maximum restock quantity
            max_restock = predicted_demand * self.config['MAX_RESTOCK_MULTIPLIER']

            # Final restock quantity
            restock_qty = min(safety_stock, max_restock)

            # Only restock if predicted demand exceeds threshold
            if predicted_demand > self.config['DEMAND_THRESHOLD']:
                return int(restock_qty)

            return 0

        except Exception as e:
            logger.warning(f"Error calculating restock for row: {str(e)}")
            return 0

    def determine_discount_strategy(self, row: pd.Series) -> Dict:
        """Determine discount strategy based on expiry risk and demand."""
        try:
            expiry_risk = row['expiry_risk']
            predicted_demand = row['predicted_demand']
            days_to_expiry = row.get('days_to_expiry', 30)

            discount_info = {
                'apply_discount': 0,
                'discount_percentage': 0,
                'discount_reason': 'none'
            }

            # High expiry risk with low demand
            if expiry_risk > self.config['EXPIRY_RISK_THRESHOLD']:
                if predicted_demand < self.config['DEMAND_THRESHOLD']:
                    discount_info['apply_discount'] = 1
                    discount_info['discount_percentage'] = min(50, int(expiry_risk * 70))
                    discount_info['discount_reason'] = 'high_expiry_risk'

            # Items close to expiry
            elif days_to_expiry <= 3:
                discount_info['apply_discount'] = 1
                discount_info['discount_percentage'] = 30
                discount_info['discount_reason'] = 'close_to_expiry'

            # Overstocked items
            elif predicted_demand < row['unit_sales'] * self.config['DISCOUNT_THRESHOLD']:
                discount_info['apply_discount'] = 1
                discount_info['discount_percentage'] = 20
                discount_info['discount_reason'] = 'overstocked'

            return discount_info

        except Exception as e:
            logger.warning(f"Error determining discount strategy: {str(e)}")
            return {'apply_discount': 0, 'discount_percentage': 0, 'discount_reason': 'error'}

    def apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smart restocking and discount logic."""
        try:
            logger.info("Applying business rules...")

            # Calculate restock quantities
            df['restock_qty'] = df.apply(self.calculate_restock_quantity, axis=1)

            # Determine discount strategies
            discount_info = df.apply(self.determine_discount_strategy, axis=1)
            df['apply_discount'] = discount_info.apply(lambda x: x['apply_discount'])
            df['discount_percentage'] = discount_info.apply(lambda x: x['discount_percentage'])
            df['discount_reason'] = discount_info.apply(lambda x: x['discount_reason'])

            # Add priority flags
            df['high_priority'] = df.apply(
                lambda row: 1 if (
                        row['predicted_demand'] > self.config['DEMAND_THRESHOLD'] * 1.5 or
                        row.get('family', '') in self.config['HIGH_PRIORITY_FAMILIES']
                ) else 0, axis=1
            )

            # Add urgency score (0-100)
            df['urgency_score'] = np.clip(
                (df['predicted_demand'] / df['unit_sales'].replace(0, 1)) *
                (1 - df['days_to_expiry'] / 30) * 100, 0, 100
            ).astype(int)

            logger.info("Business rules applied successfully")
            return df

        except Exception as e:
            logger.error(f"Error applying business rules: {str(e)}")
            raise

    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the restock plan."""
        try:
            total_items = len(df)
            items_to_restock = len(df[df['restock_qty'] > 0])
            items_for_discount = len(df[df['apply_discount'] == 1])
            high_priority_items = len(df[df['high_priority'] == 1])

            total_restock_value = df['restock_qty'].sum()
            avg_expiry_risk = df['expiry_risk'].mean()
            avg_predicted_demand = df['predicted_demand'].mean()

            # Add donation-related statistics if columns exist
            donation_stats = {}
            if 'donation_eligible' in df.columns:
                donation_eligible_items = len(df[df['donation_eligible'] == True])
                donation_stats['donation_eligible_items'] = donation_eligible_items
                donation_stats['donation_eligible_percentage'] = round((donation_eligible_items / total_items) * 100, 2) if total_items > 0 else 0
            
            if 'donation_status' in df.columns:
                donation_status_counts = df['donation_status'].value_counts().to_dict()
                donation_stats['donation_status_breakdown'] = donation_status_counts

            summary = {
                'total_items': total_items,
                'items_to_restock': items_to_restock,
                'items_for_discount': items_for_discount,
                'high_priority_items': high_priority_items,
                'total_restock_quantity': total_restock_value,
                'avg_expiry_risk': round(avg_expiry_risk, 3),
                'avg_predicted_demand': round(avg_predicted_demand, 2),
                'restock_percentage': round((items_to_restock / total_items) * 100, 2),
                'discount_percentage': round((items_for_discount / total_items) * 100, 2),
                **donation_stats  # Include donation statistics
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return {}

    def save_results(self, df: pd.DataFrame, summary: Dict):
        """Save restocking suggestions and summary report."""
        try:
            logger.info("Saving results...")

            # Select relevant columns for output
            output_cols = [
                'date', 'store_nbr', 'item_nbr', 'family', 'unit_sales',
                'predicted_demand', 'days_to_expiry', 'expiry_risk',
                'restock_qty', 'apply_discount', 'discount_percentage', 'discount_reason',
                'high_priority', 'urgency_score'
            ]
            
            # Add donation columns if they exist
            if 'donation_eligible' in df.columns:
                output_cols.append('donation_eligible')
            if 'donation_status' in df.columns:
                output_cols.append('donation_status')

            # Filter columns that exist in dataframe
            available_cols = [col for col in output_cols if col in df.columns]
            restock_df = df[available_cols].copy()

            # Sort by urgency score and predicted demand
            restock_df = restock_df.sort_values(
                ['urgency_score', 'predicted_demand'],
                ascending=[False, False]
            )

            # Save main results
            output_path = Path(self.config['PATHS']['output_dir']) / 'restocking_suggestions.csv'
            restock_df.to_csv(output_path, index=False)

            # Save high-priority items separately
            high_priority_df = restock_df[restock_df['high_priority'] == 1]
            if not high_priority_df.empty:
                priority_path = Path(self.config['PATHS']['output_dir']) / 'high_priority_restock.csv'
                high_priority_df.to_csv(priority_path, index=False)

            # Save summary report
            summary_path = Path(self.config['PATHS']['output_dir']) / 'restock_summary.csv'
            pd.DataFrame([summary]).to_csv(summary_path, index=False)

            logger.info(f"✅ Results saved to {self.config['PATHS']['output_dir']}")
            logger.info(f"📊 Summary: {summary}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def filter_donation_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out items that are donation-eligible and have pending/donated status.
        These items should not be restocked as they're handled through donation channels.
        """
        try:
            logger.info("Filtering donation-eligible items from restock consideration...")
            
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
            
            # Count items that will be excluded
            excluded_items = df[
                (df['donation_eligible'] == True) & 
                (df['donation_status'].isin(['Pending', 'Donated']))
            ]
            excluded_count = len(excluded_items)
            
            # Filter out donation items that are pending or already donated
            filtered_df = df[
                ~((df['donation_eligible'] == True) & 
                  (df['donation_status'].isin(['Pending', 'Donated'])))
            ].copy()
            
            final_count = len(filtered_df)
            
            logger.info(f"📊 Donation filtering results:")
            logger.info(f"   Initial items: {initial_count:,}")
            logger.info(f"   Excluded (donation pending/donated): {excluded_count:,}")
            logger.info(f"   Remaining for restock consideration: {final_count:,}")
            
            if excluded_count > 0:
                # Log breakdown by donation status
                status_breakdown = excluded_items['donation_status'].value_counts()
                logger.info(f"   Excluded breakdown: {status_breakdown.to_dict()}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering donation items: {str(e)}")
            # Return original dataframe if filtering fails
            return df

    def run(self):
        """Execute the complete restock plan generation process."""
        try:
            logger.info("🚀 Starting restock plan generation...")

            # Setup
            self._ensure_directories()

            # Load data and models
            df = self.load_data()
            self.load_models()

            # Filter out donation items before processing
            df = self.filter_donation_items(df)

            # Process data
            df = self.prepare_features(df)
            df = self.generate_predictions(df)
            df = self.apply_business_rules(df)

            # Generate summary and save results
            summary = self.generate_summary_report(df)
            self.save_results(df, summary)

            logger.info("✅ Restock plan generation completed successfully!")
            return df, summary

        except Exception as e:
            logger.error(f"❌ Error in restock plan generation: {str(e)}")
            raise


# Main execution
if __name__ == "__main__":
    # Custom configuration (optional)
    custom_config = {
        'DEMAND_THRESHOLD': 8,
        'EXPIRY_RISK_THRESHOLD': 0.6,
        'SAFETY_STOCK_MULTIPLIER': 1.3,
        'HIGH_PRIORITY_FAMILIES': ['GROCERY I', 'BEVERAGES', 'DAIRY', 'PRODUCE']
    }

    # Initialize and run
    generator = RestockPlanGenerator(custom_config)
    df_result, summary_result = generator.run()

    print(f"\n🎯 Restock Plan Summary:")
    print(f"Total Items: {summary_result.get('total_items', 'N/A')}")
    print(f"Items to Restock: {summary_result.get('items_to_restock', 'N/A')}")
    print(f"Items for Discount: {summary_result.get('items_for_discount', 'N/A')}")
    print(f"High Priority Items: {summary_result.get('high_priority_items', 'N/A')}")
    print(f"Total Restock Quantity: {summary_result.get('total_restock_quantity', 'N/A')}")
    
    # Print donation-related information if available
    if 'donation_eligible_items' in summary_result:
        print(f"\n🤝 Donation Information:")
        print(f"Donation Eligible Items: {summary_result.get('donation_eligible_items', 'N/A')}")
        print(f"Donation Eligible %: {summary_result.get('donation_eligible_percentage', 'N/A')}%")
        
        if 'donation_status_breakdown' in summary_result:
            print("Donation Status Breakdown:")
            for status, count in summary_result['donation_status_breakdown'].items():
                if status and status != "":  # Only show non-empty statuses
                    print(f"  {status}: {count} items")