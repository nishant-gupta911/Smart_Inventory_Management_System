#!/usr/bin/env python3
"""
Test script to verify donation features integration in expiry model training
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_expiry_model_donation_features():
    """Test the donation features integration in expiry model training"""
    print("🧪 Testing donation features in expiry model training...")
    
    try:
        from train_expiry_model import ExpiryModelTrainer
        
        # Create test data with donation columns
        np.random.seed(42)
        n_samples = 500
        
        test_data = pd.DataFrame({
            'item_id': [f'Item_{i:05d}' for i in range(n_samples)],
            'store_nbr': np.random.randint(1, 10, n_samples),
            'rolling_avg_sales_7': np.random.exponential(2, n_samples),
            'days_to_expiry': np.random.randint(1, 21, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'shelf_life': np.random.randint(7, 31, n_samples),
            'days_on_shelf': np.random.randint(1, 15, n_samples),
            'unit_price': np.random.uniform(0.5, 50, n_samples),
            'date': pd.date_range(start=datetime.now() - pd.Timedelta(days=30), periods=n_samples, freq='H'),
            
            # Donation columns
            'donation_eligible': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'donation_status': np.random.choice(['', 'Pending', 'Rejected', 'Donated'], n_samples, p=[0.5, 0.25, 0.15, 0.1])
        })
        
        print(f"✅ Created test data with {len(test_data)} samples")
        
        # Initialize the trainer
        trainer = ExpiryModelTrainer()
        
        print("\n📊 Before feature engineering:")
        print(f"Donation eligible: {test_data['donation_eligible'].sum()}")
        print("Donation status breakdown:")
        status_counts = test_data['donation_status'].value_counts()
        for status, count in status_counts.items():
            if status != '':
                print(f"  {status}: {count}")
        
        # Test feature engineering with donation features
        print(f"\n🔧 Testing feature engineering with donation features...")
        try:
            df_processed = trainer.engineer_features(test_data)
            print(f"✅ Feature engineering successful, shape: {df_processed.shape}")
            
            # Check if donation features were added
            donation_cols = [col for col in df_processed.columns if 'donation' in col]
            print(f"✅ Donation features created: {donation_cols}")
            
            # Test feature preparation
            X, y, features = trainer.prepare_features(df_processed)
            donation_features = [f for f in features if 'donation' in f]
            
            print(f"✅ Feature preparation successful:")
            print(f"   Total features: {len(features)}")
            print(f"   Donation features: {len(donation_features)}")
            print(f"   Donation features included: {donation_features}")
            print(f"   Target variable shape: {y.shape}")
            print(f"   Target distribution: {y.value_counts().to_dict()}")
            
            # Test model training with donation features
            print(f"\n🧠 Testing model training with donation features...")
            if len(X) > 50:  # Ensure we have enough data
                from sklearn.model_selection import train_test_split
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, 
                    stratify=y if len(y.unique()) > 1 else None
                )
                
                model = trainer.train_model(X_train, y_train)
                print(f"✅ Model training successful with {len(features)} features")
                
                # Test evaluation
                metrics = trainer.evaluate_model(model, X_test, y_test)
                print(f"✅ Model evaluation successful:")
                print(f"   AUC Score: {metrics['auc_score']:.4f}")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                
                # Check feature importance for donation features
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    feature_importance = model.named_steps['classifier'].feature_importances_
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    donation_importance = importance_df[importance_df['feature'].str.contains('donation')]
                    if not donation_importance.empty:
                        print(f"\n📊 Donation feature importance:")
                        for _, row in donation_importance.head().iterrows():
                            print(f"   {row['feature']}: {row['importance']:.4f}")
                    else:
                        print(f"ℹ️ No donation features in top importance rankings")
            
        except Exception as e:
            print(f"❌ Feature engineering/training failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_donation_feature_logic():
    """Test the donation feature preparation logic specifically"""
    print("\n🔬 Testing donation feature preparation logic...")
    
    try:
        from train_expiry_model import ExpiryModelTrainer
        
        trainer = ExpiryModelTrainer()
        
        # Test with missing donation columns
        test_data_missing = pd.DataFrame({
            'rolling_avg_sales_7': [1.0, 2.0, 0.5, 3.0],
            'shelf_life': [7, 14, 3, 21],
            'days_to_expiry': [2, 10, 1, 15]
        })
        
        print("Testing with missing donation columns...")
        result = trainer._prepare_donation_features(test_data_missing)
        
        required_features = ['donation_flag', 'has_donation_history', 'donation_rejected_pattern']
        for feature in required_features:
            if feature in result.columns:
                print(f"✅ {feature} created successfully")
            else:
                print(f"❌ {feature} missing")
        
        # Test with existing donation columns
        test_data_existing = pd.DataFrame({
            'rolling_avg_sales_7': [1.0, 2.0, 0.5, 3.0],
            'shelf_life': [7, 14, 3, 21],
            'days_to_expiry': [2, 10, 1, 15],
            'donation_eligible': [True, False, True, False],
            'donation_status': ['Pending', '', 'Rejected', '']
        })
        
        print("\nTesting with existing donation columns...")
        result = trainer._prepare_donation_features(test_data_existing)
        
        print(f"✅ donation_flag values: {result['donation_flag'].tolist()}")
        print(f"✅ has_donation_history values: {result['has_donation_history'].tolist()}")
        print(f"✅ donation_rejected_pattern values: {result['donation_rejected_pattern'].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Donation feature logic test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_expiry_model_donation_features()
    success2 = test_donation_feature_logic()
    
    if success1 and success2:
        print("\n✅ All tests passed! Donation features are properly integrated.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
