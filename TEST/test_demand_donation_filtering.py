#!/usr/bin/env python3
"""
Test script to verify donation filtering in demand model training
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_demand_model_donation_filtering():
    """Test the donation filtering functionality in demand model training"""
    print("🧪 Testing donation filtering in demand model training...")
    
    try:
        from train_demand_model import DemandForecastModel
        
        # Create test data with donation columns
        np.random.seed(42)
        n_samples = 200
        
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
            'item_id': [f'Item_{i%20:03d}' for i in range(n_samples)],
            'store_id': np.random.randint(1, 6, n_samples),
            'shelf_life': np.random.randint(7, 31, n_samples),
            'days_to_expiry': np.random.randint(1, 21, n_samples),
            'family': np.random.choice(['Grocery', 'Electronics', 'Clothing', 'Home'], n_samples),
            'unit_sales': np.random.exponential(2, n_samples),
            
            # Donation columns
            'donation_eligible': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'donation_status': np.random.choice(['', 'Pending', 'Donated', 'Rejected'], n_samples, p=[0.5, 0.2, 0.2, 0.1])
        })
        
        print(f"✅ Created test data with {len(test_data)} samples")
        
        # Initialize the model
        model = DemandForecastModel()
        
        # Test the filtering function
        print("\n📊 Before filtering:")
        print(f"Total samples: {len(test_data)}")
        print(f"Donation eligible: {test_data['donation_eligible'].sum()}")
        print("Donation status breakdown:")
        status_counts = test_data['donation_status'].value_counts()
        for status, count in status_counts.items():
            if status != '':
                print(f"  {status}: {count}")
        
        # Count items that should be excluded (donated items)
        donated_items = test_data[
            (test_data['donation_eligible'] == True) & 
            (test_data['donation_status'] == 'Donated')
        ]
        expected_excluded = len(donated_items)
        
        print(f"Expected to exclude: {expected_excluded} donated items")
        
        # Apply donation filtering
        filtered_data = model.filter_donation_data(test_data)
        
        print(f"\n📊 After filtering:")
        print(f"Total samples: {len(filtered_data)}")
        print(f"Donation eligible: {filtered_data['donation_eligible'].sum()}")
        
        # Check if donation features were added
        if 'donation_flag' in filtered_data.columns:
            print(f"✅ donation_flag feature added: {filtered_data['donation_flag'].sum()} eligible items")
        
        donation_status_cols = [col for col in filtered_data.columns if col.startswith('donation_status_')]
        if donation_status_cols:
            print(f"✅ Donation status features added: {donation_status_cols}")
        
        # Verify filtering logic
        expected_remaining = len(test_data) - expected_excluded
        actual_remaining = len(filtered_data)
        
        print(f"\n✅ Verification:")
        print(f"Expected samples after filtering: {expected_remaining}")
        print(f"Actual samples after filtering: {actual_remaining}")
        print(f"Items excluded (donated): {expected_excluded}")
        
        if expected_remaining == actual_remaining:
            print("✅ Donation filtering working correctly!")
        else:
            print("❌ Donation filtering has issues!")
            
        # Check that no donated items remain
        remaining_donated = filtered_data[
            (filtered_data['donation_eligible'] == True) & 
            (filtered_data['donation_status'] == 'Donated')
        ]
        
        if len(remaining_donated) == 0:
            print("✅ No donated items in filtered training data!")
        else:
            print(f"❌ Found {len(remaining_donated)} donated items still in training data!")
        
        # Test feature engineering with donation features
        print(f"\n🔧 Testing feature engineering with donation features...")
        try:
            df_processed = model.engineer_features(filtered_data)
            print(f"✅ Feature engineering successful, shape: {df_processed.shape}")
            
            # Test feature selection
            features = model.select_features(df_processed)
            donation_features = [f for f in features if 'donation' in f]
            if donation_features:
                print(f"✅ Donation features selected: {donation_features}")
            else:
                print("ℹ️ No donation features in final feature set")
                
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_demand_model_donation_filtering()
