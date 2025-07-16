#!/usr/bin/env python3
"""
Test script to verify the donation filtering functionality in the restock plan generator
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_donation_filtering():
    """Test the donation filtering functionality"""
    print("🧪 Testing donation filtering in restock plan generator...")
    
    try:
        from generate_restock_plan import RestockPlanGenerator
        
        # Create test data with donation columns
        np.random.seed(42)
        n_items = 100
        
        test_data = pd.DataFrame({
            'date': [datetime.now().date()] * n_items,
            'store_nbr': np.random.randint(1, 6, n_items),
            'item_nbr': [f'ITEM_{i:04d}' for i in range(n_items)],
            'family': np.random.choice(['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY'], n_items),
            'unit_sales': np.random.randint(0, 50, n_items),
            'days_to_expiry': np.random.randint(0, 30, n_items),
            'shelf_life': np.random.choice([3, 7, 14, 30], n_items),
            'rolling_avg_sales_7': np.random.exponential(5, n_items),
            'days_on_shelf': np.random.randint(0, 20, n_items),
            
            # Donation columns
            'donation_eligible': np.random.choice([True, False], n_items, p=[0.3, 0.7]),
            'donation_status': np.random.choice(['Pending', 'Donated', 'Rejected', ''], n_items, p=[0.15, 0.1, 0.05, 0.7])
        })
        
        print(f"✅ Created test data with {len(test_data)} items")
        
        # Initialize the generator
        generator = RestockPlanGenerator()
        
        # Test the filtering function
        print("\n📊 Before filtering:")
        print(f"Total items: {len(test_data)}")
        print(f"Donation eligible: {test_data['donation_eligible'].sum()}")
        print("Donation status breakdown:")
        status_counts = test_data['donation_status'].value_counts()
        for status, count in status_counts.items():
            if status != '':
                print(f"  {status}: {count}")
        
        # Apply donation filtering
        filtered_data = generator.filter_donation_items(test_data)
        
        print(f"\n📊 After filtering:")
        print(f"Total items: {len(filtered_data)}")
        print(f"Donation eligible: {filtered_data['donation_eligible'].sum()}")
        
        # Verify filtering logic
        excluded_items = test_data[
            (test_data['donation_eligible'] == True) & 
            (test_data['donation_status'].isin(['Pending', 'Donated']))
        ]
        
        expected_remaining = len(test_data) - len(excluded_items)
        actual_remaining = len(filtered_data)
        
        print(f"\n✅ Verification:")
        print(f"Expected items after filtering: {expected_remaining}")
        print(f"Actual items after filtering: {actual_remaining}")
        print(f"Items excluded (donation pending/donated): {len(excluded_items)}")
        
        if expected_remaining == actual_remaining:
            print("✅ Donation filtering working correctly!")
        else:
            print("❌ Donation filtering has issues!")
            
        # Check that no pending or donated items remain
        remaining_problematic = filtered_data[
            (filtered_data['donation_eligible'] == True) & 
            (filtered_data['donation_status'].isin(['Pending', 'Donated']))
        ]
        
        if len(remaining_problematic) == 0:
            print("✅ No pending/donated items in filtered results!")
        else:
            print(f"❌ Found {len(remaining_problematic)} pending/donated items still in results!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_donation_filtering()
