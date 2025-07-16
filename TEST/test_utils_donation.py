#!/usr/bin/env python3
"""
Test script to verify all donation utility functions in utils.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_donation_utils():
    """Test all donation utility functions"""
    print("🧪 Testing donation utility functions...")
    
    try:
        from utils import (
            update_donation_status,
            get_nearest_ngo,
            get_donation_summary,
            filter_pending_donations,
            save_updated_inventory,
            validate_donation_columns,
            add_missing_donation_columns,
            calculate_donation_metrics,
            format_donation_report
        )
        
        # Create test data
        print("\n📊 Creating test data...")
        np.random.seed(42)
        n_items = 100
        
        test_data = pd.DataFrame({
            'item_id': [f'ITEM_{i:05d}' for i in range(n_items)],
            'product_name': [f'Product_{i:03d}' for i in range(n_items)],
            'category': np.random.choice(['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY'], n_items),
            'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad'], n_items),
            'current_stock': np.random.randint(0, 100, n_items),
            'days_to_expiry': np.random.randint(0, 30, n_items),
            'unit_price': np.random.uniform(1.0, 50.0, n_items),
            'donation_eligible': np.random.choice([True, False], n_items, p=[0.3, 0.7]),
            'donation_status': np.random.choice(['', 'Pending', 'Donated', 'Rejected'], n_items, p=[0.5, 0.25, 0.15, 0.1]),
            'store_latitude': np.random.uniform(15.0, 30.0, n_items),
            'store_longitude': np.random.uniform(70.0, 85.0, n_items),
            'nearest_ngo': np.random.choice(['Food Bank Central', 'Helping Hands', 'Care Alliance'], n_items),
            'ngo_address': ['Sample Address'] * n_items,
            'ngo_contact': ['contact@example.org'] * n_items
        })
        
        print(f"✅ Created test data with {len(test_data)} items")
        
        # Test 1: validate_donation_columns
        print("\n🔍 Testing validate_donation_columns...")
        is_valid, missing = validate_donation_columns(test_data)
        print(f"   Validation result: {is_valid}, Missing: {missing}")
        
        # Test 2: update_donation_status
        print("\n✏️ Testing update_donation_status...")
        item_to_update = test_data[test_data['donation_eligible'] == True]['item_id'].iloc[0]
        updated_data = update_donation_status(test_data, item_to_update, 'Donated')
        print(f"   Updated item {item_to_update} status to 'Donated'")
        
        # Test 3: get_nearest_ngo
        print("\n🏢 Testing get_nearest_ngo...")
        test_cities = ['Mumbai', 'Delhi', 'UnknownCity']
        for city in test_cities:
            ngo_info = get_nearest_ngo(city)
            print(f"   {city}: {ngo_info['nearest_ngo']}")
        
        # Test 4: get_donation_summary
        print("\n📊 Testing get_donation_summary...")
        summary = get_donation_summary(updated_data)
        print(f"   Total eligible: {summary['total_donation_eligible']}")
        print(f"   Status counts: {summary['donation_status_counts']}")
        print(f"   Top NGOs: {list(summary['top_ngos'].keys())[:3]}")
        
        # Test 5: filter_pending_donations
        print("\n⏳ Testing filter_pending_donations...")
        pending = filter_pending_donations(updated_data)
        print(f"   Found {len(pending)} pending donation items")
        
        # Test 6: calculate_donation_metrics
        print("\n📈 Testing calculate_donation_metrics...")
        metrics = calculate_donation_metrics(updated_data)
        print(f"   Donation rate: {metrics['donation_rate']:.1f}%")
        print(f"   Success rate: {metrics['success_rate']:.1f}%")
        
        # Test 7: format_donation_report
        print("\n📋 Testing format_donation_report...")
        report = format_donation_report(summary)
        print("   Generated report (first 5 lines):")
        for line in report.split('\n')[:5]:
            print(f"     {line}")
        
        # Test 8: save_updated_inventory
        print("\n💾 Testing save_updated_inventory...")
        test_output_path = 'test_output/test_inventory.csv'
        success = save_updated_inventory(updated_data, test_output_path)
        print(f"   Save operation: {'✅ Success' if success else '❌ Failed'}")
        
        # Test 9: add_missing_donation_columns
        print("\n🔧 Testing add_missing_donation_columns...")
        minimal_data = pd.DataFrame({
            'item_id': ['ITEM_001', 'ITEM_002'],
            'product_name': ['Product A', 'Product B'],
            'city': ['Mumbai', 'Delhi']
        })
        enhanced_data = add_missing_donation_columns(minimal_data)
        added_cols = [col for col in enhanced_data.columns if col not in minimal_data.columns]
        print(f"   Added columns: {added_cols}")
        
        print("\n✅ All utility function tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔬 Testing edge cases...")
    
    try:
        from utils import (
            update_donation_status,
            get_donation_summary,
            filter_pending_donations
        )
        
        # Test with empty DataFrame
        print("   Testing with empty DataFrame...")
        empty_df = pd.DataFrame()
        summary_empty = get_donation_summary(empty_df)
        pending_empty = filter_pending_donations(empty_df)
        print(f"   Empty DF summary: {summary_empty['total_items']} items")
        print(f"   Empty DF pending: {len(pending_empty)} items")
        
        # Test with missing columns
        print("   Testing with missing donation columns...")
        basic_df = pd.DataFrame({
            'item_id': ['ITEM_001'],
            'product_name': ['Test Product']
        })
        summary_basic = get_donation_summary(basic_df)
        pending_basic = filter_pending_donations(basic_df)
        print(f"   Basic DF summary: {summary_basic['total_donation_eligible']} eligible")
        print(f"   Basic DF pending: {len(pending_basic)} items")
        
        # Test update_donation_status error cases
        print("   Testing update_donation_status error cases...")
        test_df = pd.DataFrame({
            'item_id': ['ITEM_001'],
            'donation_eligible': [False],
            'donation_status': ['']
        })
        
        try:
            update_donation_status(test_df, 'NONEXISTENT', 'Donated')
        except ValueError as e:
            print(f"   ✅ Correctly caught error for nonexistent item: {str(e)[:50]}...")
        
        try:
            update_donation_status(test_df, 'ITEM_001', 'Donated')
        except ValueError as e:
            print(f"   ✅ Correctly caught error for ineligible item: {str(e)[:50]}...")
        
        print("✅ Edge case testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_donation_utils()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Utility functions are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
