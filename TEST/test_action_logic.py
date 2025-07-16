#!/usr/bin/env python3

# Test script to verify the new Action logic
import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the transformation function
from transform_inventory_data import apply_donation_logic_to_dataframe

def test_action_logic():
    """Test the new Action column logic"""
    print("Testing Action column logic...")
    
    # Create test data with various scenarios
    test_data = [
        # Test case 1: Too expired (< -5)
        {'days_to_expiry': -10, 'perishable': 1, 'category': 'DAIRY', 'donation_eligible': False, 'Stock_Level': 'High', 'Reorder': 'No'},
        # Test case 2: Recently expired and donation eligible (-5 to -1)
        {'days_to_expiry': -3, 'perishable': 1, 'category': 'DAIRY', 'donation_eligible': True, 'Stock_Level': 'High', 'Reorder': 'No'},
        # Test case 3: Near expiry (0 to 5)
        {'days_to_expiry': 2, 'perishable': 1, 'category': 'DAIRY', 'donation_eligible': False, 'Stock_Level': 'High', 'Reorder': 'No'},
        # Test case 4: Fresh and low stock (> 5)
        {'days_to_expiry': 10, 'perishable': 1, 'category': 'DAIRY', 'donation_eligible': False, 'Stock_Level': 'Low', 'Reorder': 'No'},
        # Test case 5: Fresh and high stock (> 5)
        {'days_to_expiry': 15, 'perishable': 1, 'category': 'DAIRY', 'donation_eligible': False, 'Stock_Level': 'High', 'Reorder': 'No'},
    ]
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Apply transformation
    result_df = apply_donation_logic_to_dataframe(df)
    
    # Print results
    print("\nTest Results:")
    print("=" * 80)
    for idx, row in result_df.iterrows():
        print(f"Case {idx+1}: days_to_expiry={row['days_to_expiry']}, donation_eligible={row['donation_eligible']}, Action='{row['Action']}'")
    
    print("\nExpected vs Actual:")
    expected_actions = ['Remove', 'Donate', 'Apply Discount', 'Restock', 'No Action']
    for idx, (expected, actual) in enumerate(zip(expected_actions, result_df['Action'])):
        status = "✓" if expected == actual else "✗"
        print(f"{status} Case {idx+1}: Expected='{expected}', Actual='{actual}'")

if __name__ == "__main__":
    test_action_logic()
