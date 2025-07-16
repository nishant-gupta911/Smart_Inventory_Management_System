#!/usr/bin/env python3
"""
Test script for donation features
"""
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import add_donation_features, validate_donation_columns, log_donation_summary

def test_donation_features():
    """Test the donation features functionality"""
    print("Testing donation features...")
    
    # Create test data
    np.random.seed(42)
    test_data = {
        'item_id': range(1, 11),
        'store_nbr': [1, 2, 1, 2, 3, 1, 2, 3, 1, 2],
        'days_to_expiry': [1, 2, 3, 5, 7, 0, 15, 20, 1, 2],
        'perishable': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        'sales': [10.5, 20.3, 5.0, 12.1, 8.5, 15.2, 25.0, 30.1, 7.8, 13.4]
    }
    
    df = pd.DataFrame(test_data)
    print(f"Created test dataframe with {len(df)} rows")
    print("Original columns:", df.columns.tolist())
    
    # Add donation features
    print("\nAdding donation features...")
    df_with_donations = add_donation_features(df)
    print("Columns after adding donation features:", df_with_donations.columns.tolist())
    
    # Validate donation columns
    print("\nValidating donation columns...")
    df_validated = validate_donation_columns(df_with_donations)
    print("Columns after validation:", df_validated.columns.tolist())
    
    # Show sample data
    print("\nSample data with donation features:")
    donation_cols = ['donation_eligible', 'donation_status', 'store_latitude', 'store_longitude', 'nearest_ngo']
    sample_cols = ['item_id', 'days_to_expiry', 'perishable'] + donation_cols
    print(df_validated[sample_cols].head(10))
    
    # Log summary
    print("\nDonation summary:")
    log_donation_summary(df_validated)
    
    # Save to CSV
    output_file = "test_donation_output.csv"
    df_validated.to_csv(output_file, index=False)
    print(f"\nSaved test data to {output_file}")
    
    return df_validated

if __name__ == "__main__":
    result = test_donation_features()
    print("Test completed successfully!")
