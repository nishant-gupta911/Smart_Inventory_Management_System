#!/usr/bin/env python3
"""
Generate fresh inventory data with correct Action logic
"""

import pandas as pd
from transform_inventory_data import generate_additional_rows

def create_fresh_data():
    """Create completely fresh data with correct Action logic"""
    print("Generating fresh inventory data with correct Action logic...")
    
    # Generate 1000 fresh rows
    fresh_df = generate_additional_rows(1000)
    
    # Save the fresh data
    output_file = 'data/processed/inventory_fresh_data.csv'
    fresh_df.to_csv(output_file, index=False)
    
    print(f"Fresh dataset saved to: {output_file}")
    print(f"Total rows: {len(fresh_df)}")
    
    # Analyze the data
    print("\nAction distribution:")
    action_counts = fresh_df['Action'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action}: {count}")
    
    print(f"\nDonation eligible items: {fresh_df['donation_eligible'].sum()}")
    
    # Verify Action logic
    print("\nVerifying Action logic:")
    
    # Check Remove items
    remove_items = fresh_df[fresh_df['Action'] == 'Remove']
    if len(remove_items) > 0:
        print(f"Remove items: {len(remove_items)}")
        days_range = f"Days to expiry range: {remove_items['days_to_expiry'].min()} to {remove_items['days_to_expiry'].max()}"
        print(f"  {days_range}")
        # Should all be < -5
        correct_remove = (remove_items['days_to_expiry'] < -5).all()
        print(f"  All correctly < -5 days: {correct_remove}")
    
    # Check Donate items
    donate_items = fresh_df[fresh_df['Action'] == 'Donate']
    if len(donate_items) > 0:
        print(f"Donate items: {len(donate_items)}")
        days_range = f"Days to expiry range: {donate_items['days_to_expiry'].min()} to {donate_items['days_to_expiry'].max()}"
        print(f"  {days_range}")
        # Should all be -5 <= days <= -1 AND donation_eligible == True
        correct_range = ((donate_items['days_to_expiry'] >= -5) & (donate_items['days_to_expiry'] <= -1)).all()
        correct_eligible = (donate_items['donation_eligible'] == True).all()
        print(f"  All in range -5 to -1: {correct_range}")
        print(f"  All donation eligible: {correct_eligible}")
    
    # Check Apply Discount items
    discount_items = fresh_df[fresh_df['Action'] == 'Apply Discount']
    if len(discount_items) > 0:
        print(f"Apply Discount items: {len(discount_items)}")
        days_range = f"Days to expiry range: {discount_items['days_to_expiry'].min()} to {discount_items['days_to_expiry'].max()}"
        print(f"  {days_range}")
        # Should all be 0 <= days <= 5
        correct_range = ((discount_items['days_to_expiry'] >= 0) & (discount_items['days_to_expiry'] <= 5)).all()
        print(f"  All in range 0 to 5: {correct_range}")
    
    # Check Restock items
    restock_items = fresh_df[fresh_df['Action'] == 'Restock']
    if len(restock_items) > 0:
        print(f"Restock items: {len(restock_items)}")
        days_range = f"Days to expiry range: {restock_items['days_to_expiry'].min()} to {restock_items['days_to_expiry'].max()}"
        print(f"  {days_range}")
        # Should all be > 5
        correct_range = (restock_items['days_to_expiry'] > 5).all()
        print(f"  All > 5 days: {correct_range}")
    
    # Show sample data
    print("\nSample data:")
    sample_cols = ['days_to_expiry', 'Action', 'donation_eligible', 'category', 'city']
    print(fresh_df[sample_cols].head(10))

if __name__ == "__main__":
    create_fresh_data()
