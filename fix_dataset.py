#!/usr/bin/env python3
"""
Dataset correction script - fixing Action logic and donation eligibility
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("DATASET VERIFICATION AND CORRECTION")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/processed/inventory_analysis_results_enhanced.csv')
    print(f"   Loaded {len(df)} rows")
    
    # Step 2: Check current state
    print("\n2. Current Action distribution:")
    action_counts = df['Action'].value_counts()
    for action, count in action_counts.items():
        print(f"   {action}: {count}")
    
    print(f"\n3. Current donation eligible items: {df['donation_eligible'].sum()}")
    
    # Step 3: Fix donation eligibility
    print("\n4. Fixing donation eligibility...")
    
    # Correct donation eligibility logic
    edible_categories = ["DAIRY", "MEATS", "PRODUCE", "BAKERY", "FROZEN FOODS", "SNACKS", "BREAD/BAKERY", "FROZEN"]
    
    old_eligible = df['donation_eligible'].copy()
    df['donation_eligible'] = (
        (df['days_to_expiry'] >= -5) &
        (df['days_to_expiry'] <= -1) &
        (df['perishable'] == 1) &
        (df['category'].isin(edible_categories))
    )
    
    eligibility_changes = (old_eligible != df['donation_eligible']).sum()
    print(f"   Fixed {eligibility_changes} donation eligibility assignments")
    print(f"   New donation eligible count: {df['donation_eligible'].sum()}")
    
    # Step 4: Fix Action logic
    print("\n5. Fixing Action logic...")
    
    def get_correct_action(row):
        days = row['days_to_expiry']
        eligible = row['donation_eligible']
        
        if days < -5:
            return 'Remove'
        elif -5 <= days <= -1 and eligible:
            return 'Donate'
        elif 0 <= days <= 5:
            return 'Apply Discount'
        else:
            return 'Restock'
    
    old_actions = df['Action'].copy()
    df['Action'] = df.apply(get_correct_action, axis=1)
    
    action_changes = (old_actions != df['Action']).sum()
    print(f"   Fixed {action_changes} Action assignments")
    
    # Step 5: Show new distribution
    print("\n6. New Action distribution:")
    new_action_counts = df['Action'].value_counts()
    for action, count in new_action_counts.items():
        print(f"   {action}: {count}")
    
    # Step 6: Fix other data quality issues
    print("\n7. Fixing data quality issues...")
    
    # Fix Suggested_Discount NaN values
    nan_discounts = df['Suggested_Discount'].isna().sum()
    if nan_discounts > 0:
        df['Suggested_Discount'] = df['Suggested_Discount'].fillna(0)
        print(f"   Fixed {nan_discounts} NaN values in Suggested_Discount")
    
    # Fix negative/zero prices
    bad_prices = (df['unit_price'] <= 0) | df['unit_price'].isna()
    if bad_prices.sum() > 0:
        median_price = df[df['unit_price'] > 0]['unit_price'].median()
        df.loc[bad_prices, 'unit_price'] = median_price
        print(f"   Fixed {bad_prices.sum()} bad unit_price values")
    
    # Fix negative stock
    bad_stock = (df['current_stock'] < 0) | df['current_stock'].isna()
    if bad_stock.sum() > 0:
        median_stock = df[df['current_stock'] >= 0]['current_stock'].median()
        df.loc[df['current_stock'] < 0, 'current_stock'] = 0
        df.loc[df['current_stock'].isna(), 'current_stock'] = median_stock
        print(f"   Fixed {bad_stock.sum()} bad current_stock values")
    
    # Step 7: Save corrected dataset
    print("\n8. Saving corrected dataset...")
    output_file = 'data/processed/inventory_analysis_results_enhanced.csv'
    df.to_csv(output_file, index=False)
    print(f"   Saved to: {output_file}")
    
    # Step 8: Final verification
    print("\n9. Final verification - sample of corrected data:")
    print("-" * 60)
    
    # Show samples of each action type
    for action in ['Remove', 'Donate', 'Apply Discount', 'Restock']:
        action_items = df[df['Action'] == action]
        if len(action_items) > 0:
            sample = action_items.iloc[0]
            print(f"{action:15} | days_to_expiry: {sample['days_to_expiry']:3d} | eligible: {sample['donation_eligible']}")
    
    print("\n✅ DATASET CORRECTION COMPLETED!")
    print(f"   Total changes: {action_changes + eligibility_changes}")
    print(f"   Action changes: {action_changes}")
    print(f"   Eligibility changes: {eligibility_changes}")

if __name__ == "__main__":
    main()
