#!/usr/bin/env python3
"""
Simple simulation to update Action column with new logic
"""
import pandas as pd

def simulate_new_action_logic():
    # Read the CSV file  
    df = pd.read_csv("data/processed/inventory_analysis_results_enhanced.csv")
    
    print(f"Loaded {len(df)} rows")
    print("\nCurrent Action distribution:")
    print(df['Action'].value_counts())
    
    # Apply new logic
    removed_count = 0
    donated_count = 0
    discount_count = 0
    restock_count = 0
    no_action_count = 0
    
    for idx, row in df.iterrows():
        days_to_expiry = row['days_to_expiry']
        donation_eligible = row.get('donation_eligible', False)
        stock_level = row.get('Stock_Level', 'Normal')
        reorder = row.get('Reorder', 'No')
        
        if days_to_expiry < -5:
            new_action = 'Remove'
            removed_count += 1
        elif -5 <= days_to_expiry <= -1 and donation_eligible:
            new_action = 'Donate'
            donated_count += 1
        elif 0 <= days_to_expiry <= 5:
            new_action = 'Apply Discount'
            discount_count += 1
        elif days_to_expiry > 5:
            if stock_level == 'Low' or reorder == 'Yes':
                new_action = 'Restock'
                restock_count += 1
            else:
                new_action = 'No Action'
                no_action_count += 1
        else:
            new_action = 'No Action'
            no_action_count += 1
        
        df.at[idx, 'Action'] = new_action
    
    print(f"\nNew Action distribution (simulation):")
    print(f"Remove: {removed_count}")
    print(f"Donate: {donated_count}")
    print(f"Apply Discount: {discount_count}")
    print(f"Restock: {restock_count}")
    print(f"No Action: {no_action_count}")
    
    # Save the updated file
    df.to_csv("data/processed/inventory_analysis_results_enhanced.csv", index=False)
    print(f"\nFile updated successfully!")
    
    # Show some examples
    print(f"\nSample of updated data:")
    sample_cols = ['product_name', 'days_to_expiry', 'donation_eligible', 'Stock_Level', 'Reorder', 'Action']
    print(df[sample_cols].head(10).to_string())
    
    return df

if __name__ == "__main__":
    simulate_new_action_logic()
