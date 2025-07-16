#!/usr/bin/env python3
"""
Quick script to apply new Action logic to existing data
"""
import pandas as pd
import os

def main():
    # Read existing data
    data_file = "data/processed/inventory_analysis_results_enhanced.csv"
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows")
    
    # Apply new Action logic
    print("Applying new Action logic...")
    for idx, row in df.iterrows():
        days_to_expiry = row['days_to_expiry']
        donation_eligible = row.get('donation_eligible', False)
        stock_level = row.get('Stock_Level', 'Normal')
        reorder = row.get('Reorder', 'No')
        
        if days_to_expiry < -5:
            df.at[idx, 'Action'] = 'Remove'
        elif -5 <= days_to_expiry <= -1 and donation_eligible:
            df.at[idx, 'Action'] = 'Donate'
        elif 0 <= days_to_expiry <= 5:
            df.at[idx, 'Action'] = 'Apply Discount'
        elif days_to_expiry > 5:
            if stock_level == 'Low' or reorder == 'Yes':
                df.at[idx, 'Action'] = 'Restock'
            else:
                df.at[idx, 'Action'] = 'No Action'
        else:
            df.at[idx, 'Action'] = 'No Action'
    
    # Save updated data
    print("Saving updated data...")
    df.to_csv(data_file, index=False)
    
    # Print summary
    action_counts = df['Action'].value_counts()
    print("\nAction Summary:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")
    
    # Check donation eligible items
    donation_eligible_count = (df['donation_eligible'] == True).sum()
    remove_count = (df['Action'] == 'Remove').sum()
    donate_count = (df['Action'] == 'Donate').sum()
    
    print(f"\nDonation Analysis:")
    print(f"  Total donation eligible: {donation_eligible_count}")
    print(f"  Items to remove: {remove_count}")
    print(f"  Items to donate: {donate_count}")
    
    print("Update complete!")

if __name__ == "__main__":
    main()
