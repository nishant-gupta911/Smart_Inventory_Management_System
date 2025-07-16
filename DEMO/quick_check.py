import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/processed/inventory_analysis_results_enhanced.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total rows: {len(df)}")

# Check Action distribution
print("\nCurrent Action distribution:")
print(df['Action'].value_counts())

# Sample of critical columns
critical_cols = ['product_name', 'days_to_expiry', 'Action', 'donation_eligible', 'donation_status', 'Suggested_Discount', 'unit_price', 'current_stock']
print(f"\nFirst 10 rows of critical columns:")
available_cols = [col for col in critical_cols if col in df.columns]
print(df[available_cols].head(10).to_string())

# Check Action logic
print("\nChecking Action logic...")

def get_expected_action(row):
    days = row['days_to_expiry']
    eligible = row.get('donation_eligible', False)
    
    if days < -5:
        return 'Remove'
    elif -5 <= days <= -1 and eligible:
        return 'Donate'
    elif 0 <= days <= 5:
        return 'Apply Discount'
    else:
        return 'Restock'

# Calculate expected actions
df['Expected_Action'] = df.apply(get_expected_action, axis=1)

# Find mismatches
action_mismatches = df[df['Action'] != df['Expected_Action']]
print(f"Action mismatches found: {len(action_mismatches)}")

if len(action_mismatches) > 0:
    print(f"\nSample mismatches:")
    sample_cols = ['days_to_expiry', 'donation_eligible', 'Action', 'Expected_Action']
    print(action_mismatches[sample_cols].head(20).to_string())

print(f"\nDonation eligible items: {df['donation_eligible'].sum()}")
print(f"Items with Action=Donate: {(df['Action'] == 'Donate').sum()}")
print(f"Items with Action=Apply Discount: {(df['Action'] == 'Apply Discount').sum()}")
print(f"Items with Action=Remove: {(df['Action'] == 'Remove').sum()}")
print(f"Items with Action=Restock: {(df['Action'] == 'Restock').sum()}")
