import pandas as pd

# Read the data
df = pd.read_csv("data/processed/inventory_analysis_results_enhanced.csv")
print(f"Loaded {len(df)} rows")

# Check current Action values
print("Current Action distribution:")
print(df['Action'].value_counts())

# Apply new logic to first 5 rows as test
for idx in range(5):
    days_to_expiry = df.iloc[idx]['days_to_expiry']
    donation_eligible = df.iloc[idx].get('donation_eligible', False)
    stock_level = df.iloc[idx].get('Stock_Level', 'Normal')
    reorder = df.iloc[idx].get('Reorder', 'No')
    
    if days_to_expiry < -5:
        new_action = 'Remove'
    elif -5 <= days_to_expiry <= -1 and donation_eligible:
        new_action = 'Donate'
    elif 0 <= days_to_expiry <= 5:
        new_action = 'Apply Discount'
    elif days_to_expiry > 5:
        if stock_level == 'Low' or reorder == 'Yes':
            new_action = 'Restock'
        else:
            new_action = 'No Action'
    else:
        new_action = 'No Action'
    
    old_action = df.iloc[idx]['Action']
    print(f"Row {idx}: days_to_expiry={days_to_expiry}, donation_eligible={donation_eligible}, old_action='{old_action}', new_action='{new_action}'")

# Apply to all rows
print("\nApplying to all rows...")
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

# Save the updated data
df.to_csv("data/processed/inventory_analysis_results_enhanced.csv", index=False)

print("New Action distribution:")
print(df['Action'].value_counts())

# Summary
donate_count = (df['Action'] == 'Donate').sum()
remove_count = (df['Action'] == 'Remove').sum()
discount_count = (df['Action'] == 'Apply Discount').sum()
restock_count = (df['Action'] == 'Restock').sum()

print(f"\nSummary:")
print(f"Items to Remove: {remove_count}")
print(f"Items to Donate: {donate_count}")
print(f"Items for Discount: {discount_count}")
print(f"Items to Restock: {restock_count}")

print("Update completed successfully!")
