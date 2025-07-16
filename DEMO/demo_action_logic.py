import pandas as pd

# Test data showing different scenarios
test_data = [
    # Case 1: Too expired to donate/sell (days < -5)
    {'product_name': 'Expired Milk', 'days_to_expiry': -10, 'perishable': 1, 'category': 'DAIRY', 'donation_eligible': False, 'Stock_Level': 'High', 'Reorder': 'No'},
    
    # Case 2: Recently expired, donation eligible (-5 <= days <= -1)
    {'product_name': 'Near-Expired Bread', 'days_to_expiry': -3, 'perishable': 1, 'category': 'BREAD/BAKERY', 'donation_eligible': True, 'Stock_Level': 'High', 'Reorder': 'No'},
    
    # Case 3: Near expiry for discount (0 <= days <= 5)
    {'product_name': 'Discount Yogurt', 'days_to_expiry': 2, 'perishable': 1, 'category': 'DAIRY', 'donation_eligible': False, 'Stock_Level': 'High', 'Reorder': 'No'},
    
    # Case 4: Fresh but low stock (days > 5, needs restock)
    {'product_name': 'Low Stock Apples', 'days_to_expiry': 10, 'perishable': 1, 'category': 'PRODUCE', 'donation_eligible': False, 'Stock_Level': 'Low', 'Reorder': 'No'},
    
    # Case 5: Fresh and good stock (days > 5, no action needed)
    {'product_name': 'Fresh Bananas', 'days_to_expiry': 15, 'perishable': 1, 'category': 'PRODUCE', 'donation_eligible': False, 'Stock_Level': 'High', 'Reorder': 'No'},
    
    # Case 6: Fresh but needs reorder
    {'product_name': 'Reorder Chicken', 'days_to_expiry': 20, 'perishable': 1, 'category': 'MEATS', 'donation_eligible': False, 'Stock_Level': 'Medium', 'Reorder': 'Yes'},
]

df = pd.DataFrame(test_data)

# Apply the new logic
for idx, row in df.iterrows():
    days_to_expiry = row['days_to_expiry']
    donation_eligible = row['donation_eligible']
    stock_level = row['Stock_Level']
    reorder = row['Reorder']
    
    if days_to_expiry < -5:
        action = 'Remove'
    elif -5 <= days_to_expiry <= -1 and donation_eligible:
        action = 'Donate'
    elif 0 <= days_to_expiry <= 5:
        action = 'Apply Discount'
    elif days_to_expiry > 5:
        if stock_level == 'Low' or reorder == 'Yes':
            action = 'Restock'
        else:
            action = 'No Action'
    else:
        action = 'No Action'
    
    df.at[idx, 'Action'] = action

print("NEW ACTION LOGIC DEMONSTRATION")
print("=" * 80)
for idx, row in df.iterrows():
    print(f"\n{idx+1}. {row['product_name']}")
    print(f"   Days to expiry: {row['days_to_expiry']}")
    print(f"   Donation eligible: {row['donation_eligible']}")
    print(f"   Stock level: {row['Stock_Level']}")
    print(f"   Reorder needed: {row['Reorder']}")
    print(f"   → ACTION: {row['Action']}")

print("\n" + "=" * 80)
print("SUMMARY OF NEW LOGIC:")
print("• days_to_expiry < -5        → Remove (too expired)")
print("• -5 ≤ days_to_expiry ≤ -1 + donation_eligible → Donate")  
print("• 0 ≤ days_to_expiry ≤ 5     → Apply Discount")
print("• days_to_expiry > 5 + (Low stock OR Reorder=Yes) → Restock")
print("• days_to_expiry > 5 + (Good stock AND Reorder=No) → No Action")

# Save as demo file
df.to_csv('demo_new_action_logic.csv', index=False)
print(f"\nDemo data saved to: demo_new_action_logic.csv")
