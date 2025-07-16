#!/usr/bin/env python3
"""
Data verification and correction script for inventory dataset
"""

import pandas as pd
import numpy as np

def load_and_inspect_dataset():
    """Load and inspect the inventory dataset"""
    print("=" * 60)
    print("STEP 1: LOADING AND INSPECTING DATASET")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('data/processed/inventory_analysis_results_enhanced.csv')
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display first few rows of critical columns
    critical_cols = ['product_name', 'days_to_expiry', 'Action', 'donation_eligible', 
                    'donation_status', 'Suggested_Discount', 'unit_price', 'current_stock']
    
    print(f"\nFirst 10 rows of critical columns:")
    print("-" * 80)
    available_cols = [col for col in critical_cols if col in df.columns]
    print(df[available_cols].head(10).to_string())
    
    return df

def validate_critical_columns(df):
    """Validate critical columns according to business rules"""
    print("\n" + "=" * 60)
    print("STEP 2: VALIDATING CRITICAL COLUMNS")
    print("=" * 60)
    
    issues_found = []
    
    # Check Action logic
    print("\n1. Checking Action column logic...")
    
    # Define expected actions based on business rules
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
    
    print(f"   Total rows: {len(df)}")
    print(f"   Action mismatches found: {len(action_mismatches)}")
    
    if len(action_mismatches) > 0:
        print(f"\n   Sample mismatches:")
        sample_cols = ['days_to_expiry', 'donation_eligible', 'Action', 'Expected_Action']
        print(action_mismatches[sample_cols].head(10).to_string())
        issues_found.append(f"Action logic: {len(action_mismatches)} incorrect assignments")
    
    # Check donation_eligible logic
    print("\n2. Checking donation_eligible logic...")
    
    # Define expected donation eligibility
    edible_categories = ["DAIRY", "MEATS", "PRODUCE", "BAKERY", "FROZEN FOODS", "SNACKS", "BREAD/BAKERY", "FROZEN"]
    
    expected_donation_eligible = (
        (df['days_to_expiry'] >= -5) &
        (df['days_to_expiry'] <= -1) &
        (df['perishable'] == 1) &
        (df['category'].isin(edible_categories))
    )
    
    df['Expected_Donation_Eligible'] = expected_donation_eligible
    
    # Find mismatches
    donation_mismatches = df[df['donation_eligible'] != df['Expected_Donation_Eligible']]
    
    print(f"   Donation eligibility mismatches found: {len(donation_mismatches)}")
    
    if len(donation_mismatches) > 0:
        print(f"\n   Sample mismatches:")
        sample_cols = ['days_to_expiry', 'perishable', 'category', 'donation_eligible', 'Expected_Donation_Eligible']
        print(donation_mismatches[sample_cols].head(10).to_string())
        issues_found.append(f"Donation eligibility: {len(donation_mismatches)} incorrect assignments")
    
    # Check Suggested_Discount
    print("\n3. Checking Suggested_Discount column...")
    
    discount_issues = 0
    if 'Suggested_Discount' in df.columns:
        # Check for NaN values
        nan_discounts = df['Suggested_Discount'].isna().sum()
        if nan_discounts > 0:
            print(f"   NaN values in Suggested_Discount: {nan_discounts}")
            discount_issues += nan_discounts
        
        # Check for non-numeric values
        try:
            df['Suggested_Discount'] = pd.to_numeric(df['Suggested_Discount'], errors='coerce')
            new_nan = df['Suggested_Discount'].isna().sum() - nan_discounts
            if new_nan > 0:
                print(f"   Non-numeric values converted to NaN: {new_nan}")
                discount_issues += new_nan
        except:
            print("   Error processing Suggested_Discount column")
            discount_issues += 1
    
    if discount_issues > 0:
        issues_found.append(f"Suggested_Discount: {discount_issues} issues")
    else:
        print("   Suggested_Discount column looks good!")
    
    # Check unit_price and current_stock
    print("\n4. Checking unit_price and current_stock...")
    
    price_issues = 0
    stock_issues = 0
    
    if 'unit_price' in df.columns:
        negative_prices = (df['unit_price'] <= 0).sum()
        nan_prices = df['unit_price'].isna().sum()
        price_issues = negative_prices + nan_prices
        if price_issues > 0:
            print(f"   unit_price issues: {negative_prices} negative/zero, {nan_prices} NaN")
    
    if 'current_stock' in df.columns:
        negative_stock = (df['current_stock'] < 0).sum()
        nan_stock = df['current_stock'].isna().sum()
        stock_issues = negative_stock + nan_stock
        if stock_issues > 0:
            print(f"   current_stock issues: {negative_stock} negative, {nan_stock} NaN")
    
    if price_issues == 0 and stock_issues == 0:
        print("   unit_price and current_stock columns look good!")
    else:
        if price_issues > 0:
            issues_found.append(f"unit_price: {price_issues} issues")
        if stock_issues > 0:
            issues_found.append(f"current_stock: {stock_issues} issues")
    
    print(f"\n" + "-" * 60)
    print(f"VALIDATION SUMMARY:")
    if issues_found:
        print(f"Issues found: {len(issues_found)}")
        for issue in issues_found:
            print(f"  • {issue}")
    else:
        print("✅ No issues found! Dataset is valid.")
    
    return df, issues_found

def fix_problems(df, issues_found):
    """Fix any problems found in the dataset"""
    if not issues_found:
        print("\n" + "=" * 60)
        print("STEP 3: NO FIXES NEEDED - DATASET IS CLEAN!")
        print("=" * 60)
        return df
    
    print("\n" + "=" * 60)
    print("STEP 3: FIXING PROBLEMS")
    print("=" * 60)
    
    df_fixed = df.copy()
    fixes_applied = []
    
    # Fix Action column if needed
    if any("Action logic" in issue for issue in issues_found):
        print("\n1. Fixing Action column...")
        
        # Apply correct action logic
        def get_correct_action(row):
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
        
        # Count changes
        old_actions = df_fixed['Action'].copy()
        df_fixed['Action'] = df_fixed.apply(get_correct_action, axis=1)
        changes = (old_actions != df_fixed['Action']).sum()
        
        print(f"   ✅ Fixed {changes} Action assignments")
        fixes_applied.append(f"Action: {changes} corrections")
    
    # Fix donation_eligible if needed
    if any("Donation eligibility" in issue for issue in issues_found):
        print("\n2. Fixing donation_eligible column...")
        
        # Apply correct donation eligibility logic
        edible_categories = ["DAIRY", "MEATS", "PRODUCE", "BAKERY", "FROZEN FOODS", "SNACKS", "BREAD/BAKERY", "FROZEN"]
        
        old_eligible = df_fixed['donation_eligible'].copy()
        df_fixed['donation_eligible'] = (
            (df_fixed['days_to_expiry'] >= -5) &
            (df_fixed['days_to_expiry'] <= -1) &
            (df_fixed['perishable'] == 1) &
            (df_fixed['category'].isin(edible_categories))
        )
        
        changes = (old_eligible != df_fixed['donation_eligible']).sum()
        print(f"   ✅ Fixed {changes} donation_eligible assignments")
        fixes_applied.append(f"Donation eligibility: {changes} corrections")
    
    # Fix Suggested_Discount if needed
    if any("Suggested_Discount" in issue for issue in issues_found):
        print("\n3. Fixing Suggested_Discount column...")
        
        # Fill NaN values with 0
        nan_count = df_fixed['Suggested_Discount'].isna().sum()
        df_fixed['Suggested_Discount'] = df_fixed['Suggested_Discount'].fillna(0)
        
        print(f"   ✅ Fixed {nan_count} NaN values in Suggested_Discount")
        fixes_applied.append(f"Suggested_Discount: {nan_count} NaN fixes")
    
    # Fix unit_price if needed
    if any("unit_price" in issue for issue in issues_found):
        print("\n4. Fixing unit_price column...")
        
        # Fix negative/zero prices (set to median price)
        median_price = df_fixed[df_fixed['unit_price'] > 0]['unit_price'].median()
        bad_prices = (df_fixed['unit_price'] <= 0) | df_fixed['unit_price'].isna()
        fix_count = bad_prices.sum()
        
        df_fixed.loc[bad_prices, 'unit_price'] = median_price
        
        print(f"   ✅ Fixed {fix_count} unit_price issues (set to median: ${median_price:.2f})")
        fixes_applied.append(f"unit_price: {fix_count} fixes")
    
    # Fix current_stock if needed
    if any("current_stock" in issue for issue in issues_found):
        print("\n5. Fixing current_stock column...")
        
        # Fix negative stock (set to 0) and NaN (set to median)
        median_stock = df_fixed[df_fixed['current_stock'] >= 0]['current_stock'].median()
        
        negative_stock = df_fixed['current_stock'] < 0
        nan_stock = df_fixed['current_stock'].isna()
        
        df_fixed.loc[negative_stock, 'current_stock'] = 0
        df_fixed.loc[nan_stock, 'current_stock'] = median_stock
        
        fix_count = (negative_stock | nan_stock).sum()
        print(f"   ✅ Fixed {fix_count} current_stock issues")
        fixes_applied.append(f"current_stock: {fix_count} fixes")
    
    # After fixing, recalculate Action if donation_eligible was changed
    if any("Donation eligibility" in issue for issue in issues_found):
        print("\n6. Recalculating Action after donation_eligible fixes...")
        
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
        
        old_actions = df_fixed['Action'].copy()
        df_fixed['Action'] = df_fixed.apply(get_correct_action, axis=1)
        additional_changes = (old_actions != df_fixed['Action']).sum()
        
        if additional_changes > 0:
            print(f"   ✅ Additional {additional_changes} Action corrections after donation_eligible fixes")
            fixes_applied.append(f"Action (post-donation): {additional_changes} corrections")
    
    # Clean up temporary columns
    if 'Expected_Action' in df_fixed.columns:
        df_fixed = df_fixed.drop('Expected_Action', axis=1)
    if 'Expected_Donation_Eligible' in df_fixed.columns:
        df_fixed = df_fixed.drop('Expected_Donation_Eligible', axis=1)
    
    print(f"\n" + "-" * 60)
    print(f"FIXES SUMMARY:")
    for fix in fixes_applied:
        print(f"  ✅ {fix}")
    
    return df_fixed

def save_corrected_dataset(df_fixed):
    """Save the corrected dataset"""
    output_file = 'data/processed/inventory_analysis_results_enhanced.csv'
    df_fixed.to_csv(output_file, index=False)
    
    print(f"\n✅ Corrected dataset saved to: {output_file}")
    
    # Final validation
    print(f"\nFinal dataset summary:")
    print(f"  Shape: {df_fixed.shape[0]} rows × {df_fixed.shape[1]} columns")
    print(f"  Action distribution:")
    action_counts = df_fixed['Action'].value_counts()
    for action, count in action_counts.items():
        print(f"    {action}: {count}")
    
    if 'donation_eligible' in df_fixed.columns:
        eligible_count = df_fixed['donation_eligible'].sum()
        print(f"  Donation eligible items: {eligible_count}")

def main():
    """Main verification and correction workflow"""
    try:
        # Step 1: Load and inspect
        df = load_and_inspect_dataset()
        
        # Step 2: Validate
        df, issues_found = validate_critical_columns(df)
        
        # Step 3: Fix problems
        df_fixed = fix_problems(df, issues_found)
        
        # Save corrected dataset
        save_corrected_dataset(df_fixed)
        
        print("\n" + "=" * 60)
        print("✅ DATASET VERIFICATION AND CORRECTION COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
