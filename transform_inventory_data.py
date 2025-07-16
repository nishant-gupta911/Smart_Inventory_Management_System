"""
Inventory Data Transformation Script with Donation Logic

This script transforms inventory data to include expired food donation functionality.
Key features implemented:

1. Donation Eligibility Logic:
   - Items are eligible for donation if:
     * days_to_expiry is between -5 and -1 (inclusive)
     * perishable == 1 (is a perishable item)
     * category is in edible categories: ['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY', 'FROZEN FOODS', 'SNACKS', 'BEVERAGES']

2. New Columns Added:
   - donation_eligible: Boolean indicating if item can be donated
   - donation_status: 'Pending' for eligible items, empty for others
   - store_latitude, store_longitude: Store location coordinates
   - nearest_ngo, ngo_address, ngo_contact: NGO information for donations

3. Data Quality:
   - Ensures no NaN values in critical columns
   - Maintains data schema integrity
   - Provides both transformation of existing data and generation of new test data

4. Usage:
   - Use apply_donation_logic_to_dataframe(df) for transforming any DataFrame
   - Run main() to process the full dataset and generate enhanced output
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Real food product names by category
FOOD_PRODUCTS = {
    'DAIRY': ['Milk', 'Greek Yogurt', 'Cheddar Cheese', 'Butter', 'Paneer', 'Cream Cheese', 'Cottage Cheese', 'Mozzarella'],
    'MEATS': ['Chicken Breast', 'Ground Beef', 'Lamb Chops', 'Pork Ribs', 'Turkey Slices', 'Mutton', 'Sausages', 'Bacon'],
    'PRODUCE': ['Tomatoes', 'Bananas', 'Apples', 'Onions', 'Potatoes', 'Carrots', 'Spinach', 'Mangoes', 'Oranges', 'Lettuce'],
    'BREAD/BAKERY': ['White Bread', 'Whole Wheat Bread', 'Croissants', 'Bagels', 'Dinner Rolls', 'Pita Bread', 'Naan', 'Baguette'],
    'SEAFOOD': ['Salmon Fillet', 'Prawns', 'Tuna Steaks', 'Cod Fish', 'Crab Meat', 'Lobster', 'Mackerel', 'Sardines'],
    'FROZEN': ['Frozen Peas', 'Ice Cream', 'Frozen Pizza', 'Frozen Berries', 'Frozen Corn', 'Frozen Fish Sticks']
}

# Indian cities with coordinates
INDIAN_CITIES = {
    'Mumbai': {'state': 'Maharashtra', 'lat': 19.0760, 'lon': 72.8777},
    'Delhi': {'state': 'Delhi', 'lat': 28.7041, 'lon': 77.1025},
    'Bangalore': {'state': 'Karnataka', 'lat': 12.9716, 'lon': 77.5946},
    'Chennai': {'state': 'Tamil Nadu', 'lat': 13.0827, 'lon': 80.2707},
    'Pune': {'state': 'Maharashtra', 'lat': 18.5204, 'lon': 73.8567},
    'Hyderabad': {'state': 'Telangana', 'lat': 17.3850, 'lon': 78.4867},
    'Kolkata': {'state': 'West Bengal', 'lat': 22.5726, 'lon': 88.3639},
    'Ahmedabad': {'state': 'Gujarat', 'lat': 23.0225, 'lon': 72.5714},
    'Jaipur': {'state': 'Rajasthan', 'lat': 26.9124, 'lon': 75.7873},
    'Kochi': {'state': 'Kerala', 'lat': 9.9312, 'lon': 76.2673},
    'Lucknow': {'state': 'Uttar Pradesh', 'lat': 26.8467, 'lon': 80.9462},
    'Bhopal': {'state': 'Madhya Pradesh', 'lat': 23.2599, 'lon': 77.4126},
    'Chandigarh': {'state': 'Punjab', 'lat': 30.7333, 'lon': 76.7794},
    'Goa': {'state': 'Goa', 'lat': 15.2993, 'lon': 74.1240},
    'Surat': {'state': 'Gujarat', 'lat': 21.1702, 'lon': 72.8311}
}

# Indian NGOs with contact info
INDIAN_NGOS = [
    {'name': 'Animal Aid Unlimited', 'address': 'Badi Village, Udaipur, Rajasthan 313001', 'contact': '+91-9414049887'},
    {'name': 'People For Animals', 'address': 'B-4/30, Safdarjung Enclave, New Delhi 110029', 'contact': '+91-11-26161816'},
    {'name': 'Blue Cross of India', 'address': '1 Nelson Manickam Road, Chennai 600029', 'contact': '+91-44-24611026'},
    {'name': 'World For All Animal Care', 'address': 'Juhu-Versova Link Road, Mumbai 400049', 'contact': '+91-22-26215211'},
    {'name': 'CARE India', 'address': '14 Paschimi Marg, Vasant Vihar, New Delhi 110057', 'contact': '+91-11-40801000'},
    {'name': 'Wildlife SOS', 'address': 'H-6, Sector 6, Noida, Uttar Pradesh 201301', 'contact': '+91-9871963535'},
    {'name': 'Friendicoes SECA', 'address': 'Jangpura Extension, New Delhi 110014', 'contact': '+91-11-24376307'},
    {'name': 'Karuna Society for Animals', 'address': 'Ananda Ashram, Puttaparthi, Andhra Pradesh 515134', 'contact': '+91-8555-287771'},
    {'name': 'Bharatiya Animal Welfare Board', 'address': 'Shanti Kunj, Haridwar, Uttarakhand 249411', 'contact': '+91-1334-246001'},
    {'name': 'Animal Welfare Board of India', 'address': 'Pocket A, Sector 8, Dwarka, New Delhi 110077', 'contact': '+91-11-25056789'}
]

# Perishable categories
PERISHABLE_CATEGORIES = ['DAIRY', 'MEATS', 'PRODUCE', 'BREAD/BAKERY', 'SEAFOOD', 'FROZEN']

# Edible categories for donation eligibility
EDIBLE_CATEGORIES = ['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY', 'FROZEN FOODS', 'SNACKS', 'BEVERAGES']

def load_and_transform_data():
    """Load and transform the inventory data with donation logic"""
    # Load existing data
    df = pd.read_csv('data/processed/inventory_analysis_results.csv')
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Apply the complete transformation with donation logic
    df = transform_inventory_with_donation_logic(df)
    
    # Generate additional rows
    additional_rows = generate_additional_rows(500)
    
    # Combine original and new data
    final_df = pd.concat([df, additional_rows], ignore_index=True)
    
    print(f"Final dataset size: {len(final_df)} rows")
    
    return final_df

def transform_product_names(df):
    """Replace generic product names with real food names"""
    for idx, row in df.iterrows():
        category = row['category']
        if category in FOOD_PRODUCTS:
            df.at[idx, 'product_name'] = random.choice(FOOD_PRODUCTS[category])
        else:
            # For non-food categories, create generic names
            df.at[idx, 'product_name'] = f"General Product {idx}"

def add_donation_columns(df):
    """
    Add donation-related columns based on specified business rules.
    
    Donation eligibility criteria:
    - days_to_expiry between -5 and -1 (inclusive)
    - perishable == 1
    - category in edible items list
    """
    # Initialize donation columns if they don't exist
    if 'donation_eligible' not in df.columns:
        df['donation_eligible'] = False
    if 'donation_status' not in df.columns:
        df['donation_status'] = ''
    
    for idx, row in df.iterrows():
        days_to_expiry = row['days_to_expiry']
        category = row['category']
        perishable = row.get('perishable', 0)  # Default to 0 if column doesn't exist
        
        # Check if eligible for donation based on business rules
        is_expired_in_range = -5 <= days_to_expiry <= -1
        is_perishable = perishable == 1
        is_edible_category = category in EDIBLE_CATEGORIES
        
        if is_expired_in_range and is_perishable and is_edible_category:
            df.at[idx, 'donation_eligible'] = True
            # Set donation_status to "Pending" for newly eligible items
            if pd.isna(df.at[idx, 'donation_status']) or df.at[idx, 'donation_status'] == '':
                df.at[idx, 'donation_status'] = 'Pending'
        else:
            df.at[idx, 'donation_eligible'] = False
            # Set donation_status to empty string for non-eligible items
            df.at[idx, 'donation_status'] = ''

def add_location_columns(df):
    """Add realistic Indian coordinates based on city"""
    # Initialize location columns if they don't exist
    if 'store_latitude' not in df.columns:
        df['store_latitude'] = 0.0
    if 'store_longitude' not in df.columns:
        df['store_longitude'] = 0.0
    
    for idx, row in df.iterrows():
        # Use Mumbai as default if city not in our list
        city_name = row.get('city', 'Mumbai')
        city_data = INDIAN_CITIES.get(city_name, INDIAN_CITIES['Mumbai'])
        
        # Add some random variation to coordinates
        lat_variation = random.uniform(-0.1, 0.1)
        lon_variation = random.uniform(-0.1, 0.1)
        
        df.at[idx, 'store_latitude'] = round(city_data['lat'] + lat_variation, 6)
        df.at[idx, 'store_longitude'] = round(city_data['lon'] + lon_variation, 6)
        
        # Update city and state to Indian locations if not already set
        if pd.isna(df.at[idx, 'city']) or df.at[idx, 'city'] == '':
            city_name = random.choice(list(INDIAN_CITIES.keys()))
            df.at[idx, 'city'] = city_name
            df.at[idx, 'state'] = INDIAN_CITIES[city_name]['state']

def add_ngo_columns(df):
    """Add NGO-related columns and ensure no NaN values"""
    # Initialize NGO columns if they don't exist
    if 'nearest_ngo' not in df.columns:
        df['nearest_ngo'] = ''
    if 'ngo_address' not in df.columns:
        df['ngo_address'] = ''
    if 'ngo_contact' not in df.columns:
        df['ngo_contact'] = ''
    
    for idx in df.index:
        # Only assign NGO if not already populated or if empty/NaN
        if pd.isna(df.at[idx, 'nearest_ngo']) or df.at[idx, 'nearest_ngo'] == '':
            ngo = random.choice(INDIAN_NGOS)
            df.at[idx, 'nearest_ngo'] = ngo['name']
            df.at[idx, 'ngo_address'] = ngo['address']
            df.at[idx, 'ngo_contact'] = ngo['contact']
    
    # Ensure no NaN values remain in NGO columns
    df['nearest_ngo'] = df['nearest_ngo'].fillna('')
    df['ngo_address'] = df['ngo_address'].fillna('')
    df['ngo_contact'] = df['ngo_contact'].fillna('')

def add_action_column(df):
    """
    Add or update the Action column based on business rules:
    
    1. If days_to_expiry < -5: Action = "Remove" (too expired)
    2. If -5 <= days_to_expiry <= -1 AND donation_eligible is True: Action = "Donate"
    3. If 0 <= days_to_expiry <= 5: Action = "Apply Discount"
    4. Else: Action = "Restock"
    """
    # Always reset the Action column to ensure clean recalculation
    df['Action'] = ''
    
    for idx, row in df.iterrows():
        days_to_expiry = row['days_to_expiry']
        donation_eligible = row.get('donation_eligible', False)
        
        if days_to_expiry < -5:
            # Too expired to donate or sell
            df.at[idx, 'Action'] = 'Remove'
        elif -5 <= days_to_expiry <= -1 and donation_eligible:
            # Recently expired but donation eligible
            df.at[idx, 'Action'] = 'Donate'
        elif 0 <= days_to_expiry <= 5:
            # Near expiry, apply discount
            df.at[idx, 'Action'] = 'Apply Discount'
        else:
            # All other cases: restock
            df.at[idx, 'Action'] = 'Restock'

def transform_inventory_with_donation_logic(df):
    """
    Main function to transform inventory data with donation logic.
    
    This function:
    1. Ensures all required columns exist
    2. Computes donation eligibility based on business rules
    3. Populates donation status appropriately
    4. Ensures no NaN values in key columns
    
    Args:
        df (pandas.DataFrame): Input inventory dataframe
        
    Returns:
        pandas.DataFrame: Transformed dataframe with donation logic
    """
    print("Applying donation logic transformation...")
    
    # Ensure required columns exist with default values
    required_columns = {
        'days_to_expiry': 0,
        'perishable': 0,
        'category': 'UNKNOWN',
        'donation_eligible': False,
        'donation_status': '',
        'store_latitude': 0.0,
        'store_longitude': 0.0,
        'nearest_ngo': '',
        'ngo_address': '',
        'ngo_contact': ''
    }
    
    for col, default_val in required_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Apply transformations
    transform_product_names(df)
    add_donation_columns(df)
    add_location_columns(df)
    add_ngo_columns(df)
    add_action_column(df)
    
    # Final cleanup to ensure no NaN values
    df['donation_eligible'] = df['donation_eligible'].fillna(False)
    df['donation_status'] = df['donation_status'].fillna('')
    
    print(f"Transformation complete. Donation eligible items: {df['donation_eligible'].sum()}")
    
    return df

def generate_additional_rows(num_rows):
    """Generate additional rows with realistic data"""
    additional_data = []
    
    for i in range(num_rows):
        # Select random city
        city_name = random.choice(list(INDIAN_CITIES.keys()))
        city_data = INDIAN_CITIES[city_name]
        
        # Select random category and product
        category = random.choice(list(FOOD_PRODUCTS.keys()))
        product_name = random.choice(FOOD_PRODUCTS[category])
        
        # Generate realistic expiry scenarios
        expiry_scenario = random.choice(['fresh', 'near_expiry', 'recently_expired', 'long_expired'])
        
        if expiry_scenario == 'fresh':
            days_to_expiry = random.randint(5, 30)
        elif expiry_scenario == 'near_expiry':
            days_to_expiry = random.randint(0, 4)
        elif expiry_scenario == 'recently_expired':
            days_to_expiry = random.randint(-5, -1)
        else:  # long_expired
            days_to_expiry = random.randint(-30, -8)
        
        # Determine donation eligibility using proper business logic
        perishable = 1 if category in PERISHABLE_CATEGORIES else 0
        is_expired_in_range = -5 <= days_to_expiry <= -1
        is_perishable = perishable == 1
        is_edible_category = category in EDIBLE_CATEGORIES
        
        donation_eligible = is_expired_in_range and is_perishable and is_edible_category
        donation_status = 'Pending' if donation_eligible else ''
        
        # Generate stock level and reorder for data completeness
        stock_level = random.choice(['Low', 'Medium', 'High'])
        reorder = 'Yes' if random.random() > 0.6 else 'No'
        
        # Determine Action based on business rules (no random choices)
        if days_to_expiry < -5:
            action = 'Remove'
        elif -5 <= days_to_expiry <= -1 and donation_eligible:
            action = 'Donate'
        elif 0 <= days_to_expiry <= 5:
            action = 'Apply Discount'
        else:
            action = 'Restock'
        
        # Select NGO
        ngo = random.choice(INDIAN_NGOS)
        
        # Generate coordinates with variation
        lat_variation = random.uniform(-0.1, 0.1)
        lon_variation = random.uniform(-0.1, 0.1)
        
        # Create row data
        row_data = {
            'id': 2000 + i,
            'date': '2013-01-01',
            'store_nbr': random.randint(1, 55),
            'family_x': random.choice(['GROCERY', 'DAIRY', 'MEATS', 'PRODUCE']),
            'sales': round(random.uniform(0, 1000), 2),
            'onpromotion': random.choice([0, 1]),
            'item_id': 2000.0 + i,
            'family_y': random.choice(['GROCERY', 'DAIRY', 'MEATS', 'PRODUCE']),
            'perishable': perishable,
            'shelf_life': random.choice([2, 3, 4, 5, 7, 14, 30, 90, 365]),
            'city': city_name,
            'state': city_data['state'],
            'type': random.choice(['A', 'B', 'C', 'D']),
            'cluster': random.randint(1, 17),
            'type_y': random.choice(['Holiday', 'Work Day', '']),
            'day_of_week': random.randint(1, 7),
            'month': 1,
            'year': 2013,
            'is_weekend': random.choice([0, 1]),
            'rolling_avg_sales_7': round(random.uniform(0, 500), 2),
            'days_on_shelf': random.randint(0, 30),
            'days_to_expiry': days_to_expiry,
            'product_name': product_name,
            'current_stock': random.randint(5, 200),
            'unit_price': round(random.uniform(0.5, 50), 2),
            'category': category,
            'Stock_Level': stock_level,
            'Expiry_Risk': 'Expired' if days_to_expiry < 0 else ('Near Expiry' if days_to_expiry <= 5 else 'Safe'),
            'Suggested_Discount': random.choice([0, 15, 25, 40, 50]) if days_to_expiry <= 5 else 0,
            'Reorder': reorder,
            'Action': action,
            'donation_eligible': donation_eligible,
            'donation_status': donation_status,
            'store_latitude': round(city_data['lat'] + lat_variation, 6),
            'store_longitude': round(city_data['lon'] + lon_variation, 6),
            'nearest_ngo': ngo['name'],
            'ngo_address': ngo['address'],
            'ngo_contact': ngo['contact']
        }
        
        additional_data.append(row_data)
    
    return pd.DataFrame(additional_data)

def main():
    """Main function to execute the transformation"""
    print("Starting inventory data transformation...")
    
    # Transform the data
    final_df = load_and_transform_data()
    
    # Save the transformed data
    output_file = 'data/processed/inventory_analysis_results_enhanced.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"Enhanced dataset saved to: {output_file}")
    print(f"Total rows: {len(final_df)}")
    print(f"Donation eligible items: {final_df['donation_eligible'].sum()}")
    print(f"Items with 'Pending' donation status: {(final_df['donation_status'] == 'Pending').sum()}")
    print(f"Unique cities: {final_df['city'].nunique()}")
    print(f"Unique products: {final_df['product_name'].nunique()}")
    print(f"Perishable items: {(final_df['perishable'] == 1).sum()}")
    
    # Donation eligibility breakdown
    donation_breakdown = final_df.groupby(['category', 'donation_eligible']).size().unstack(fill_value=0)
    if True in donation_breakdown.columns:
        print("\nDonation eligible items by category:")
        for category in donation_breakdown.index:
            eligible_count = donation_breakdown.loc[category, True] if True in donation_breakdown.columns else 0
            print(f"  {category}: {eligible_count}")
    
    # Display sample of the new data
    print("\nSample of enhanced data:")
    sample_cols = ['product_name', 'category', 'perishable', 'days_to_expiry', 
                  'donation_eligible', 'donation_status', 'city', 'nearest_ngo']
    print(final_df[sample_cols].head(10).to_string())

def apply_donation_logic_to_dataframe(df):
    """
    Standalone function to apply donation logic to any inventory DataFrame.
    
    This function can be used independently to transform an existing DataFrame
    with the donation logic without loading from CSV or generating additional rows.
    
    Args:
        df (pandas.DataFrame): Input inventory dataframe
        
    Returns:
        pandas.DataFrame: Transformed dataframe with donation logic
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Apply the donation logic transformation
    df_transformed = transform_inventory_with_donation_logic(df_copy)
    
    return df_transformed

if __name__ == "__main__":
    main()
