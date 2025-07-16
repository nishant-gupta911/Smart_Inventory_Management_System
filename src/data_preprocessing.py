import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def load_data_with_fallback():
    """Load data with fallback to sample data generation - NO LIMITS"""
    
    # Try to load existing cleaned data first
    possible_paths = [
        os.path.join(project_root, "data", "cleaned_inventory_data.csv"),
        os.path.join(project_root, "cleaned_inventory_data.csv"),
        os.path.join(project_root, "data", "raw", "cleaned_inventory_data.csv")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logger.info(f"✅ Found existing cleaned data: {path} with {len(df):,} rows")
                return df, "existing_data"
            except Exception as e:
                logger.warning(f"Failed to load {path}: {str(e)}")
                continue
    
    # Try to load ALL raw data files - NO ROW LIMITS
    raw_data_paths = {
        'train': [
            os.path.join(project_root, "data", "raw", "train.csv"),
            os.path.join(project_root, "train.csv"),
            "train.csv"
        ],
        'items': [
            os.path.join(project_root, "data", "raw", "items.csv"),
            os.path.join(project_root, "items.csv"),
            "items.csv"
        ],
        'stores': [
            os.path.join(project_root, "data", "raw", "stores.csv"),
            os.path.join(project_root, "stores.csv"),
            "stores.csv"
        ],
        'holidays': [
            os.path.join(project_root, "data", "raw", "holidays_events.csv"),
            os.path.join(project_root, "holidays_events.csv"),
            "holidays_events.csv"
        ],
        'oil': [
            os.path.join(project_root, "data", "raw", "oil.csv"),
            os.path.join(project_root, "oil.csv"),
            "oil.csv"
        ],
        'transactions': [
            os.path.join(project_root, "data", "raw", "transactions.csv"),
            os.path.join(project_root, "transactions.csv"),
            "transactions.csv"
        ]
    }
    
    raw_data = {}
    for data_type, paths in raw_data_paths.items():
        found = False
        for path in paths:
            if os.path.exists(path):
                try:
                    logger.info(f"🔄 Loading {data_type} from: {path}")
                    
                    # Load ALL data - no row limits
                    if data_type == 'train':
                        # Check for date column first
                        sample = pd.read_csv(path, nrows=1)
                        has_date = any('date' in col.lower() for col in sample.columns)
                        if has_date:
                            raw_data[data_type] = pd.read_csv(path, parse_dates=['date'])
                        else:
                            raw_data[data_type] = pd.read_csv(path)
                    elif data_type in ['holidays', 'oil', 'transactions']:
                        # Check for date columns
                        sample = pd.read_csv(path, nrows=1)
                        date_cols = [col for col in sample.columns if 'date' in col.lower()]
                        if date_cols:
                            raw_data[data_type] = pd.read_csv(path, parse_dates=date_cols)
                        else:
                            raw_data[data_type] = pd.read_csv(path)
                    else:
                        raw_data[data_type] = pd.read_csv(path)
                    
                    logger.info(f"✅ Loaded {data_type}: {raw_data[data_type].shape[0]:,} rows, {raw_data[data_type].shape[1]} columns")
                    found = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {str(e)}")
                    continue
        
        if not found:
            logger.warning(f"❌ Could not find {data_type} data file")
    
    if len(raw_data) >= 2:  # At least have sales and items data
        return raw_data, "raw_data"
    else:
        logger.warning("⚠️ No suitable data files found. Generating comprehensive sample data.")
        return generate_comprehensive_sample_data(), "sample_data"

def generate_comprehensive_sample_data():
    """Generate comprehensive large sample data for the inventory system"""
    logger.info("🔧 Generating comprehensive sample inventory data...")
    
    np.random.seed(42)
    
    # Generate extended date range (2 years of data for manageability)
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2023-12-31')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define comprehensive product families and their typical shelf lives
    families = {
        'DAIRY': 7,
        'PRODUCE': 5,
        'MEATS': 4,
        'BREAD/BAKERY': 3,
        'SEAFOOD': 2,
        'FROZEN FOODS': 90,
        'BEVERAGES': 30,
        'DELI': 4,
        'POULTRY': 3,
        'EGGS': 14,
        'GROCERY I': 365,
        'GROCERY II': 365,
        'PREPARED FOODS': 2,
        'CLEANING': 365,
        'PERSONAL CARE': 365,
        'HOME CARE': 365,
        'PET SUPPLIES': 365,
        'AUTOMOTIVE': 365,
        'HARDWARE': 365,
        'BOOKS': 365,
        'MAGAZINES': 30,
        'SCHOOL AND OFFICE SUPPLIES': 365,
        'CELEBRATION': 365,
        'HOME AND KITCHEN I': 365,
        'HOME AND KITCHEN II': 365,
        'HOME APPLIANCES': 365,
        'LADIESWEAR': 365,
        'LINGERIE': 365,
        'LAWN AND GARDEN': 365,
        'LIQUOR,WINE,BEER': 365,
        'PLAYERS AND ELECTRONICS': 365
    }
    
    # Generate items (reasonable number per family)
    items_data = []
    item_id = 1
    for family, shelf_life in families.items():
        # Generate 50-200 items per family
        num_items = 200 if family in ['GROCERY I', 'GROCERY II'] else 100 if family in ['DAIRY', 'BEVERAGES', 'PRODUCE'] else 50
        
        for i in range(num_items):
            items_data.append({
                'item_nbr': item_id,
                'family': family,
                'perishable': 1 if shelf_life <= 30 else 0,
                'shelf_life': shelf_life + np.random.randint(-2, 3),
                'class': np.random.randint(1, 5),
                'unit_price': round(np.random.uniform(0.5, 100.0), 2),
                'brand': f'Brand_{np.random.randint(1, 21)}',
                'description': f'{family}_Product_{i:04d}'
            })
            item_id += 1
    
    items_df = pd.DataFrame(items_data)
    logger.info(f"Generated {len(items_df):,} items across {len(families)} families")
    
    # Generate stores (50 stores)
    stores_data = []
    cities = ['Quito', 'Guayaquil', 'Cuenca', 'Santo Domingo', 'Machala', 'Manta', 'Portoviejo', 'Ambato', 'Riobamba', 'Loja']
    states = ['Pichincha', 'Guayas', 'Azuay', 'Santo Domingo de los Tsachilas', 'El Oro', 'Manabi', 'Tungurahua', 'Chimborazo', 'Loja']
    
    for store_id in range(1, 51):  # 50 stores
        stores_data.append({
            'store_nbr': store_id,
            'city': np.random.choice(cities),
            'state': np.random.choice(states),
            'type': np.random.choice(['A', 'B', 'C', 'D', 'E'], p=[0.1, 0.2, 0.3, 0.3, 0.1]),
            'cluster': np.random.randint(1, 18),
            'size_category': np.random.choice(['Small', 'Medium', 'Large'], p=[0.3, 0.5, 0.2])
        })
    
    stores_df = pd.DataFrame(stores_data)
    logger.info(f"Generated {len(stores_df):,} stores")
    
    # Generate holidays and events
    holidays_data = []
    holiday_types = ['Holiday', 'Event', 'Transfer', 'Additional', 'Bridge']
    
    for year in [2022, 2023]:
        # Major holidays
        major_holidays = [
            f'{year}-01-01',  # New Year
            f'{year}-12-25',  # Christmas
            f'{year}-07-04',  # Independence
            f'{year}-11-24',  # Thanksgiving
        ]
        
        for holiday_date in major_holidays:
            try:
                holidays_data.append({
                    'date': pd.to_datetime(holiday_date),
                    'type': 'Holiday',
                    'locale': 'National',
                    'transferred': False
                })
            except:
                continue
        
        # Regional events (monthly)
        for month in range(1, 13):
            for event_day in np.random.randint(1, 29, 2):  # 2 events per month
                try:
                    holidays_data.append({
                        'date': pd.to_datetime(f'{year}-{month:02d}-{event_day:02d}'),
                        'type': np.random.choice(holiday_types),
                        'locale': np.random.choice(['Local', 'Regional', 'National']),
                        'transferred': np.random.choice([True, False], p=[0.1, 0.9])
                    })
                except:
                    continue
    
    holidays_df = pd.DataFrame(holidays_data)
    logger.info(f"Generated {len(holidays_df):,} holidays and events")
    
    # Generate oil prices
    oil_data = []
    base_oil_price = 60.0
    for date in date_range:
        daily_change = np.random.normal(0, 1)
        base_oil_price = max(20, min(150, base_oil_price + daily_change))
        
        oil_data.append({
            'date': date,
            'dcoilwtico': round(base_oil_price, 2)
        })
    
    oil_df = pd.DataFrame(oil_data)
    logger.info(f"Generated {len(oil_df):,} oil price records")
    
    # Generate transactions data
    transactions_data = []
    for date in date_range:
        for store_id in range(1, 51):  # All stores
            daily_transactions = max(100, np.random.poisson(500))  # Minimum 100 transactions
            transactions_data.append({
                'date': date,
                'store_nbr': store_id,
                'transactions': daily_transactions
            })
    
    transactions_df = pd.DataFrame(transactions_data)
    logger.info(f"Generated {len(transactions_df):,} transaction records")
    
    # Generate sales data (manageable size)
    logger.info("🔄 Generating sales data...")
    sales_data = []
    
    # Generate sales for a subset of combinations
    num_sales_records = 500_000  # 500K records for manageable processing
    logger.info(f"Generating {num_sales_records:,} sales records...")
    
    for i in range(num_sales_records):
        date = np.random.choice(date_range)
        store_id = np.random.randint(1, 51)
        item_idx = int(np.random.randint(0, len(items_data)))
        item_info = items_data[item_idx]
        
        # Generate realistic sales
        base_sales = max(0, np.random.exponential(2))
        
        # Add patterns
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
        weekend_factor = 1.3 if date.weekday() >= 5 else 1.0
        holiday_factor = 1.8 if np.random.random() < 0.03 else 1.0
        perishable_factor = 1.2 if item_info['perishable'] else 1.0
        
        unit_sales = base_sales * seasonal_factor * weekend_factor * holiday_factor * perishable_factor
        
        sales_data.append({
            'id': i + 1,
            'date': date,
            'store_nbr': store_id,
            'item_nbr': item_info['item_nbr'],
            'unit_sales': round(unit_sales, 3),
            'onpromotion': np.random.choice([0, 1], p=[0.8, 0.2])
        })
        
        if (i + 1) % 100_000 == 0:
            logger.info(f"Generated {i + 1:,} sales records...")
    
    sales_df = pd.DataFrame(sales_data)
    logger.info(f"✅ Generated {len(sales_df):,} total sales records")
    
    return {
        'train': sales_df,
        'items': items_df,
        'stores': stores_df,
        'holidays': holidays_df,
        'oil': oil_df,
        'transactions': transactions_df
    }

def find_column(df, possible_names):
    """Find the correct column name from a list of possibilities"""
    if df is None or df.empty:
        return None
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def safe_merge(left_df, right_df, left_on, right_on, how='left', suffixes=('', '_y')):
    """Safely merge two dataframes with error handling"""
    try:
        if right_df is None or right_df.empty:
            return left_df
        if left_on not in left_df.columns or right_on not in right_df.columns:
            return left_df
        return left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes)
    except Exception as e:
        logger.warning(f"Merge failed: {str(e)}")
        return left_df

def process_raw_data(raw_data):
    """Process raw data files into cleaned format"""
    logger.info("📊 Processing and merging raw data...")
    
    if 'train' not in raw_data or 'items' not in raw_data:
        raise ValueError("Missing required 'train' or 'items' data")
    
    sales = raw_data['train'].copy()
    items = raw_data['items'].copy()
    stores = raw_data.get('stores')
    holidays = raw_data.get('holidays')
    oil = raw_data.get('oil')
    transactions = raw_data.get('transactions')
    
    # Clean column names
    for df_name, df in raw_data.items():
        if df is not None:
            df.columns = df.columns.str.strip()
            logger.info(f"{df_name.title()} shape: {df.shape}")
    
    # Find correct column names
    sales_item_col = find_column(sales, ['item_nbr', 'item_number', 'item_id', 'id'])
    sales_store_col = find_column(sales, ['store_nbr', 'store_number', 'store_id', 'store'])
    sales_date_col = find_column(sales, ['date', 'Date', 'DATE'])
    sales_units_col = find_column(sales, ['unit_sales', 'units', 'sales', 'quantity'])
    sales_promo_col = find_column(sales, ['onpromotion', 'promotion', 'promo'])
    
    items_item_col = find_column(items, ['item_nbr', 'item_number', 'item_id', 'id'])
    items_family_col = find_column(items, ['family', 'category', 'Family', 'FAMILY'])
    
    # Validate required columns
    if not all([sales_item_col, sales_store_col, sales_date_col, sales_units_col]):
        raise ValueError(f"Missing required sales columns. Available: {sales.columns.tolist()}")
    
    if not items_item_col:
        raise ValueError(f"Missing required item column. Available: {items.columns.tolist()}")
    
    logger.info("✅ Found all required columns")
    
    # Start with sales data
    logger.info(f"🔗 Starting with sales data: {sales.shape[0]:,} records")
    df = sales.copy()
    
    # Ensure date column is datetime
    if sales_date_col in df.columns:
        df[sales_date_col] = pd.to_datetime(df[sales_date_col], errors='coerce')
    
    # Merge with items data
    logger.info("🔗 Merging with items data...")
    df = safe_merge(df, items, sales_item_col, items_item_col, suffixes=('', '_items'))
    logger.info(f"After items merge: {df.shape[0]:,} records")
    
    # Merge with stores data if available
    if stores is not None and not stores.empty:
        logger.info("🔗 Merging with stores data...")
        stores_store_col = find_column(stores, ['store_nbr', 'store_number', 'store_id', 'store'])
        if stores_store_col:
            df = safe_merge(df, stores, sales_store_col, stores_store_col, suffixes=('', '_stores'))
            logger.info(f"After stores merge: {df.shape[0]:,} records")
    
    # Merge with holidays data if available
    if holidays is not None and not holidays.empty:
        logger.info("🔗 Merging with holidays data...")
        holidays_date_col = find_column(holidays, ['date', 'Date', 'DATE'])
        holidays_type_col = find_column(holidays, ['type', 'Type', 'TYPE', 'holiday_type'])
        
        if holidays_date_col and holidays_type_col:
            # Ensure date column is datetime
            holidays[holidays_date_col] = pd.to_datetime(holidays[holidays_date_col], errors='coerce')
            holidays_subset = holidays[[holidays_date_col, holidays_type_col]].drop_duplicates()
            df = safe_merge(df, holidays_subset, sales_date_col, holidays_date_col, suffixes=('', '_holiday'))
            if holidays_type_col in df.columns:
                df[holidays_type_col] = df[holidays_type_col].fillna('None')
            logger.info(f"After holidays merge: {df.shape[0]:,} records")
    
    # Merge with oil data if available
    if oil is not None and not oil.empty:
        logger.info("🔗 Merging with oil price data...")
        oil_date_col = find_column(oil, ['date', 'Date', 'DATE'])
        oil_price_col = find_column(oil, ['dcoilwtico', 'oil_price', 'price'])
        
        if oil_date_col and oil_price_col:
            oil[oil_date_col] = pd.to_datetime(oil[oil_date_col], errors='coerce')
            oil_subset = oil[[oil_date_col, oil_price_col]].drop_duplicates()
            df = safe_merge(df, oil_subset, sales_date_col, oil_date_col, suffixes=('', '_oil'))
            logger.info(f"After oil merge: {df.shape[0]:,} records")
    
    # Merge with transactions data if available
    if transactions is not None and not transactions.empty:
        logger.info("🔗 Merging with transactions data...")
        trans_date_col = find_column(transactions, ['date', 'Date', 'DATE'])
        trans_store_col = find_column(transactions, ['store_nbr', 'store_number', 'store_id', 'store'])
        trans_count_col = find_column(transactions, ['transactions', 'transaction_count', 'trans'])
        
        if trans_date_col and trans_store_col and trans_count_col:
            transactions[trans_date_col] = pd.to_datetime(transactions[trans_date_col], errors='coerce')
            trans_subset = transactions[[trans_date_col, trans_store_col, trans_count_col]].drop_duplicates()
            df = safe_merge(df, trans_subset, 
                           [sales_date_col, sales_store_col], 
                           [trans_date_col, trans_store_col], 
                           suffixes=('', '_trans'))
            logger.info(f"After transactions merge: {df.shape[0]:,} records")
    
    # Add time-based features
    logger.info("📅 Adding time-based features...")
    if sales_date_col in df.columns:
        df['day_of_week'] = df[sales_date_col].dt.dayofweek
        df['month'] = df[sales_date_col].dt.month
        df['year'] = df[sales_date_col].dt.year
        df['quarter'] = df[sales_date_col].dt.quarter
        df['day_of_month'] = df[sales_date_col].dt.day
        df['day_of_year'] = df[sales_date_col].dt.dayofyear
        try:
            df['week_of_year'] = df[sales_date_col].dt.isocalendar().week
        except:
            df['week_of_year'] = df[sales_date_col].dt.week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[sales_date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[sales_date_col].dt.is_month_end.astype(int)
    
    # Add shelf life features
    logger.info("🏪 Adding shelf life features...")
    shelf_life_mapping = {
        'DAIRY': 7, 'BEVERAGES': 30, 'PRODUCE': 5, 'MEATS': 4, 'DELI': 4,
        'SEAFOOD': 2, 'FROZEN FOODS': 90, 'BREAD/BAKERY': 3, 'PREPARED FOODS': 2,
        'POULTRY': 3, 'EGGS': 14, 'GROCERY I': 365, 'GROCERY II': 365,
        'CLEANING': 365, 'PERSONAL CARE': 365, 'HOME CARE': 365
    }
    
    if items_family_col and items_family_col in df.columns:
        family_col = items_family_col
    else:
        df['family'] = 'GROCERY I'
        family_col = 'family'
    
    df[family_col] = df[family_col].fillna('GROCERY I')
    df['shelf_life'] = df[family_col].map(shelf_life_mapping).fillna(365)
    
    # Add basic inventory features
    df['days_on_shelf'] = np.random.randint(0, 30, len(df))
    df['days_to_expiry'] = df['shelf_life'] - df['days_on_shelf']
    df['days_to_expiry'] = df['days_to_expiry'].clip(lower=0)
    df['current_stock'] = np.random.randint(0, 200, len(df))
    
    # Add perishable indicator if not exists
    if 'perishable' not in df.columns:
        perishable_families = ['DAIRY', 'PRODUCE', 'MEATS', 'SEAFOOD', 'BREAD/BAKERY', 'DELI', 'PREPARED FOODS', 'POULTRY', 'EGGS']
        df['perishable'] = df[family_col].isin(perishable_families).astype(int)
    
    # Add price information if missing
    if 'unit_price' not in df.columns:
        price_ranges = {
            'DAIRY': (1, 10), 'BEVERAGES': (0.5, 5), 'PRODUCE': (0.5, 8), 
            'MEATS': (5, 50), 'SEAFOOD': (8, 80), 'DEFAULT': (1, 20)
        }
        
        def get_price(family):
            price_range = price_ranges.get(family, price_ranges['DEFAULT'])
            return round(np.random.uniform(price_range[0], price_range[1]), 2)
        
        df['unit_price'] = df[family_col].apply(get_price)
    
    # Add donation features
    df = add_donation_features(df)
    
    # Validate and clean donation columns
    df = validate_donation_columns(df)
    
    # Data quality improvements
    logger.info("🔍 Performing data quality improvements...")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Handle negative sales
    if sales_units_col in df.columns:
        negative_sales = df[sales_units_col] < 0
        if negative_sales.any():
            logger.info(f"Found {negative_sales.sum():,} negative sales, setting to 0")
            df.loc[negative_sales, sales_units_col] = 0
    
    # Standardize column names
    column_mapping = {
        sales_item_col: 'item_id',
        sales_store_col: 'store_nbr',
        sales_date_col: 'date',
        sales_units_col: 'sales'
    }
    
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=existing_mapping)
    
    logger.info(f"✅ Data processing complete: {len(df):,} records")
    return df

def add_donation_features(df):
    """Add donation-related features to the dataset"""
    logger.info("🎁 Adding donation features...")
    
    # Add donation eligibility based on days to expiry and perishable status
    if 'days_to_expiry' in df.columns:
        # Items close to expiry (1-3 days) and perishable are donation eligible
        df['donation_eligible'] = (
            (df['days_to_expiry'] <= 3) & 
            (df['days_to_expiry'] > 0) & 
            (df.get('perishable', 1) == 1)
        )
    else:
        df['donation_eligible'] = False
    
    # Add donation status
    def assign_donation_status(row):
        if row.get('donation_eligible', False):
            # Random assignment for simulation
            return np.random.choice(['Pending', 'Donated', 'Rejected'], p=[0.5, 0.3, 0.2])
        return ""
    
    df['donation_status'] = df.apply(assign_donation_status, axis=1)
    
    # Add store coordinates (simulated)
    if 'store_nbr' in df.columns:
        # Ecuador coordinates range: lat -5 to 2, lon -92 to -75
        np.random.seed(42)  # For consistent coordinates per store
        unique_stores = df['store_nbr'].unique()
        store_coords = {}
        
        for store in unique_stores:
            store_coords[store] = {
                'latitude': round(np.random.uniform(-4.5, 1.5), 6),
                'longitude': round(np.random.uniform(-91, -76), 6)
            }
        
        df['store_latitude'] = df['store_nbr'].map(lambda x: store_coords.get(x, {}).get('latitude', 0.0))
        df['store_longitude'] = df['store_nbr'].map(lambda x: store_coords.get(x, {}).get('longitude', 0.0))
    else:
        df['store_latitude'] = 0.0
        df['store_longitude'] = 0.0
    
    # Add NGO information
    ngos = [
        {'name': 'Food Bank Ecuador', 'address': 'Av. 10 de Agosto, Quito', 'contact': '+593-2-123-4567'},
        {'name': 'Minga por la Vida', 'address': 'Malecón 2000, Guayaquil', 'contact': '+593-4-234-5678'},
        {'name': 'Fundación Alimenta', 'address': 'Av. España, Cuenca', 'contact': '+593-7-345-6789'},
        {'name': 'Caritas Ecuador', 'address': 'Av. Patria, Quito', 'contact': '+593-2-456-7890'},
        {'name': 'Banco de Alimentos', 'address': 'Av. 9 de Octubre, Guayaquil', 'contact': '+593-4-567-8901'}
    ]
    
    def assign_ngo(row):
        if row.get('donation_eligible', False):
            ngo = np.random.choice(len(ngos))
            return ngos[ngo]['name'], ngos[ngo]['address'], ngos[ngo]['contact']
        return "Unknown", "Unknown", "Unknown"
    
    ngo_assignments = df.apply(assign_ngo, axis=1, result_type='expand')
    df['nearest_ngo'] = ngo_assignments[0]
    df['ngo_address'] = ngo_assignments[1]
    df['ngo_contact'] = ngo_assignments[2]
    
    return df

def validate_donation_columns(df):
    """Validate and clean donation-related columns"""
    logger.info("🔍 Validating donation columns...")
    
    # Required donation columns
    required_donation_cols = [
        'donation_eligible', 'donation_status', 'store_latitude', 
        'store_longitude', 'nearest_ngo', 'ngo_address', 'ngo_contact'
    ]
    
    # Ensure all donation columns exist
    for col in required_donation_cols:
        if col not in df.columns:
            logger.warning(f"Missing donation column: {col}")
            if col == 'donation_eligible':
                df[col] = False
            elif col == 'donation_status':
                df[col] = ""
            elif col in ['store_latitude', 'store_longitude']:
                df[col] = 0.0
            else:
                df[col] = "Unknown"
    
    # Data type casting and validation
    logger.info("🔧 Casting donation column types...")
    
    # Boolean column
    df['donation_eligible'] = df['donation_eligible'].astype(bool)
    
    # String columns - fill NaN with appropriate defaults
    string_cols = ['donation_status', 'nearest_ngo', 'ngo_address', 'ngo_contact']
    for col in string_cols:
        if col == 'donation_status':
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = df[col].fillna("Unknown").astype(str)
    
    # Float columns - validate coordinates
    for col in ['store_latitude', 'store_longitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Validate coordinate ranges for Ecuador
        if col == 'store_latitude':
            df[col] = df[col].clip(-5.0, 2.0)  # Valid latitude range for Ecuador
        else:  # longitude
            df[col] = df[col].clip(-92.0, -75.0)  # Valid longitude range for Ecuador
    
    # Drop rows with critical missing values
    critical_cols = ['days_to_expiry']
    if 'family' in df.columns:
        critical_cols.append('family')
    if 'category' in df.columns:
        critical_cols.append('category')
    
    initial_rows = len(df)
    
    # Drop rows with missing critical values
    for col in critical_cols:
        if col in df.columns:
            before_drop = len(df)
            df = df.dropna(subset=[col])
            dropped = before_drop - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped:,} rows with missing {col}")
    
    # Drop rows with invalid days_to_expiry
    if 'days_to_expiry' in df.columns:
        before_drop = len(df)
        df = df[df['days_to_expiry'] >= 0]
        dropped = before_drop - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped:,} rows with negative days_to_expiry")
    
    final_rows = len(df)
    total_dropped = initial_rows - final_rows
    
    if total_dropped > 0:
        logger.info(f"Total rows dropped: {total_dropped:,} ({100*total_dropped/initial_rows:.1f}%)")
    
    return df

def log_donation_summary(df):
    """Log summary statistics for donation features"""
    logger.info("=" * 60)
    logger.info("🎁 DONATION FEATURE SUMMARY")
    logger.info("=" * 60)
    
    if 'donation_eligible' in df.columns:
        eligible_count = df['donation_eligible'].sum()
        total_count = len(df)
        logger.info(f"📦 Donation Eligible Items: {eligible_count:,} ({100*eligible_count/total_count:.1f}%)")
    
    if 'donation_status' in df.columns:
        status_counts = df['donation_status'].value_counts()
        logger.info("📊 Donation Status Breakdown:")
        for status, count in status_counts.items():
            if status:  # Don't show empty string counts
                logger.info(f"   {status}: {count:,}")
    
    if 'nearest_ngo' in df.columns:
        ngo_counts = df[df['donation_eligible'] == True]['nearest_ngo'].value_counts()
        if len(ngo_counts) > 0:
            logger.info("🏢 NGO Assignment for Eligible Items:")
            for ngo, count in ngo_counts.head().items():
                if ngo != "Unknown":
                    logger.info(f"   {ngo}: {count:,}")
    
    if all(col in df.columns for col in ['store_latitude', 'store_longitude']):
        valid_coords = (
            (df['store_latitude'] != 0.0) & 
            (df['store_longitude'] != 0.0)
        ).sum()
        logger.info(f"📍 Stores with Valid Coordinates: {valid_coords:,}")
    
    logger.info("=" * 60)

# ...existing code...
def main():
    """Main preprocessing pipeline"""
    try:
        logger.info("🚀 Starting Data Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Create directories
        os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "data", "raw"), exist_ok=True)
        
        # Load data
        data, data_type = load_data_with_fallback()
        
        if data_type == "existing_data":
            logger.info("✅ Using existing cleaned data")
            df = data
            # Add donation features to existing data if they don't exist
            if isinstance(df, pd.DataFrame) and 'donation_eligible' not in df.columns:
                logger.info("🎁 Adding donation features to existing data...")
                df = add_donation_features(df)
                df = validate_donation_columns(df)
        elif data_type == "raw_data":
            logger.info("📊 Processing raw data files")
            df = process_raw_data(data)
        else:  # sample_data
            logger.info("🎲 Using generated sample data")
            df = process_raw_data(data)
        
        # Ensure df is a DataFrame at this point
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame but got {type(df)}. Processing failed.")
        
        # Save cleaned dataset
        output_paths = [
            os.path.join(project_root, "data", "cleaned_inventory_data.csv"),
            os.path.join(project_root, "cleaned_inventory_data.csv")
        ]
        
        for output_path in output_paths:
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"💾 Saving to {output_path}...")
                df.to_csv(output_path, index=False)
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ Saved: {output_path} ({file_size:.1f} MB)")
            except Exception as e:
                logger.warning(f"Failed to save to {output_path}: {str(e)}")
        
        # Display summary
        logger.info("=" * 60)
        logger.info("📊 DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📦 Total Records: {len(df):,}")
        logger.info(f"📋 Total Columns: {len(df.columns)}")
        logger.info(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        if 'date' in df.columns:
            logger.info(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
        
        if 'store_nbr' in df.columns:
            logger.info(f"🏪 Stores: {df['store_nbr'].nunique():,}")
        
        if 'item_id' in df.columns:
            logger.info(f"📦 Items: {df['item_id'].nunique():,}")
        
        if 'family' in df.columns:
            logger.info(f"🏷️  Families: {df['family'].nunique()}")
        
        if 'sales' in df.columns:
            logger.info(f"💰 Total Sales: {df['sales'].sum():,.2f}")
        
        # Log donation feature summary
        log_donation_summary(df)
        
        logger.info("=" * 60)
        logger.info("✅ PREPROCESSING COMPLETE!")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()