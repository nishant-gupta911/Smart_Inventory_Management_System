import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InventoryAnalyzer:
    def __init__(self, project_root=None):
        if project_root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(current_dir)
        else:
            self.project_root = project_root
        
        # Thresholds for analysis
        self.overstock_multiplier = 1.5
        self.understock_multiplier = 0.5
        self.near_expiry_days = 15
        self.max_discount_pct = 40
        
    def load_inventory_data(self) -> pd.DataFrame:
        """Load inventory data from available sources"""
        logger.info("📥 Loading inventory data...")
        
        # Try multiple possible data paths
        possible_paths = [
            os.path.join(self.project_root, "data", "cleaned_inventory_data.csv"),
            os.path.join(self.project_root, "cleaned_inventory_data.csv"),
            os.path.join(self.project_root, "data", "processed", "expiry_risk_predictions.csv"),
            os.path.join(self.project_root, "data", "raw", "cleaned_inventory_data.csv")
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    logger.info(f"✅ Data loaded from: {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {str(e)}")
                    continue
        
        if df is None:
            logger.warning("⚠️ No data file found. Generating sample data.")
            df = self._generate_sample_inventory()
        
        return self._prepare_data(df)
    
    def _generate_sample_inventory(self) -> pd.DataFrame:
        """Generate sample inventory data for demonstration"""
        logger.info("🔧 Generating sample inventory data...")
        
        np.random.seed(42)
        n_items = 1000
        
        # Generate realistic inventory data
        df = pd.DataFrame({
            'item_id': [f'ITEM_{i:05d}' for i in range(n_items)],
            'product_name': [f'Product_{i:03d}' for i in range(n_items)],
            'store_nbr': np.random.randint(1, 11, n_items),
            'category': np.random.choice(['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY', 'SEAFOOD', 'FROZEN FOODS'], n_items),
            'current_stock': np.random.randint(0, 200, n_items),
            'rolling_avg_sales_7': np.random.exponential(5, n_items),
            'shelf_life': np.random.choice([2, 3, 4, 7, 14, 30, 90], n_items),
            'days_to_expiry': np.random.randint(0, 31, n_items),
            'unit_price': np.random.uniform(0.5, 50, n_items),
            'date': datetime.now().date()
        })
        
        # Add donation-related columns
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego']
        ngos = ['Food Bank Central', 'Helping Hands', 'Community Kitchen', 'Hope Foundation', 'Care Alliance']
        
        # Donation eligibility based on expiry
        df['donation_eligible'] = (df['days_to_expiry'] <= 15) & (df['category'].isin(['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY']))
        
        # Donation status
        df['donation_status'] = df.apply(lambda x: 
            np.random.choice(['Pending', 'Donated', 'Rejected'], p=[0.4, 0.4, 0.2]) if x['donation_eligible'] 
            else 'N/A', axis=1)
        
        # Store location data
        df['city'] = np.random.choice(cities, n_items)
        df['store_latitude'] = np.random.uniform(25.0, 45.0, n_items)  # US latitude range
        df['store_longitude'] = np.random.uniform(-125.0, -65.0, n_items)  # US longitude range
        
        # NGO data
        df['nearest_ngo'] = np.random.choice(ngos, n_items)
        df['ngo_address'] = df.apply(lambda x: f"{np.random.randint(100, 9999)} {np.random.choice(['Main St', 'Oak Ave', 'Pine Rd', 'Elm St'])}, {x['city']}", axis=1)
        df['ngo_contact'] = df['nearest_ngo'].apply(lambda x: f"contact@{x.lower().replace(' ', '')}.org")
        
        return df
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and standardize the data"""
        logger.info("🔧 Preparing data for analysis...")
        
        # Ensure required columns exist
        required_columns = {
            'item_id': lambda: [f'ITEM_{i:05d}' for i in range(len(df))],
            'product_name': lambda: [f'Product_{i:03d}' for i in range(len(df))],
            'store_nbr': lambda: np.random.randint(1, 11, len(df)),
            'current_stock': lambda: np.random.randint(0, 200, len(df)),
            'rolling_avg_sales_7': lambda: np.random.exponential(5, len(df)),
            'days_to_expiry': lambda: np.random.randint(0, 31, len(df)),
            'unit_price': lambda: np.random.uniform(0.5, 50, len(df)),
            'category': lambda: np.random.choice(['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY'], len(df))
        }
        
        # Add donation-related columns if missing
        if 'donation_eligible' not in df.columns:
            df['donation_eligible'] = (df['days_to_expiry'] <= 15) & (df['category'].isin(['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY']))
        
        if 'donation_status' not in df.columns:
            df['donation_status'] = df.apply(lambda x: 
                np.random.choice(['Pending', 'Donated', 'Rejected'], p=[0.4, 0.4, 0.2]) if x['donation_eligible'] 
                else 'N/A', axis=1)
        
        if 'city' not in df.columns:
            cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
            df['city'] = np.random.choice(cities, len(df))
        
        if 'nearest_ngo' not in df.columns:
            ngos = ['Food Bank Central', 'Helping Hands', 'Community Kitchen', 'Hope Foundation', 'Care Alliance']
            df['nearest_ngo'] = np.random.choice(ngos, len(df))
        
        if 'ngo_contact' not in df.columns:
            df['ngo_contact'] = df['nearest_ngo'].apply(lambda x: f"contact@{x.lower().replace(' ', '')}.org")
        
        # Add missing columns
        for col, generator in required_columns.items():
            if col not in df.columns:
                logger.warning(f"Missing column '{col}', generating default values")
                df[col] = generator()
        
        # Handle current_stock - derive from rolling_avg_sales_7 if not available
        if 'current_stock' not in df.columns:
            # Estimate current stock based on sales patterns
            df['current_stock'] = (df['rolling_avg_sales_7'] * np.random.uniform(0.5, 3, len(df))).astype(int)
        
        # Clean and validate data
        df['rolling_avg_sales_7'] = pd.to_numeric(df['rolling_avg_sales_7'], errors='coerce').fillna(0)
        df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce').fillna(0).astype(int)
        df['days_to_expiry'] = pd.to_numeric(df['days_to_expiry'], errors='coerce').fillna(30).astype(int)
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce').fillna(1.0)
        
        # Ensure no negative values
        df['current_stock'] = df['current_stock'].clip(lower=0)
        df['rolling_avg_sales_7'] = df['rolling_avg_sales_7'].clip(lower=0)
        df['days_to_expiry'] = df['days_to_expiry'].clip(lower=0)
        
        logger.info(f"Data prepared: {len(df)} items across {df['store_nbr'].nunique()} stores")
        return df
    
    def analyze_stock_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze stock levels and categorize as High/Low/Normal"""
        logger.info("📊 Analyzing stock levels...")
        
        # Calculate weekly demand (avoid division by zero)
        weekly_demand = df['rolling_avg_sales_7'] * 7
        weekly_demand = weekly_demand.replace(0, 0.1)  # Minimum demand to avoid division by zero
        
        # Calculate thresholds
        overstock_threshold = weekly_demand * self.overstock_multiplier
        understock_threshold = weekly_demand * self.understock_multiplier
        
        # Categorize stock levels
        conditions = [
            df['current_stock'] > overstock_threshold,
            df['current_stock'] < understock_threshold
        ]
        choices = ['High', 'Low']
        df['Stock_Level'] = np.select(conditions, choices, default='Normal')
        
        # Log results
        stock_counts = df['Stock_Level'].value_counts()
        logger.info(f"Stock Level Analysis: {stock_counts.to_dict()}")
        
        return df
    
    def analyze_expiry_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze expiry risk based on days to expiry"""
        logger.info("⏰ Analyzing expiry risk...")
        
        # Categorize expiry risk
        conditions = [
            df['days_to_expiry'] <= 0,
            df['days_to_expiry'] <= self.near_expiry_days
        ]
        choices = ['Expired', 'Near Expiry']
        df['Expiry_Risk'] = np.select(conditions, choices, default='Safe')
        
        # Log results
        expiry_counts = df['Expiry_Risk'].value_counts()
        logger.info(f"Expiry Risk Analysis: {expiry_counts.to_dict()}")
        
        return df
    
    def calculate_discount_suggestions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate suggested discounts based on stock level and expiry risk"""
        logger.info("💰 Calculating discount suggestions...")
        
        # Initialize discount column
        df['Suggested_Discount'] = 0
        
        # Discount logic
        for idx, row in df.iterrows():
            discount = 0
            
            # High stock items get discount
            if row['Stock_Level'] == 'High':
                discount += 15
            
            # Near expiry items get higher discount
            if row['Expiry_Risk'] == 'Near Expiry':
                discount += 20
            elif row['Expiry_Risk'] == 'Expired':
                discount += 40
            
            # Overstocked + near expiry gets maximum discount
            if row['Stock_Level'] == 'High' and row['Expiry_Risk'] in ['Near Expiry', 'Expired']:
                discount = min(self.max_discount_pct, discount + 10)
            
            # Never discount understocked items
            if row['Stock_Level'] == 'Low':
                discount = 0
            
            # Cap at maximum discount
            df.at[idx, 'Suggested_Discount'] = min(discount, self.max_discount_pct)
        
        # Log results
        discount_stats = df['Suggested_Discount'].describe()
        items_with_discount = (df['Suggested_Discount'] > 0).sum()
        logger.info(f"Discount Suggestions: {items_with_discount} items need discounts")
        logger.info(f"Average suggested discount: {df['Suggested_Discount'].mean():.1f}%")
        
        return df
    
    def determine_reorder_needs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determine which items need reordering"""
        logger.info("📦 Determining reorder needs...")
        
        # Reorder logic
        conditions = [
            (df['Stock_Level'] == 'Low') | 
            (df['Expiry_Risk'] == 'Expired') |
            ((df['current_stock'] < df['rolling_avg_sales_7'] * 3) & (df['Expiry_Risk'] == 'Safe'))
        ]
        
        df['Reorder'] = np.where(conditions[0], 'Yes', 'No')
        
        # Log results
        reorder_counts = df['Reorder'].value_counts()
        logger.info(f"Reorder Analysis: {reorder_counts.to_dict()}")
        
        return df
    
    def determine_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determine recommended actions for each item"""
        logger.info("⚡ Determining recommended actions...")
        
        def get_action(row):
            # Priority 1: Expired items
            if row['Expiry_Risk'] == 'Expired':
                return 'Remove'
            
            # Priority 2: Understocked items
            elif row['Stock_Level'] == 'Low':
                return 'Restock'
            
            # Priority 3: Overstocked items
            elif row['Stock_Level'] == 'High':
                if row['Expiry_Risk'] == 'Near Expiry':
                    return 'Apply Discount'
                else:
                    return 'Redistribute'
            
            # Priority 4: Near expiry items
            elif row['Expiry_Risk'] == 'Near Expiry' and row['Suggested_Discount'] > 0:
                return 'Apply Discount'
            
            # Default: No action needed
            else:
                return 'No Action'
        
        df['Action'] = df.apply(get_action, axis=1)
        
        # Log results
        action_counts = df['Action'].value_counts()
        logger.info(f"Action Recommendations: {action_counts.to_dict()}")
        
        return df
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the analysis"""
        logger.info("📋 Generating summary report...")
        
        # Get donation summary
        donation_summary = self.get_donation_summary(df)
        
        summary = {
            'total_items': len(df),
            'total_stores': df['store_nbr'].nunique(),
            'stock_levels': df['Stock_Level'].value_counts().to_dict(),
            'expiry_risks': df['Expiry_Risk'].value_counts().to_dict(),
            'actions_needed': df['Action'].value_counts().to_dict(),
            'items_needing_discount': (df['Suggested_Discount'] > 0).sum(),
            'items_needing_reorder': (df['Reorder'] == 'Yes').sum(),
            'avg_suggested_discount': df['Suggested_Discount'].mean(),
            'total_inventory_value': (df['current_stock'] * df['unit_price']).sum(),
            'at_risk_value': df[df['Action'].isin(['Remove', 'Apply Discount'])]['current_stock'].sum() * df[df['Action'].isin(['Remove', 'Apply Discount'])]['unit_price'].mean(),
            # Add donation statistics
            'donation_summary': donation_summary
        }
        
        return summary
    
    def save_updated_inventory(self, df: pd.DataFrame) -> str:
        """Save the updated inventory with new analysis columns"""
        logger.info("💾 Saving updated inventory data...")
        
        # Create output directory
        output_dir = os.path.join(self.project_root, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save updated inventory
        output_path = os.path.join(output_dir, 'inventory_analysis_results.csv')
        df.to_csv(output_path, index=False)
        
        # Also update the main cleaned data file
        main_output_path = os.path.join(self.project_root, 'data', 'cleaned_inventory_data.csv')
        df.to_csv(main_output_path, index=False)
        
        logger.info(f"✅ Updated inventory saved to: {output_path}")
        return output_path
    
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete inventory analysis pipeline"""
        logger.info("🚀 Starting comprehensive inventory analysis...")
        
        try:
            # Load data
            df = self.load_inventory_data()
            
            # Run analysis steps
            df = self.analyze_stock_levels(df)
            df = self.analyze_expiry_risk(df)
            df = self.calculate_discount_suggestions(df)
            df = self.determine_reorder_needs(df)
            df = self.determine_actions(df)
            
            # Generate summary
            summary = self.generate_summary_report(df)
            
            # Save results
            output_path = self.save_updated_inventory(df)
            
            # Print summary report
            self.print_summary_report(summary)
            
            logger.info("✅ Inventory analysis completed successfully!")
            return df, summary
            
        except Exception as e:
            logger.error(f"❌ Inventory analysis failed: {str(e)}")
            raise
    
    def print_summary_report(self, summary: Dict):
        """Print a formatted summary report"""
        print("\n" + "=" * 80)
        print("📊 INVENTORY ANALYSIS SUMMARY REPORT")
        print("=" * 80)
        
        print(f"📦 Total Items Analyzed: {summary['total_items']:,}")
        print(f"🏪 Total Stores: {summary['total_stores']}")
        print(f"💰 Total Inventory Value: ${summary['total_inventory_value']:,.2f}")
        print(f"⚠️  At-Risk Inventory Value: ${summary.get('at_risk_value', 0):,.2f}")
        
        print("\n📊 STOCK LEVEL DISTRIBUTION:")
        for level, count in summary['stock_levels'].items():
            print(f"   {level}: {count:,} items ({count/summary['total_items']*100:.1f}%)")
        
        print("\n⏰ EXPIRY RISK DISTRIBUTION:")
        for risk, count in summary['expiry_risks'].items():
            print(f"   {risk}: {count:,} items ({count/summary['total_items']*100:.1f}%)")
        
        print("\n⚡ ACTION RECOMMENDATIONS:")
        for action, count in summary['actions_needed'].items():
            print(f"   {action}: {count:,} items ({count/summary['total_items']*100:.1f}%)")
        
        print(f"\n💸 DISCOUNT RECOMMENDATIONS:")
        print(f"   Items needing discount: {summary['items_needing_discount']:,}")
        print(f"   Average suggested discount: {summary['avg_suggested_discount']:.1f}%")
        
        print(f"\n📦 REORDER RECOMMENDATIONS:")
        print(f"   Items needing reorder: {summary['items_needing_reorder']:,}")
        
        # Add donation summary section
        donation_summary = summary.get('donation_summary', {})
        print(f"\n🤝 DONATION ANALYSIS:")
        print(f"   Total donation-eligible items: {donation_summary.get('total_donation_eligible', 0):,}")
        
        if 'donation_status_counts' in donation_summary:
            print("   Donation status breakdown:")
            for status, count in donation_summary['donation_status_counts'].items():
                print(f"     {status}: {count:,} items")
        
        if 'top_ngos' in donation_summary and donation_summary['top_ngos']:
            print("   Top NGOs receiving donations:")
            for ngo, count in list(donation_summary['top_ngos'].items())[:3]:
                print(f"     {ngo}: {count} items")
        
        print("\n" + "=" * 80)
    
    def get_donation_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive donation summary including totals, status counts, 
        city/category breakdown, and top NGOs
        """
        logger.info("🤝 Analyzing donation summary...")
        
        # Ensure donation columns exist
        if 'donation_eligible' not in df.columns:
            logger.warning("donation_eligible column not found, creating default values")
            df['donation_eligible'] = df['Expiry_Risk'].isin(['Expired', 'Near Expiry'])
        
        if 'donation_status' not in df.columns:
            logger.warning("donation_status column not found, creating default values")
            # Set status based on expiry risk for simulation
            df['donation_status'] = df.apply(lambda x: 
                'Pending' if x['donation_eligible'] and x['Expiry_Risk'] == 'Near Expiry'
                else 'Donated' if x['donation_eligible'] and x['Expiry_Risk'] == 'Expired' and np.random.random() > 0.3
                else 'Rejected' if x['donation_eligible'] else 'N/A', axis=1)
        
        # Add missing location columns if not present
        if 'city' not in df.columns:
            df['city'] = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], len(df))
        
        if 'nearest_ngo' not in df.columns:
            ngos = ['Food Bank Central', 'Helping Hands', 'Community Kitchen', 'Hope Foundation', 'Care Alliance']
            df['nearest_ngo'] = np.random.choice(ngos, len(df))
        
        # Filter donation-eligible items
        donation_eligible_df = df[df['donation_eligible'] == True]
        
        summary = {
            'total_donation_eligible': len(donation_eligible_df),
            'donation_status_counts': donation_eligible_df['donation_status'].value_counts().to_dict(),
            'city_category_breakdown': {},
            'top_ngos': {}
        }
        
        # City and category breakdown
        if len(donation_eligible_df) > 0:
            city_category_pivot = donation_eligible_df.groupby(['city', 'category']).size().reset_index(name='count')
            summary['city_category_breakdown'] = city_category_pivot.to_dict('records')
            
            # Top 5 NGOs receiving most donations
            donated_items = donation_eligible_df[donation_eligible_df['donation_status'] == 'Donated']
            if len(donated_items) > 0:
                top_ngos = donated_items['nearest_ngo'].value_counts().head(5)
                summary['top_ngos'] = top_ngos.to_dict()
        
        # Log summary
        logger.info(f"Donation Summary: {summary['total_donation_eligible']} eligible items")
        logger.info(f"Status distribution: {summary['donation_status_counts']}")
        
        return summary
    
    def get_pending_donations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get all items that are donation-eligible and have pending status
        """
        logger.info("⏳ Filtering pending donations...")
        
        # Ensure required columns exist
        if 'donation_eligible' not in df.columns:
            df['donation_eligible'] = df['Expiry_Risk'].isin(['Expired', 'Near Expiry'])
        
        if 'donation_status' not in df.columns:
            df['donation_status'] = 'Pending'
        
        # Add missing columns with default values
        if 'city' not in df.columns:
            df['city'] = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], len(df))
        
        if 'nearest_ngo' not in df.columns:
            ngos = ['Food Bank Central', 'Helping Hands', 'Community Kitchen', 'Hope Foundation', 'Care Alliance']
            df['nearest_ngo'] = np.random.choice(ngos, len(df))
        
        if 'ngo_contact' not in df.columns:
            df['ngo_contact'] = df['nearest_ngo'].apply(lambda x: f"contact@{x.lower().replace(' ', '')}.org")
        
        # Filter for pending donations
        pending_donations = df[
            (df['donation_eligible'] == True) & 
            (df['donation_status'] == 'Pending')
        ].copy()
        
        # Select relevant columns
        columns_to_include = ['product_name', 'days_to_expiry', 'city', 'nearest_ngo', 'ngo_contact']
        
        # Add additional useful columns if they exist
        optional_columns = ['item_id', 'category', 'current_stock', 'unit_price', 'store_nbr']
        for col in optional_columns:
            if col in df.columns:
                columns_to_include.append(col)
        
        result_df = pending_donations[columns_to_include].copy()
        
        logger.info(f"Found {len(result_df)} items with pending donation status")
        
        return result_df
    
    def analyze_donation_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze donation-eligible items by category using pivot table
        """
        logger.info("📊 Analyzing donations by category...")
        
        # Ensure required columns exist
        if 'donation_eligible' not in df.columns:
            df['donation_eligible'] = df['Expiry_Risk'].isin(['Expired', 'Near Expiry'])
        
        if 'donation_status' not in df.columns:
            df['donation_status'] = 'Pending'
        
        # Filter donation-eligible items
        donation_eligible_df = df[df['donation_eligible'] == True].copy()
        
        if len(donation_eligible_df) == 0:
            logger.warning("No donation-eligible items found")
            return pd.DataFrame()
        
        # Create pivot table
        category_analysis = donation_eligible_df.groupby(['category', 'donation_status']).agg({
            'item_id': 'count',
            'current_stock': 'sum',
            'unit_price': 'mean'
        }).round(2)
        
        category_analysis.columns = ['item_count', 'total_stock', 'avg_unit_price']
        category_analysis = category_analysis.reset_index()
        
        # Add percentage of total donations per category
        total_donations = len(donation_eligible_df)
        category_totals = category_analysis.groupby('category')['item_count'].sum()
        category_analysis['percentage_of_total'] = category_analysis.apply(
            lambda row: round((category_totals[row['category']] / total_donations) * 100, 1), axis=1
        )
        
        logger.info(f"Category analysis complete for {category_analysis['category'].nunique()} categories")
        
        return category_analysis

def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = InventoryAnalyzer()
        
        # Run full analysis
        df, summary = analyzer.run_full_analysis()
        
        # Display sample results
        print("\n📋 SAMPLE ANALYSIS RESULTS:")
        print("-" * 50)
        
        # Show items needing action
        action_items = df[df['Action'] != 'No Action'].head(10)
        if len(action_items) > 0:
            print(action_items[['item_id', 'product_name', 'current_stock', 'Stock_Level', 
                             'Expiry_Risk', 'Suggested_Discount', 'Action']].to_string(index=False))
        
        # Demonstrate donation analysis
        print("\n🤝 DONATION ANALYSIS EXAMPLES:")
        print("-" * 50)
        
        # Get pending donations
        pending_donations = analyzer.get_pending_donations(df)
        if len(pending_donations) > 0:
            print(f"\n📋 PENDING DONATIONS ({len(pending_donations)} items):")
            print(pending_donations.head(5)[['product_name', 'days_to_expiry', 'city', 'nearest_ngo']].to_string(index=False))
        
        # Get category analysis
        category_analysis = analyzer.analyze_donation_by_category(df)
        if len(category_analysis) > 0:
            print(f"\n📊 DONATIONS BY CATEGORY:")
            print(category_analysis.head(10).to_string(index=False))
        
        print("\n✅ Inventory analysis completed! Check the dashboard for detailed insights.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {str(e)}")
        print(f"❌ Analysis failed: {str(e)}")
        return False


if __name__ == "__main__":
    main()