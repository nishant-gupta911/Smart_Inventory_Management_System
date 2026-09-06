"""
Smart Inventory Management System - Main Pipeline
This script orchestrates the entire inventory management workflow including
data preprocessing, expiry prediction, and comprehensive analysis.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

# Setup project paths - use pathlib for better cross-platform compatibility
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_PATH = PROJECT_ROOT / 'src'

# Add paths to Python path for module imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

def setup_logging() -> logging.Logger:
    """Configure logging with proper error handling"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "main_pipeline.log"
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing logging config
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()

def import_modules() -> Tuple[Any, Any, Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """
    Import required modules with proper fallback handling.
    Returns:
        (train_expiry_model, inventory_analyzer, data_preprocessing,
         transform_inventory_data, utils, train_demand_model, generate_restock_plan)
    """
    train_expiry_model       = None
    inventory_analyzer       = None
    data_preprocessing       = None
    transform_inventory_data = None
    utils                    = None
    train_demand_model       = None
    generate_restock_plan    = None

    # ── Core modules (required) ───────────────────────────────────────────────
    try:
        from src import train_expiry_model, inventory_analyzer, data_preprocessing
        logger.info("✅ Successfully imported core modules from src package")
    except ImportError as e:
        logger.error(f"❌ Failed to import core modules: {e}")
        logger.error("Required files:")
        logger.error(f"  - {SRC_PATH / 'train_expiry_model.py'}")
        logger.error(f"  - {SRC_PATH / 'inventory_analyzer.py'}")
        logger.error(f"  - {SRC_PATH / 'data_preprocessing.py'}")
        raise ImportError("Cannot proceed without required modules") from e

    # ── Demand model ──────────────────────────────────────────────────────────
    try:
        from src import train_demand_model
        logger.info("✅ Successfully imported train_demand_model module")
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import train_demand_model module: {e}")

    # ── Restock plan generator ────────────────────────────────────────────────
    try:
        from src import generate_restock_plan
        logger.info("✅ Successfully imported generate_restock_plan module")
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import generate_restock_plan module: {e}")

    # ── Donation utilities ────────────────────────────────────────────────────
    try:
        from src import utils
        logger.info("✅ Successfully imported utils module")
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import utils module: {e}")
        utils = None

    try:
        import transform_inventory_data
        logger.info("✅ Successfully imported transform_inventory_data module")
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import transform_inventory_data module: {e}")
        transform_inventory_data = None

    return (train_expiry_model, inventory_analyzer, data_preprocessing,
            transform_inventory_data, utils, train_demand_model, generate_restock_plan)

def ensure_directories() -> bool:
    """Create required directories with proper error handling"""
    required_dirs = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "plots",
        PROJECT_ROOT / "dashboard" / "cache"
    ]
    
    logger.info("🔧 Ensuring required directories exist...")
    
    for directory in required_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"✅ Directory ready: {directory}")
        except PermissionError:
            logger.error(f"❌ Permission denied creating: {directory}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    logger.info("✅ All required directories verified")
    return True

def check_data_files() -> bool:
    """Check if required data files exist"""
    data_file_locations = [
        PROJECT_ROOT / "data" / "cleaned_inventory_data.csv",
        PROJECT_ROOT / "data" / "processed" / "inventory_analysis_results_enhanced.csv",
        PROJECT_ROOT / "data" / "processed" / "inventory_data.csv",
        PROJECT_ROOT / "data" / "raw" / "inventory_data.csv",
    ]
    
    for data_file in data_file_locations:
        if data_file.exists():
            logger.info(f"✅ Found data file: {data_file}")
            return True
    
    logger.warning("⚠️ No data files found in expected locations:")
    for location in data_file_locations:
        logger.warning(f"  - {location}")
    
    return False

def run_data_preprocessing(data_preprocessing_module) -> bool:
    """Run data preprocessing if needed and available"""
    try:
        if check_data_files():
            logger.info("✅ Data files already exist, skipping preprocessing")
            return True
        
        logger.info("🔄 Running data preprocessing...")
        
        if hasattr(data_preprocessing_module, 'preprocess_data'):
            data_preprocessing_module.preprocess_data()
        elif hasattr(data_preprocessing_module, 'main'):
            data_preprocessing_module.main()
        else:
            logger.error("❌ data_preprocessing module missing 'preprocess_data()' or 'main()' function")
            return False
        
        if check_data_files():
            logger.info("✅ Data preprocessing completed successfully")
            return True
        else:
            logger.error("❌ Data preprocessing completed but no output files found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in data preprocessing: {e}")
        logger.exception("Preprocessing error details:")
        return False

def run_expiry_prediction(train_expiry_model) -> bool:
    """Run the expiry risk prediction model and log achieved accuracy"""
    try:
        logger.info("🔍 Phase 1: Running expiry risk prediction model...")
        
        if hasattr(train_expiry_model, 'train_and_predict'):
            mean_acc = train_expiry_model.train_and_predict()
            acc_pct  = mean_acc * 100 if isinstance(mean_acc, float) else 0.0
            logger.info(f"✅ Expiry model training complete — "
                        f"CV Accuracy: {acc_pct:.2f}%")
            print(f"\n🎯 [Phase 1] Expiry Model Accuracy: {acc_pct:.2f}%\n")
            return True
        else:
            logger.error("❌ train_expiry_model module missing 'train_and_predict()' function")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error in expiry risk prediction: {e}")
        logger.exception("Expiry prediction error details:")
        return False


def run_demand_model(train_demand_model) -> bool:
    """Run the demand forecasting model and log achieved accuracy"""
    try:
        logger.info("📈 Phase 1.5: Running demand forecasting model...")

        if train_demand_model is None:
            logger.warning("⚠️ train_demand_model module not available, skipping demand training")
            return False

        if hasattr(train_demand_model, 'train_and_evaluate'):
            mean_r2  = train_demand_model.train_and_evaluate()
            acc_pct  = max(0.0, mean_r2) * 100 if isinstance(mean_r2, float) else 0.0
            logger.info(f"✅ Demand model training complete — "
                        f"CV R²: {mean_r2:.4f}  (~{acc_pct:.2f}%)")
            print(f"\n🎯 [Phase 1.5] Demand Model Accuracy (R²): "
                  f"{mean_r2:.4f}  (~{acc_pct:.2f}%)\n")
            return True
        else:
            logger.error("❌ train_demand_model module missing 'train_and_evaluate()' function")
            return False

    except Exception as e:
        logger.error(f"❌ Error in demand model training: {e}")
        logger.exception("Demand model training error details:")
        return False


def run_inventory_analysis(inventory_analyzer) -> Tuple[bool, Optional[pd.DataFrame], Optional[Dict]]:
    """Run comprehensive inventory analysis"""
    try:
        logger.info("📊 Phase 2: Running comprehensive inventory analysis...")
        
        if hasattr(inventory_analyzer, 'InventoryAnalyzer'):
            analyzer = inventory_analyzer.InventoryAnalyzer(str(PROJECT_ROOT))
            df, summary = analyzer.run_full_analysis()
            
            logger.info("✅ Inventory analysis completed")
            return True, df, summary
        else:
            logger.error("❌ inventory_analyzer module missing 'InventoryAnalyzer' class")
            return False, None, None
        
    except Exception as e:
        logger.error(f"❌ Error in inventory analysis: {e}")
        logger.exception("Inventory analysis error details:")
        return False, None, None

def save_dashboard_data(df: pd.DataFrame) -> bool:
    """Save data for dashboard consumption"""
    try:
        logger.info("📈 Phase 3: Preparing dashboard data...")
        
        dashboard_cache_dir = PROJECT_ROOT / "dashboard" / "cache"
        dashboard_cache_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard_data_path = dashboard_cache_dir / "dashboard_data.csv"
        df.to_csv(dashboard_data_path, index=False)
        
        logger.info(f"✅ Dashboard data saved: {dashboard_data_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving dashboard data: {e}")
        logger.exception("Dashboard data save error:")
        return False

def safe_get_summary_value(summary: Dict[str, Any], key: str, default: Any = "N/A") -> str:
    """Safely extract and format summary values"""
    try:
        value = summary.get(key, default)
        if key in ['total_inventory_value'] and isinstance(value, (int, float)):
            return f"${value:,.2f}"
        elif key in ['total_items'] and isinstance(value, (int, float)):
            return f"{value:,}"
        else:
            return str(value)
    except Exception:
        return str(default)

def display_summary(summary: Dict[str, Any]) -> None:
    """Display analysis summary with safe error handling"""
    try:
        logger.info("📊 Analysis Summary:")
        logger.info(f"   📦 Total Items: {safe_get_summary_value(summary, 'total_items')}")
        logger.info(f"   💰 Total Value: {safe_get_summary_value(summary, 'total_inventory_value')}")
        
        actions_needed = summary.get('actions_needed', {})
        if isinstance(actions_needed, dict):
            high_risk = actions_needed.get('Remove', 0) + actions_needed.get('Apply Discount', 0)
            logger.info(f"   ⚠️  High Risk Items: {high_risk}")
        else:
            logger.info("   ⚠️  High Risk Items: N/A")
        
        logger.info(f"   📦 Reorder Needed: {safe_get_summary_value(summary, 'items_needing_reorder')}")
        
    except Exception as e:
        logger.warning(f"Could not display complete summary: {e}")
        logger.info("   📊 Summary data available but formatting incomplete")

def run_donation_data_processing(transform_inventory_data) -> Tuple[bool, Optional[pd.DataFrame]]:
    """Run donation data processing and transformation"""
    try:
        logger.info("🤝 PROCESSING DONATION DATA...")
        logger.info("═" * 50)
        
        if transform_inventory_data is None:
            logger.warning("⚠️ transform_inventory_data module not available, skipping donation processing")
            return False, None
        
        if hasattr(transform_inventory_data, 'load_and_transform_data'):
            # Load and transform data with donation logic
            df = transform_inventory_data.load_and_transform_data()
            
            if df is not None and len(df) > 0:
                logger.info(f"✅ Donation data processing completed: {len(df):,} rows")
                
                # Quick preview of the new Action logic results
                if 'Action' in df.columns:
                    action_counts = df['Action'].value_counts()
                    logger.info("⚡ ACTION DISTRIBUTION PREVIEW:")
                    for action, count in action_counts.items():
                        if action == 'Remove':
                            logger.info(f"   🗑️  {action}: {count:,}")
                        elif action == 'Donate':
                            logger.info(f"   🤝 {action}: {count:,}")
                        elif action == 'Apply Discount':
                            logger.info(f"   💸 {action}: {count:,}")
                        elif action == 'Restock':
                            logger.info(f"   📦 {action}: {count:,}")
                        else:
                            logger.info(f"   ✅ {action}: {count:,}")
                
                return True, df
            else:
                logger.error("❌ Donation data processing returned empty or None DataFrame")
                return False, None
        else:
            logger.error("❌ transform_inventory_data module missing 'load_and_transform_data()' function")
            return False, None
        
    except Exception as e:
        logger.error(f"❌ Error in donation data processing: {e}")
        logger.exception("Donation processing error details:")
        return False, None

def analyze_donation_data(df: pd.DataFrame, utils_module=None) -> None:
    """Analyze and log donation data statistics with enhanced validation and metrics"""
    try:
        logger.info("📊 ANALYZING DONATION DATA...")
        logger.info("═" * 60)
        
        # Validate donation data first
        is_valid, validation_message = validate_donation_data(df)
        if not is_valid:
            logger.error(f"❌ Donation data validation failed: {validation_message}")
            return
        
        # Display comprehensive donation metrics
        display_donation_metrics(df)
        
        # Top 5 cities with most donations (ACTUAL DONATED ITEMS)
        if 'city' in df.columns and 'donation_status' in df.columns:
            donated_items = df[df['donation_status'] == 'Donated']
            if len(donated_items) > 0:
                top_cities = donated_items['city'].value_counts().head(5)
                logger.info("🏙️ TOP 5 CITIES WITH MOST DONATIONS (Actual Donated):")
                logger.info("─" * 50)
                for i, (city, count) in enumerate(top_cities.items(), 1):
                    logger.info(f"   {i}. {city:15}: {count:4} donations")
            else:
                logger.info("🏙️ No donated items found for city analysis")
        
        # Top 5 NGOs by donation count (ACTUAL DONATED ITEMS)
        if 'nearest_ngo' in df.columns and 'donation_status' in df.columns:
            donated_items = df[df['donation_status'] == 'Donated']
            if len(donated_items) > 0:
                top_ngos = donated_items['nearest_ngo'].value_counts().head(5)
                logger.info("🏢 TOP 5 NGOs BY ACTUAL DONATIONS:")
                logger.info("─" * 50)
                for i, (ngo, count) in enumerate(top_ngos.items(), 1):
                    ngo_name = str(ngo)[:30] if len(str(ngo)) > 30 else str(ngo)
                    logger.info(f"   {i}. {ngo_name:30}: {count:4} donations")
            else:
                logger.info("🏢 No donated items found for NGO analysis")
        
        # Expiry-based action analysis
        if 'days_to_expiry' in df.columns and 'Action' in df.columns:
            logger.info("📅 EXPIRY-BASED ACTION ANALYSIS:")
            logger.info("─" * 50)
            
            # Items to remove (too expired)
            remove_items = df[df['Action'] == 'Remove']
            logger.info(f"🗑️  Items to Remove (too expired):     {len(remove_items):,}")
            
            # Items for donation
            donate_items = df[df['Action'] == 'Donate']
            logger.info(f"🤝 Items for Donation:               {len(donate_items):,}")
            
            # Items for discount
            discount_items = df[df['Action'] == 'Apply Discount']
            logger.info(f"💸 Items for Discount:               {len(discount_items):,}")
            
            # Calculate potential value loss from removed items
            if 'unit_price' in df.columns and 'current_stock' in df.columns:
                removed_value = (remove_items['unit_price'] * remove_items['current_stock']).sum()
                donated_value = (donate_items['unit_price'] * donate_items['current_stock']).sum()
                discount_value = (discount_items['unit_price'] * discount_items['current_stock']).sum()
                
                logger.info(f"💰 Value Analysis:")
                logger.info(f"   💸 Potential Loss (Remove):        ${removed_value:,.2f}")
                logger.info(f"   🤝 Value Recovered (Donate):       ${donated_value:,.2f}")
                logger.info(f"   🏷️  Discount Opportunity:           ${discount_value:,.2f}")
        
        # Use utils module for comprehensive summary if available
        if utils_module and hasattr(utils_module, 'get_donation_summary'):
            try:
                donation_summary = utils_module.get_donation_summary(df)
                logger.info("📊 COMPREHENSIVE DONATION SUMMARY (from utils):")
                logger.info("─" * 50)
                
                # Extract key metrics from utils summary
                eligible_pct = donation_summary.get('donation_eligible_percentage', 0)
                if eligible_pct > 0:
                    logger.info(f"   ✅ Donation eligible percentage: {eligible_pct:.1f}%")
                
                status_counts = donation_summary.get('donation_status_counts', {})
                if status_counts:
                    logger.info(f"   📋 Status distribution: {dict(status_counts)}")
                
                top_ngos_from_summary = donation_summary.get('top_ngos', {})
                if top_ngos_from_summary:
                    logger.info("🏆 Top NGOs from comprehensive analysis:")
                    for ngo, count in list(top_ngos_from_summary.items())[:3]:
                        logger.info(f"   • {ngo}: {count} items")
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not generate comprehensive donation summary: {e}")
        
        logger.info("✅ DONATION DATA ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("═" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error analyzing donation data: {e}")
        logger.exception("Donation analysis error details:")

def save_enhanced_dataset(df: pd.DataFrame, utils_module=None) -> bool:
    """Save the enhanced dataset with donation data and separate output files"""
    try:
        logger.info("💾 SAVING ENHANCED DATASETS...")
        logger.info("═" * 50)
        
        # Save main enhanced dataset
        output_path = PROJECT_ROOT / "data" / "processed" / "inventory_analysis_results_enhanced.csv"
        
        if utils_module and hasattr(utils_module, 'save_updated_inventory'):
            success = utils_module.save_updated_inventory(df, str(output_path))
            if not success:
                logger.warning("⚠️ Utils save failed, falling back to direct pandas save")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Enhanced dataset saved: {output_path}")
        logger.info(f"   📊 Total rows: {len(df):,}")
        
        # Save removed items separately (Action == "Remove")
        if 'Action' in df.columns:
            removed_items_df = df[df['Action'] == 'Remove']
            if len(removed_items_df) > 0:
                removed_items_path = PROJECT_ROOT / "data" / "processed" / "removed_items.csv"
                removed_items_df.to_csv(removed_items_path, index=False)
                logger.info(f"🗑️  Removed items saved: {removed_items_path}")
                logger.info(f"   📊 Items to remove: {len(removed_items_df):,}")
                
                # Calculate total value of removed items
                if 'unit_price' in removed_items_df.columns and 'current_stock' in removed_items_df.columns:
                    total_removed_value = (removed_items_df['unit_price'] * removed_items_df['current_stock']).sum()
                    logger.info(f"   💰 Total value loss: ${total_removed_value:,.2f}")
            else:
                logger.info("✅ No items marked for removal found")
        
        # Save donation summary CSV (only donation-eligible items)
        if 'donation_eligible' in df.columns:
            donation_eligible_df = df[df['donation_eligible'] == True]
            if len(donation_eligible_df) > 0:
                donation_summary_path = PROJECT_ROOT / "data" / "processed" / "donation_summary.csv"
                donation_eligible_df.to_csv(donation_summary_path, index=False)
                logger.info(f"🤝 Donation summary saved: {donation_summary_path}")
                logger.info(f"   📊 Donation eligible items: {len(donation_eligible_df):,}")
                
                # Break down by donation status
                if 'donation_status' in donation_eligible_df.columns:
                    status_breakdown = donation_eligible_df['donation_status'].value_counts()
                    for status, count in status_breakdown.items():
                        if status == 'Pending':
                            logger.info(f"   🟡 Pending: {count:,}")
                        elif status == 'Donated':
                            logger.info(f"   🟢 Donated: {count:,}")
                        elif status == 'Rejected':
                            logger.info(f"   🔴 Rejected: {count:,}")
                        else:
                            logger.info(f"   ⚪ {status}: {count:,}")
            else:
                logger.info("ℹ️ No donation-eligible items found, skipping donation summary CSV")
        else:
            logger.warning("⚠️ No donation_eligible column found, skipping donation summary CSV")
        
        # Save pending donations for NGO coordination
        if 'donation_status' in df.columns and 'donation_eligible' in df.columns:
            pending_donations_df = df[(df['donation_eligible'] == True) & (df['donation_status'] == 'Pending')]
            if len(pending_donations_df) > 0:
                pending_donations_path = PROJECT_ROOT / "data" / "processed" / "pending_donations.csv"
                
                # Include key columns for NGO coordination
                key_columns = ['item_id', 'product_name', 'category', 'current_stock', 'days_to_expiry', 
                              'city', 'nearest_ngo', 'ngo_contact', 'ngo_address', 'donation_status']
                available_columns = [col for col in key_columns if col in pending_donations_df.columns]
                
                pending_donations_df[available_columns].to_csv(pending_donations_path, index=False)
                logger.info(f"🟡 Pending donations saved: {pending_donations_path}")
                logger.info(f"   📊 Pending items: {len(pending_donations_df):,}")
                
                # Top NGOs for pending donations
                if 'nearest_ngo' in pending_donations_df.columns:
                    top_ngos_pending = pending_donations_df['nearest_ngo'].value_counts().head(3)
                    logger.info("   🏢 Top NGOs for pending donations:")
                    for i, (ngo, count) in enumerate(top_ngos_pending.items(), 1):
                        ngo_name = str(ngo)[:25] if len(str(ngo)) > 25 else str(ngo)
                        logger.info(f"      {i}. {ngo_name}: {count} items")
        
        logger.info("✅ ALL DATASETS SAVED SUCCESSFULLY")
        logger.info("═" * 50)
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving enhanced dataset: {e}")
        logger.exception("Enhanced dataset save error:")
        return False

def validate_donation_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate donation data integrity and completeness
    Returns: (is_valid, error_message)
    """
    try:
        # Check required columns exist
        required_columns = ['donation_eligible', 'donation_status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required donation columns: {missing_columns}"
        
        # Check data types
        if df['donation_eligible'].dtype != bool:
            # Try to convert if it's 0/1 or True/False strings
            try:
                df['donation_eligible'] = df['donation_eligible'].astype(bool)
            except:
                return False, "donation_eligible column is not boolean and cannot be converted"
        
        # Check donation status values
        valid_statuses = ['', 'Pending', 'Donated', 'Cancelled']
        invalid_statuses = df[~df['donation_status'].isin(valid_statuses)]['donation_status'].unique()
        
        if len(invalid_statuses) > 0:
            logger.warning(f"⚠️ Found unexpected donation statuses: {list(invalid_statuses)}")
        
        # Check for logical consistency
        eligible_count = df['donation_eligible'].sum()
        pending_donated_count = df[df['donation_status'].isin(['Pending', 'Donated'])].shape[0]
        
        if pending_donated_count > eligible_count:
            logger.warning("⚠️ More items with Pending/Donated status than eligible items")
        
        # Check NGO data completeness for donated items
        if 'nearest_ngo' in df.columns:
            donated_items = df[df['donation_status'] == 'Donated']
            missing_ngo_data = donated_items['nearest_ngo'].isna().sum()
            
            if missing_ngo_data > 0:
                logger.warning(f"⚠️ {missing_ngo_data} donated items missing NGO information")
        
        logger.info("✅ Donation data validation passed")
        return True, "Validation successful"
        
    except Exception as e:
        return False, f"Validation error: {e}"

def display_donation_metrics(df: pd.DataFrame) -> None:
    """Display comprehensive donation metrics in a formatted way"""
    try:
        logger.info("🤝 DONATION METRICS DASHBOARD")
        logger.info("═" * 60)
        
        total_items = len(df)
        
        if 'donation_eligible' in df.columns:
            eligible_items = df['donation_eligible'].sum()
            eligible_percentage = (eligible_items / total_items * 100) if total_items > 0 else 0
            logger.info(f"📦 Total Items:              {total_items:,}")
            logger.info(f"🤝 Donation Eligible:        {eligible_items:,} ({eligible_percentage:.1f}%)")
        
        if 'donation_status' in df.columns:
            status_counts = df['donation_status'].value_counts()
            logger.info("📋 DONATION STATUS BREAKDOWN:")
            logger.info("─" * 30)
            
            for status in ['Pending', 'Donated', 'Rejected', 'Cancelled']:
                count = status_counts.get(status, 0)
                percentage = (count / total_items * 100) if total_items > 0 else 0
                if status == 'Pending':
                    logger.info(f"   🟡 {status:10}: {count:6,} ({percentage:4.1f}%)")
                elif status == 'Donated':
                    logger.info(f"   🟢 {status:10}: {count:6,} ({percentage:4.1f}%)")
                elif status == 'Rejected':
                    logger.info(f"   🔴 {status:10}: {count:6,} ({percentage:4.1f}%)")
                else:
                    logger.info(f"   ⚪ {status:10}: {count:6,} ({percentage:4.1f}%)")
        
        # Action analysis
        if 'Action' in df.columns:
            action_counts = df['Action'].value_counts()
            logger.info("⚡ ACTION BREAKDOWN:")
            logger.info("─" * 30)
            
            action_emojis = {
                'Remove': '🗑️',
                'Donate': '🤝',
                'Apply Discount': '💸',
                'Restock': '📦',
                'No Action': '✅'
            }
            
            for action in ['Remove', 'Donate', 'Apply Discount', 'Restock', 'No Action']:
                count = action_counts.get(action, 0)
                percentage = (count / total_items * 100) if total_items > 0 else 0
                emoji = action_emojis.get(action, '📋')
                logger.info(f"   {emoji} {action:15}: {count:6,} ({percentage:4.1f}%)")
        
        # Category analysis for donations
        if 'category' in df.columns and 'donation_eligible' in df.columns:
            eligible_by_category = df[df['donation_eligible']]['category'].value_counts().head(5)
            if len(eligible_by_category) > 0:
                logger.info("🏷️ TOP 5 CATEGORIES (Donation Eligible):")
                logger.info("─" * 30)
                for i, (category, count) in enumerate(eligible_by_category.items(), 1):
                    logger.info(f"   {i}. {category}: {count:,} items")
        
        # Geographic distribution
        if 'city' in df.columns and 'donation_eligible' in df.columns:
            eligible_by_city = df[df['donation_eligible']]['city'].value_counts().head(5)
            if len(eligible_by_city) > 0:
                logger.info("🏙️ TOP 5 CITIES (Donation Eligible):")
                logger.info("─" * 30)
                for i, (city, count) in enumerate(eligible_by_city.items(), 1):
                    logger.info(f"   {i}. {city}: {count:,} items")
        
        logger.info("═" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error displaying donation metrics: {e}")

def run_restock_plan(generate_restock_plan_module) -> bool:
    """Phase 4: Generate smart restocking plan using trained models."""
    try:
        logger.info("📦 Phase 4: Generating smart restock plan...")

        if generate_restock_plan_module is None:
            logger.warning("⚠️ generate_restock_plan module not available, skipping restock plan")
            return False

        if hasattr(generate_restock_plan_module, 'RestockPlanGenerator'):
            generator = generate_restock_plan_module.RestockPlanGenerator()
            df_restock, restock_summary = generator.run()

            logger.info("✅ Restock plan generated successfully")
            logger.info(f"   📊 Total items evaluated:   {restock_summary.get('total_items', 0):,}")
            logger.info(f"   📦 Items to restock:        {restock_summary.get('items_to_restock', 0):,}")
            logger.info(f"   💸 Items for discount:      {restock_summary.get('items_for_discount', 0):,}")
            logger.info(f"   ⭐ High-priority items:     {restock_summary.get('high_priority_items', 0):,}")
            logger.info(f"   📈 Avg expiry risk:         {restock_summary.get('avg_expiry_risk', 0):.3f}")
            return True
        else:
            logger.error("❌ generate_restock_plan module missing 'RestockPlanGenerator' class")
            return False

    except Exception as e:
        logger.error(f"❌ Error generating restock plan: {e}")
        logger.exception("Restock plan error details:")
        return False


def main() -> bool:
    """Main pipeline execution function"""
    logger.info("🚀 Starting Smart Inventory Management Pipeline")
    logger.info("=" * 60)

    try:
        # Import required modules
        (train_expiry_model, inventory_analyzer, data_preprocessing,
         transform_inventory_data, utils, train_demand_model,
         generate_restock_plan) = import_modules()
        
        # Ensure directory structure
        if not ensure_directories():
            logger.error("❌ Failed to create required directories")
            return False
        
        # Phase 0: Data Preprocessing
        logger.info("🔧 Phase 0: Ensuring data is prepared...")
        if not run_data_preprocessing(data_preprocessing):
            logger.error("❌ Data preprocessing failed")
            return False
        
        # Phase 1: Expiry Risk Prediction
        if not run_expiry_prediction(train_expiry_model):
            return False
        
        # Phase 1.5: Demand Forecasting Model
        logger.info("=" * 60)
        logger.info("📈 Phase 1.5: Training demand forecasting model...")
        demand_ok = run_demand_model(train_demand_model)
        if not demand_ok:
            logger.warning("⚠️ Demand model training failed or skipped — continuing pipeline")
        
        # Phase 2: Inventory Analysis
        success, df, summary = run_inventory_analysis(inventory_analyzer)
        if not success:
            return False
        
        # Phase 3: Dashboard Data
        if df is not None:
            if not save_dashboard_data(df):
                logger.warning("⚠️ Dashboard data save failed, but continuing...")
        
        # Phase 3.5: Donation Data Processing and Analysis
        logger.info("🤝 PHASE 3.5: PROCESSING AND ANALYZING DONATION DATA...")
        logger.info("═" * 70)
        donation_success, donation_df = run_donation_data_processing(transform_inventory_data)
        if donation_success and donation_df is not None:
            # Validate donation data
            is_valid, validation_message = validate_donation_data(donation_df)
            if not is_valid:
                logger.error(f"❌ Donation data validation failed: {validation_message}")
                return False
            
            analyze_donation_data(donation_df, utils)
            
            # Save enhanced dataset with donation data
            if save_enhanced_dataset(donation_df, utils):
                logger.info("✅ Enhanced dataset with donation data saved successfully")
            else:
                logger.warning("⚠️ Failed to save enhanced dataset")
                
            # Log final donation integration summary
            eligible_count = donation_df['donation_eligible'].sum() if 'donation_eligible' in donation_df.columns else 0
            pending_count = len(donation_df[donation_df['donation_status'] == 'Pending']) if 'donation_status' in donation_df.columns else 0
            donated_count = len(donation_df[donation_df['donation_status'] == 'Donated']) if 'donation_status' in donation_df.columns else 0
            removed_count = len(donation_df[donation_df['Action'] == 'Remove']) if 'Action' in donation_df.columns else 0
            
            logger.info("🎯 FINAL DONATION INTEGRATION SUMMARY:")
            logger.info("═" * 50)
            logger.info(f"   📊 Total items processed:       {len(donation_df):,}")
            logger.info(f"   🤝 Donation eligible items:     {eligible_count:,}")
            logger.info(f"   🟡 Pending donations:           {pending_count:,}")
            logger.info(f"   🟢 Successfully donated:        {donated_count:,}")
            logger.info(f"   🗑️  Items marked for removal:    {removed_count:,}")
            logger.info(f"   💾 Enhanced dataset saved to:   data/processed/")
            logger.info("═" * 50)
        else:
            logger.warning("⚠️ Donation data processing failed, continuing without donation features")
            logger.warning("   This may be due to missing donation modules or data files")
        
        # Display results
        logger.info("🎯 ALL PHASES EXECUTED SUCCESSFULLY!")
        logger.info("═" * 70)
        if summary:
            display_summary(summary)

        # Phase 4: Restock Plan Generation
        logger.info("=" * 60)
        logger.info("📦 Phase 4: Generating smart restock plan...")
        restock_ok = run_restock_plan(generate_restock_plan)
        if not restock_ok:
            logger.warning("⚠️ Restock plan generation failed or skipped — continuing pipeline")

        # ML model accuracy summary
        logger.info("🤖 ML MODEL ACCURACY SUMMARY:")
        logger.info("─" * 40)
        logger.info("   ✅ Expiry Classifier  — see [Phase 1]  log above for CV accuracy")
        logger.info("   ✅ Demand Forecaster  — see [Phase 1.5] log above for CV R²")

        # Final output summary
        logger.info("📋 OUTPUT FILES GENERATED:")
        logger.info("─" * 40)
        logger.info("   📄 inventory_analysis_results_enhanced.csv")
        logger.info("   📄 donation_summary.csv (donation eligible items)")
        logger.info("   📄 removed_items.csv (items marked for removal)")
        logger.info("   📄 pending_donations.csv (items pending donation)")
        logger.info("   📄 dashboard_data.csv (for dashboard)")
        logger.info("   📄 restocking_suggestions.csv (restock plan)")
        logger.info("   📄 high_priority_restock.csv (urgent restock items)")

        logger.info("🌐 NEXT STEPS:")
        logger.info("─" * 40)
        logger.info("   1. Run dashboard: streamlit run dashboard/app.py")
        logger.info("   2. Review removed items for disposal")
        logger.info("   3. Coordinate with NGOs for pending donations")
        logger.info("   4. Apply discounts to near-expiry items")
        logger.info("   5. Execute restock plan from restocking_suggestions.csv")
        logger.info("═" * 70)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("⚠️ Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected pipeline error: {e}")
        logger.exception("Pipeline error details:")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        
        if success:
            logger.info("🎉 Pipeline execution completed successfully!")
        else:
            logger.error("❌ Pipeline execution failed")
        
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        logger.exception("Critical error details:")
        sys.exit(1)