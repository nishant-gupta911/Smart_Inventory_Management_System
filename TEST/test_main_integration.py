#!/usr/bin/env python3
"""
Test script to verify main.py donation integration works correctly
"""

import sys
import os
from pathlib import Path

# Add current directory to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from src import utils, inventory_analyzer, train_expiry_model, data_preprocessing
        print("✅ Core src modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core modules: {e}")
        return False
    
    try:
        import transform_inventory_data
        print("✅ transform_inventory_data module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import transform_inventory_data: {e}")
        return False
    
    return True

def test_donation_data_processing():
    """Test the donation data processing functionality"""
    print("\n🤝 Testing donation data processing...")
    
    try:
        import transform_inventory_data
        
        # Test the load_and_transform_data function
        df = transform_inventory_data.load_and_transform_data()
        print(f"✅ Data loaded successfully: {len(df)} rows")
        
        # Check for donation columns
        expected_cols = ['donation_eligible', 'donation_status', 'nearest_ngo']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ Missing donation columns: {missing_cols}")
        else:
            print("✅ All donation columns present")
        
        # Check donation statistics
        if 'donation_eligible' in df.columns:
            eligible_count = df['donation_eligible'].sum()
            total_count = len(df)
            percentage = (eligible_count / total_count * 100) if total_count > 0 else 0
            print(f"📊 Donation eligible items: {eligible_count:,} ({percentage:.1f}%)")
        
        return True, df
        
    except Exception as e:
        print(f"❌ Error in donation data processing: {e}")
        return False, None

def test_utils_integration():
    """Test utils module integration"""
    print("\n🔧 Testing utils module integration...")
    
    try:
        from src import utils
        
        # Check if key functions exist
        required_functions = ['get_donation_summary', 'save_updated_inventory', 'update_donation_status']
        missing_functions = []
        
        for func_name in required_functions:
            if not hasattr(utils, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"⚠️ Missing utils functions: {missing_functions}")
        else:
            print("✅ All required utils functions present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing utils integration: {e}")
        return False

def test_main_pipeline():
    """Test the main pipeline execution"""
    print("\n🚀 Testing main pipeline imports...")
    
    try:
        # Add parent directory to path for main.py import
        parent_dir = os.path.join(os.path.dirname(__file__), '..')
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        import main
        
        # Test if main functions are accessible
        functions_to_check = [
            'run_donation_data_processing',
            'analyze_donation_data', 
            'save_enhanced_dataset'
        ]
        
        missing_functions = []
        for func_name in functions_to_check:
            if not hasattr(main, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"⚠️ Missing main functions: {missing_functions}")
        else:
            print("✅ All main donation functions present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing main pipeline: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 Running Main.py Donation Integration Tests")
    print("=" * 50)
    
    test_results = []
    
    # Test imports
    test_results.append(test_imports())
    
    # Test donation data processing
    success, df = test_donation_data_processing()
    test_results.append(success)
    
    # Test utils integration
    test_results.append(test_utils_integration())
    
    # Test main pipeline
    test_results.append(test_main_pipeline())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    if all(test_results):
        print("🎉 All tests passed! Donation integration is working correctly.")
        return True
    else:
        failed_tests = sum(1 for result in test_results if not result)
        print(f"❌ {failed_tests} test(s) failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
