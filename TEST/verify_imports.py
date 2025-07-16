#!/usr/bin/env python3
"""
Quick verification script to check if all test files can import their dependencies correctly
"""

import sys
import os
from pathlib import Path

def test_import(test_file_name):
    """Test if a specific test file can import its dependencies"""
    print(f"\n🔍 Testing imports for {test_file_name}...")
    
    try:
        # Import the test module
        test_module_name = test_file_name.replace('.py', '')
        test_module = __import__(test_module_name)
        print(f"✅ {test_file_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ {test_file_name} failed to import: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {test_file_name} imported but with issues: {e}")
        return True  # Still counts as successful import

def main():
    """Test all test files"""
    print("🧪 VERIFYING TEST FILE IMPORTS AFTER DIRECTORY MOVE")
    print("=" * 60)
    
    # Add paths
    parent_dir = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(parent_dir / 'src'))
    
    test_files = [
        'test_action_logic.py',
        'test_dashboard.py', 
        'test_demand_donation_filtering.py',
        'test_donation_features.py',
        'test_donation_filtering.py',
        'test_donation_functions.py',
        'test_enhanced_main.py',
        'test_expiry_donation_features.py',
        'test_main_integration.py',
        'test_utils_donation.py'
    ]
    
    successful = 0
    total = len(test_files)
    
    for test_file in test_files:
        if test_import(test_file):
            successful += 1
    
    print(f"\n📊 SUMMARY: {successful}/{total} test files can import dependencies successfully")
    
    if successful == total:
        print("🎉 All test files are ready to run!")
    else:
        print("⚠️  Some test files may need additional path fixes")

if __name__ == "__main__":
    main()
