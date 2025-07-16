#!/usr/bin/env python3
"""
Simple test script to verify the dashboard functionality
"""

import pandas as pd
import os
import sys

# Add the dashboard directory to the path
dashboard_dir = os.path.join(os.path.dirname(__file__), '..', 'dashboard')
sys.path.append(dashboard_dir)

def test_data_loading():
    """Test if the enhanced dataset can be loaded"""
    try:
        # Check if enhanced dataset exists
        enhanced_path = os.path.join(os.path.dirname(__file__), "data", "processed", "inventory_analysis_results_enhanced.csv")
        
        if os.path.exists(enhanced_path):
            df = pd.read_csv(enhanced_path)
            print(f"✅ Enhanced dataset loaded successfully: {len(df)} rows")
            
            # Check for donation columns
            donation_cols = ['donation_eligible', 'donation_status', 'nearest_ngo', 'ngo_contact']
            missing_cols = [col for col in donation_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️  Missing donation columns: {missing_cols}")
            else:
                print("✅ All donation columns present")
            
            # Check donation data
            eligible_items = df[df['donation_eligible'] == True]
            print(f"📊 Donation-eligible items: {len(eligible_items)}")
            
            if len(eligible_items) > 0:
                status_counts = eligible_items['donation_status'].value_counts()
                print("📈 Donation status distribution:")
                for status, count in status_counts.items():
                    print(f"   {status}: {count}")
                
                # Show sample NGOs
                if 'nearest_ngo' in eligible_items.columns:
                    unique_ngos = eligible_items['nearest_ngo'].nunique()
                    print(f"🏢 Unique NGOs: {unique_ngos}")
                
                # Show sample cities
                if 'city' in eligible_items.columns:
                    unique_cities = eligible_items['city'].nunique()
                    print(f"🏙️  Unique cities: {unique_cities}")
                
                return True
            else:
                print("⚠️  No donation-eligible items found")
                return False
        else:
            print(f"❌ Enhanced dataset not found at: {enhanced_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data loading: {str(e)}")
        return False

def test_dashboard_imports():
    """Test if all required packages are available"""
    try:
        import streamlit
        import plotly.express
        import plotly.graph_objects
        print("✅ All required packages available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Dashboard Functionality")
    print("=" * 50)
    
    # Test 1: Package imports
    print("\n1. Testing package imports...")
    imports_ok = test_dashboard_imports()
    
    # Test 2: Data loading
    print("\n2. Testing data loading...")
    data_ok = test_data_loading()
    
    # Summary
    print("\n" + "=" * 50)
    if imports_ok and data_ok:
        print("✅ All tests passed! Dashboard should work correctly.")
        print("\nTo run the dashboard:")
        print("cd dashboard")
        print("streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        if not imports_ok:
            print("Install missing packages with: pip install streamlit plotly")
        if not data_ok:
            print("Run the main pipeline to generate the enhanced dataset")

if __name__ == "__main__":
    main()
