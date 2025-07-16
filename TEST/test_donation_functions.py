#!/usr/bin/env python3
"""
Test script to verify the new donation analysis functions work correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inventory_analyzer import InventoryAnalyzer
import pandas as pd

def test_donation_functions():
    """Test the new donation analysis functions"""
    print("🧪 Testing donation analysis functions...")
    
    # Initialize analyzer
    analyzer = InventoryAnalyzer()
    
    # Load or generate sample data
    df = analyzer.load_inventory_data()
    print(f"✅ Data loaded: {len(df)} items")
    
    # Run basic analysis to prepare data
    df = analyzer.analyze_stock_levels(df)
    df = analyzer.analyze_expiry_risk(df)
    
    # Test donation summary
    print("\n📊 Testing get_donation_summary()...")
    donation_summary = analyzer.get_donation_summary(df)
    print(f"   Total donation-eligible items: {donation_summary['total_donation_eligible']}")
    print(f"   Status counts: {donation_summary['donation_status_counts']}")
    print(f"   Top NGOs: {list(donation_summary['top_ngos'].keys())[:3] if donation_summary['top_ngos'] else 'None'}")
    
    # Test pending donations
    print("\n⏳ Testing get_pending_donations()...")
    pending_donations = analyzer.get_pending_donations(df)
    print(f"   Found {len(pending_donations)} pending donation items")
    if len(pending_donations) > 0:
        print("   Sample pending donations:")
        print(pending_donations[['product_name', 'days_to_expiry', 'city', 'nearest_ngo']].head(3).to_string(index=False))
    
    # Test category analysis
    print("\n📈 Testing analyze_donation_by_category()...")
    category_analysis = analyzer.analyze_donation_by_category(df)
    print(f"   Category analysis shape: {category_analysis.shape}")
    if len(category_analysis) > 0:
        print("   Sample category breakdown:")
        print(category_analysis.head(5)[['category', 'donation_status', 'item_count']].to_string(index=False))
    
    print("\n✅ All donation functions tested successfully!")
    return True

if __name__ == "__main__":
    try:
        test_donation_functions()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
