#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced main.py functionality
"""
import pandas as pd
import sys
from pathlib import Path

# Add project path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def test_enhanced_main():
    """Test the enhanced donation processing logic"""
    print("🚀 TESTING ENHANCED MAIN.PY FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Import the transform module
        import transform_inventory_data
        print("✅ Successfully imported transform_inventory_data")
        
        # Test the load and transform function
        if hasattr(transform_inventory_data, 'load_and_transform_data'):
            df = transform_inventory_data.load_and_transform_data()
            
            if df is not None and len(df) > 0:
                print(f"✅ Data loaded successfully: {len(df):,} rows")
                
                # Test Action column analysis
                if 'Action' in df.columns:
                    action_counts = df['Action'].value_counts()
                    print("\n⚡ ACTION DISTRIBUTION:")
                    print("-" * 30)
                    for action, count in action_counts.items():
                        if action == 'Remove':
                            print(f"🗑️  {action:15}: {count:6,}")
                        elif action == 'Donate':
                            print(f"🤝 {action:15}: {count:6,}")
                        elif action == 'Apply Discount':
                            print(f"💸 {action:15}: {count:6,}")
                        elif action == 'Restock':
                            print(f"📦 {action:15}: {count:6,}")
                        else:
                            print(f"✅ {action:15}: {count:6,}")
                
                # Test donation analysis
                if 'donation_eligible' in df.columns:
                    eligible_count = df['donation_eligible'].sum()
                    print(f"\n🤝 Donation eligible items: {eligible_count:,}")
                
                if 'donation_status' in df.columns:
                    status_counts = df['donation_status'].value_counts()
                    print("\n📋 DONATION STATUS:")
                    print("-" * 30)
                    for status, count in status_counts.items():
                        if status == 'Pending':
                            print(f"🟡 {status:10}: {count:6,}")
                        elif status == 'Donated':
                            print(f"🟢 {status:10}: {count:6,}")
                        elif status == 'Rejected':
                            print(f"🔴 {status:10}: {count:6,}")
                        else:
                            print(f"⚪ {status:10}: {count:6,}")
                
                # Test city analysis
                if 'city' in df.columns and 'donation_status' in df.columns:
                    donated_items = df[df['donation_status'] == 'Donated']
                    if len(donated_items) > 0:
                        top_cities = donated_items['city'].value_counts().head(5)
                        print("\n🏙️ TOP 5 CITIES (Actual Donations):")
                        print("-" * 35)
                        for i, (city, count) in enumerate(top_cities.items(), 1):
                            print(f"   {i}. {city:15}: {count:4} donations")
                
                # Test NGO analysis
                if 'nearest_ngo' in df.columns and 'donation_status' in df.columns:
                    donated_items = df[df['donation_status'] == 'Donated']
                    if len(donated_items) > 0:
                        top_ngos = donated_items['nearest_ngo'].value_counts().head(5)
                        print("\n🏢 TOP 5 NGOs (Actual Donations):")
                        print("-" * 35)
                        for i, (ngo, count) in enumerate(top_ngos.items(), 1):
                            ngo_name = str(ngo)[:25]
                            print(f"   {i}. {ngo_name:25}: {count:4} donations")
                
                # Test save functionality
                print(f"\n💾 Testing save functionality...")
                try:
                    output_dir = PROJECT_ROOT / "data" / "processed"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save main dataset
                    main_path = output_dir / "test_enhanced_results.csv"
                    df.to_csv(main_path, index=False)
                    print(f"✅ Main dataset saved: {main_path}")
                    
                    # Save removed items
                    if 'Action' in df.columns:
                        removed_items = df[df['Action'] == 'Remove']
                        if len(removed_items) > 0:
                            removed_path = output_dir / "test_removed_items.csv"
                            removed_items.to_csv(removed_path, index=False)
                            print(f"🗑️  Removed items saved: {removed_path} ({len(removed_items)} items)")
                    
                    # Save donation summary
                    if 'donation_eligible' in df.columns:
                        donation_items = df[df['donation_eligible'] == True]
                        if len(donation_items) > 0:
                            donation_path = output_dir / "test_donation_summary.csv"
                            donation_items.to_csv(donation_path, index=False)
                            print(f"🤝 Donation summary saved: {donation_path} ({len(donation_items)} items)")
                    
                    print("✅ All save operations completed successfully")
                    
                except Exception as e:
                    print(f"❌ Save error: {e}")
                
                print("\n🎯 TEST COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                
            else:
                print("❌ Failed to load data")
        else:
            print("❌ load_and_transform_data function not found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_main()
