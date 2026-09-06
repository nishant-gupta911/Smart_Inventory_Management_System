"""
Convert CSV data files to JSON format for React frontend integration
This script reads the processed inventory data and creates optimized JSON for the dashboard
Uses only built-in libraries (csv, json) for maximum compatibility
"""

import csv
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# Paths
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
# Frontend is in a sibling directory, not inside this project
OUTPUT_DIR = Path(__file__).parent.parent / "smart_inventory_frontend" / "src" / "services"

def load_csv(filepath):
    """Load CSV file and return list of dictionaries"""
    rows = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        print(f"✓ Loaded {filepath.name}: {len(rows)} rows")
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
    return rows

def create_inventory_items(items_list, transactions_list):
    """Create inventory items for dashboard"""
    print("\n🏭 Creating inventory items...")
    
    items_data = []
    avg_transaction = int(sum(int(t['transactions']) for t in transactions_list) / max(len(transactions_list), 1)) if transactions_list else 100
    
    # Use first 50 items
    for idx, item in enumerate(items_list[:50]):
        item_id = int(item['item_id'])
        family = item['family']
        perishable = int(item['perishable'])
        shelf_life = int(item['shelf_life'])
        
        quantity = random.randint(10, avg_transaction * 2)
        reorder_level = max(10, int(quantity * 0.2))
        
        # Status based on quantity
        if quantity == 0:
            status = "out-of-stock"
        elif quantity < reorder_level:
            status = "low-stock"
        else:
            status = "in-stock"
        
        items_data.append({
            "id": item_id,
            "sku": f"ITEM-{item_id:05d}",
            "name": f"{family} Product {item_id}",
            "category": family,
            "quantity": quantity,
            "reorderLevel": reorder_level,
            "unitPrice": round(random.uniform(5, 500), 2),
            "lastRestockDate": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "status": status,
            "perishable": bool(perishable),
            "shelfLife": shelf_life,
            "expiryRisk": perishable == 1 and random.random() > 0.7
        })
    
    print(f"   Created {len(items_data)} inventory items")
    return items_data

def create_suppliers(stores_list):
    """Create suppliers from stores data"""
    print("🚚 Creating suppliers...")
    
    suppliers_data = []
    cities = list(set([s['city'] for s in stores_list]))[:10]  # Top 10 unique cities
    
    for idx, city in enumerate(cities):
        suppliers_data.append({
            "id": f"SUP-{idx+1:03d}",
            "name": f"{city} Distribution Center",
            "email": f"supplier-{idx+1}@inventory.local",
            "phone": f"+593 {random.randint(1000000, 9999999)}",
            "leadTime": random.randint(1, 14),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "activeOrders": random.randint(0, 20)
        })
    
    print(f"   Created {len(suppliers_data)} suppliers")
    return suppliers_data

def create_orders(transactions_list, stores_list):
    """Create orders from transaction data"""
    print("📦 Creating orders...")
    
    orders_data = []
    statuses = ["pending", "shipped", "delivered"]
    
    # Use last 50 transactions as orders
    for idx, trans in enumerate(transactions_list[-50:]):
        store_nbr = int(trans['store_nbr'])
        date_str = trans['date']
        
        try:
            date = datetime.fromisoformat(date_str)
        except:
            date = datetime.now()
        
        orders_data.append({
            "id": f"ORD-{idx+1:06d}",
            "supplierId": f"SUP-{random.randint(1, 10):03d}",
            "storeId": store_nbr,
            "itemsCount": random.randint(5, 50),
            "orderDate": date.isoformat(),
            "expectedDelivery": (date + timedelta(days=random.randint(3, 14))).isoformat(),
            "totalCost": round(random.uniform(500, 50000), 2),
            "status": statuses[random.randint(0, 2)]
        })
    
    print(f"   Created {len(orders_data)} orders")
    return orders_data

def create_alerts(items_list, expiry_list):
    """Create alerts from inventory and expiry data"""
    print("⚠️ Creating alerts...")
    
    alerts_data = []
    
    # Create alerts from items with high expiry risk
    risk_count = min(20, len(expiry_list))
    for idx, expiry_row in enumerate(expiry_list[:risk_count]):
        if expiry_row.get('expiry_risk') == '1.0' or expiry_row.get('expiry_risk') == '1':
            days_to_expiry = int(expiry_row.get('days_to_expiry', 30))
            severity = "critical" if days_to_expiry < 7 else "warning"
            
            alerts_data.append({
                "id": f"ALT-{idx+1:05d}",
                "type": severity,
                "message": f"Expiry risk - {days_to_expiry} days remaining",
                "itemReference": f"ITEM-{int(expiry_row.get('store_nbr', 1001)):05d}",
                "timestamp": datetime.now().isoformat(),
                "dismissed": False,
                "severity": severity
            })
    
    # Add info alert
    alerts_data.append({
        "id": "ALT-99999",
        "type": "info",
        "message": "Inventory sync completed successfully",
        "timestamp": datetime.now().isoformat(),
        "dismissed": False,
        "severity": "info"
    })
    
    print(f"   Created {len(alerts_data)} alerts")
    return alerts_data

def create_charts_data(transactions_list, items_list):
    """Create data for charts"""
    print("📊 Creating chart data...")
    
    # Stock trend - aggregate transactions by unique dates
    stock_trend_data = []
    date_totals = {}
    
    for trans in transactions_list[-30:]:
        date = trans['date']
        trans_count = int(trans['transactions'])
        if date not in date_totals:
            date_totals[date] = 0
        date_totals[date] += trans_count
    
    for date in sorted(date_totals.keys()):
        stock_trend_data.append({
            "date": date,
            "totalValue": int(date_totals[date] * random.uniform(10, 50))
        })
    
    # Category distribution from items
    category_totals = {}
    for item in items_list[:50]:
        family = item['family']
        if family not in category_totals:
            category_totals[family] = 0
        category_totals[family] += random.randint(5000, 20000)
    
    colors = ["#00D4FF", "#00E676", "#F5A623", "#FF3D57", "#A855F7", "#EC4899"]
    category_data = [
        {
            "name": cat,
            "value": value,
            "color": colors[idx % len(colors)]
        }
        for idx, (cat, value) in enumerate(category_totals.items())
    ]
    
    return {
        "stockTrend": stock_trend_data,
        "categoryDistribution": category_data
    }

def main():
    """Main conversion function"""
    print("=" * 60)
    print("🔄 Converting CSV data to JSON for React Frontend")
    print("=" * 60)
    print()
    
    # Load data files
    print("📂 Loading data files...")
    items_list = load_csv(RAW_DIR / "items.csv")
    stores_list = load_csv(RAW_DIR / "stores.csv")
    transactions_list = load_csv(RAW_DIR / "transactions.csv")
    expiry_list = load_csv(PROCESSED_DIR / "expiry_risk_predictions.csv")
    
    if not all([items_list, stores_list, transactions_list]):
        print("\n✗ Error: Missing required data files")
        return
    
    # Create datasets
    inventory_items = create_inventory_items(items_list, transactions_list)
    suppliers = create_suppliers(stores_list)
    orders = create_orders(transactions_list, stores_list)
    alerts = create_alerts(items_list, expiry_list)
    charts = create_charts_data(transactions_list, items_list)
    
    print("\n📊 Creating additional metrics...")
    mock_data = {
        "mockInventory": inventory_items,
        "mockSuppliers": suppliers,
        "mockOrders": orders,
        "mockAlerts": alerts,
        "mockStockTrendData": charts["stockTrend"],
        "mockCategoryData": charts["categoryDistribution"],
        "mockDemandForecast": [
            {
                "date": (datetime.now() + timedelta(days=i)).isoformat(),
                "quantity": random.randint(1000, 5000),
                "confidence": round(random.uniform(0.75, 0.95), 2)
            }
            for i in range(7)
        ],
        "mockTurnoverData": [
            {"name": item["category"], "value": random.randint(1, 8)}
            for item in inventory_items[:6]
        ],
        "mockFastMovingItems": inventory_items[:5],
        "mockDeadStockItems": [
            {
                "sku": item["sku"],
                "name": item["name"],
                "category": item["category"],
                "daysInactive": random.randint(60, 180),
                "blockedValue": round(item["unitPrice"] * item["quantity"], 2)
            }
            for item in inventory_items[10:16]
        ]
    }
    
    # Save to JSON file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "realData.json"
    
    with open(output_file, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"\n✅ Data conversion complete!")
    print(f"\n📄 Generated: {output_file}")
    print(f"   ├─ Inventory items: {len(inventory_items)}")
    print(f"   ├─ Suppliers: {len(suppliers)}")
    print(f"   ├─ Orders: {len(orders)}")
    print(f"   ├─ Alerts: {len(alerts)}")
    print(f"   ├─ Stock trend points: {len(charts['stockTrend'])}")
    print(f"   └─ Categories: {len(charts['categoryDistribution'])}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
