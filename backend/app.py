"""
Smart Inventory Management System — FastAPI Backend
Deployed on Render | Frontend on Vercel

Endpoints mirror the logic in dashboard/app.py.
Data: uses the same seeded mock generation as the React frontend's api.js
      (real CSV is 700MB — not suitable for Render free tier).
      Swap generate_data() with pd.read_csv() when you have a DB/storage.
"""

import os
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Inventory API",
    description="AI-powered inventory management — Sparkathon 2025",
    version="1.0.0",
)

# ─── CORS — allow Vercel frontend (and local dev) ─────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data generation (mirrors api.js logic + app.py fallback) ─────────────────
CATEGORIES  = ["DAIRY", "PRODUCE", "MEATS", "SNACKS", "BAKERY", "BEVERAGES", "HOUSEHOLD"]
ACTIONS     = ["No Action", "Apply Discount", "Restock", "Remove", "Donate"]
STOCK_LVLS  = ["High", "Normal", "Low"]
EXPIRY_RISK = ["Safe", "Near Expiry", "Expired"]
STORES      = [str(i) for i in range(1, 11)]
CITIES      = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad"]
NGOS        = ["Akshaya Patra", "Robin Hood Army", "Feeding India", "No Food Waste"]
PRODUCTS    = [
    "Organic Bananas", "Whole Milk 2%", "Sliced Bread", "Greek Yogurt", "Baby Spinach",
    "Orange Juice", "Cheddar Block", "Chicken Breast", "Brown Rice 5kg", "Cooking Oil",
    "Paper Towels", "Cereal Box", "Bottled Water", "Potato Chips", "Frozen Peas",
    "Almond Milk", "Butter 500g", "Eggs 12pk", "Pasta 500g", "Tomato Sauce",
    "Laundry Detergent", "Shampoo 400ml", "Instant Noodles", "Coffee 250g", "Orange 1kg",
    "Apple Juice", "Strawberry Jam", "Peanut Butter", "Canned Tuna", "Olive Oil",
]


def _seeded(i: int, offset: int) -> float:
    x = math.sin(i * 37 + offset + 1) * 10000
    return x - math.floor(x)


def _choice(lst, i, offset):
    return lst[int(_seeded(i, offset) * len(lst))]


def generate_data(n: int = 300) -> pd.DataFrame:
    rows = []
    for i in range(n):
        stock      = int(_seeded(i, 6) * 200)
        days_exp   = int(_seeded(i, 7) * 31)
        unit_price = round(_seeded(i, 8) * 49.5 + 0.5, 2)
        sales_avg  = round(_seeded(i, 9) * 20, 1)
        city       = _choice(CITIES, i, 4)
        ngo        = _choice(NGOS,   i, 5)
        product    = _choice(PRODUCTS,   i, 3)
        category   = _choice(CATEGORIES, i, 2)
        store      = _choice(STORES,     i, 1)

        stock_level = "Low" if stock < 20 else ("High" if stock > 120 else "Normal")
        expiry_risk = "Expired" if days_exp == 0 else ("Near Expiry" if days_exp <= 3 else "Safe")

        action, discount = "No Action", 0
        if expiry_risk == "Expired":
            action = "Remove"
        elif expiry_risk == "Near Expiry":
            action   = "Donate" if _seeded(i, 10) > 0.4 else "Apply Discount"
            discount = [20, 30, 40][int(_seeded(i, 11) * 3)] if action == "Apply Discount" else 0
        elif stock_level == "Low":
            action = "Restock"
        elif stock_level == "High":
            action   = "Apply Discount"
            discount = [10, 15, 20][int(_seeded(i, 12) * 3)]

        donation_eligible = action == "Donate"
        donation_status   = (
            ["Pending", "Pending", "Donated", "Rejected"][int(_seeded(i, 13) * 4)]
            if donation_eligible else "N/A"
        )

        rows.append({
            "item_id":             f"ITEM_{i:05d}",
            "product_name":        product,
            "store_nbr":           store,
            "current_stock":       stock,
            "rolling_avg_sales_7": sales_avg,
            "days_to_expiry":      days_exp,
            "unit_price":          unit_price,
            "Stock_Level":         stock_level,
            "Expiry_Risk":         expiry_risk,
            "Suggested_Discount":  discount,
            "Reorder":             "Yes" if stock_level == "Low" else "No",
            "Action":              action,
            "donation_eligible":   donation_eligible,
            "donation_status":     donation_status,
            "city":                city,
            "category":            category,
            "nearest_ngo":         ngo,
            "ngo_contact":         "+91-9999999999",
            "ngo_address":         f"{city} Central",
            "store_latitude":      round(8.0  + _seeded(i, 14) * 27, 4),
            "store_longitude":     round(68.0 + _seeded(i, 15) * 29, 4),
        })
    return pd.DataFrame(rows)


# Singleton — generated once at startup
DF = generate_data(300)


# ─── Helper: apply filters (mirrors app.py sidebar) ──────────────────────────
def apply_filters(
    df: pd.DataFrame,
    store: str,
    stock_level: str,
    expiry_risk: str,
    action: str,
) -> pd.DataFrame:
    fdf = df.copy()
    if store       != "All": fdf = fdf[fdf["store_nbr"]   == store]
    if stock_level != "All": fdf = fdf[fdf["Stock_Level"] == stock_level]
    if expiry_risk != "All": fdf = fdf[fdf["Expiry_Risk"] == expiry_risk]
    if action      != "All": fdf = fdf[fdf["Action"]      == action]
    return fdf


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Smart Inventory API — Sparkathon 2025"}


@app.get("/api/health")
def health():
    return {"status": "healthy", "records": len(DF)}


@app.get("/api/filters/options")
def filter_options():
    """Return unique values for all filter dropdowns."""
    return {
        "stores":       ["All"] + sorted(DF["store_nbr"].unique().tolist()),
        "stock_levels": ["All"] + sorted(DF["Stock_Level"].unique().tolist()),
        "expiry_risks": ["All"] + sorted(DF["Expiry_Risk"].unique().tolist()),
        "actions":      ["All"] + sorted(DF["Action"].unique().tolist()),
        "cities":       ["All"] + sorted(DF["city"].dropna().unique().tolist()),
        "categories":   ["All"] + sorted(DF["category"].dropna().unique().tolist()),
    }


@app.get("/api/kpis")
def kpis(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
):
    """KPIs — mirrors app.py lines 209-220."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)
    disc_items = fdf[fdf["Suggested_Discount"] > 0]
    avg_disc   = float(disc_items["Suggested_Discount"].mean()) if len(disc_items) else 0.0

    return {
        "total":       int(len(fdf)),
        "total_value": float((fdf["current_stock"] * fdf["unit_price"]).sum()),
        "high_risk":   int(len(fdf[fdf["Action"].isin(["Remove", "Apply Discount"])])),
        "reorder":     int(len(fdf[fdf["Reorder"] == "Yes"])),
        "overstock":   int(len(fdf[fdf["Stock_Level"] == "High"])),
        "understock":  int(len(fdf[fdf["Stock_Level"] == "Low"])),
        "near_expiry": int(len(fdf[fdf["Expiry_Risk"] == "Near Expiry"])),
        "avg_discount": round(avg_disc, 1),
    }


@app.get("/api/financial")
def financial(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
):
    """Financial summary — mirrors app.py lines 179-202."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)

    disc_items    = fdf[fdf["Action"] == "Apply Discount"]
    donated_items = fdf[fdf["donation_status"] == "Donated"]
    removed_items = fdf[fdf["Action"] == "Remove"]

    disc_revenue = float((disc_items["unit_price"] * disc_items["current_stock"] * (1 - disc_items["Suggested_Discount"] / 100)).sum())
    donated_val  = float((donated_items["unit_price"] * donated_items["current_stock"]).sum())
    removed_val  = float((removed_items["unit_price"] * removed_items["current_stock"]).sum())
    net_impact   = disc_revenue + donated_val - removed_val
    total_at_risk = disc_revenue + donated_val + removed_val
    recovery_rate = ((disc_revenue + donated_val) / total_at_risk * 100) if total_at_risk > 0 else 0.0
    items_processed = len(disc_items) + len(donated_items) + len(removed_items)

    return {
        "disc_revenue":     round(disc_revenue, 2),
        "donated_val":      round(donated_val, 2),
        "removed_val":      round(removed_val, 2),
        "net_impact":       round(net_impact, 2),
        "recovery_rate":    round(recovery_rate, 1),
        "items_processed":  items_processed,
    }


@app.get("/api/charts")
def charts(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
):
    """All chart data — stock dist, expiry dist, action dist, value by action, category risk."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)

    def dist(col):
        return [{"name": k, "value": int(v)} for k, v in fdf[col].value_counts().items()]

    value_by_action = (
        fdf.groupby("Action")
           .apply(lambda x: float((x["current_stock"] * x["unit_price"]).sum()), include_groups=False)
           .reset_index()
    )
    value_by_action.columns = ["name", "value"]

    cat_agg = fdf.groupby("category").apply(
        lambda x: {
            "category": x.name,
            "riskCount": int((x["Expiry_Risk"] != "Safe").sum()),
            "total": len(x),
            "riskPct": int((x["Expiry_Risk"] != "Safe").mean() * 100),
        }, include_groups=False
    )

    demand_forecast = [
        {"week": "W1", "actual": 1200, "predicted": 1180},
        {"week": "W2", "actual": 1350, "predicted": 1320},
        {"week": "W3", "actual": 1100, "predicted": 1150},
        {"week": "W4", "actual": 1480, "predicted": 1450},
        {"week": "W5", "actual": 1600, "predicted": 1580},
        {"week": "W6", "actual": 1750, "predicted": 1700},
        {"week": "W7", "actual": None, "predicted": 1820},
        {"week": "W8", "actual": None, "predicted": 1950},
    ]

    return {
        "stock_dist":      dist("Stock_Level"),
        "expiry_dist":     dist("Expiry_Risk"),
        "action_dist":     dist("Action"),
        "value_by_action": value_by_action.to_dict(orient="records"),
        "category_risk":   list(cat_agg.values),
        "demand_forecast": demand_forecast,
    }


@app.get("/api/items/urgent")
def urgent_items(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
    limit:        int = Query(100),
):
    """Urgent tab — mirrors app.py tab1."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)
    urgent = fdf[fdf["Action"].isin(["Remove", "Apply Discount", "Restock"])].sort_values("days_to_expiry")
    cols = ["item_id","product_name","store_nbr","current_stock","Stock_Level","Expiry_Risk","days_to_expiry","Action"]
    return {"total": len(urgent), "items": urgent[cols].head(limit).to_dict(orient="records")}


@app.get("/api/items/discounts")
def discount_items(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
    limit:        int = Query(100),
):
    """Discounts tab — mirrors app.py tab2 with computed columns."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)
    di = fdf[fdf["Action"] == "Apply Discount"].sort_values("Suggested_Discount", ascending=False).copy()
    di["original_value"]  = (di["current_stock"] * di["unit_price"]).round(2)
    di["discount_amount"] = (di["original_value"] * di["Suggested_Discount"] / 100).round(2)
    di["revenue_after"]   = (di["original_value"] - di["discount_amount"]).round(2)

    cols = ["item_id","product_name","store_nbr","current_stock","unit_price",
            "Suggested_Discount","original_value","discount_amount","revenue_after","Expiry_Risk"]
    subset = di[cols].head(limit)
    return {
        "total": len(di),
        "items": subset.to_dict(orient="records"),
        "totals": {
            "original":  round(float(di["original_value"].sum()), 2),
            "discount":  round(float(di["discount_amount"].sum()), 2),
            "revenue":   round(float(di["revenue_after"].sum()), 2),
        }
    }


@app.get("/api/items/restock")
def restock_items(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
    limit:        int = Query(100),
):
    """Restock tab — mirrors app.py tab3."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)
    ri = fdf[fdf["Reorder"] == "Yes"].sort_values("current_stock")
    cols = ["item_id","product_name","store_nbr","current_stock","rolling_avg_sales_7","Stock_Level","days_to_expiry","category"]
    return {"total": len(ri), "items": ri[cols].head(limit).to_dict(orient="records")}


@app.get("/api/items/remove")
def remove_items(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
    limit:        int = Query(100),
):
    """Remove tab — mirrors app.py tab4."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)
    rmi = fdf[fdf["Action"] == "Remove"].sort_values("days_to_expiry").copy()
    rmi["loss_value"] = (rmi["current_stock"] * rmi["unit_price"]).round(2)
    cols = ["item_id","product_name","store_nbr","current_stock","days_to_expiry","unit_price","loss_value"]
    subset = rmi[cols].head(limit)
    return {
        "total": len(rmi),
        "items": subset.to_dict(orient="records"),
        "total_loss": round(float(rmi["loss_value"].sum()), 2),
    }


@app.get("/api/items/donations")
def donation_items(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
    don_city:     str = Query("All"),
    don_category: str = Query("All"),
    limit:        int = Query(100),
):
    """Donation Management Center — mirrors app.py tab5."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)
    don = fdf[(fdf["donation_eligible"] == True) & (fdf["Action"] == "Donate")].copy()

    if don_city     != "All": don = don[don["city"]     == don_city]
    if don_category != "All": don = don[don["category"] == don_category]

    status_counts = don["donation_status"].value_counts().to_dict()
    top_cities = don["city"].value_counts().head(5).to_dict()
    top_ngos   = don["nearest_ngo"].value_counts().head(5).to_dict()

    cols = ["item_id","product_name","days_to_expiry","city","nearest_ngo",
            "ngo_contact","ngo_address","donation_status","category",
            "store_latitude","store_longitude","current_stock","unit_price"]

    return {
        "total":         len(don),
        "status_counts": status_counts,
        "top_cities":    [{"city": k, "count": v} for k,v in top_cities.items()],
        "top_ngos":      [{"ngo": k, "count": v}  for k,v in top_ngos.items()],
        "items":         don[cols].head(limit).to_dict(orient="records"),
    }


@app.get("/api/items/donations/export")
def export_donations(
    store:        str = Query("All"),
    stock_level:  str = Query("All"),
    expiry_risk:  str = Query("All"),
    action:       str = Query("All"),
):
    """CSV export — mirrors app.py download_button."""
    fdf = apply_filters(DF, store, stock_level, expiry_risk, action)
    don = fdf[(fdf["donation_eligible"] == True) & (fdf["Action"] == "Donate")]
    buf = io.StringIO()
    don.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=donations.csv"},
    )
