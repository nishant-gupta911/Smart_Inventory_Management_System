import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="Smart Inventory Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Load data with aggressive caching ─────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Loading inventory data...")
def load_data():
    root = _root()
    paths = [
        os.path.join(root, "data", "processed", "inventory_analysis_results_enhanced.csv"),
        os.path.join(root, "data", "processed", "inventory_analysis_results.csv"),
        os.path.join(root, "dashboard", "cache", "dashboard_data.csv"),
        os.path.join(root, "data", "cleaned_inventory_data.csv"),
    ]

    df = None
    source = None
    for p in paths:
        if os.path.exists(p):
            try:
                # Only load columns we actually need — much faster
                needed = [
                    'item_id', 'product_name', 'store_nbr', 'current_stock',
                    'rolling_avg_sales_7', 'days_to_expiry', 'unit_price',
                    'Stock_Level', 'Expiry_Risk', 'Suggested_Discount',
                    'Reorder', 'Action', 'donation_eligible', 'donation_status',
                    'city', 'category', 'nearest_ngo', 'ngo_contact', 'ngo_address',
                    'store_latitude', 'store_longitude', 'date'
                ]
                # Read header first to check which columns exist
                header = pd.read_csv(p, nrows=0, low_memory=False).columns.tolist()
                use_cols = [c for c in needed if c in header]
                df = pd.read_csv(p, usecols=use_cols, low_memory=False)
                source = p
                break
            except Exception as e:
                continue

    if df is None:
        # Fallback sample data
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'item_id': [f'ITEM_{i:05d}' for i in range(n)],
            'product_name': [f'Product_{i:03d}' for i in range(n)],
            'store_nbr': np.random.randint(1, 11, n).astype(str),
            'current_stock': np.random.randint(0, 200, n),
            'rolling_avg_sales_7': np.random.exponential(5, n),
            'days_to_expiry': np.random.randint(0, 31, n),
            'unit_price': np.random.uniform(0.5, 50, n),
            'Stock_Level': np.random.choice(['High', 'Normal', 'Low'], n),
            'Expiry_Risk': np.random.choice(['Safe', 'Near Expiry', 'Expired'], n),
            'Suggested_Discount': np.random.choice([0, 10, 20, 30, 40], n),
            'Reorder': np.random.choice(['No', 'Yes'], n),
            'Action': np.random.choice(['No Action', 'Apply Discount', 'Restock', 'Remove'], n),
            'donation_eligible': np.random.choice([True, False], n, p=[0.1, 0.9]),
            'donation_status': 'Pending',
            'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune'], n),
            'category': np.random.choice(['DAIRY', 'PRODUCE', 'MEATS', 'SNACKS'], n),
            'nearest_ngo': np.random.choice(['Akshaya Patra', 'Robin Hood Army', 'Feeding India'], n),
            'ngo_contact': '+91-9999999999',
            'store_latitude': np.random.uniform(8.0, 35.0, n),
            'store_longitude': np.random.uniform(68.0, 97.0, n),
        })
        source = "Sample Data"

    # Ensure required columns exist
    df.columns = df.columns.str.strip()
    defaults = {
        'Stock_Level': 'Normal', 'Expiry_Risk': 'Safe',
        'Suggested_Discount': 0, 'Reorder': 'No', 'Action': 'No Action',
        'donation_eligible': False, 'donation_status': 'N/A',
        'city': 'Unknown', 'category': 'Unknown',
        'nearest_ngo': 'Unknown', 'ngo_contact': 'N/A',
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # Type coercion
    df['store_nbr']           = df['store_nbr'].astype(str)
    df['current_stock']       = pd.to_numeric(df['current_stock'], errors='coerce').fillna(0).astype(int)
    df['days_to_expiry']      = pd.to_numeric(df['days_to_expiry'], errors='coerce').fillna(7).astype(int)
    df['unit_price']          = pd.to_numeric(df['unit_price'], errors='coerce').fillna(1.0)
    df['rolling_avg_sales_7'] = pd.to_numeric(df['rolling_avg_sales_7'], errors='coerce').fillna(0)
    df['Suggested_Discount']  = pd.to_numeric(df['Suggested_Discount'], errors='coerce').fillna(0).astype(int)

    return df, source


# ── Persist donation status ────────────────────────────────────────────────────
def save_donation_status(df):
    root = _root()
    processed = os.path.join(root, "data", "processed")
    os.makedirs(processed, exist_ok=True)
    try:
        df.to_csv(os.path.join(processed, "inventory_analysis_results_enhanced.csv"), index=False)
        if "donation_eligible" in df.columns:
            don = df[df["donation_eligible"] == True]
            if not don.empty:
                don.to_csv(os.path.join(processed, "donation_summary.csv"), index=False)
            pending = don[don["donation_status"] == "Pending"]
            if not pending.empty:
                cols = ["item_id","product_name","category","current_stock",
                        "days_to_expiry","city","nearest_ngo","ngo_contact","donation_status"]
                pending[[c for c in cols if c in pending.columns]].to_csv(
                    os.path.join(processed, "pending_donations.csv"), index=False)
        return True
    except Exception as e:
        st.error(f"❌ Save failed: {e}")
        return False


# ── Load ───────────────────────────────────────────────────────────────────────
try:
    df, data_source = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Apply session-state changes
if 'df_changes' in st.session_state:
    for idx, status in st.session_state.df_changes.items():
        if idx in df.index and 'donation_status' in df.columns:
            df.at[idx, 'donation_status'] = status

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filter Options")
st.sidebar.success(f"✅ {os.path.basename(data_source)}")
st.sidebar.info(f"📊 {len(df):,} records")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

stores = ['All'] + sorted(df['store_nbr'].dropna().unique().tolist())
selected_store = st.sidebar.selectbox("Select Store", stores)

stock_opts = ['All'] + df['Stock_Level'].dropna().unique().tolist()
selected_stock = st.sidebar.selectbox("Stock Level", stock_opts)

expiry_opts = ['All'] + df['Expiry_Risk'].dropna().unique().tolist()
selected_expiry = st.sidebar.selectbox("Expiry Risk", expiry_opts)

action_opts = ['All'] + df['Action'].dropna().unique().tolist()
selected_action = st.sidebar.selectbox("Required Action", action_opts)

# Apply filters
fdf = df
if selected_store  != 'All': fdf = fdf[fdf['store_nbr']   == selected_store]
if selected_stock  != 'All': fdf = fdf[fdf['Stock_Level'] == selected_stock]
if selected_expiry != 'All': fdf = fdf[fdf['Expiry_Risk'] == selected_expiry]
if selected_action != 'All': fdf = fdf[fdf['Action']      == selected_action]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Smart Inventory Management Dashboard")
st.markdown("### AI-Powered Inventory Analysis & Recommendations")
st.markdown("---")

# ── Financial Summary ──────────────────────────────────────────────────────────
st.subheader("💰 Financial Impact Summary")

disc_items    = fdf[fdf['Action'] == 'Apply Discount']
donated_items = fdf[fdf['donation_status'] == 'Donated']
removed_items = fdf[fdf['Action'] == 'Remove']

orig_val     = disc_items['unit_price'] * disc_items['current_stock']
disc_revenue = (orig_val * (1 - disc_items['Suggested_Discount'] / 100)).sum()
donated_val  = (donated_items['unit_price'] * donated_items['current_stock']).sum()
removed_val  = (removed_items['unit_price'] * removed_items['current_stock']).sum()
net_impact   = disc_revenue + donated_val - removed_val

c1, c2, c3, c4 = st.columns(4)
c1.metric("💸 Revenue from Discounts",  f"₹{disc_revenue:,.0f}")
c2.metric("🧡 Donated Goods Value",     f"₹{donated_val:,.0f}")
c3.metric("🗑️ Loss from Removals",     f"₹{removed_val:,.0f}",
          delta=f"-₹{removed_val:,.0f}" if removed_val > 0 else None, delta_color="inverse")
c4.metric("📊 Net Financial Impact",    f"₹{net_impact:,.0f}",
          delta=f"₹{net_impact:,.0f}", delta_color="normal" if net_impact >= 0 else "inverse")

total_at_risk = disc_revenue + donated_val + removed_val
if total_at_risk > 0:
    recovery = (disc_revenue + donated_val) / total_at_risk * 100
    rc1, rc2 = st.columns(2)
    rc1.metric("📈 Value Recovery Rate", f"{recovery:.1f}%")
    rc2.metric("📦 Items Processed", f"{len(disc_items)+len(donated_items)+len(removed_items):,}")

st.markdown("---")

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.subheader("📌 Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Items",        f"{len(fdf):,}")
k2.metric("Inventory Value",    f"₹{(fdf['current_stock']*fdf['unit_price']).sum():,.0f}")
k3.metric("High Risk Items",    f"{len(fdf[fdf['Action'].isin(['Remove','Apply Discount'])]):,}")
k4.metric("Reorder Needed",     f"{len(fdf[fdf['Reorder']=='Yes']):,}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Overstocked",        f"{len(fdf[fdf['Stock_Level']=='High']):,}")
k6.metric("Understocked",       f"{len(fdf[fdf['Stock_Level']=='Low']):,}")
k7.metric("Near Expiry",        f"{len(fdf[fdf['Expiry_Risk']=='Near Expiry']):,}")
avg_disc = fdf[fdf['Suggested_Discount']>0]['Suggested_Discount'].mean()
k8.metric("Avg Discount",       f"{avg_disc:.1f}%" if not pd.isna(avg_disc) else "0%")

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("📊 Stock Level Distribution")
    sc = fdf['Stock_Level'].value_counts()
    fig1 = px.pie(values=sc.values, names=sc.index,
                  color_discrete_map={'High':'#FF6B6B','Normal':'#4ECDC4','Low':'#45B7D1'})
    st.plotly_chart(fig1, use_container_width=True)

with col_r:
    st.subheader("⚠️ Expiry Risk Analysis")
    ec = fdf['Expiry_Risk'].value_counts()
    fig2 = px.pie(values=ec.values, names=ec.index,
                  color_discrete_map={'Safe':'#2E8B57','Near Expiry':'#FFA500','Expired':'#DC143C'})
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("⚡ Action Recommendations")
ac = fdf['Action'].value_counts()
fig3 = px.bar(x=ac.index, y=ac.values,
              labels={'x':'Action','y':'Items'},
              color=ac.values, color_continuous_scale='RdYlBu_r')
st.plotly_chart(fig3, use_container_width=True)

st.subheader("💰 Inventory Value by Action")
val_by_action = fdf.groupby('Action').apply(
    lambda x: (x['current_stock'] * x['unit_price']).sum(), include_groups=False
).reset_index()
val_by_action.columns = ['Action', 'Value']
fig4 = px.bar(val_by_action, x='Action', y='Value',
              color='Value', color_continuous_scale='Viridis',
              labels={'Value':'Inventory Value (₹)'})
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Action Tabs ────────────────────────────────────────────────────────────────
st.subheader("🚨 Critical Items")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Urgent", "💸 Discounts", "📦 Restock", "🗑️ Remove", "🤝 Donations"
])

DISP_LIMIT = 500  # show max 500 rows in tables — much faster rendering

with tab1:
    urgent = fdf[fdf['Action'].isin(['Remove','Apply Discount','Restock'])].sort_values(['Action','days_to_expiry'])
    st.write(f"**{len(urgent):,} items** requiring immediate attention (showing first {DISP_LIMIT})")
    cols = [c for c in ['item_id','product_name','store_nbr','current_stock',
                         'Stock_Level','Expiry_Risk','days_to_expiry','Action'] if c in urgent.columns]
    st.dataframe(urgent[cols].head(DISP_LIMIT), use_container_width=True)

with tab2:
    di = fdf[fdf['Action']=='Apply Discount'].sort_values('Suggested_Discount', ascending=False)
    st.write(f"**{len(di):,} items** with discount recommendations (showing first {DISP_LIMIT})")
    if len(di) > 0:
        dd = di[['item_id','product_name','store_nbr','current_stock',
                  'Suggested_Discount','Expiry_Risk','unit_price']].head(DISP_LIMIT).copy()
        dd['Original_Value']   = (dd['current_stock'] * dd['unit_price']).round(2)
        dd['Discount_Amount']  = (dd['Original_Value'] * dd['Suggested_Discount'] / 100).round(2)
        dd['Revenue_After']    = (dd['Original_Value'] - dd['Discount_Amount']).round(2)
        st.dataframe(dd, use_container_width=True)
        dc1, dc2, dc3 = st.columns(3)
        dc1.info(f"💰 Original: ₹{dd['Original_Value'].sum():,.0f}")
        dc2.warning(f"🏷️ Discount: ₹{dd['Discount_Amount'].sum():,.0f}")
        dc3.success(f"💵 Revenue: ₹{dd['Revenue_After'].sum():,.0f}")

with tab3:
    ri = fdf[fdf['Reorder']=='Yes'].sort_values('current_stock')
    st.write(f"**{len(ri):,} items** needing restock (showing first {DISP_LIMIT})")
    cols = [c for c in ['item_id','product_name','store_nbr','current_stock',
                         'rolling_avg_sales_7','Stock_Level','days_to_expiry'] if c in ri.columns]
    st.dataframe(ri[cols].head(DISP_LIMIT), use_container_width=True)

with tab4:
    rmi = fdf[fdf['Action']=='Remove'].sort_values('days_to_expiry')
    st.write(f"**{len(rmi):,} expired items** to remove (showing first {DISP_LIMIT})")
    if len(rmi) > 0:
        rd = rmi[['item_id','product_name','store_nbr','current_stock',
                   'days_to_expiry','unit_price']].head(DISP_LIMIT).copy()
        rd['Loss_Value'] = (rd['current_stock'] * rd['unit_price']).round(2)
        st.dataframe(rd, use_container_width=True)
        st.error(f"⚠️ Total loss: ₹{rd['Loss_Value'].sum():,.0f}")

with tab5:
    st.markdown("### 🤝 Donation Management Center")
    don_df = fdf[(fdf['donation_eligible']==True) & (fdf['Action']=='Donate')].copy()

    if len(don_df) > 0:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("🎁 Eligible",  len(don_df))
        d2.metric("🟡 Pending",   len(don_df[don_df['donation_status']=='Pending']))
        d3.metric("🟢 Donated",   len(don_df[don_df['donation_status']=='Donated']))
        d4.metric("🔴 Rejected",  len(don_df[don_df['donation_status']=='Rejected']))

        # Status pie
        sc2 = don_df['donation_status'].value_counts()
        fig5 = px.pie(values=sc2.values, names=sc2.index, title="Donation Status",
                      color_discrete_map={'Pending':'#FFA500','Donated':'#28a745','Rejected':'#dc3545'})
        st.plotly_chart(fig5, use_container_width=True)

        col_city, col_ngo = st.columns(2)
        with col_city:
            st.subheader("🏙️ Top Cities")
            if 'city' in don_df.columns:
                for i, (city, cnt) in enumerate(don_df['city'].value_counts().head(5).items()):
                    st.write(f"{i+1}. **{city}**: {cnt}")
        with col_ngo:
            st.subheader("🏢 Top NGOs")
            if 'nearest_ngo' in don_df.columns:
                for i, (ngo, cnt) in enumerate(don_df['nearest_ngo'].value_counts().head(5).items()):
                    st.write(f"{i+1}. **{ngo}**: {cnt}")

        st.markdown("---")

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            city_opts = ['All'] + (sorted(don_df['city'].unique().tolist()) if 'city' in don_df.columns else [])
            sel_city = st.selectbox("Filter by City", city_opts, key="don_city")
        with f2:
            cat_opts = ['All'] + (sorted(don_df['category'].unique().tolist()) if 'category' in don_df.columns else [])
            sel_cat = st.selectbox("Filter by Category", cat_opts, key="don_cat")

        fdon = don_df.copy()
        if sel_city != 'All': fdon = fdon[fdon['city'] == sel_city]
        if sel_cat  != 'All': fdon = fdon[fdon['category'] == sel_cat]

        dcols = [c for c in ['item_id','product_name','days_to_expiry','city',
                              'nearest_ngo','ngo_contact','donation_status'] if c in fdon.columns]
        st.dataframe(fdon[dcols].head(DISP_LIMIT), use_container_width=True)

        # Bulk actions
        pending = fdon[fdon['donation_status']=='Pending']
        if len(pending) > 0:
            ba1, ba2, ba3 = st.columns(3)
            with ba1:
                if st.button("✅ Mark All Pending → Donated", type="primary"):
                    mask = (df['donation_eligible']==True) & (df['donation_status']=='Pending')
                    df.loc[mask, 'donation_status'] = 'Donated'
                    save_donation_status(df)
                    st.cache_data.clear()
                    st.rerun()
            with ba2:
                if st.button("❌ Mark All Pending → Rejected"):
                    mask = (df['donation_eligible']==True) & (df['donation_status']=='Pending')
                    df.loc[mask, 'donation_status'] = 'Rejected'
                    save_donation_status(df)
                    st.cache_data.clear()
                    st.rerun()
            with ba3:
                csv = fdon.to_csv(index=False)
                st.download_button("📥 Export CSV", csv,
                                   f"donations_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

            # Individual actions (only if small set)
            if len(pending) <= 10:
                st.write("**Individual actions:**")
                for idx, row in pending.iterrows():
                    ic1, ic2, ic3 = st.columns([3,1,1])
                    ic1.write(f"**{row.get('product_name','?')}** — {row.get('city','?')}")
                    if ic2.button("✅ Donate", key=f"d_{idx}"):
                        df.at[idx, 'donation_status'] = 'Donated'
                        save_donation_status(df)
                        st.cache_data.clear()
                        st.rerun()
                    if ic3.button("❌ Reject", key=f"r_{idx}"):
                        df.at[idx, 'donation_status'] = 'Rejected'
                        save_donation_status(df)
                        st.cache_data.clear()
                        st.rerun()

        # Map
        if 'store_latitude' in fdon.columns and 'store_longitude' in fdon.columns:
            map_df = fdon[['store_latitude','store_longitude','city','nearest_ngo','donation_status']].dropna()
            if len(map_df) > 0:
                fig_map = px.scatter_mapbox(
                    map_df, lat="store_latitude", lon="store_longitude",
                    color="donation_status", hover_name="city",
                    hover_data=["nearest_ngo"],
                    color_discrete_map={'Pending':'#FFA500','Donated':'#28a745','Rejected':'#dc3545'},
                    zoom=4, height=450, title="Donation Locations"
                )
                fig_map.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("ℹ️ No donation-eligible items in current filter.")

# ── Footer ─────────────────────────────────────────────────────────────────────
with st.expander("📋 Dataset Summary"):
    st.write(f"**Shape:** {fdf.shape[0]:,} rows × {fdf.shape[1]} columns")
    st.write(f"**Source:** {data_source}")
    if st.checkbox("Show raw sample (first 10 rows)"):
        st.dataframe(fdf.head(10))

st.markdown("---")
st.success("✅ Smart Inventory Dashboard — Sparkathon 2025 | Walmart Hackathon")
st.info("🔄 Run `python main.py` to refresh analysis data")