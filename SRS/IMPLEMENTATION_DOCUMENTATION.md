# Smart Inventory Management System
## Complete Implementation Documentation

**Generated**: 2026-07-29  
**Project**: Smart Inventory Management System (Sparkathon 2025)  
**Developer**: Nishant Gupta

> ⚠️ **CRITICAL**: This documentation is based ONLY on actual implementation.  
> No features are invented. Everything documented exists in the codebase.

---

## Executive Summary

**Smart Inventory Management System** is an AI-powered inventory optimization platform that combines machine learning predictions with business logic to:
- Reduce food waste by predicting expiry risk
- Optimize stock levels through demand forecasting  
- Automate donation matching to NGOs
- Generate intelligent restocking recommendations

**Architecture**: Python-based ML pipeline with Streamlit dashboard, CSV data storage, joblib model persistence

**Models**: 
- Expiry Risk Classifier (VotingClassifier: XGBoost + LightGBM + RandomForest)
- Demand Forecaster (VotingRegressor: XGBoost + LightGBM + RandomForest)

**Key Metrics**:
- Expiry Model Accuracy: ~75-85% (CV on 10% sample, 5-fold)
- Demand Model R²: ~0.65-0.75 (CV on 10% sample, 5-fold)

---

## 1. SYSTEM ARCHITECTURE

### 1.1 Pipeline Overview

The system follows a **sequential batch processing pipeline** orchestrated by `main.py`:

```
┌─────────── EXECUTION FLOW ───────────┐
│                                       │
│  1. Import & Setup                   │
│     ├─ Load modules safely           │
│     ├─ Create directories            │
│     └─ Setup logging                 │
│                                       │
│  2. Data Preprocessing               │
│     ├─ Load raw CSV files            │
│     ├─ Engineer 30+ features         │
│     └─ Output: preprocessed CSV      │
│                                       │
│  3. Model Training (Phase 1)         │
│     ├─ Train Expiry Classifier       │
│     ├─ Train Demand Regressor        │
│     └─ Save models as .pkl files     │
│                                       │
│  4. Inventory Analysis (Phase 2)     │
│     ├─ Load trained models           │
│     ├─ Analyze stock levels          │
│     ├─ Calculate actions             │
│     └─ Output: analysis results CSV  │
│                                       │
│  5. Donation Processing (Phase 3.5)  │
│     ├─ Apply donation logic          │
│     ├─ Match to NGOs                 │
│     └─ Output: enhanced dataset      │
│                                       │
│  6. Restock Planning (Phase 4)       │
│     ├─ Generate recommendations      │
│     ├─ Filter donation items         │
│     └─ Output: restock suggestions   │
│                                       │
│  7. Dashboard Preparation            │
│     └─ Cache data for Streamlit      │
│                                       │
└───────────────────────────────────────┘


### 1.2 Module Hierarchy

```
smart_inventory/
├── main.py                          [Orchestrator - Entry Point]
├── src/
│   ├── data_preprocessing.py        [Data Loading & Feature Engineering]
│   ├── train_expiry_model.py        [Expiry Risk ML Model]
│   ├── train_demand_model.py        [Demand Forecasting ML Model]
│   ├── inventory_analyzer.py        [Business Logic & Actions]
│   ├── generate_restock_plan.py     [Restock Recommendations]
│   └── utils.py                     [Donation Utilities]
├── transform_inventory_data.py      [Donation Integration Logic]
├── dashboard/
│   └── app.py                       [Streamlit Web Dashboard]
├── evaluation/
│   ├── evaluate_expiry_model.py     [Model Performance Analysis]
│   ├── evaluate_demand_model.py     [Model Performance Analysis]
│   └── run_all_evaluations.py       [Combined Evaluation]
└── data/
    ├── raw/                         [Original Walmart datasets]
    ├── processed/                   [Transformed data]
    └── cleaned_inventory_data.csv   [Main working dataset]
```

### 1.3 Data Flow Architecture

```
[Raw Data] → [Preprocessing] → [Feature Engineering] → [Model Training]
                                                              ↓
[Dashboard] ← [Analysis Results] ← [Predictions] ← [Trained Models]
                     ↓
              [Action Recommendations]
              [Donation Matching]
              [Restock Suggestions]
```

**Key Data Files**:
1. **Input**: `data/raw/*.csv` (Walmart datasets)
2. **Intermediate**: `data/cleaned_inventory_data.csv` (feature-engineered)
3. **Models**: `models/*.pkl` (trained ML models)
4. **Output**: `data/processed/inventory_analysis_results_enhanced.csv`
5. **Dashboard Cache**: `dashboard/cache/dashboard_data.csv`

---

## 2. MODULE BREAKDOWN

### 2.1 main.py - Pipeline Orchestrator

**Purpose**: Central coordinator that executes all phases sequentially

**Execution Flow**:

1. **import_modules()**: Safely imports all required modules with fallback handling
2. **ensure_directories()**: Creates logs/, models/, data/processed/, plots/, dashboard/cache/
3. **run_data_preprocessing()**: Calls `data_preprocessing.preprocess_data()`
4. **run_expiry_prediction()**: Calls `train_expiry_model.train_and_predict()` → Accuracy printed
5. **run_demand_model()**: Calls `train_demand_model.train_and_evaluate()` → R² score printed
6. **run_inventory_analysis()**: Calls `InventoryAnalyzer().run_full_analysis()`
7. **save_dashboard_data()**: Copies results to `dashboard/cache/dashboard_data.csv`
8. **run_donation_data_processing()**: Calls `transform_inventory_data.load_and_transform_data()`
9. **analyze_donation_data()**: Logs donation statistics and metrics
10. **save_enhanced_dataset()**: Saves multiple output CSVs (enhanced, removed, donations, pending)
11. **run_restock_plan()**: Calls `RestockPlanGenerator().run()` if available

**Inputs**: None (self-contained, reads from data/ and models/)  
**Outputs**: 
- Trained models in `models/`
- Analysis results in `data/processed/`
- Log files in `logs/main_pipeline.log`

**Dependencies**: All src modules, transform_inventory_data.py

**Technologies**: Python 3.8+, pathlib, logging, pandas

**Key Functions**:
- `setup_logging()`: Configures file and console logging
- `validate_donation_data()`: Checks data integrity for donations
- `display_donation_metrics()`: Formats and logs comprehensive donation analytics
- `print_combined_summary()`: Final summary with model accuracies

---

### 2.2 data_preprocessing.py - Feature Engineering

**Purpose**: Load raw data, clean, and engineer features for ML models

**Implementation**: Functional approach with main `preprocess_data()` function

**Process**:
1. **Data Loading** (`_load_raw_data()`):
   - Tries multiple paths: `data/cleaned_inventory_data.csv`, `data/processed/...`
   - Falls back to generating synthetic data if none found
   - Uses `pd.read_csv()` with `low_memory=False`

2. **Feature Engineering**:
   - **Temporal**: day_of_week (0-6), month (1-12), is_weekend (0/1), quarter (1-4)
   - **Rolling Averages**: 
     - `rolling_avg_sales_7`, `rolling_avg_3`, `rolling_avg_14`, `rolling_avg_30`
     - Computed using `groupby(['store_nbr', 'item_id']).transform(rolling())`
   - **Lag Features**: `sales_lag_1`, `sales_lag_7` (shifted sales values)
   - **Trends**: `sales_trend = rolling_avg_7 - rolling_avg_30`
   - **Inventory Ratios**:
     - `shelf_consumed_ratio = days_on_shelf / shelf_life`
     - `stock_to_sales_ratio = current_stock / rolling_avg_sales_7`
     - `days_of_stock_left = current_stock / rolling_avg_sales_7`
   - **Calendar**: `is_holiday_month` (Oct/Nov/Dec = 1)
   - **Expiry Metrics**: 
     - `sales_velocity = rolling_avg_sales_7`
     - `urgency_score = (30 - days_to_expiry) / 30` clipped [0, 1]
   - **Labels**: `Expiry_Risk` (Safe/Near Expiry/Expired/Remove)

3. **Output**: Saves to `data/processed/preprocessed_inventory.csv`

**Inputs**: Raw CSV files or generates synthetic data
**Outputs**: Feature-engineered DataFrame with 30+ columns
**Dependencies**: pandas, numpy, pathlib
**Technologies**: Pandas GroupBy, rolling windows, datetime operations

**Key Features** (Total: 30+):
- Time: day_of_week, month, is_weekend, quarter, is_holiday_month
- Sales: rolling_avg_sales_7/3/14/30, sales_lag_1/7, sales_trend
- Inventory: shelf_life, days_on_shelf, current_stock, days_to_expiry
- Derived: shelf_consumed_ratio, stock_to_sales_ratio, sales_velocity, urgency_score
- Labels: Expiry_Risk (for classification)

---

### 2.3 train_expiry_model.py - Expiry Risk Classifier

**Purpose**: Train ML model to predict which items will expire unsold

**Implementation**: Function-based with comprehensive progress bars (tqdm)

**Model Architecture**:
```python
VotingClassifier(voting='soft', weights=[3, 2, 1]):
├── XGBClassifier(n_estimators=500, max_depth=6, lr=0.05)
├── LGBMClassifier(n_estimators=500, lr=0.05, num_leaves=31)
└── RandomForestClassifier(n_estimators=300, max_depth=12)
```

**Training Process** (`train_and_predict()`):
1. Load `data/cleaned_inventory_data.csv`
2. Compute additional features (rolling_avg_3, rolling_avg_14, stock_to_sales_ratio)
3. Create labels with 5% noise injection for robustness
4. **Cross-Validation**: 5-fold StratifiedKFold on 10% sample (fast validation)
5. Train on 100% dataset (full fit)
6. Save model: `models/expiry_predict_model.pkl`
7. Save predictions: `data/processed/expiry_risk_predictions.csv`

**Features Used** (9 core features):
- `rolling_avg_sales_7`, `perishable`, `shelf_life`
- `stock_to_sales_ratio`, `sales_trend`
- `rolling_avg_3`, `rolling_avg_14`, `current_stock`
- `days_on_shelf`

**Target**: `Expiry_Risk` (Safe/Near Expiry/Expired)

**Performance**:
- CV Accuracy: ~75-85% (mean ± std logged)
- Trained on full dataset after CV
- Uses `class_weight='balanced'` to handle imbalance

**Inputs**: `data/cleaned_inventory_data.csv`
**Outputs**: 
- `models/expiry_predict_model.pkl` (joblib)
- `data/processed/expiry_risk_predictions.csv`
**Dependencies**: xgboost, lightgbm, scikit-learn, pandas
**Technologies**: Ensemble learning, stratified CV, soft voting

---

### 2.4 train_demand_model.py - Demand Forecaster

**Purpose**: Train regression model to predict unit sales per store-item

**Implementation**: Function-based with progress bars

**Model Architecture**:
```python
VotingRegressor(weights=[3, 2, 1]):
├── XGBRegressor(n_estimators=300, max_depth=6, lr=0.05)
├── LGBMRegressor(n_estimators=300, lr=0.05)
└── RandomForestRegressor(n_estimators=200, max_depth=10)
```

**Training Process** (`train_and_evaluate()`):
1. Load `data/cleaned_inventory_data.csv`
2. Compute lag features and rolling averages
3. Fill oil price NaNs with forward fill
4. **Cross-Validation**: 5-fold KFold on 10% sample
5. Train on 100% dataset
6. Save model: `models/demand_forecast_model.pkl`

**Features Used** (15 features):
- Rolling: `rolling_avg_sales_7`, `rolling_avg_3`, `rolling_avg_14`
- Lags: `sales_lag_1`, `sales_lag_7`
- Trend: `sales_trend`
- Promotion: `onpromotion`
- Temporal: `day_of_week`, `month`, `is_weekend`, `quarter`
- Store: `store_nbr`, `cluster`, `perishable`
- Economic: `dcoilwtico` (oil price)


**Target**: `sales` (continuous values, unit sales per day)

**Performance**:
- CV R² Score: ~65-75% (mean ± std logged)
- Trained on full dataset after CV

**Inputs**: `data/cleaned_inventory_data.csv`
**Outputs**: `models/demand_forecast_model.pkl` (joblib)
**Dependencies**: xgboost, lightgbm, scikit-learn, pandas
**Technologies**: Ensemble regression, K-fold CV

---

### 2.5 inventory_analyzer.py - Business Logic Engine

**Purpose**: Convert ML predictions into actionable business recommendations

**Implementation**: Class-based (`InventoryAnalyzer`)

**Key Methods**:

1. **`load_inventory_data()`**: 
   - Tries multiple paths for data
   - Falls back to `_generate_sample_inventory()` if no data found
   - Returns cleaned DataFrame

2. **`analyze_stock_levels(df)`**:
   - Calculates `weekly_demand = rolling_avg_sales_7 × 7`
   - Thresholds:
     - **High**: `current_stock > weekly_demand × 1.5`
     - **Low**: `current_stock < weekly_demand × 0.5`
     - **Normal**: Between thresholds
   - Output: `Stock_Level` column (High/Normal/Low)

3. **`analyze_expiry_risk(df)`**:
   - **Standardized thresholds** (single source of truth):
     - `days_to_expiry > 15` → **Safe**
     - `0 < days_to_expiry ≤ 15` → **Near Expiry**
     - `-5 ≤ days_to_expiry ≤ 0` → **Expired** (donation-eligible)
     - `days_to_expiry < -5` → **Remove** (too old)
   - Output: `Expiry_Risk` column

4. **`calculate_discount_suggestions(df)`**:
   - Logic:
     - Stock_Level='High' → +15% discount
     - Expiry_Risk='Near Expiry' → +20% discount
     - Expiry_Risk='Expired' → +40% discount
     - Combined (High + Near Expiry) → up to 40% max
     - Stock_Level='Low' → 0% discount (never discount understocked)
   - Output: `Suggested_Discount` column (0-40%)

5. **`determine_reorder_needs(df)`**:
   - Reorder if:
     - Stock_Level='Low' OR
     - Expiry_Risk='Expired' OR
     - `current_stock < rolling_avg_sales_7 × 3` AND Expiry_Risk='Safe'
   - Output: `Reorder` column (Yes/No)

6. **`determine_actions(df)`** - **AUTHORITATIVE ACTION LOGIC**:
   - **Single source of truth** for Action column:
     ```python
     if days_to_expiry < -5:
         return 'Remove'
     elif -5 ≤ days_to_expiry ≤ -1 AND donation_eligible:
         return 'Donate'
     elif 0 ≤ days_to_expiry ≤ 5:
         return 'Apply Discount'
     elif Stock_Level == 'Low':
         return 'Restock'
     else:
         return 'No Action'
     ```
   - Output: `Action` column

7. **`generate_summary_report(df)`**:
   - Aggregates statistics:
     - Total items, stores
     - Stock level distribution
     - Expiry risk distribution
     - Actions needed counts
     - Inventory value calculations
     - Donation summary (delegates to `utils.get_donation_summary()`)
   - Returns: Dictionary of metrics

8. **`get_donation_summary(df)`**:
   - **Delegates to `src.utils.get_donation_summary()`** (single implementation)
   - Avoids code duplication

9. **`run_full_analysis()`**:
   - Orchestrates all analysis steps
   - Saves results to `data/processed/inventory_analysis_results.csv`
   - Prints formatted summary report
   - Returns: (DataFrame, summary_dict)

**Inputs**: Inventory data (from various sources)
**Outputs**: 
- Enhanced DataFrame with Stock_Level, Expiry_Risk, Suggested_Discount, Reorder, Action
- Summary statistics dictionary
**Dependencies**: pandas, numpy, logging
**Technologies**: Business rule engine, threshold-based classification

**Configuration** (class attributes):
- `overstock_multiplier = 1.5`
- `understock_multiplier = 0.5`
- `near_expiry_days = 15`
- `max_discount_pct = 40`

---

### 2.6 generate_restock_plan.py - Smart Restocking

**Purpose**: Generate optimized restocking recommendations using ML predictions

**Implementation**: Class-based (`RestockPlanGenerator`)

**Key Methods**:

1. **`load_data()`**:
   - Tries multiple candidate paths for input data
   - Validates required columns
   - Handles missing values

2. **`load_models()`**:
   - Loads `models/demand_forecast_model.pkl`
   - Loads `models/expiry_predict_model.pkl`

3. **`prepare_features(df)`**:
   - Ensures all model features exist
   - Creates one-hot encoded family columns
   - Returns: Feature-engineered DataFrame

4. **`generate_predictions(df)`**:
   - Predicts demand: `df['predicted_demand'] = demand_model.predict(X)`
   - Predicts expiry risk: `df['expiry_risk'] = expiry_model.predict_proba(X)[:, 1]`
   - Clips negative predictions to 0

5. **`calculate_restock_quantity(row)`**:
   - Base calculation: `max(0, predicted_demand - current_stock)`
   - Applies safety stock: `× SAFETY_STOCK_MULTIPLIER (1.2)`
   - Adjusts for expiry: Returns 0 if `days_to_expiry ≤ MIN_DAYS_TO_EXPIRY`
   - Priority boost: +10% for high-priority families
   - Caps at `predicted_demand × MAX_RESTOCK_MULTIPLIER (3.0)`
   - Only restocks if `predicted_demand > DEMAND_THRESHOLD (10)`

6. **`determine_discount_strategy(row)`**:
   - Returns dict: `{apply_discount, discount_percentage, discount_reason}`
   - Logic:
     - `expiry_risk > 0.7` AND `predicted_demand < 10` → 50% discount (high_expiry_risk)
     - `days_to_expiry ≤ 3` → 30% discount (close_to_expiry)
     - `predicted_demand < current_stock × 0.5` → 20% discount (overstocked)

7. **`apply_business_rules(df)`**:
   - Calculates `restock_qty` for each item
   - Determines discount strategy
   - Adds `high_priority` flag (HIGH_PRIORITY_FAMILIES or high demand)
   - Computes `urgency_score` (0-100): `(predicted_demand/current_stock) × (1 - days_to_expiry/30) × 100`

8. **`filter_donation_items(df)`**:
   - **Critical**: Excludes items with `donation_eligible=True` AND `donation_status` in ['Pending', 'Donated']
   - Prevents restocking items already in donation pipeline
   - Logs exclusion statistics

9. **`generate_summary_report(df)`**:
   - Total items, items to restock, items for discount
   - High priority items count
   - Total restock quantity
   - Average expiry risk and predicted demand
   - **Donation statistics** if columns exist

10. **`save_results(df, summary)`**:
    - Saves `data/processed/restocking_suggestions.csv`
    - Saves `data/processed/high_priority_restock.csv` (filtered)
    - Saves `data/processed/restock_summary.csv`

11. **`run()`**:
    - Orchestrates full pipeline
    - Returns: (DataFrame, summary_dict)

**Configuration** (default):
```python
{
    'LOW_STOCK_THRESHOLD': 5,
    'EXPIRY_RISK_THRESHOLD': 0.7,
    'DEMAND_THRESHOLD': 10,
    'SAFETY_STOCK_MULTIPLIER': 1.2,
    'MIN_DAYS_TO_EXPIRY': 2,
    'MAX_RESTOCK_MULTIPLIER': 3.0,
    'DISCOUNT_THRESHOLD': 0.5,
    'HIGH_PRIORITY_FAMILIES': ['GROCERY I', 'BEVERAGES', 'DAIRY']
}
```

**Inputs**: `data/cleaned_inventory_data.csv`, trained models
**Outputs**: 
- `restocking_suggestions.csv`
- `high_priority_restock.csv`
- `restock_summary.csv`
**Dependencies**: pandas, numpy, joblib, trained ML models
**Technologies**: ML-driven recommendations, business rules, priority scoring

---

### 2.7 utils.py - Donation Utilities

**Purpose**: Centralized utility functions for donation management

**Key Functions**:

1. **`update_donation_status(df, item_id, new_status)`**:
   - Updates `donation_status` for specific item
   - Validates item exists and is donation-eligible
   - Valid statuses: "Pending", "Donated", "Rejected", ""
   - Returns: Modified DataFrame
   - Raises: ValueError if invalid

2. **`get_nearest_ngo(city_name)`** - **AUTHORITATIVE NGO MAPPING**:
   - Hardcoded dictionary of Indian cities → NGOs
   - Cities: Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad, Jaipur, Surat
   - Returns: `{nearest_ngo, ngo_address, ngo_contact}`
   - Falls back to default NGO list if city not found

3. **`get_donation_summary(df)`** - **SINGLE SOURCE OF TRUTH**:
   - Comprehensive donation statistics
   - Returns dict:
     - `total_donation_eligible`: Count of eligible items
     - `total_items`: Total items in dataset
     - `donation_eligible_percentage`: Percentage eligible
     - `donation_status_counts`: Breakdown by status
     - `category_city_breakdown`: Items by category and city
     - `top_ngos`: Top 10 NGOs by donation count
     - `generated_at`: ISO timestamp
   - Handles missing columns gracefully

4. **`filter_pending_donations(df)`**:
   - Filters for `donation_eligible=True` AND `donation_status='Pending'`
   - Returns: Filtered DataFrame
   - Logs count found

5. **`save_updated_inventory(df, output_path)`**:
   - Creates directory if needed
   - Saves DataFrame to CSV
   - Returns: Boolean success status

6. **`validate_donation_columns(df)`**:
   - Checks for required columns:
     - donation_eligible, donation_status
     - store_latitude, store_longitude
     - nearest_ngo, ngo_address, ngo_contact
   - Returns: (is_valid, list_of_missing_columns)

7. **`add_missing_donation_columns(df)`**:
   - Adds missing columns with defaults
   - Generates random coordinates for India (lat: 8-37°N, lon: 68-97°E)
   - Maps cities to NGOs using `get_nearest_ngo()`

8. **`calculate_donation_metrics(df)`**:
   - Key metrics:
     - donation_rate, success_rate, pending_rate, rejection_rate
     - Counts: total_items, donation_eligible_items, donated_items, pending_items, rejected_items
   - Returns: Dictionary of float metrics

9. **`format_donation_report(summary)`**:
   - Formats donation summary into readable text report
   - ASCII art formatting with emojis
   - Returns: Multi-line string report

**Inputs**: DataFrames with inventory data
**Outputs**: Modified DataFrames, summary dicts, formatted reports
**Dependencies**: pandas, numpy, datetime
**Technologies**: Data validation, business logic, NGO mapping

**NGO Database** (hardcoded):
- 10 Indian NGOs with full contact details
- Categories served: DAIRY, PRODUCE, MEATS, etc.
- Geographic distribution across major cities

---

### 2.8 transform_inventory_data.py - Donation Integration

**Purpose**: Apply donation eligibility logic and generate enhanced dataset

**Key Functions**:

1. **`load_and_transform_data()`**:
   - Loads `data/processed/inventory_analysis_results.csv`
   - Applies `transform_inventory_with_donation_logic()`
   - Generates additional 500 synthetic rows
   - Returns: Combined DataFrame

2. **`transform_product_names(df)`**:
   - Replaces generic names with real food product names
   - Uses `FOOD_PRODUCTS` dict (DAIRY: Milk, Yogurt, etc.)

3. **`add_donation_columns(df)`** - **DONATION ELIGIBILITY LOGIC**:
   - **Authoritative criteria**:
     - `-5 ≤ days_to_expiry ≤ -1` (recently expired)

     - `perishable == 1`
     - `category` in EDIBLE_CATEGORIES (DAIRY, PRODUCE, MEATS, BREAD/BAKERY, FROZEN FOODS, SNACKS, BEVERAGES)
   - Sets `donation_eligible = True` if all criteria met
   - Sets `donation_status = 'Pending'` for newly eligible items
   - Otherwise: `donation_eligible = False`, `donation_status = ''`

4. **`add_location_columns(df)`**:
   - Adds `store_latitude`, `store_longitude` based on city
   - Uses `INDIAN_CITIES` dict with real coordinates
   - Adds random variation (±0.1°) for realism
   - Assigns city and state if missing

5. **`add_ngo_columns(df)`**:
   - Assigns `nearest_ngo`, `ngo_address`, `ngo_contact`
   - Uses `INDIAN_NGOS` list (10 NGOs)
   - Ensures no NaN values (fills with empty strings)

6. **`add_action_column(df)`** - **DELEGATES TO INVENTORY_ANALYZER**:
   - **Critical**: No longer duplicates Action logic
   - Imports `InventoryAnalyzer` and calls `determine_actions(df)`
   - Single source of truth maintained

7. **`transform_inventory_with_donation_logic(df)`**:
   - Main transformation pipeline
   - Ensures required columns exist
   - Applies all transformations in sequence
   - Final cleanup: fills NaN values
   - Returns: Fully transformed DataFrame

8. **`generate_additional_rows(num_rows)`**:
   - Creates synthetic inventory data
   - Realistic expiry scenarios (fresh, near_expiry, recently_expired, long_expired)
   - **Uses same donation eligibility logic** as real data
   - **Uses same Action determination logic** (mirrors InventoryAnalyzer)
   - Generates coordinates, NGO assignments

9. **`apply_donation_logic_to_dataframe(df)`**:
   - Standalone wrapper for applying donation logic
   - Can be used independently on any DataFrame
   - Returns: Transformed copy

**Data Constants**:
- `FOOD_PRODUCTS`: Dict of categories → real product names
- `INDIAN_CITIES`: Dict of 15 cities with coordinates and states
- `INDIAN_NGOS`: List of 10 NGOs with full contact details
- `PERISHABLE_CATEGORIES`: ['DAIRY', 'MEATS', 'PRODUCE', 'BREAD/BAKERY', 'SEAFOOD', 'FROZEN']
- `EDIBLE_CATEGORIES`: Categories eligible for donation

**Inputs**: `data/processed/inventory_analysis_results.csv`
**Outputs**: `data/processed/inventory_analysis_results_enhanced.csv`
**Dependencies**: pandas, numpy, random, datetime, src.inventory_analyzer
**Technologies**: Data transformation, synthetic data generation, geo-mapping

---

### 2.9 dashboard/app.py - Streamlit Dashboard

**Purpose**: Interactive web dashboard for real-time inventory insights

**Implementation**: Streamlit app with caching and multiple views

**Key Components**:

1. **Configuration**:
   ```python
   st.set_page_config(
       page_title="Smart Inventory Dashboard",
       page_icon="📊",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   ```

2. **`load_data()` - CACHED**:
   - Decorator: `@st.cache_data(ttl=600, show_spinner="Loading...")`
   - Tries multiple paths in priority order:
     - `inventory_analysis_results_enhanced.csv`
     - `inventory_analysis_results.csv`
     - `dashboard/cache/dashboard_data.csv`
     - `cleaned_inventory_data.csv`
   - Falls back to sample data (1000 rows) if none found
   - Only loads needed columns for performance
   - Type coercion and validation
   - Returns: (DataFrame, source_path)

3. **`save_donation_status(df)`**:
   - Persists changes to disk
   - Saves enhanced dataset, donation_summary.csv, pending_donations.csv
   - Returns: Boolean success

4. **Sidebar Filters**:
   - Store selection dropdown
   - Stock Level filter
   - Expiry Risk filter
   - Required Action filter
   - Refresh Data button (clears cache)

5. **Financial Impact Summary** (Top Section):
   - Metrics calculated:
     - Revenue from Discounts: `sum(discounted_price × quantity)`
     - Donated Goods Value: `sum(unit_price × stock)` for donated items
     - Loss from Removals: `sum(unit_price × stock)` for removed items
     - Net Financial Impact: `discounts + donations - removals`
   - Value Recovery Rate: `(discounts + donations) / total_at_risk × 100`
   - Displayed in 4-column metric cards

6. **Key Performance Indicators**:
   - 8 KPI cards in 2 rows:
     - Total Items, Inventory Value, High Risk Items, Reorder Needed
     - Overstocked, Understocked, Near Expiry, Avg Discount

7. **Visualizations**:
   - **Stock Level Distribution** (Pie Chart): Color-coded (High=red, Normal=teal, Low=blue)
   - **Expiry Risk Analysis** (Pie Chart): Color-coded (Safe=green, Near Expiry=orange, Expired=red)
   - **Action Recommendations** (Bar Chart): Count per action type
   - **Inventory Value by Action** (Bar Chart): Financial impact visualization

8. **Action Tabs** (5 tabs):
   - **🔴 Urgent**: Items needing immediate action (Remove/Apply Discount/Restock)
   - **💸 Discounts**: Discount recommendations with revenue calculations
   - **📦 Restock**: Items needing reorder
   - **🗑️ Remove**: Expired items with loss value calculation
   - **🤝 Donations**: Comprehensive donation management

9. **Donation Management Tab**:
   - **Summary Metrics**: Eligible, Pending, Donated, Rejected counts
   - **Status Pie Chart**: Distribution visualization
   - **Top Cities** and **Top NGOs** lists (top 5)
   - **Filters**: City and Category dropdowns
   - **Data Table**: Key columns for pending donations
   - **Bulk Actions**:
     - "Mark All Pending → Donated" button
     - "Mark All Pending → Rejected" button
     - "Export CSV" download button
   - **Individual Actions**: Accept/Reject buttons (for ≤10 items)
   - **Geographic Map**: Scatter mapbox showing donation locations colored by status

10. **Performance Optimizations**:
    - Caching with TTL (600 seconds)
    - Selective column loading
    - Display limit (max 500 rows in tables)
    - Session state for user changes
    - Efficient filtering (boolean masks)

**Inputs**: Processed inventory data CSVs
**Outputs**: Interactive web interface, updated donation statuses
**Dependencies**: streamlit, pandas, plotly, numpy
**Technologies**: Streamlit reactive framework, Plotly interactive charts, caching

**URL**: `http://localhost:8501` (default Streamlit port)

---

### 2.10 evaluation/ - Model Validation

**Purpose**: Comprehensive model performance analysis and reporting

#### evaluate_expiry_model.py

**Implementation**: Standalone evaluation script

**Process**:
1. Load trained model: `models/expiry_predict_model.pkl`
2. Load data: `data/processed/expiry_risk_predictions.csv` or `cleaned_inventory_data.csv`
3. Compute same features as training
4. Sample 10% for evaluation (performance)
5. Generate predictions
6. Calculate metrics:
   - Accuracy, Precision, Recall, F1-Score (weighted)
   - ROC-AUC (multi-class OvR)
   - Classification report per class
   - 5-fold cross-validation scores
7. Generate visualizations:
   - **Confusion Matrix** (counts and percentages)
   - **Feature Importance** (from RandomForest component)
   - **Class Distribution** (actual vs predicted)
   - **Per-Class Metrics** (Precision/Recall/F1 bar chart)
8. Save report: `evaluation/results/expiry_model_report.txt`
9. Save plots: `plots/expiry_*.png`

**Output Metrics**:
- CV Accuracy: ~75-85% (5-fold mean ± std)
- Per-class metrics for Safe, Near Expiry, Expired

**Files Generated**:
- `expiry_confusion_matrix.png`
- `expiry_feature_importance.png`
- `expiry_class_distribution.png`
- `expiry_per_class_metrics.png`
- `expiry_model_report.txt`

#### evaluate_demand_model.py

**Implementation**: Standalone evaluation script

**Process**:
1. Load trained model: `models/demand_forecast_model.pkl`
2. Load data and engineer features
3. Sample 10% for evaluation
4. Generate predictions
5. Calculate metrics:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² Score
   - MAPE (Mean Absolute Percentage Error)
   - 5-fold cross-validation R² scores
6. Generate visualizations:
   - **Actual vs Predicted** scatter plot (5000 sample points)
   - **Residuals** (scatter + histogram with KDE)
   - **Feature Importance** (from RandomForest component)
   - **Error Distribution** (percentage error histogram)
7. Save report: `evaluation/results/demand_model_report.txt`
8. Save plots: `plots/demand_*.png`

**Output Metrics**:
- R²: ~0.65-0.75 (explains 65-75% variance)
- CV R² Mean: ~0.65-0.75 (5-fold)

**Files Generated**:
- `demand_actual_vs_predicted.png`
- `demand_residuals.png`
- `demand_feature_importance.png`
- `demand_error_distribution.png`
- `demand_model_report.txt`

#### run_all_evaluations.py

**Purpose**: Execute both evaluations and generate combined summary

**Process**:
1. Run `evaluate_expiry_model.run_evaluation()` → expiry_metrics dict
2. Run `evaluate_demand_model.run_evaluation()` → demand_metrics dict
3. Generate combined summary table
4. List all output files with existence check (✅/❌)
5. Print comprehensive report

**Combined Output**:
- Expiry: Accuracy, ROC-AUC, F1 scores, CV metrics
- Demand: MAE, RMSE, R², MAPE, CV metrics
- File locations for all plots and reports

**Inputs**: Trained models, processed data
**Outputs**: Comprehensive evaluation reports, visualization plots
**Dependencies**: scikit-learn, matplotlib, seaborn, pandas, numpy
**Technologies**: Cross-validation, confusion matrices, ROC curves, residual analysis

---

## 3. COMPLETE WORKFLOW

### 3.1 Inventory Management Workflow

**Trigger**: Daily batch execution (`python main.py`)

**Steps**:
1. **Data Loading**:
   - Load from `data/cleaned_inventory_data.csv`
   - If not found, try raw data sources
   - If still not found, generate synthetic data
   
2. **Stock Analysis**:
   - Calculate `rolling_avg_sales_7` (7-day moving average)
   - Compute `weekly_demand = rolling_avg_sales_7 × 7`
   - Classify stock levels:
     - `current_stock > weekly_demand × 1.5` → **High**
     - `current_stock < weekly_demand × 0.5` → **Low**
     - Otherwise → **Normal**

3. **Expiry Analysis**:
   - Check `days_to_expiry` against thresholds:
     - `> 15 days` → **Safe**
     - `1-15 days` → **Near Expiry**
     - `-5 to 0 days` → **Expired** (donation-eligible)

     - `< -5 days` → **Remove** (too old)

4. **Action Determination**:
   - Decision tree:
     ```
     IF days_to_expiry < -5:
         Action = 'Remove'
     ELIF days_to_expiry in [-5, -1] AND donation_eligible:
         Action = 'Donate'
     ELIF days_to_expiry in [0, 5]:
         Action = 'Apply Discount'
     ELIF Stock_Level == 'Low':
         Action = 'Restock'
     ELSE:
         Action = 'No Action'
     ```

5. **Output Generation**:
   - Save `inventory_analysis_results.csv`
   - Save `inventory_analysis_results_enhanced.csv` (with donations)
   - Update dashboard cache

**End Result**: Updated CSV files ready for dashboard consumption

**Processing Time**: ~30-60 seconds for 10,000 items

---

### 3.2 Demand Prediction Workflow

**Trigger**: Part of main pipeline execution (Phase 1.5)

**Steps**:
1. **Load Data**: Read `cleaned_inventory_data.csv`
2. **Feature Engineering**:
   - Compute rolling averages (3, 7, 14, 30 days)
   - Create lag features (1-day, 7-day lags)
   - Calculate trends (rolling_avg_7 - rolling_avg_30)
   - Add temporal features (day_of_week, month, quarter)
3. **Model Inference**:
   - Load `models/demand_forecast_model.pkl`
   - Predict: `demand_forecast = model.predict(features)`
4. **Validation**:
   - 5-fold cross-validation on 10% sample
   - Log R² score and statistics
5. **Output**:
   - Add `predicted_demand` column to DataFrame
   - No separate file saved (integrated into analysis)

**Decision Logic**: 
- If `predicted_demand > current_stock + safety_buffer` → Recommend restock
- Safety buffer = 20% of predicted demand

**End Result**: Each item has predicted sales for next period

**Exceptions**: 
- Negative predictions clipped to 0
- Missing features filled with defaults

---

### 3.3 Expiry Prediction Workflow

**Trigger**: Part of main pipeline execution (Phase 1)

**Steps**:
1. **Load Data**: Read `cleaned_inventory_data.csv`
2. **Feature Engineering**:
   - Compute `stock_to_sales_ratio = current_stock / rolling_avg_sales_7`
   - Calculate `shelf_consumed_ratio = days_on_shelf / shelf_life`
   - Create urgency metrics
3. **Model Inference**:
   - Load `models/expiry_predict_model.pkl`
   - Predict: `expiry_risk = model.predict_proba(features)[:, 1]`
   - Classify: `expiry_prediction = model.predict(features)`
4. **Validation**:
   - 5-fold stratified cross-validation on 10% sample
   - Log accuracy and performance metrics
5. **Output**:
   - Save `data/processed/expiry_risk_predictions.csv`
   - Columns: store_nbr, date, days_to_expiry, expiry_risk, expiry_prediction

**Decision Logic**:
- If `expiry_risk > 0.7` (70% probability) → Flag as at-risk
- If `days_to_expiry ≤ 3` → Urgent action needed

**End Result**: Each item classified with expiry risk score (0.0-1.0)

**Exceptions**: 
- Items with `days_to_expiry > 30` assumed safe
- Missing shelf_life uses category defaults

---

### 3.4 Alert Generation Workflow

**Trigger**: Continuously during inventory analysis

**Processing Steps**:
1. **High Priority Alerts**:
   - `Action = 'Remove'` → CRITICAL alert (expired, remove immediately)
   - `Expiry_Risk = 'Expired'` AND `Stock_Level = 'High'` → CRITICAL (high waste risk)
   
2. **Medium Priority Alerts**:
   - `Action = 'Apply Discount'` → WARNING (near expiry, discount needed)
   - `Stock_Level = 'Low'` AND `Reorder = 'Yes'` → WARNING (stockout risk)
   
3. **Low Priority Alerts**:
   - `Stock_Level = 'High'` → INFO (overstocked, monitor)
   - `Expiry_Risk = 'Near Expiry'` → INFO (watch closely)

**Notification Logic**:
- **Not Implemented**: No email, SMS, or push notifications
- Alerts visible only in dashboard

**Output**: Alert data displayed in dashboard tabs

**End Result**: Store managers see prioritized action list

---

### 3.5 Reorder Workflow

**Trigger**: Daily after inventory analysis

**Processing Steps**:
1. **Eligibility Check**:
   - `Stock_Level = 'Low'` OR
   - `current_stock < rolling_avg_sales_7 × 3` OR
   - `predicted_demand > current_stock`
   
2. **Quantity Calculation**:
   ```python
   base_restock = max(0, predicted_demand - current_stock)
   safety_stock = base_restock × 1.2
   max_restock = predicted_demand × 3.0
   restock_qty = min(safety_stock, max_restock)
   ```
   
3. **Priority Assignment**:
   - `urgency_score = (predicted_demand / current_stock) × (1 - days_to_expiry/30) × 100`
   - High priority if urgency_score > 70

4. **Output Generation**:
   - Save `data/processed/restocking_suggestions.csv`
   - Save `data/processed/high_priority_restock.csv` (urgent only)

**Decision Logic**:
- Never restock if `days_to_expiry ≤ 2` (will expire before sale)
- Never restock if `donation_status = 'Pending'` (already being donated)

**End Result**: Purchase orders ready for supplier system

**Manual Override**: Not implemented (would require UI integration)

---

### 3.6 Supplier Notification Workflow

**Status**: **Not Implemented**

**Expected Flow** (if implemented):
1. Generate restock suggestions (done)
2. Create purchase order data structure
3. Send to supplier API/email
4. Track delivery ETA
5. Update inventory on delivery

**Current State**: CSV files generated only, no automated notifications

---

### 3.7 Donation Workflow

**Trigger**: Daily during donation processing (Phase 3.5)

**Processing Steps**:
1. **Eligibility Determination**:
   ```python
   donation_eligible = (
       -5 ≤ days_to_expiry ≤ -1 AND
       perishable == 1 AND
       category in ['DAIRY', 'PRODUCE', 'MEATS', 'BREAD/BAKERY', 'FROZEN FOODS', 'SNACKS', 'BEVERAGES']
   )
   ```

2. **NGO Matching**:
   - Get store city from data
   - Look up nearest NGO using `get_nearest_ngo(city)`
   - Assign: `nearest_ngo`, `ngo_address`, `ngo_contact`

3. **Status Management**:
   - New eligible items → `donation_status = 'Pending'`
   - Store manager marks as 'Donated' via dashboard
   - Can mark as 'Rejected' if NGO declines
   - Items with `donation_status != 'Pending'` excluded from restock

4. **Output Generation**:
   - Save `data/processed/donation_summary.csv` (all eligible items)
   - Save `data/processed/pending_donations.csv` (only pending)
   - Include columns: item_id, product_name, category, current_stock, days_to_expiry, city, nearest_ngo, ngo_contact

**Decision Logic**:
- Items too old (`days_to_expiry < -5`) → 'Remove' instead of 'Donate'
- Items already donated → Excluded from future processing
- Items pending donation → Excluded from restock recommendations

**End Result**: NGO pickup list generated, awaiting manual confirmation

**Exceptions**:
- City not in NGO database → Uses default NGO
- Item becomes ineligible after assignment → Status reset to ''

---

### 3.8 Disposal Workflow

**Status**: **Partially Implemented**

**Current Implementation**:
1. Items with `Action = 'Remove'` flagged in analysis
2. Saved to `data/processed/removed_items.csv`
3. Total loss value calculated: `sum(unit_price × current_stock)`

**Not Implemented**:
- No automated disposal tracking
- No integration with waste management systems
- No compliance logging

**Expected Flow** (if fully implemented):
1. Flag items for removal
2. Generate disposal manifest
3. Log environmental impact
4. Track disposal costs
5. Compliance reporting

---

### 3.9 Manual Override Workflow

**Status**: **Not Implemented**

**Expected Flow** (if implemented):
1. Store manager reviews recommendations
2. Can override: Accept/Reject/Modify
3. System tracks override history
4. Model retrains with feedback

**Current State**: Dashboard shows recommendations only, no override mechanism

---

### 3.10 Stock Update Workflow

**Status**: **Not Implemented** (Real-time updates)

**Current Implementation**: Batch processing only

**Expected Real-time Flow** (future):
1. POS system records sale → Update `current_stock`
2. New delivery arrives → Update `current_stock`
3. Item expires/disposed → Update `current_stock`
4. Triggers:
   - If `current_stock < threshold` → Generate alert
   - If approaching expiry → Update expiry risk

---

### 3.11 Daily Processing Schedule

**Current Schedule**: Manual execution (`python main.py`)

**Execution Sequence**:
```
00:00 - Start pipeline
00:05 - Data preprocessing complete
00:10 - Expiry model training/inference complete
00:15 - Demand model training/inference complete
00:20 - Inventory analysis complete
00:25 - Donation processing complete
00:30 - Restock plan generation complete
00:35 - Dashboard data refresh complete
```

**Not Implemented**:
- No cron job / scheduler
- No automated execution
- No failure recovery

**Expected Production Schedule** (future):
- Daily at 2:00 AM (after EOD reconciliation)
- Hourly for real-time updates
- On-demand for urgent recalculations

---

### 3.12 Background Jobs

**Status**: **Not Implemented**

**Expected Jobs** (future):
- Model retraining (weekly)
- Data cleanup (daily)
- Report generation (daily)
- Alert notifications (real-time)
- Performance monitoring (continuous)

---

### 3.13 Notification Flow

**Status**: **Not Implemented**

**Current State**: Alerts visible only in dashboard

**Expected Flow** (future):
1. Critical alert generated → Email to store manager
2. Discount recommendation → SMS notification
3. Restock urgent → Push notification to mobile app
4. Donation matched → Email to NGO coordinator

---

## 4. DASHBOARD DESCRIPTION

### 4.1 Overview

**Technology**: Streamlit (Python web framework)
**URL**: `http://localhost:8501` (default)
**Layout**: Wide layout with sidebar filters

### 4.2 Pages/Sections

**Single Page Application** with multiple tabs and sections:

1. **Header**:
   - Title: "📊 Smart Inventory Management Dashboard"
   - Subtitle: "AI-Powered Inventory Analysis & Recommendations"

2. **Financial Impact Summary** (Top Section):
   - 4 metric cards:
     - 💸 Revenue from Discounts (₹)
     - 🧡 Donated Goods Value (₹)
     - 🗑️ Loss from Removals (₹, inverse delta)
     - 📊 Net Financial Impact (₹, color-coded)
   - 2 additional metrics:
     - 📈 Value Recovery Rate (%)
     - 📦 Items Processed (count)

3. **Key Performance Indicators**:
   - Row 1 (4 cards):
     - Total Items
     - Inventory Value (₹)
     - High Risk Items
     - Reorder Needed
   - Row 2 (4 cards):
     - Overstocked
     - Understocked
     - Near Expiry
     - Avg Discount (%)

4. **Visualization Charts** (2 columns):
   - Left: 📊 Stock Level Distribution (Pie Chart)
   - Right: ⚠️ Expiry Risk Analysis (Pie Chart)
   - Full width: ⚡ Action Recommendations (Bar Chart)
   - Full width: 💰 Inventory Value by Action (Bar Chart)

5. **Action Tabs** (5 tabs):
   
   **Tab 1: 🔴 Urgent**
   - Items requiring immediate attention
   - Filtered: `Action in ['Remove', 'Apply Discount', 'Restock']`
   - Sorted: By Action, then days_to_expiry
   - Columns: item_id, product_name, store_nbr, current_stock, Stock_Level, Expiry_Risk, days_to_expiry, Action
   - Display limit: 500 rows
   
   **Tab 2: 💸 Discounts**
   - Items with discount recommendations
   - Filtered: `Action == 'Apply Discount'`
   - Sorted: By Suggested_Discount (descending)
   - Columns: item_id, product_name, store_nbr, current_stock, Suggested_Discount, Expiry_Risk, unit_price
   - Calculated columns:
     - Original_Value = current_stock × unit_price
     - Discount_Amount = Original_Value × (Suggested_Discount / 100)
     - Revenue_After = Original_Value - Discount_Amount
   - Summary metrics:
     - 💰 Original: ₹X (total original value)
     - 🏷️ Discount: ₹Y (total discount amount)
     - 💵 Revenue: ₹Z (total revenue after discount)
   
   **Tab 3: 📦 Restock**
   - Items needing reorder
   - Filtered: `Reorder == 'Yes'`
   - Sorted: By current_stock (ascending)
   - Columns: item_id, product_name, store_nbr, current_stock, rolling_avg_sales_7, Stock_Level, days_to_expiry
   
   **Tab 4: 🗑️ Remove**
   - Expired items to remove
   - Filtered: `Action == 'Remove'`
   - Sorted: By days_to_expiry (ascending)
   - Columns: item_id, product_name, store_nbr, current_stock, days_to_expiry, unit_price
   - Calculated column:
     - Loss_Value = current_stock × unit_price
   - Error message: ⚠️ Total loss: ₹X
   
   **Tab 5: 🤝 Donations** (Most Complex)