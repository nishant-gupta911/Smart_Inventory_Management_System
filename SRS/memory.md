# Memory aid

## 30-second version

This project is a Python-based retail inventory optimization system. It reads inventory data, engineers features, trains demand and expiry models, and then recommends actions like restocking, discounting, or donation. The results are shown in a Streamlit dashboard and saved as CSV files for business review.

## 2-minute version

The repository is a local, file-based analytics pipeline for retail inventory risk management. [main.py](main.py) orchestrates the flow, [src/data_preprocessing.py](src/data_preprocessing.py) creates the feature set, [src/train_demand_model.py](src/train_demand_model.py) and [src/train_expiry_model.py](src/train_expiry_model.py) train prediction models, and [src/inventory_analyzer.py](src/inventory_analyzer.py) converts their outputs into business actions. The project uses CSV data, joblib model files, and a Streamlit dashboard rather than a full app backend or database. It also contains a donation workflow that identifies near-expiry items suitable for charity with NGO mapping in [src/utils.py](src/utils.py).

## 5-minute revision version

Think of the system as four layers:

1. Data and preprocessing: [src/data_preprocessing.py](src/data_preprocessing.py), [transform_inventory_data.py](transform_inventory_data.py)
2. Prediction: [src/train_demand_model.py](src/train_demand_model.py), [src/train_expiry_model.py](src/train_expiry_model.py)
3. Decision logic: [src/inventory_analyzer.py](src/inventory_analyzer.py), [src/generate_restock_plan.py](src/generate_restock_plan.py)
4. Presentation: [dashboard/app.py](dashboard/app.py)

The business problem is reducing waste and stock imbalance in retail. The repo emphasizes operational recommendations such as low-stock restock, discounts for near-expiry inventory, removal of expired items, and donation opportunities for eligible goods. The architecture is intentionally simple and local-first, which is good for demos and prototyping but not yet a full production system.

## One-page project cheat sheet

### Core purpose

Minimize retail waste and improve inventory decision making through ML-based demand forecasting and expiry-risk analysis.

### Target users

- store managers
- supply chain or inventory admins
- business stakeholders evaluating risk and waste
- donation and CSR workflows

### Main modules

| Module | What it does |
| --- | --- |
| [main.py](main.py) | orchestration | 
| [src/data_preprocessing.py](src/data_preprocessing.py) | feature engineering and data preparation |
| [src/train_demand_model.py](src/train_demand_model.py) | sales prediction model |
| [src/train_expiry_model.py](src/train_expiry_model.py) | expiry-risk model |
| [src/inventory_analyzer.py](src/inventory_analyzer.py) | action logic and classification |
| [src/generate_restock_plan.py](src/generate_restock_plan.py) | restock recommendations |
| [src/utils.py](src/utils.py) | donation info and summaries |
| [dashboard/app.py](dashboard/app.py) | marketing/demo dashboard |
| [transform_inventory_data.py](transform_inventory_data.py) | augmentation and donation transformation |

### Key entities and relationships

| Entity | Meaning | Key fields |
| --- | --- | --- |
| Item | inventory record | item_id, product_name, category |
| Store | site context | store_nbr, city |
| Inventory snapshot | current stock and sales | current_stock, rolling_avg_sales_7, sales |
| Product risk | expiry and stock urgency | days_to_expiry, Stock_Level, Expiry_Risk |
| Action | recommended operational response | Action, Reorder, Suggested_Discount |
| Donation record | donation eligibility and coordination | donation_eligible, donation_status, nearest_ngo |

### Critical files to remember

- [main.py](main.py)
- [src/data_preprocessing.py](src/data_preprocessing.py)
- [src/train_demand_model.py](src/train_demand_model.py)
- [src/train_expiry_model.py](src/train_expiry_model.py)
- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [dashboard/app.py](dashboard/app.py)
- [src/utils.py](src/utils.py)
- [transform_inventory_data.py](transform_inventory_data.py)
- [requirement.txt](requirement.txt)
- [README.md](README.md)

### Key dependencies and why they matter

- pandas: data wrangling and CSV handling
- numpy: numeric feature engineering
- scikit-learn: ML model training and CV
- xgboost and lightgbm: stronger ensemble models
- streamlit: dashboard UI
- plotly: charts and business visualization
- joblib: model persistence

### Security/auth summary

There is no explicit auth model visible in the repo. The project is local and demo-first; there are no login, RBAC, API keys, or secrets management patterns in the codebase.

### Deployment summary

- local Python execution
- CSV and joblib persistence
- Streamlit dashboard
- no CI, Dockerfile, or cloud manifests found in the repo evidence

## Top 10 things to remember for interviews

1. The project solves retail waste and stock imbalance using AI.
2. It has a local Python pipeline rather than a web API backend.
3. Demand and expiry are modeled separately, then merged in operational logic.
4. The repo uses file-based data interchange and fallback logic.
5. The action engine is in [src/inventory_analyzer.py](src/inventory_analyzer.py).
6. Donation logic is a business extension, not just a model output.
7. The dashboard is a demo product layer, not a full web application.
8. The repo is more prototype-oriented than production-oriented.
9. Model training is ensemble-based and local.
10. There is no formal auth or deployment strategy in the repository evidence.

## Top 10 easy-to-forget details

1. The app often falls back to synthetic data when files are missing.
2. Action logic uses column names like `Action`, `Reorder`, `Suggested_Discount`, and `donation_status`.
3. Donation eligibility is tied to expiry and category semantics.
4. The repo uses repo-relative paths and expects a consistent working directory.
5. The model training code uses sample-based cross-validation, not necessarily full dataset training.
6. The app is local-first and not a multi-user system.
7. A “React frontend” mention in docs is not the actual implementation evidence in the repo.
8. `joblib` is used for model serialization in addition to CSV outputs.
9. The dashboard reads from processed CSVs and sometimes writes back status updates.
10. The strongest architecture is the combination of model predictions + rule-based actions.

## Mnemonic-style summaries

### “D-A-D-A”

Data -> Action -> Dashboard -> Analysis

### “P-D-M-A”

Preprocess -> Detect demand -> Model expiry -> Act on results

### “W-S-D”

Waste reduction, stock optimization, donation workflow

## What to say in an interview

- “This is a retail decision-support pipeline with demand forecasting, expiry-risk scoring, and action recommendation.”
- “The core pattern is file-based ML + rules + dashboard, not microservices.”
- “The repo is a prototype-focused system optimized for clarity and demo value.”

## What to study next

- [main.py](main.py)
- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [src/data_preprocessing.py](src/data_preprocessing.py)
- [dashboard/app.py](dashboard/app.py)
- [README.md](README.md)

## Open questions / uncertainties

- The exact chronology of the project’s phases is inferred from code and docs rather than from formal project history.
- Some files may have overlapping responsibilities, suggesting iterative growth and partial refactoring.
- The project describes extension to a more production-grade architecture, but the current repo evidence remains local and prototype-oriented.
