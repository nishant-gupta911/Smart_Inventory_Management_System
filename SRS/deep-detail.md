# Deep detail

## Table of contents

- [Repository-level orientation](#repository-level-orientation)
- [Folder-by-folder breakdown](#folder-by-folder-breakdown)
- [Most important files in the repo](#most-important-20-files-in-this-repo)
- [Critical path through the code](#critical-path-through-the-code)
- [Key classes, functions, and responsibilities](#key-classes-functions-and-responsibilities)
- [Core algorithms and logic flows](#core-algorithms-and-logic-flows)
- [Important schemas and contracts](#important-schemas-and-contracts)
- [API surface and behavior](#api-surface-and-behavior)
- [Important state transitions](#important-state-transitions)
- [Edge cases and hidden complexity](#edge-cases-and-hidden-complexity)
- [Test strategy and gaps](#test-strategy-and-gaps)
- [Known or likely risk areas](#known-or-likely-risk-areas)
- [Questions to ask original developers](#questions-to-ask-the-original-developers)
- [What to say in an interview](#what-to-say-in-an-interview)
- [What to study next](#what-to-study-next)
- [Open questions / uncertainties](#open-questions--uncertainties)

## Repository-level orientation

This repo is a single-machine retail AI demo with a business-oriented decision layer. The code is not organized around authenticated services or a multi-tier application. Instead, it is built around Python modules, DataFrame transformations, model training, a dashboard, and CSV outputs.

The repo is not small enough to ignore, but it also is not a large platform. Most of the important complexity is concentrated in a few files rather than across dozens of independent services.

## Folder-by-folder breakdown

### Root directory

The root includes the orchestration entry point, documentation, sample and generated data, and one-off scripts. Files like [main.py](main.py), [README.md](README.md), [transform_inventory_data.py](transform_inventory_data.py), and [convert_real_data.py](convert_real_data.py) are not just config; they are working components of the project lifecycle.

### [src](src)

This is the main implementation folder. It contains the core data and ML logic. The most critical files are:

- [src/data_preprocessing.py](src/data_preprocessing.py)
- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [src/train_demand_model.py](src/train_demand_model.py)
- [src/train_expiry_model.py](src/train_expiry_model.py)
- [src/generate_restock_plan.py](src/generate_restock_plan.py)
- [src/utils.py](src/utils.py)

This is the technical center of the project.

### [data](data)

This contains both raw and processed data. It is the project’s durable state layer. Most of the repo’s “system memory” lives here, in CSV format.

### [dashboard](dashboard)

This is the display layer. The code is a Streamlit app that consumes processed data and renders business-facing charts and metrics.

### [models](models)

This is the model artifact directory, but the file list in the workspace snippet is not fully detailed. This is the expected location for trained joblib model files.

### [TEST](TEST)

This is the project’s test folder and it is one of the clearest signals of the repo’s maturity. It has targeted tests for dashboard loading, donation logic, action logic, and main integration.

### [logs](logs)

This is the operational logging area. It records runtime information created by the pipeline.

### [notebooks](notebooks)

This folder suggests experimentation and exploratory modeling. The notebooks are likely earlier-phase prototypes or notebooks used during training experiments.

## Most important 20 files in this repo

1. [main.py](main.py)
2. [README.md](README.md)
3. [src/data_preprocessing.py](src/data_preprocessing.py)
4. [src/inventory_analyzer.py](src/inventory_analyzer.py)
5. [src/train_demand_model.py](src/train_demand_model.py)
6. [src/train_expiry_model.py](src/train_expiry_model.py)
7. [src/generate_restock_plan.py](src/generate_restock_plan.py)
8. [src/utils.py](src/utils.py)
9. [transform_inventory_data.py](transform_inventory_data.py)
10. [dashboard/app.py](dashboard/app.py)
11. [requirement.txt](requirement.txt)
12. [data/cleaned_inventory_data.csv](data/cleaned_inventory_data.csv)
13. [data/processed/inventory_analysis_results_enhanced.csv](data/processed/inventory_analysis_results_enhanced.csv)
14. [data/processed/inventory_analysis_results.csv](data/processed/inventory_analysis_results.csv)
15. [data/raw/train.csv](data/raw/train.csv)
16. [data/raw/items.csv](data/raw/items.csv)
17. [data/raw/stores.csv](data/raw/stores.csv)
18. [data/raw/oil.csv](data/raw/oil.csv)
19. [TEST/test_main_integration.py](TEST/test_main_integration.py)
20. [TEST/test_action_logic.py](TEST/test_action_logic.py)

This list is designed to reflect actual project gravity rather than every utility file.

## Critical path through the code

The most important runtime path is:

```text
main.py
  -> import modules
  -> ensure directories
  -> run preprocessing
  -> train demand and expiry models
  -> run inventory analysis
  -> save dashboard data
  -> launch dashboard/app.py
``` 

The key logic path inside the analysis is:

```text
InventoryAnalyzer.load_inventory_data()
  -> _prepare_data()
  -> analyze_stock_levels()
  -> analyze_expiry_risk()
  -> calculate_discount_suggestions()
  -> determine_reorder_needs()
  -> determine_actions()
  -> generate_summary()
```

This is the real heart of the application.

## Key classes, functions, and responsibilities

### `InventoryAnalyzer`

[Main responsibilities]

- load data from multiple paths
- generate synthetic fallback data when none exists
- normalize and validate values
- classify stock
- classify expiry risk
- compute suggested discounts
- decide on reorder needs and action outputs

This is the clearest central class in the repo.

### `RestockPlanGenerator`

This class is more opinionated about restocking logic. It loads data, fills feature gaps, computes predictions, and produces a restock recommendation. The design is simpler than a real inventory optimizer but useful as a prototype.

### `update_donation_status`

This utility updates a donation state for a specific item and enforces that the item is donation-eligible before modifying the status.

### `get_nearest_ngo`

This maps city names to NGO metadata, which is a practical business integration point for donation coordination.

### `preprocess_data`

This function computes the core demand and expiry features. It is effectively the project’s default schema builder for operational metrics.

### `train_and_evaluate` and `train_and_predict`

These functions are the machine-learning entry points. They create a sample of data, build an ensemble model, evaluate, fit on the full dataset, and save the artifact.

## Core algorithms and logic flows

### Feature engineering

The preprocessing layer strongly emphasizes features like:

- rolling averages over 3, 7, 14, and 30 windows
- sales lag 1 and 7
- sales trend
- shelf and stock ratios
- day-of-week, month, weekend indicators
- urgency score and sales velocity

This is sensible for retail forecasting and is clearly designed around business understanding.

### Action decision logic

The action logic is rule-heavy rather than model-driven. It assigns actions based on stock and expiry thresholds. This is not a pure ML system; it is a hybrid of prediction + rules.

The result is a system that is easy to explain and easier to validate in practice.

### Donation logic

Donation eligibility is based on a combination of expiry-window, perishability, and category. This is explicitly encoded in [transform_inventory_data.py](transform_inventory_data.py) and indirectly enforced elsewhere.

## Important schemas and contracts

Because there is no database schema file, the practical schema is defined by DataFrames. The most important columns are:

- `item_id`
- `product_name`
- `store_nbr`
- `city`
- `category`
- `current_stock`
- `rolling_avg_sales_7`
- `days_to_expiry`
- `unit_price`
- `Stock_Level`
- `Expiry_Risk`
- `Action`
- `Suggested_Discount`
- `Reorder`
- `donation_eligible`
- `donation_status`
- `nearest_ngo`

This is the effective contract between modules.

## API surface and behavior

The repo does not have a formal REST or GraphQL interface. The API surface is Pythonic:

- import modules and call functions
- call `InventoryAnalyzer().run_full_analysis()` or similar
- direct reads and writes to CSVs
- Streamlit app handles UI interactions in-process

This means the interviews should describe the project as a workflow and local product, not a web service.

## Important state transitions

### Data processing state

Raw data -> cleaned/engineered data -> trained model artifacts -> processed inventory dataset -> dashboard data.

### Donation state

Donation-eligible item -> status Pending -> Donated or Rejected.

### Operational state

Overstock or near-expiry item -> recommendation -> user action or summary.

## Edge cases and hidden complexity

- Missing or malformed columns are common, so fallback logic is necessary.
- Data sample generation can hide underlying issues in demos.
- Donation logic is spread across multiple modules and may drift from one source of truth.
- Action logic uses thresholds that are not explicitly centralized in one config file.
- The code frequently assumes a stable root directory structure and working directory.

## Test strategy and gaps

There is a legitimate test directory and a clear attempt to validate functionality. However, the tests read mostly as focused behavior checks rather than a comprehensive QA strategy.

Key tests in [TEST](TEST) cover:

- action logic
- dashboard expectations
- donation filtering
- donation functions
- main integration

The gap is that there is no visible CI or automated environment configuration that would enforce these checks consistently.

## Known or likely risk areas

1. Data contract drift between modules.
2. Duplicate business logic for action and donation calculations.
3. Synthetic data fallback masking true quality issues.
4. Local file dependence making the project brittle in different working environments.
5. No strong security model or auth model.
6. No prod deployment or orchestration layer.

## Questions to ask the original developers

- What is the real source-of-truth for action logic?
- Is the donation logic intentionally duplicated or is it still in migration?
- Why is the dashboard built with Streamlit rather than a backend + frontend split?
- Was the project always intended to be local-first, or is this a temporary prototype state?
- What was the actual production target if the system were to scale beyond a demo?

## What to say in an interview

- “The repo’s core complexity is in feature engineering and action logic, not in web plumbing.”
- “The project is a hybrid of ML and rules-based decisioning.”
- “The biggest weakness is that the system is optimized for demo clarity rather than enterprise operational hardening.”

## What to study next

- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [src/data_preprocessing.py](src/data_preprocessing.py)
- [transform_inventory_data.py](transform_inventory_data.py)
- [dashboard/app.py](dashboard/app.py)
- [TEST](TEST)

## Open questions / uncertainties

- The precise source-of-truth between `InventoryAnalyzer` and `transform_inventory_data` is not fully resolved from the visible code.
- The repo’s documentation suggests a more ambitious architecture than the actual implementation supports.
- There is no clear deployment or release story despite the project claiming broader business value.
