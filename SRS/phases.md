# Project phases

## Table of contents

- [Overview](#overview)
- [Product and build phases](#product-and-build-phases)
- [Runtime and user flow phases](#runtime-and-user-flow-phases)
- [If I had to rebuild this project from scratch](#if-i-had-to-rebuild-this-project-from-scratch-in-what-order-would-i-build-it)
- [Why this phased approach makes sense](#why-this-phased-approach-makes-sense)
- [Alternative build orders and why they may be worse](#alternative-build-orders-and-why-they-may-be-worse)
- [What to say in an interview](#what-to-say-in-an-interview)
- [What to study next](#what-to-study-next)
- [Open questions / uncertainties](#open-questions--uncertainties)

## Overview

The repository reads like a staged project rather than a single giant implementation. The progression implied by the code is:

1. collect and clean inventory data
2. derive business signals from raw signals
3. train models for demand and expiry
4. turn predictions into operational actions
5. show the outputs in a dashboard
6. support donation and waste-reduction workflows

This sequence is visible in [main.py](main.py), [src/data_preprocessing.py](src/data_preprocessing.py), [src/train_demand_model.py](src/train_demand_model.py), [src/train_expiry_model.py](src/train_expiry_model.py), [src/inventory_analyzer.py](src/inventory_analyzer.py), and [dashboard/app.py](dashboard/app.py).

## Product and build phases

### Phase 1: data and feasibility

Likely first phase: get a usable inventory dataset and prove the core retail problem.

Evidence:

- [data/raw](data/raw) contains likely source datasets.
- [src/data_preprocessing.py](src/data_preprocessing.py) builds the basic feature layer.
- [README.md](README.md) explicitly frames the problem as retail waste and stock optimization.

This phase likely focused on: obtaining data, cleaning it, normalizing fields, and confirming the key variables such as `current_stock`, `rolling_avg_sales_7`, and `days_to_expiry`.

### Phase 2: prediction models

Likely second phase: train the specific predictive models.

Evidence:

- [src/train_demand_model.py](src/train_demand_model.py) is a sales forecasting model.
- [src/train_expiry_model.py](src/train_expiry_model.py) is an expiry-risk classification model.
- [main.py](main.py) orchestrates model execution after preprocessing.

This phase likely established the “AI” portion of the project: the project is not just a dashboard; it tries to predict outcomes and create decisions based on them.

### Phase 3: decision logic and action planning

Likely third phase: take the model outputs and convert them into actions.

Evidence:

- [src/inventory_analyzer.py](src/inventory_analyzer.py) computes `Stock_Level`, `Expiry_Risk`, `Suggested_Discount`, `Reorder`, and `Action`.
- [src/generate_restock_plan.py](src/generate_restock_plan.py) focuses on restock planning.
- [transform_inventory_data.py](transform_inventory_data.py) adds donation and action logic.

This phase is the business layer. It turns raw model predictions into something a manager can act on.

### Phase 4: dashboard and usage flow

Likely fourth phase: present insights in a way that is easy to use in demos and business meetings.

Evidence:

- [dashboard/app.py](dashboard/app.py) renders KPIs, filters, charts, and action cards.
- [README.md](README.md) emphasizes visual analytics and business demo use cases.

This is the phase that makes the project product-like rather than purely model-centric.

### Phase 5: donation and sustainability angle

The donation logic is a clear later extension. It adds a social-impact layer to the same operational logic.

Evidence:

- [src/utils.py](src/utils.py) includes NGO mapping and donation summaries.
- [transform_inventory_data.py](transform_inventory_data.py) adds `donation_eligible` and `donation_status` columns.
- [dashboard/app.py](dashboard/app.py) includes donation filtering and status updates.

This likely came after the core inventory logic because it adds business and social impact narratives to the project.

## Runtime and user flow phases

### Phase A: app start and initialization

When the user starts the project:

1. [main.py](main.py) sets up logging and ensures directories exist.
2. It attempts to import the main modules.
3. It checks for existing data or creates a fallback dataset if needed.
4. It may train models or skip if artifacts already exist.

This is a start-up and safety phase.

### Phase B: preprocessing and feature generation

The real computational work begins when data is transformed:

- [src/data_preprocessing.py](src/data_preprocessing.py) computes rolling averages, lag features, time features, shelf-life features, and expiry labels.
- The transformed data is saved to [data/processed](data/processed).

### Phase C: prediction and scoring

Next, models produce outputs:

- demand model predicts sales values
- expiry model predicts risk categories or labels
- output artifacts are saved for reuse

The project stores the trained artifacts in [models](models) and derived CSVs in [data/processed](data/processed).

### Phase D: operational recommendation

Then the business rules take over:

- `Stock_Level` is assigned
- `Expiry_Risk` is assigned
- discount recommendations are computed
- restock recommendations are estimated
- donation eligibility is evaluated

This is the point where the model output becomes an action plan.

### Phase E: dashboard use

Finally, the user interacts with [dashboard/app.py](dashboard/app.py):

- select store or risk filter
- inspect KPIs and charts
- review recommended actions
- see donation opportunities and status
- optionally update donation state

This is the human-facing phase that gives the system its business value.

### Phase F: failure handling

The project handles failure mainly by fallback and logging rather than a robust retry system.

Example:

- missing data -> generate synthetic data
- missing model -> warn and continue if possible
- invalid values -> coerce to numeric and fill missing

This means the app is designed to keep running during demos rather than to recover from strongly degraded production conditions.

## If I had to rebuild this project from scratch, in what order would I build it?

1. Define the business logic and data model.
   - item, store, stock, expiry, action, donation status.
2. Build a small, deterministic pipeline around one clean CSV file.
3. Implement the preprocessing and feature engineering layer.
4. Build the demand and expiry models.
5. Write the action evaluation layer.
6. Add the dashboard UI.
7. Add donation workflow and export files.
8. Add robustness, fallback data, and tests.

This order matches the codebase structure. It gives the highest value early while preserving a clean path to product demo and operational insight.

## Why this phased approach makes sense

This approach is sensible because the project is first and foremost a decision-support system. The AI is useful only when it is connected to a coherent operational action layer. The repo’s structure reinforces that sequence: data -> model -> decision -> dashboard -> social-impact add-on.

It also matches typical prototype development workflows: prove the ML idea, then show a decision or product story to stakeholders.

## Alternative build orders and why they may be worse

### Alternative: build the dashboard first

This would produce a good UI but a weak logic foundation. It would make the product look polished before the actual decision quality is proven, which is risky when the scientific core is still uncertain.

### Alternative: start with a full production stack

This would slow the project down with auth, deployment, database schema, and service APIs before the business problem is validated. The repo does not suggest those were the main constraints.

### Alternative: start with donation logic only

This would bias the project toward social-impact features before the core inventory model is working, which would be less compelling and more fragile.

## What to say in an interview

- “The project evolved in phases: data foundation, ML forecasting, decision logic, and dashboard UI.”
- “The architecture was intentionally staged so the business question remained the driver.”
- “In a rebuild, I would keep the same sequence: data, features, models, action logic, dashboard.”

## What to study next

- [main.py](main.py) for orchestration ordering
- [src/data_preprocessing.py](src/data_preprocessing.py) for early-phase logic
- [src/inventory_analyzer.py](src/inventory_analyzer.py) for operational decision phase
- [dashboard/app.py](dashboard/app.py) for end-user experience phase

## Open questions / uncertainties

- There is no visible version history in the repo, so the exact build chronology is inferred from code structure rather than committed milestones.
- Some files appear to be demonstration-focused or duplicated across the project, which suggests the build may have evolved iteratively rather than through a strict planned lifecycle.
- The project reads as a prototype with social-impact and demo layers added later, not as a formal staged product roadmap.
