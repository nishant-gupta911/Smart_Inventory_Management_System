# Architecture

## Table of contents

- [Executive summary](#executive-summary)
- [High-level architecture overview](#high-level-architecture-overview)
- [Major components and responsibilities](#major-components-and-responsibilities)
- [Data flow through the system](#data-flow-through-the-system)
- [Runtime lifecycle](#runtime-lifecycle)
- [Persistence and data architecture](#persistence-and-data-architecture)
- [External integrations](#external-integrations)
- [Authentication and authorization](#authentication-and-authorization)
- [Error handling strategy](#error-handling-strategy)
- [Configuration and environment strategy](#configuration-and-environment-strategy)
- [Deployment and runtime model](#deployment-and-runtime-model)
- [Scalability considerations](#scalability-considerations)
- [Security considerations](#security-considerations)
- [Technical debt and architecture risks](#technical-debt-and-architecture-risks)
- [Why this architecture?](#why-this-architecture)
- [Architecture in plain English](#architecture-in-plain-english)
- [2-minute interview answer](#how-to-explain-this-architecture-in-a-2-minute-interview-answer)
- [Deep technical interview answer](#how-to-explain-this-architecture-in-a-deep-technical-interview)
- [What to say in an interview](#what-to-say-in-an-interview)
- [What to study next](#what-to-study-next)
- [Open questions / uncertainties](#open-questions--uncertainties)

## Executive summary

This repository is a Python-based retail intelligence pipeline built to reduce waste, forecast demand, and suggest operational actions such as restocking, discounting, or donation. The evidence in [main.py](main.py), [src/data_preprocessing.py](src/data_preprocessing.py), [src/train_demand_model.py](src/train_demand_model.py), [src/train_expiry_model.py](src/train_expiry_model.py), [src/inventory_analyzer.py](src/inventory_analyzer.py), and [dashboard/app.py](dashboard/app.py) shows a system that reads inventory data, engineers features, trains ML models, produces action recommendations, and exposes results through a Streamlit dashboard.

The architecture is intentionally simple and local-first. There is no web API layer, no database server, no authentication system, and no orchestration framework. The project uses CSV files as the primary persistence layer, scikit-learn and gradient-boosting models for prediction, and a dashboard for business-facing output. This is a good fit for a hackathon or prototype, and it is also a useful interview example because it demonstrates end-to-end ML + product thinking without hiding behind enterprise infrastructure.

## High-level architecture overview

The project follows a batch data science pipeline pattern:

1. Raw or synthetic data is loaded.
2. Features are engineered and validated.
3. ML models are trained and saved to disk.
4. Inventory analysis computes stock, expiry, discount, and donation actions.
5. Results are exported to CSV for the dashboard and downstream review.

The architecture is best described as:

- Python orchestration layer: [main.py](main.py)
- Data preparation layer: [src/data_preprocessing.py](src/data_preprocessing.py), [transform_inventory_data.py](transform_inventory_data.py)
- ML training layer: [src/train_demand_model.py](src/train_demand_model.py), [src/train_expiry_model.py](src/train_expiry_model.py)
- Business logic layer: [src/inventory_analyzer.py](src/inventory_analyzer.py), [src/generate_restock_plan.py](src/generate_restock_plan.py)
- Utility and donation layer: [src/utils.py](src/utils.py)
- UI layer: [dashboard/app.py](dashboard/app.py)
- Dataset and model artifacts: [data](data), [models](models), [logs](logs)

A simplified flow is:

```text
CSV data
  -> preprocessing + feature engineering
  -> demand model + expiry model
  -> action logic (stock, expiry, discount, donation)
  -> processed CSVs + dashboard cache
  -> Streamlit dashboard
```

## Major components and responsibilities

### 1. Orchestration layer

[main.py](main.py) acts as the entry point. It creates directories, imports modules, optionally trains models, runs a full inventory analysis, and saves dashboard-ready data. It has a strong fallback pattern: if real data is missing, it logs a warning and continues with synthetic data or generated sample data.

This matters because the project is designed to run in a local demo environment without requiring extensive infrastructure or a production data platform.

### 2. Data ingestion and preprocessing

[src/data_preprocessing.py](src/data_preprocessing.py) is the primary feature engineering module. It loads data, checks columns, creates rolling averages, lag features, shelf-life features, date features, and a derived expiry label. It saves a processed dataset under [data/processed](data/processed).

The project also contains [transform_inventory_data.py](transform_inventory_data.py), which appears to be a more explicit donation and action-generation transform layer. It adds columns such as donation eligibility, NGO contact data, and action recommendations.

### 3. Model training layer

[src/train_demand_model.py](src/train_demand_model.py) builds a voting ensemble regressor using XGBoost, LightGBM, and Random Forest to predict sales quantity. It uses cross-validation on a sample and saves a model to [models](models).

[src/train_expiry_model.py](src/train_expiry_model.py) does the same for a classification task with an ensemble classifier for expiry risk categories. It saves a model to [models](models) and writes CSV predictions.

This design suggests the project treats demand and expiry as separate but related prediction problems that feed a later business-logic decision layer.

### 4. Business logic layer

[src/inventory_analyzer.py](src/inventory_analyzer.py) is the most important business-logic module. It defines the actual decision rules for:

- stock level classification
- expiry risk classification
- discount suggestions
- reorder suggestions
- action selection such as Remove, Apply Discount, Restock, or Donate

This file is effectively the “decision engine” of the repository.

### 5. Donation and operational utility layer

[src/utils.py](src/utils.py) implements donation-related helper functions, including identifying the nearest NGO by city and generating a donation summary. This is tied directly to the “waste reduction and donation opportunity” theme.

### 6. Dashboard layer

[dashboard/app.py](dashboard/app.py) is the user-facing frontend. It loads processed inventory data, filters it by store, stock level, expiry risk, and action, and renders KPI cards, charts, and recommendation tables. It also supports a donation-status update flow.

The dashboard uses Plotly and Streamlit, with some CSV outputs being written back to [data/processed](data/processed).

## Data flow through the system

### Data sources

The repo includes data under [data/raw](data/raw), [data/processed](data/processed), and root-level CSVs such as [cleaned_inventory_data.csv](cleaned_inventory_data.csv). The raw folder includes files such as:

- [data/raw/train.csv](data/raw/train.csv)
- [data/raw/items.csv](data/raw/items.csv)
- [data/raw/stores.csv](data/raw/stores.csv)
- [data/raw/transactions.csv](data/raw/transactions.csv)
- [data/raw/oil.csv](data/raw/oil.csv)
- [data/raw/holidays_events.csv](data/raw/holidays_events.csv)

The code expects these to represent retail inventory and market signals, but the project is also resilient to missing data: many functions fall back to synthetic data generation.

### Flow sequence

```text
Load raw CSVs or synthetic fallback
  -> normalize columns and types
  -> engineer sales, stock, date, and expiry features
  -> train/score demand and expiry models
  -> generate stock + expiry action logic
  -> produce processed CSVs
  -> render dashboard tables and charts
```

### Decision logic flow

In [src/inventory_analyzer.py](src/inventory_analyzer.py), each row is evaluated with a few business rules. The code compares current stock to a calculated threshold, checks expiry risk, and then chooses an action. This is a classic rules-heavy scoring layer rather than a formal state machine.

## Runtime lifecycle

The project has no server process or event bus. The runtime lifecycle is straightforward:

1. Run [main.py](main.py) or use the app entry point.
2. Ensure required folders exist: [logs](logs), [models](models), [data/processed](data/processed), [dashboard/cache](dashboard/cache).
3. Load data or generate sample fallback.
4. Run preprocessing and model training when necessary.
5. Run inventory analysis.
6. Save dashboard CSV cache for Streamlit.
7. Launch [dashboard/app.py](dashboard/app.py) with Streamlit.

The app itself is interactive but not multi-user. There is no stateless API runtime or worker queue.

## Persistence and data architecture

### Primary persistence model

The repository is file-centric:

| Storage type | Evidence | Role |
| --- | --- | --- |
| CSV files | [data](data), root CSVs | Main data interchange and dashboard input |
| Pickle/joblib models | [models](models) | Saved trained ML artifacts |
| Logs | [logs](logs) | Operational diagnostics |
| Cache files | [dashboard/cache](dashboard/cache) | Dashboard-serving data snapshots |

This is a pragmatic choice for a data-science prototype and is strongly supported by the code: model files are written with joblib, and CSVs are used throughout the analysis pipeline.

### Data model

The logical domain model is centered on a retail item record with attributes such as:

- item_id, product_name
- store_nbr, city, category
- current_stock, rolling_avg_sales_7
- days_to_expiry, shelf_life, perishable
- unit_price
- donation_eligible, donation_status
- Action, Suggested_Discount, Reorder

This is consistent through [src/inventory_analyzer.py](src/inventory_analyzer.py), [src/utils.py](src/utils.py), [dashboard/app.py](dashboard/app.py), and [transform_inventory_data.py](transform_inventory_data.py).

## External integrations

This project does not have a production-grade external integration layer. It is closer to a local analytic workflow than a service-oriented system.

The code does mention external signals such as:

- oil price via dcoilwtico in [src/train_demand_model.py](src/train_demand_model.py)
- holiday data, store metadata, and item metadata in [data/raw](data/raw)
- NGO/charity information in [src/utils.py](src/utils.py) and [transform_inventory_data.py](transform_inventory_data.py)

The “external integration” here is mostly dataset-driven rather than HTTP or API-driven. In other words, the project integrates multiple CSV sources and uses a local donation mapping rather than a real service integration.

## Authentication and authorization

There is no explicit authentication or authorization model in the repository. There are no login flows, session tokens, RBAC definitions, environment keys, or middleware.

That is an important architectural fact: this is not an authenticated application; it is a local or demo analytics system. The likely assumption is that the user runs it in a trusted environment, not in a production web app context.

## Error handling strategy

The code uses a pragmatic, defensive pattern:

- logging with `logging.basicConfig` and file + console handlers
- try/except blocks around major steps
- fallback sample data generation when files are missing
- checks before training or processing
- safe conversion functions for numeric fields

This is visible in [main.py](main.py) and [src/inventory_analyzer.py](src/inventory_analyzer.py). The repo favors resilient local execution over strict operational guarantees.

## Configuration and environment strategy

There is no environment file or deployment config such as `.env`, Docker Compose, Kubernetes manifests, or Terraform. The project relies on:

- Python packages listed in [requirement.txt](requirement.txt)
- repo-relative file paths
- fixed directory names like [data](data), [logs](logs), [models](models)
- hard-coded thresholds inside model and analyzer classes

This is a simple, convention-based configuration strategy. It is fit for a prototype but not for multi-environment production deployment.

## Deployment and runtime model

This repo is not packaged as a system service. It is designed to run locally as:

- `python main.py` for the orchestration pipeline
- `streamlit run dashboard/app.py` for the dashboard

This indicates a desktop-style or local server runtime model. There is no containerization, no CI pipeline, and no “release pipeline” visible in the repo.

## Scalability considerations

The project is built for a single machine and local batch processing. It is not designed around distributed workers, asynchronous jobs, or a database-backed API.

That creates some scalability tradeoffs:

- Data size can grow, but the implementation mostly leverages pandas, which is memory-intensive for very large tables.
- Model training is local and CPU-bound rather than cloud-managed.
- Dashboard queries are in-memory and file-backed, not optimized for multi-user concurrent access.

The likely preference was speed of demonstration and simplicity over production-scale architecture.

## Security considerations

The repository has minimal security logic. There are no credentials, no secret management, no encryption at rest, no network security model, and no user isolation.

This is a significant risk if the system were ever used with real customer data. The current code assumes a trusted local environment and a demo/test dataset.

## Technical debt and architecture risks

There are several visible risks:

1. Inconsistent source-of-truth around action logic. Different modules appear to derive action rules separately, especially around donation logic and action assignment.
2. Hidden coupling to file paths. Many modules assume relative paths and repo-root conventions.
3. Synthetic data fallback creates a possibility of false confidence during demos.
4. No formal schema validation for CSV datasets.
5. Very limited modular API boundaries. Modules are functions and classes, but they are not organized as a service layer.
6. No test automation runner configured in a standard CI environment; the repo has a [TEST](TEST) folder but no visible CI config.
7. The project presents itself as a production-ready idea while still being a local prototype in implementation.

## Why this architecture?

This design likely exists because the project is a demonstration of an end-to-end AI retail application rather than a production enterprise system. The chosen architecture makes sense for:

- quick iteration in a hackathon or startup prototype
- local data science experimentation
- low operational complexity
- clear demo flow from data to dashboard

### Alternative architectures

A more conventional architecture would have been:

- a backend API with a database and authenticated user sessions
- a React frontend with a separate service layer
- a scheduler or job queue for background model retraining
- a cloud deployment with CI/CD and observability

Those alternatives would improve maintainability and scale, but they also increase complexity. The repo does not show evidence of those concerns being primary constraints. The chosen architecture favors simplicity and velocity.

## Architecture in plain English

This project is a local Python-based retail analytics engine. It reads sales and inventory data, predicts what will sell and what might expire, then recommends actions like restocking, discounting, or donating goods. The output is displayed in a dashboard so a store manager can quickly review risk and take action.

## How to explain this architecture in a 2-minute interview answer

“I built this as a retail AI workflow rather than a web app. The system loads inventory data, engineers features like rolling sales averages and days-to-expiry, trains separate demand and expiry models, and then applies business rules to decide on restocking, discounts, or donations. The results are written to CSV and shown in a Streamlit dashboard. The design is intentionally simple and local-first: there is no auth layer, no production API, and no database: instead, it uses CSVs and model artifacts for speed and demo clarity. That choice reflects the project goal of proving the end-to-end workflow quickly rather than building enterprise infrastructure.”

## How to explain this architecture in a deep technical interview

“The central pattern is a batch ML pipeline with a decision layer and a presentation layer. [main.py](main.py) orchestrates the workflow, [src/data_preprocessing.py](src/data_preprocessing.py) creates features, [src/train_demand_model.py](src/train_demand_model.py) and [src/train_expiry_model.py](src/train_expiry_model.py) train separate models, and [src/inventory_analyzer.py](src/inventory_analyzer.py) turns model outputs plus business thresholds into actions. The design is deliberately file-centric: CSVs and joblib files are the persistence boundary, and the dashboard reads processed files rather than a service API. That reduces complexity and makes the system easy to demo, but it also means the architecture is not production-hardened. If I were scaling it, I would move to a real data store, API boundary, and job orchestration layer.”

## What to say in an interview

- “This project is an end-to-end retail analytics and recommendation system built in Python.”
- “The main technical pattern is feature engineering + ML + rule-based decisioning.”
- “The project prioritizes clarity and demoability over production infra.”
- “The biggest architectural compromise is that it uses local CSV files instead of a database-backed service layer.”

## What to study next

- The interaction between [src/inventory_analyzer.py](src/inventory_analyzer.py) and [transform_inventory_data.py](transform_inventory_data.py)
- The exact feature engineering logic in [src/data_preprocessing.py](src/data_preprocessing.py)
- The action decision thresholds in [src/inventory_analyzer.py](src/inventory_analyzer.py)
- The runtime assumptions in [dashboard/app.py](dashboard/app.py)
- The gaps between demo-friendly design and production readiness

## Open questions / uncertainties

- There is no visible CI/CD pipeline or deployment manifest, so the “production” story is mostly aspirational rather than implemented.
- The repo appears to mix demo data generation, production-like logic, and experimentation, which may indicate a project still in transition from prototype to product.
- The donation logic is present in several places, suggesting some duplication or drift; the code comments acknowledge it in [transform_inventory_data.py](transform_inventory_data.py).
- The project claims a React frontend in some documentation, but the visible implementation is primarily Streamlit; the actual repository evidence is strongest around the Python pipeline.
