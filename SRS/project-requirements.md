# Project requirements

## Table of contents

- [Project purpose](#project-purpose)
- [Explicit requirements visible in code and docs](#explicit-requirements-visible-in-code-and-docs)
- [Inferred requirements](#inferred-requirements)
- [Functional requirements](#functional-requirements)
- [Non-functional requirements](#non-functional-requirements)
- [User roles and personas](#user-roles-and-personas)
- [Business rules](#business-rules)
- [Performance expectations](#performance-expectations)
- [Reliability expectations](#reliability-expectations)
- [Security expectations](#security-expectations)
- [Integration requirements](#integration-requirements)
- [Reporting and analytics requirements](#reporting-and-analytics-requirements)
- [Admin and ops requirements](#admin-and-ops-requirements)
- [Missing but likely required requirements](#missing-but-likely-required-requirements)
- [Why these requirements likely led to the chosen implementation](#why-these-requirements-likely-led-to-the-chosen-implementation)
- [What to say in an interview](#what-to-say-in-an-interview)
- [What to study next](#what-to-study-next)
- [Open questions / uncertainties](#open-questions--uncertainties)

## Project purpose

The strongest explicit requirement is to reduce inventory waste and improve operational decisions in a retail environment. The project is clearly oriented around reducing food waste, preventing stockouts, and improving inventory profitability. This is repeated in [README.md](README.md), [main.py](main.py), and [src/inventory_analyzer.py](src/inventory_analyzer.py).

## Explicit requirements visible in code and docs

### Functional requirements

1. Forecast demand for retail inventory items.
   - Evidence: [src/train_demand_model.py](src/train_demand_model.py)
2. Identify items with expiry risk.
   - Evidence: [src/train_expiry_model.py](src/train_expiry_model.py)
3. Classify inventory into stock states like High, Normal, or Low.
   - Evidence: [src/inventory_analyzer.py](src/inventory_analyzer.py)
4. Recommend action items such as Remove, Apply Discount, Restock, or Donate.
   - Evidence: [src/inventory_analyzer.py](src/inventory_analyzer.py), [dashboard/app.py](dashboard/app.py)
5. Track donation-eligible items and donation status.
   - Evidence: [src/utils.py](src/utils.py), [transform_inventory_data.py](transform_inventory_data.py)
6. Show results in a human-friendly dashboard.
   - Evidence: [dashboard/app.py](dashboard/app.py)
7. Generate processed outputs for downstream use.
   - Evidence: [main.py](main.py), [data/processed](data/processed)

### Non-functional requirements

- Keep the project easy to run locally.
- Prefer simple, readable Python code.
- Use CSV and model artifact persistence for ease of inspection.
- Avoid heavy infrastructure complexity for a demo-first workflow.

These are not formal SLOs, but they are strongly implied by the repo structure and file conventions.

## Inferred requirements

These requirements are not spelled out as formal docs, but they are likely intended:

- multi-store inventory analysis
- item-level risk tracking across categories
- support for perishable goods categories
- donation coordination for near-expiry, edible goods
- local analytical workflows rather than cloud-managed services
- business-facing operational output rather than strict backend API contracts

## Functional requirements

### 1. Demand forecasting

The system must estimate demand using previous sales and contextual features. This is visible in training features such as rolling averages, lag features, seasonality, day-of-week, month, and oil-price signals.

### 2. Expiry risk classification

The system must detect near-expiry and expired inventory. This is implemented as score-based logic in [src/inventory_analyzer.py](src/inventory_analyzer.py) and a trained classification model in [src/train_expiry_model.py](src/train_expiry_model.py).

### 3. Stock health analysis

The project must flag overstock and understock situations using threshold logic. This is an operational analytics requirement rather than a pure prediction requirement.

### 4. Action generation

A central requirement is to convert analytics into recommended actions so users can respond operationally. That is a major design choice in the project and likely a core product requirement.

### 5. Donation logic

The project requires a social-impact and sustainability pathway to recover value from near-expiry goods via donation opportunities.

## Non-functional requirements

### Performance

The project is optimized for a local demo environment. Performance expectations appear moderate and not tied to large-scale distributed systems. The key assumption is that data can be processed in memory and rendered quickly enough for a dashboard demo.

### Reliability

The repo values resilience and recoverability rather than high operational rigor. The code uses fallbacks and best-effort processing instead of strict production-grade failure management.

### Maintainability

The project tries to keep logic readable across modules. However, some duplication and overlap across action logic is visible, which suggests maintainability is still a work in progress.

### Security

The repo does not show evidence of enterprise security requirements. This is a gap if the application were to move into a real operational environment.

## User roles and personas

### Store manager

Likely primary persona. They need to know what has to be restocked, discounted, or removed.

### Supply chain / inventory analyst

Likely secondary persona. This user needs demand and expiry signals to optimize ordering and reduce waste.

### CSR / donation coordinator

This persona is likely supported by the donation logic. The project includes NGO mapping and donation status decisions.

### Demonstration stakeholder

The project also clearly serves a presentation or hackathon audience, where product storytelling matters as much as technical rigor.

## Business rules

The code evidences several business rules:

- items with low demand and high expiry risk may receive a discount
- items with significant overstock may be flagged for markdowns
- expired or near-expired items may require removal
- donation-eligible items may be assigned to a pending or donated state
- perishable and edible categories are treated as donation candidates in some logic

These rules are split across [src/inventory_analyzer.py](src/inventory_analyzer.py), [transform_inventory_data.py](transform_inventory_data.py), and [src/utils.py](src/utils.py).

## Performance expectations

The project expects the following operational behaviors:

- run in a local environment without a heavy infrastructure stack
- process tabular data quickly enough for dashboard interaction
- display charts and KPI cards with minimal delay
- allow multiple analysis actions without a dedicated backend service

This is reasonable for a prototype, but not a true high-concurrency or high-volume production requirement.

## Reliability expectations

The repo expects resilient execution under messy conditions:

- missing fields are filled or generated
- invalid numeric values are coerced
- files can be missing without crashing the project
- sample data can stand in for missing data in demos

This is a pragmatic requirement for local experimentation, but it is not the same as production-grade reliability.

## Security expectations

There are no explicit security expectations visible. No requirements for:

- user identity
- authorization
- data encryption
- auditing
- PII protection

This likely means the project’s current requirements are limited to analysis and demo use, not enterprise operational security.

## Integration requirements

The project has a strong expectation of integrating multiple internal data sources, such as:

- transaction data
- product metadata
- store metadata
- holidays and market signals
- local processed CSV outputs

This integration is file-based and local, not service-based.

## Reporting and analytics requirements

The project must provide:

- KPI summaries
- stock distribution by stock level
- expiry analysis breakdown
- action recommendation counts
- donation summary and NGO output
- inventory value and impact metrics

The dashboard in [dashboard/app.py](dashboard/app.py) is the clearest evidence of the reporting requirement.

## Admin and ops requirements

These are not fully visible, but likely include:

- ability to inspect running analysis outputs
- ability to save processed CSVs
- ability to review donation statuses
- ability to rerun or refresh the analytics process

The code supports refresh patterns and data exports but does not show a full admin console.

## Missing but likely required requirements

- authentication and role-based access
- formal API contract between UI and backend
- database persistence and schema versioning
- scheduled retraining and pipeline orchestration
- secure secret storage
- operational monitoring
- cloud deployment and scalability support

These are missing because the repo is currently more of a local prototype than an enterprise product.

## Why these requirements likely led to the chosen implementation

The project requirements are centered on analytical decision support, not on a full multi-user platform. That naturally leads to a simpler architecture:

- Python modules for data science logic
- CSV and joblib files for persistence
- Streamlit for easy dashboard delivery
- rule-based action logic rather than a large event-driven architecture

This is a direct fit for a hackathon or prototype environment and explains the observed architecture.

## What to say in an interview

- “The requirement set is focused on reducing waste and making better operational decisions with limited infrastructure.”
- “The product requirement is not a full SaaS experience; it is a decision-support and demo-first workflow.”
- “The repo matches a local analytics product with a strong business reasoning layer.”

## What to study next

- [README.md](README.md)
- [main.py](main.py)
- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [dashboard/app.py](dashboard/app.py)
- [transform_inventory_data.py](transform_inventory_data.py)

## Open questions / uncertainties

- The requirement text is strong on business impact, but the repo has limited formal product requirements or spec documents beyond README-level narrative.
- The system’s user model is not fully documented; a real product would likely need clearer persona and access boundaries.
- Some “future roadmap” statements in the docs appear stronger than the actual implementation evidence in the repo.
