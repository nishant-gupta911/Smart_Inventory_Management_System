# Design

## Table of contents

- [Design philosophy](#design-philosophy)
- [Module and responsibility boundaries](#module-and-responsibility-boundaries)
- [Separation of concerns](#separation-of-concerns)
- [Reusability patterns](#reusability-patterns)
- [UI and UX approach](#ui-and-ux-approach)
- [Data model design approach](#data-model-design-approach)
- [Extensibility strategy](#extensibility-strategy)
- [Maintainability strategy](#maintainability-strategy)
- [Developer experience choices](#developer-experience-choices)
- [Good design choices](#good-design-choices)
- [Questionable or risky design choices](#questionable-or-risky-design-choices)
- [Why this design may have been chosen](#why-this-design-may-have-been-chosen-instead-of-other-patterns)
- [What to say in an interview](#what-to-say-in-an-interview)
- [What to study next](#what-to-study-next)
- [Open questions / uncertainties](#open-questions--uncertainties)

## Design philosophy

The repository shows a strong design bias toward practical demoability and explainability. The architecture is not trying to hide complexity behind heavy abstractions. Instead, it exposes the pipeline in a way that a developer or stakeholder can follow from raw data to decision output.

That is a deliberate design choice. It is aligned with the project’s likely purpose: show that AI and analytics can improve retail operations, not build a large enterprise platform.

## Module and responsibility boundaries

The repo has a clear but not fully formalized layering:

- [main.py](main.py): orchestration
- [src/data_preprocessing.py](src/data_preprocessing.py): data preparation and feature generation
- [src/train_demand_model.py](src/train_demand_model.py): demand forecasting model training
- [src/train_expiry_model.py](src/train_expiry_model.py): expiry-risk model training
- [src/inventory_analyzer.py](src/inventory_analyzer.py): business analysis and action logic
- [src/generate_restock_plan.py](src/generate_restock_plan.py): restock planning logic
- [src/utils.py](src/utils.py): donation and summary helpers
- [transform_inventory_data.py](transform_inventory_data.py): donation augmentation and output shaping
- [dashboard/app.py](dashboard/app.py): presentation layer

This is a reasonable separation of concerns for a prototype. The key thing is that business decisions are not hidden inside the model code; they are deliberately surfaced into a separate logic layer.

## Separation of concerns

### Data quality and feature logic are separate from model training

The preprocessing step is distinct from the trained models. That separation makes the pipeline easier to reason about and easier to debug.

### Model training is separate from action generation

The repo does not directly turn raw model predictions into final business actions in the training script. Instead, [src/inventory_analyzer.py](src/inventory_analyzer.py) handles the decision layer, which is a strong design signal.

### Dashboard is separate from analytical engine

The dashboard is intentionally a consumer of processed data rather than the place where the logic originates. That makes the UI much easier to reason about.

## Reusability patterns

The project uses a few reusable patterns:

- helper functions for loading and validating data
- lightweight `Analysis`-style classes to encapsulate operational logic
- `pd.to_numeric(..., errors='coerce')` sanitization loops repeated across modules
- helper folders for outputs and cache files

This is a “good enough for research and demo” design, not a componentized product architecture.

## UI and UX approach

The UI is a Streamlit dashboard, visible in [dashboard/app.py](dashboard/app.py). The design is heavy on:

- KPI cards
- filters on store, stock state, expiry state, and action
- a few chart panels with Plotly
- operational summary cards

This is likely chosen because the user is a business operator rather than a highly technical analyst. The UI focuses on decisions and visible metrics instead of raw model internals.

The UX tradeoff is clear: strong clarity for a demo, but limited sophistication for a production large-scale business app.

## Data model design approach

The project’s effective data model is a wide inventory row with many derived features. Each row is not a relational domain object but a snapshot enriched with both raw and computed attributes.

Core attributes include:

- identity: `item_id`, `product_name`
- operational context: `store_nbr`, `city`, `category`
- stock and demand metrics: `current_stock`, `rolling_avg_sales_7`, `sales_trend`
- expiry and shelf life: `days_to_expiry`, `shelf_life`, `Expiry_Risk`
- decision outputs: `Action`, `Reorder`, `Suggested_Discount`
- donation state: `donation_eligible`, `donation_status`, `nearest_ngo`

This is a pragmatic analytics-oriented design. It is easy to read and business-friendly, but it is not a normalized relational model.

## Extensibility strategy

There are a few signs of extensibility:

- model training is factored into separate files
- preprocessing is isolated from scoring
- dashboard reads processed CSV output rather than directly depending on raw source files
- helper functions in [src/utils.py](src/utils.py) offer reusable donation logic

But the repo does not yet show a clean extension model such as service interfaces, dependency injection, or a proper plugin framework. Instead, the code relies on file conventions and direct module imports.

## Maintainability strategy

The maintainability approach is readable and local. It favors:

- direct Python modules
- explicit functions
- logs instead of hidden fail-silent behavior
- file-based artifacts for inspection

This makes the code easier for a new engineer to follow during an interview or in a short prototype cycle.

The tradeoff is that maintainability under scale is weaker than in a formal service architecture.

## Developer experience choices

The repo is designed for a developer to run it directly in a local environment. Examples:

- root-level scripts such as [main.py](main.py)
- minimal dependency management in [requirement.txt](requirement.txt)
- easy file inspection and generated CSVs
- use of standard Python logging

This is a strong developer-experience choice for a prototype, but it also increases the chance of fragile path assumptions and hidden environment dependence.

## Good design choices

- Clear separation between preprocessing, model training, and action logic.
- Explicit data transformation and output files.
- Fallback behavior for missing data makes the system more demo-resilient.
- Business logic is captured in code rather than buried in UI behavior.
- The dashboard is easy to understand and is aligned to the user’s job.

## Questionable or risky design choices

- The repo appears to duplicate donation and action logic in multiple places.
- CSV and file-path assumptions are fragile if project structure changes.
- There is no formal schema validation for all processed CSVs.
- The output layer is more ad hoc than a managed data contract.
- Some docs claim richer architecture than what the repo actually implements.

## Why this design may have been chosen instead of other patterns

The codebase strongly suggests a “fastest path to a working prototype” design. That is likely because the project’s goals were:

- validate a retail data science concept
- create a compelling demo
- produce a story around waste reduction and stock optimization
- minimize operational and deployment complexity

A more formal system would likely include a database, API layer, auth, and production deployment patterns. The repo does not show those as primary goals.

## What to say in an interview

- “The design is intentionally simple and file-based because it prioritizes fast iteration and clarity.”
- “The major boundaries are preprocessing, modeling, decision logic, and dashboard presentation.”
- “The design is strong for evidence and demo work, but not yet deeply hardened for enterprise scale.”

## What to study next

- [main.py](main.py)
- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [src/data_preprocessing.py](src/data_preprocessing.py)
- [transform_inventory_data.py](transform_inventory_data.py)
- [dashboard/app.py](dashboard/app.py)

## Open questions / uncertainties

- The repo contains signs of iterative or overlapping implementation, so some module boundaries may be less clean than the conceptual architecture suggests.
- The project seems to have a richer product story than its actual runtime architecture supports.
- Without CI or deployment config, the maintainability story is strong in code but weaker in operations.
