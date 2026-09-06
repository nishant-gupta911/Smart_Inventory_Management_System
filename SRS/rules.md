# Rules and conventions

## Table of contents

- [Project operating principles](#project-operating-principles)
- [Coding conventions](#coding-conventions)
- [Folder and module organization rules](#folder-and-module-organization-rules)
- [Naming conventions](#naming-conventions)
- [Data and API conventions](#data-and-api-conventions)
- [Validation rules](#validation-rules)
- [Error handling rules](#error-handling-rules)
- [State management rules](#state-management-rules)
- [Database and data-consistency rules](#database-and-data-consistency-rules)
- [Security and access control rules](#security-and-access-control-rules)
- [Environment and configuration rules](#environment-and-configuration-rules)
- [Testing rules](#testing-rules)
- [Release and deployment rules](#release-and-deployment-rules)
- [Implicit rules inferred from the codebase](#implicit-rules-inferred-from-the-codebase)
- [What not to do](#what-not-to-do)
- [Why these rules matter](#why-these-rules-matter)
- [Why this style may have been chosen](#why-this-style-may-have-been-chosen-over-alternatives)
- [New developer checklist](#new-developer-checklist)
- [Common mistakes someone new would make](#common-mistakes-someone-new-would-make)
- [What to say in an interview](#what-to-say-in-an-interview)
- [What to study next](#what-to-study-next)
- [Open questions / uncertainties](#open-questions--uncertainties)

## Project operating principles

The repository strongly suggests a few project principles:

- Prefer a working local demo over abstract architecture.
- Keep modules easy to run directly from Python.
- Treat CSVs as the primary interchange format.
- Make every stage resilient to missing data and partial input.
- Keep the business logic visible and readable rather than deeply abstracted.

These are visible in [main.py](main.py), [src/inventory_analyzer.py](src/inventory_analyzer.py), and [dashboard/app.py](dashboard/app.py).

## Coding conventions

### Python style

The code is mostly Pythonic, readable, and direct.

- Functions are snake_case, as in `run_data_preprocessing` and `update_donation_status`.
- Classes use PascalCase, as in `InventoryAnalyzer` and `RestockPlanGenerator`.
- Logging is explicit and consistent with `logging.getLogger(__name__)` patterns.
- There is heavy use of small helper functions rather than large complex classes.

### Fallback-first execution

A repeated rule is: if the primary data source is missing, generate a synthetic fallback instead of failing abruptly. This appears in:

- [main.py](main.py)
- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [src/utils.py](src/utils.py)
- [src/data_preprocessing.py](src/data_preprocessing.py)

This is a deliberate reliability convention for demo scenarios.

### Defensive typing and validation

The code often preserves values via calls like `pd.to_numeric(..., errors='coerce')`, `fillna(...)`, and `clip(...)`. This is a strong signal that the project values robust handling of messy data over strict type enforcement.

## Folder and module organization rules

The repo is organized around a simple data science layout:

- root entry scripts: [main.py](main.py), [transform_inventory_data.py](transform_inventory_data.py)
- source package: [src](src)
- data sources: [data](data)
- dashboard app: [dashboard](dashboard)
- tests: [TEST](TEST)
- model artifacts: [models](models)
- logs: [logs](logs)

The expected rule is: data and generated artifacts live outside source code; source code stays modular and importable.

This is visible in the use of `PROJECT_ROOT` and path-based resolution in [main.py](main.py) and [src/utils.py](src/utils.py).

## Naming conventions

The project consistently uses:

- file names that describe the purpose: `train_demand_model.py`, `generate_restock_plan.py`, `inventory_analyzer.py`
- function names that describe a task: `load_inventory_data`, `prepare_features`, `save_updated_inventory`
- columns named in business-language terms: `days_to_expiry`, `current_stock`, `stock_to_sales_ratio`, `donation_eligible`

This is good for readability and interview explainability. It also helps keep the business logic close to the data vocabulary.

## Data and API conventions

### Data conventions

The system uses CSV as the canonical interchange format. The “contract” between modules is often a DataFrame with columns like:

- `item_id`
- `store_nbr`
- `current_stock`
- `rolling_avg_sales_7`
- `days_to_expiry`
- `donation_eligible`
- `donation_status`
- `Action`

This is not a formal schema file, but it is the effective contract of the repo.

### API conventions

There are no REST endpoints or formal API controllers. The “API” is effectively a set of Python functions. That means calling conventions are based on direct import and invocation rather than HTTP routes or typed DTOs.

This is a key convention to note in interviews: the project exposes a Pythonic, not service-oriented, interface.

## Validation rules

The code validates the data in several places:

- numeric conversion and NaN replacement in [src/inventory_analyzer.py](src/inventory_analyzer.py)
- required columns checks in [src/generate_restock_plan.py](src/generate_restock_plan.py)
- donation logic expectations in [transform_inventory_data.py](transform_inventory_data.py)
- dashboard data schema checks in [dashboard/app.py](dashboard/app.py)

A key rule appears to be: handle malformed or partial values in a permissive way and continue, rather than aborting the run.

## Error handling rules

The project uses a very practical model:

- log the issue
- continue when possible
- fall back to sample data if needed
- fail only at clearly essential stages

This is honest and useful for demos, but it is not production-grade fail-fast behavior. The system is designed to keep running instead of crashing on messy data.

## State management rules

There is minimal explicit state management because the system is not an application server. The state is mostly stored in:

- DataFrame objects in memory
- CSV files on disk
- streamlit session state in [dashboard/app.py](dashboard/app.py)

The dashboard handles session-level mutation through `st.session_state` for `df_changes`, which is a direct example of UI state being managed locally in the browser-backed app state.

## Database and data-consistency rules

There is no database server in the repo. The code treats the filesystem as a data store and expects CSVs to be aligned on columns and naming. That means the hidden rule is: keep column names stable across modules.

If a module expects a column but another file produced a different name or missing field, you get degraded behavior rather than a schema error.

## Security and access control rules

There are no formal security rules visible in the tree. No auth flow, no RBAC, no token validation, no encryption, and no secrets file. The repo is not structured as a secured multi-user application.

The underlying rule is: security is out of scope for this version.

## Environment and configuration rules

The environment is effectively:

- Python 3.x local runtime
- dependencies from [requirement.txt](requirement.txt)
- repo-relative file paths
- generated artifacts in the working directory

The project uses global convention over config. There is no `.env`, no config object system, and no layered config strategy.

## Testing rules

The repo contains a dedicated [TEST](TEST) directory with tests such as:

- [TEST/test_action_logic.py](TEST/test_action_logic.py)
- [TEST/test_dashboard.py](TEST/test_dashboard.py)
- [TEST/test_main_integration.py](TEST/test_main_integration.py)
- [TEST/test_donation_functions.py](TEST/test_donation_functions.py)

This implies a rule that core actions should be tested in isolation, especially the donation and action logic. However, there is no formal CI gate, and the test suite is more demonstration-appropriate than enterprise-standard.

## Release and deployment rules

There are no visible release rules, version tags, or deployment pipeline files. This suggests:

- releases are effectively manual
- deployment is local execution, not orchestrated rollout
- there is no environment parity or smoke-test pipeline visible in the repo

## Implicit rules inferred from the codebase

These rules are likely but not explicitly documented:

- Do not assume perfect data; sanitize aggressively.
- Keep feature names understandable to business users.
- Keep output files easy to inspect manually.
- Prefer deterministic outputs where possible (random seeds are set in several places).
- When a model or file is missing, continue with a fallback path instead of failing hard.

## What not to do

- Do not hard-code new file paths without checking the repo’s root conventions.
- Do not assume the CSV schema is fully consistent across modules.
- Do not treat the synthetic fallback as if it were real data.
- Do not ignore missing or invalid values in inventory columns.
- Do not add a new report without checking whether the dashboard depends on the same output columns.
- Do not rely on unauthenticated local runs for anything beyond demo or internal analytics.

## Why these rules matter

These rules matter because the project sits squarely in the boundary between research and product demo. The same codebase is expected to handle raw, messy, partially missing warehousing data while still being readable enough for an interview or stakeholder demo.

The project is more valuable when it stays resilient and legible than when it is deeply formalized.

## Why this style may have been chosen over alternatives

The project favors convention and readability over formal architecture. That choice is likely driven by the following factors:

- prototyping speed for a hackathon or early-stage product
- Python-first execution and quick iteration in notebooks and scripts
- low operational overhead
- direct visibility into data transformation and model behavior

A more formal enterprise setup would add complexity: typed APIs, migrations, distributed deployment, auth flows, CI/CD, and a database. This repo does not show evidence that those were the primary constraints.

## New developer checklist

- Confirm the repo root and the expected working directory before running scripts.
- Check whether the code is using existing processed files or generating synthetic fallback data.
- Review [src/inventory_analyzer.py](src/inventory_analyzer.py) before changing action logic.
- Verify the dashboard expects the same columns as the CSV it loads.
- Keep path constants consistent with the repository layout.
- If you modify feature names, update both the training code and the analysis code.
- Make sure any donation logic changes do not drift from the business rules encoded in the project.
- Run the relevant tests in [TEST](TEST) or the small script checks before claiming a change is safe.

## Common mistakes someone new would make

- Trying to run a script from the wrong directory and breaking relative paths.
- Editing the dashboard without checking the processed CSV schema.
- Changing action logic in one module without updating the other modules that depend on it.
- Using synthetic sample data as an argument that the model is production-ready.
- Adding a new feature but forgetting to fill missing values consistently.
- Assuming the repo has user auth, API contracts, or RBAC even though the code shows none.

## What to say in an interview

- “The project values practical reliability over formal infrastructure.”
- “The code is explicit and readable, with fallback behavior when data is missing.”
- “The strongest conventions are file-based data interchange, Pythonic modules, and business-driven feature names.”
- “The biggest tradeoff is simplicity versus production hardening.”

## What to study next

- The source-of-truth action logic in [src/inventory_analyzer.py](src/inventory_analyzer.py)
- Column expectations in [dashboard/app.py](dashboard/app.py)
- The fallback generation paths in [main.py](main.py)
- Test assumptions and edge cases in [TEST](TEST)

## Open questions / uncertainties

- There is no formal schema versioning or migration strategy, which means data contracts are fragile.
- The project appears to have a few overlapping routes for donation logic, suggesting some duplication and possible drift.
- The dataset pipeline is convincing but not fully productionized; the code is stronger on demo flow than on operations.
