# Interview questions

## Table of contents

- [Recruiter questions](#recruiter-questions)
- [Hiring manager questions](#hiring-manager-questions)
- [Product and stakeholder questions](#product-and-stakeholder-questions)
- [Technical interviewer questions](#technical-interviewer-questions)
- [Senior engineer and architect questions](#senior-engineer--architect-questions)
- [Code walk-through questions](#code-walk-through-questions)
- [Behavioral questions tied to this project](#behavioral-questions-tied-to-this-project)
- [Follow-up trap questions](#follow-up-trap-questions-or-pressure-test-questions)
- [Explain this project to a non-technical recruiter](#explain-this-project-to-a-non-technical-recruiter)
- [Explain my contribution if I owned the whole project](#explain-my-contribution-if-i-owned-the-whole-project)
- [Explain my contribution if I only worked on part of it](#explain-my-contribution-if-i-only-worked-on-part-of-it)
- [Why did you use this approach instead of another one?](#why-did-you-use-this-approach-instead-of-another-one)
- [Tradeoff questions and ideal answers](#tradeoff-questions-and-ideal-answers)
- [System design questions they may ask from this codebase](#system-design-questions-they-may-ask-from-this-codebase)
- [What would you improve next?](#what-would-you-improve-next)
- [What was the hardest part?](#what-was-the-hardest-part)
- [If this scaled 10x, what would break first?](#if-this-scaled-10x-what-would-break-first)
- [What to say in an interview](#what-to-say-in-an-interview)
- [What to study next](#what-to-study-next)
- [Open questions / uncertainties](#open-questions--uncertainties)

## Recruiter questions

### 1. What does this project do?

- Strong sample answer: “This project is a retail inventory intelligence system that uses machine learning to forecast demand, detect expiry risk, and recommend actions such as restocking, discounting, or donation. It is designed to reduce waste and improve stock decisions.”
- Short version: “It helps retailers reduce waste and improve inventory decisions.”
- What the interviewer is testing: whether you can explain the product in business terms.
- Mistakes to avoid: avoid saying “it is just a dashboard” or leaning only on the AI model without describing the operational value.

### 2. Why would a company care about this?

- Strong sample answer: “Retailers lose margin when they overstock perishable goods, understock fast-moving items, or fail to act on near-expiry products. This project directly attacks those problems.”
- Short version: “It reduces waste, increases revenue, and supports better inventory decisions.”
- What it tests: business understanding.
- Mistakes to avoid: overclaiming savings without evidence or ignoring that this is a prototype.

### 3. What kind of problem is the project solving?

- Strong sample answer: “It solves a classic retail operations problem: balancing stock availability with product freshness and cost control.”
- Short version: “It solves waste + stock optimization.”
- What it tests: clarity of product framing.
- Mistakes to avoid: describing it as only a machine learning experiment.

## Hiring manager questions

### 4. What is your biggest technical contribution here?

- Strong sample answer: “I would say the biggest contribution is building the end-to-end decision layer: from data preparation to model training to operational rules that convert predictions into actions. The repo is organized around that flow and the action logic is the key differentiator.”
- Short version: “I built the end-to-end inventory intelligence flow and connected predictions to action rules.”
- What it tests: ownership and systems thinking.
- Mistakes to avoid: claiming to own a full enterprise product that the repo does not support.

### 5. Why did you choose this architecture?

- Strong sample answer: “Because the goal was to prove the workflow quickly and keep it understandable. The repo uses Python modules, CSV outputs, and a Streamlit dashboard rather than a full enterprise stack because the project is designed for local demo and analysis, not a production SaaS deployment.”
- Short version: “It was a clear, low-overhead prototype architecture.”
- What it tests: design tradeoff awareness.
- Mistakes to avoid: claiming it was chosen for scale when there is no evidence of that.

## Product and stakeholder questions

### 6. What would a store manager actually do with this?

- Strong sample answer: “A store manager could look at stock health, expiry risk, and action recommendations, then decide whether to restock, discount, or donate products before they become waste.”
- Short version: “They act on risk and opportunity instead of guessing.”
- What it tests: user-centered reasoning.
- Mistakes to avoid: focusing only on model performance and ignoring the end user’s operational workflow.

### 7. What are the product tradeoffs?

- Strong sample answer: “The product is easy to understand and demo, but it is not yet hardened for multi-user production operations. That tradeoff is acceptable for a prototype because it prioritizes clarity and speed over service architecture, auth, and scale.”
- Short version: “It values speed and clarity over operational hardening.”
- What it tests: real product judgment.
- Mistakes to avoid: pretending the repo is fully production-ready when it is not.

## Technical interviewer questions

### 8. Walk me through the core architecture.

- Strong sample answer: “The repo is a batch machine learning pipeline with a rule-based decision layer and a dashboard front end. [main.py](main.py) orchestrates the flow, [src/data_preprocessing.py](src/data_preprocessing.py) creates features, [src/train_demand_model.py](src/train_demand_model.py) and [src/train_expiry_model.py](src/train_expiry_model.py) train models, and [src/inventory_analyzer.py](src/inventory_analyzer.py) transforms predictions into actions. The dashboard in [dashboard/app.py](dashboard/app.py) then renders the output.”
- Short version: “The system is data -> ML -> decision logic -> dashboard.”
- What it tests: architectural understanding.
- Mistakes to avoid: summarizing only the dashboard and forgetting the analysis engine.

### 9. What makes this a hybrid system rather than a pure ML project?

- Strong sample answer: “The model handles prediction, but the business logic is rule-driven. The repo explicitly computes stock levels, expiry risk, suggested discounts, reorder needs, and donation actions. That means the project blends predictive analytics with operational decision rules.”
- Short version: “It uses ML for prediction and business rules for action.”
- What it tests: understanding of architecture patterns.
- Mistakes to avoid: saying the system is only a model training exercise.

## Senior engineer and architect questions

### 10. Where is the source of truth for action logic?

- Strong sample answer: “The clearest source of truth is [src/inventory_analyzer.py](src/inventory_analyzer.py), although the repo does show some overlapping logic in [transform_inventory_data.py](transform_inventory_data.py). That is a real architectural weakness and a clean place to improve by centralizing the rules.”
- Short version: “The repo is close to a single source of truth, but not fully centralized.”
- What it tests: judgment on architecture consistency.
- Mistakes to avoid: claiming the logic is perfectly centralized when the repo evidence suggests otherwise.

### 11. What would you change first if this had to scale 10x?

- Strong sample answer: “I would move from CSV-heavy data handling to a proper data store, define a stronger schema contract, add a service/API boundary, and decouple the analytics pipeline from the dashboard. I would also add orchestration and monitoring.”
- Short version: “I would add a real data layer, API boundary, and pipeline orchestration.”
- What it tests: ability to reason about scale.
- Mistakes to avoid: pretending the current architecture is already scalable.

## Code walk-through questions

### 12. What happens in [src/data_preprocessing.py](src/data_preprocessing.py)?

- Strong sample answer: “It creates the feature set: rolling averages, lag features, stock-to-sales ratios, date features, and expiry-related metrics. It normalizes the dataset and writes a processed file used by downstream model and business logic.”
- Short version: “It turns raw inventory data into ML-ready features.”
- What it tests: understanding of feature engineering.
- Mistakes to avoid: describing the whole repo without specific feature logic.

### 13. Why does the repo use synthetic data fallback?

- Strong sample answer: “Because the project is designed to keep running even when local files are missing. That makes it resilient for demos, but it also means the pipeline can hide data problems if someone relies on fallback data without checking.”
- Short version: “It keeps the demo alive, but can mask missing data issues.”
- What it tests: awareness of engineering tradeoffs.
- Mistakes to avoid: presenting fallback generation as equivalent to production data quality.

## Behavioral questions tied to this project

### 14. Tell me about a hard problem in this project and how you solved it.

- Strong sample answer: “The hardest problem was connecting the ML predictions to operational actions in a way that a non-technical user could trust. The solution was to separate model prediction from business logic and create a clear action layer in [src/inventory_analyzer.py](src/inventory_analyzer.py).”
- Short version: “The hardest part was converting predictions into clear operational recommendations.”
- What it tests: problem-solving and systems thinking.
- Mistakes to avoid: focusing only on training metrics and not on business usability.

### 15. What tradeoff did you make in the design?

- Strong sample answer: “I accepted a simpler file-based architecture so the system stays easy to run and explain. The cost is reduced production hardening and weaker operational scaling.”
- Short version: “I chose simplicity over production complexity.”
- What it tests: candor and engineering judgment.
- Mistakes to avoid: pretending there were no tradeoffs.

## Follow-up trap questions or pressure-test questions

### 16. Is this production-ready?

- Strong sample answer: “Not as currently structured. It is a strong prototype and demo workflow, but it does not yet include the operational standards of a production system: auth, deployment automation, database governance, monitoring, and formal data contracts.”
- Short version: “It is a strong prototype, not a hardened production system.”
- What it tests: honesty and realism.
- Mistakes to avoid: overclaiming production readiness.

### 17. Why not use a full API and database from the start?

- Strong sample answer: “Because the project’s immediate value was proving the end-to-end workflow quickly. A full service and database layer would have slowed the iteration cycle and obscured the underlying business logic. The project is a prototype-first system with a single-machine execution model.”
- Short version: “We optimized for clarity and rapid iteration.”
- What it tests: prioritization.
- Mistakes to avoid: sounding like you are ignoring production engineering fundamentals.

## Explain this project to a non-technical recruiter

- “This project helps a retailer reduce food waste and improve stock management by predicting which items are likely to sell or expire soon. It then recommends what action to take, like restocking, discounting, or donating items before they go to waste.”
- “The main idea is using data and AI to make better decisions that save money and reduce waste.”

## Explain my contribution if I owned the whole project

- “I built the end-to-end retail decision system from data ingestion through model training, business logic, and dashboard delivery. I focused on translating technical predictions into operational actions that a business user could actually use.”

## Explain my contribution if I only worked on part of it

- “I owned the analytics and decision layer, especially the feature engineering, model training, or dashboard logic. My work connected raw inventory data to business recommendations and ensured the output was interpretable and usable.”

## Why did you use this approach instead of another one?

- “The repo prioritizes clarity and quick validation over full enterprise complexity.”
- “CSV and Python modules are much easier to debug and demo than a long service stack when the main goal is proving the workflow.”
- “The decision layer matters more than a polished app shell for this kind of business problem.”

## Tradeoff questions and ideal answers

### Tradeoff: local CSVs vs. database

- Strong answer: “CSV files are easier to inspect and run locally, which fits a prototype. But they are brittle and not a strong production persistence model. I would move to a database as soon as the product needed operational scale or multi-user reliability.”

### Tradeoff: rules + ML vs. pure ML

- Strong answer: “The hybrid approach is better here because business rules are explicit and explainable, and the model is used as a signal rather than the sole decision-maker. That makes the output easier to trust.”

### Tradeoff: demo-first simplicity vs. production hardening

- Strong answer: “The repo chooses demo-first simplicity, which is appropriate for a prototype. The tradeoff is that it would need a stronger schema layer, deployment pipeline, and monitoring before production use.”

## System design questions they may ask from this codebase

- “How would you describe the architecture?”
- “Where is the data flow bottleneck?”
- “What is the source of truth for inventory decisions?”
- “What would be the biggest challenge if this ran on real enterprise data?”
- “How would you change the design to support multiple users and multiple stores at scale?”

## What would you improve next?

- Strong answer: “I would centralize the business rules, add a proper schema layer, move to a proper data store, and create a real backend service and monitorable pipeline. I would also add authentication and better deployment automation.”
- Short version: “I would harden the architecture, centralize logic, and add real operational infrastructure.”

## What was the hardest part?

- Strong answer: “The hardest part was merging model outputs with practical business actions in a way that was understandable and trustworthy. It is not enough to have high-quality predictions if the business cannot act on them clearly.”

## If this scaled 10x, what would break first?

- Strong answer: “The local file-based model and the lack of clear data contracts would become the biggest issues. As the dataset grows, the repo would become fragile in path handling, memory use, and coordination between modules. A database, queue, and service boundary would become necessary.”

## What to say in an interview

- “This project is a retail optimization system built around a hybrid ML and operational logic pipeline.”
- “The design is intentionally simple and obvious, which is a feature for a prototype but also a clear limitation for scale.”
- “I would be candid that the current architecture is demo-first and would need hardening for production.”

## What to study next

- [main.py](main.py)
- [src/inventory_analyzer.py](src/inventory_analyzer.py)
- [src/data_preprocessing.py](src/data_preprocessing.py)
- [dashboard/app.py](dashboard/app.py)
- [TEST](TEST)

## Open questions / uncertainties

- The repo suggests a more advanced product story than the actual implementation includes, so honesty about scope is important.
- The action logic is central but not fully centralized, which is a good interview weakness to acknowledge.
- There is no visible deployment path, so interview answers should avoid claiming a mature production system without evidence.
