# Starter prompt — Data Science Skills Mastery Lab

Paste the prompt below into a fresh Codex session opened in this folder.

---

You are a senior data scientist, analytics engineer, and ML systems engineer. Build a reproducible **Data Science Skills Mastery Lab** inspired by [dlmastery/data_science_examples/05_data_science_skills_lab](https://github.com/dlmastery/data_science_examples/tree/main/05_data_science_skills_lab), but demonstrate the complete installed collections:

- 15 skills from `param087/agent-ml-skills`
- 31 unique skills from `nimrodfisher/data-analytics-skills`

`skill_manifest.csv` is the contract. It maps every skill to an appropriate popular Kaggle dataset and required evidence artifact. Do not rename, omit, merge, or replace skills. Run `python verify_manifest.py` before work begins.

## Deliverable

Create a compact, polished lab with:

1. `src/` — reusable Python modules and SQL; never one giant notebook.
2. `reports/skills/<skill>/<evidence>` — the 46 required artifacts.
3. `reports/figures/` — readable labeled plots used by the artifacts.
4. `reports/final_report.md` — methods, results, limitations, and a 46-row evidence index.
5. `app/` — a simple searchable dashboard showing every skill, dataset, result, and artifact link.
6. `tests/` — manifest coverage, leakage, metric, API, and deterministic-seed tests.
7. `requirements.txt`, `.gitignore`, and `README.md` with exact run commands.

## Data

Use the Kaggle slugs in the manifest. Check for cached files under `data/raw/` first. If missing, use the Kaggle CLI/API. Never commit raw datasets. If Kaggle credentials are unavailable, stop after producing a download script and clearly list the blocked files; do not invent measurements or relabel synthetic data as Kaggle data.

For competition datasets, use these identifiers:

- `titanic`
- `house-prices-advanced-regression-techniques`
- `store-sales-time-series-forecasting`

For dataset slugs, use `kaggle datasets download -d <slug>`. Record source URL, license, file hash, row count, and retrieval date in `data/DATA_MANIFEST.md`.

## Demonstration standard

Every manifest row must produce its named evidence file. A sentence saying a skill “executed” is not evidence.

- Coding/ML skills: executable code plus measured output from held-out data.
- SQL/schema/semantic skills: executable DuckDB SQL or valid schema/model files plus validation output.
- Analysis skills: calculated tables/tests/plots with assumptions and uncertainty.
- Communication/workflow skills: dataset-specific documents citing measured results and linking upstream artifacts.
- MLOps skills: a small working implementation or test—tracking run, reproducibility check, debug case, or served prediction API.
- LLM/RAG skills: use small samples and lightweight models. If compute is inadequate, run a truthful smoke test and label its limits; never fabricate a full training result.

## Statistical and engineering rules

- Freeze train/validation/test splits before preprocessing. Fit transforms only on training folds.
- Use temporal splits for Store Sales and stratified splits for classification.
- Establish a naive baseline before tuned models.
- Match metrics to the task: PR-AUC for fraud, ROC-AUC/F1 for Titanic, RMSLE/MAE for House Prices, retrieval Recall@k/MRR for RAG, confidence intervals and effect size for A/B tests.
- Tune on train/validation only; touch test once for final evaluation.
- Set seeds, pin dependencies, log hardware/runtime, and save metrics as JSON/CSV.
- Distinguish association from causation. State assumptions, leakage risks, bias, privacy, and dataset limitations.
- Do not claim Kaggle rank, production readiness, causality, or performance without evidence.
- Keep the dashboard presentation-only; calculations belong in tested Python/SQL modules.

## Work order

Work in small verified phases:

1. Validate the 46-row manifest and create the project skeleton.
2. Acquire and validate datasets; create the data catalog/schema artifacts.
3. Run quality, EDA, cleaning, pandas, and feature demonstrations.
4. Run classical ML, imbalance, tuning, evaluation, reproducibility, debugging, tracking, and serving demonstrations.
5. Run lightweight PyTorch, fine-tuning, and RAG demonstrations.
6. Run SQL, business metrics, cohorts, funnel, segmentation, A/B, root-cause, and time-series demonstrations.
7. Build visualization, narrative, executive, stakeholder, QA, planning, review, and retrospective artifacts from real upstream results.
8. Build the dashboard and final report.
9. Run all tests and audit that every evidence path exists and is non-empty.

At the end of each phase report only: completed artifacts, key measured findings, tests run, blockers, and the next phase. Start with Phase 1 and continue autonomously unless credentials, material compute cost, or an irreversible action requires my input.
