# SegmentIQ — Customer Segmentation Studio

An end-to-end clustering assignment built with **Next.js, TypeScript, Python, scikit-learn, and FastAPI**. It recreates the learning objectives of the linked customer-segmentation reference while using the real schema of Kaggle's popular **Customer Personality Analysis** dataset and organizing the work with all six CRISP-DM phases.

The UI includes:

- Executive KPIs, segment distribution, and an interactive PCA customer landscape
- Five actionable persona profiles and a cross-segment comparison
- Data-science admin dashboard with algorithm leaderboard, candidate-k analysis, and governance checks
- Interactive CRISP-DM decision report
- Customer persona inference with a graceful demo fallback when the API is offline

> The values visible on first launch are explicitly marked **demo data**. Run the Python pipeline with the Kaggle CSV to compute and persist genuine dataset results.

## Architecture

```text
Kaggle CSV
  → Python validation + feature engineering
  → imputation + StandardScaler
  → K-Means / GMM / Agglomerative / DBSCAN benchmark
  → champion K-Means + PCA + persona profiling
  → joblib + JSON artifacts
  → FastAPI inference and dashboard endpoints
  → Next.js / TypeScript decision dashboard
```

## Quick start

### 1. Frontend

```bash
npm install
npm run dev
```

Open `http://localhost:3001`.

### 2. Dataset and Python pipeline

Download **Customer Personality Analysis** from Kaggle and save the extracted file as `backend/data/marketing_campaign.csv`.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.pipeline
```

Generated artifacts:

- `backend/artifacts/segmentation.joblib` — preprocessing, champion model, PCA, and persona mapping
- `backend/artifacts/dashboard.json` — metrics, profiles, k search, and sampled projections

### 3. API

From `backend/` with the virtual environment active:

```bash
uvicorn app.main:app --reload --port 8003
```

API docs are at `http://localhost:8003/docs`. The frontend calls the trained API when available and otherwise stays usable in clearly labeled demo mode.

## Modeling design

Nine compact behavioral features are used: age, income, tenure, children, total spend, total purchases, average basket, campaign acceptance, and web conversion. Redundant product-level totals are consolidated before scaling so categories do not receive accidental extra distance weight.

Candidate algorithms are ranked using:

- Silhouette coefficient (higher is better)
- Davies–Bouldin index (lower is better)
- Calinski–Harabasz index (higher is better)
- Business interpretability and seed stability as deployment gates

The cluster IDs produced by K-Means are arbitrary. `persona_mapping()` assigns names deterministically from cluster behavior, preventing an unstable numeric ID from becoming a business definition.

## CRISP-DM

The detailed assignment narrative is in [CRISP_DM_REPORT.md](./CRISP_DM_REPORT.md). The same six stages are also available as an interactive view in the dashboard.

## Reference alignment

This implementation retains the reference project's core scope—persona exploration, multi-model benchmarking, PCA visualization, real-time inference, model administration, and CRISP-DM documentation—while replacing its Vite/JavaScript frontend with Next.js/TypeScript and avoiding the reference's synthetic-data-as-Kaggle ambiguity.
