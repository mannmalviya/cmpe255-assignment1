# BasketLens — Associative Pattern Mining

A full-stack recreation of the associative-pattern-mining assignment using
**Next.js + TypeScript** for the decision dashboard and **Python + FastAPI** for
mining and inference. The target dataset is Kaggle's popular **Instacart Market
Basket Analysis** dataset.

Unlike the reference implementation, BasketLens distinguishes real Kaggle input
from generated preview data. The dashboard ships with a clearly labeled,
deterministic demo snapshot; assignment results should be regenerated from the
Kaggle CSVs before submission.

## Features

- Five-view data science dashboard: overview, rule explorer, basket recommender,
  model admin, and interactive CRISP-DM dossier
- Apriori and ECLAT implemented in Python with support, confidence, lift,
  leverage, and conviction
- Real Instacart relational-data loader with schema checks
- FastAPI endpoints for dashboard artifacts, rule filtering, and recommendations
- Threshold sensitivity, algorithm parity, and deployment-governance views
- Responsive UI with no charting-library dependency

## Quick start

```bash
# Terminal 1 — create a preview artifact and run the API
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.pipeline --demo --max-orders 5000
uvicorn app.main:app --reload --port 8004

# Terminal 2 — dashboard
npm install
npm run dev
```

Open `http://localhost:3004`. The UI works from its checked-in preview constants
when the API is not running; the Python endpoints remain available for direct
inspection and integration.

## Use the Kaggle data

1. Download *Instacart Market Basket Analysis* from Kaggle.
2. Follow `backend/data/README.md`.
3. Run:

```bash
cd backend
python -m ml.pipeline --max-orders 10000 --support .035 --confidence .35 --lift 1.2
uvicorn app.main:app --reload --port 8004
```

Use `--max-orders` conservatively for Apriori. ECLAT is substantially faster on
this sparse workload; production-scale runs should additionally cap the product
universe or use an optimized FP-Growth library.

## API

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/rules?limit=50&min_lift=1.5`
- `POST /api/recommend` with `{"items":["Marinara Sauce","Penne Pasta"]}`

See [CRISP_DM_REPORT.md](./CRISP_DM_REPORT.md) for methodology, assumptions, and
the deployment plan.

## Academic integrity note

The linked professor project was used as a scope reference. This implementation
uses a new Next.js/TypeScript interface, an independently structured Python
pipeline, explicit source provenance, and a corrected real-Kaggle ingestion path.
