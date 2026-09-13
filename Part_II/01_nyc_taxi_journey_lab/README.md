# Cabwise — NYC Taxi Journey Lab

An end-to-end CRISP-DM project for the [Kaggle NYC Taxi Trip Duration challenge](https://www.kaggle.com/competitions/nyc-taxi-trip-duration): data ingestion, validation, geospatial feature engineering, portable model training, typed API routes, and a polished Next.js frontend.

The visual concept was independently implemented after reviewing the architecture of [dlmastery's example](https://github.com/dlmastery/data_science_examples/tree/main/01_nyc_taxi_trip_prediction). This project uses its own code, design, model pipeline, and copy.

## Quick start

```bash
cd Part_II/nyc-taxi-journey-lab
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. The included trained JSON model is loaded directly by the Next.js server, so no Python process, API key, database, or container is required.

## Retrain the model (optional)

Python is used only to prepare data and export a new portable model:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A: make schema-compatible demo data
python -m ml.generate_demo_data
python -m ml.train --input data/processed/demo_train.csv

# Option B: use the real competition data (Kaggle credentials required)
pip install kaggle
python -m ml.download_data
python -m ml.train
```

Training overwrites `models/taxi_duration.json`; restarting Next.js activates it.

## Project map

```text
app/                 Next.js UI and typed API route handlers
lib/                 TypeScript feature engineering and inference
ml/                  data download, feature pipeline, training
data/                 raw and generated-data landing zones
models/               portable trained JSON artifact
docs/                  CRISP-DM report and model card
```

## Reproducibility

Training uses a fixed random seed, a held-out 20% validation split, and a log-transformed target. Python and TypeScript implement the same ordered feature contract; the saved JSON records that order alongside scaling parameters, coefficients, metrics, row counts, and training time.

## Build and deployment

Run `npm run build` followed by `npm start`. Because inference uses ordinary TypeScript and a bundled JSON artifact, the application can be deployed as one standard Next.js project without external services.

See [CRISP-DM.md](docs/CRISP-DM.md) for the research workflow and [MODEL_CARD.md](docs/MODEL_CARD.md) for intended use and limitations.
