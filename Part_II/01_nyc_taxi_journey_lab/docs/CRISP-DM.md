# CRISP-DM Report

## 1. Business understanding

The goal is to estimate trip duration at pickup time for a New York taxi journey. Riders benefit from clearer arrival expectations; dispatchers can compare route demand. The primary offline metric is RMSLE, matching the Kaggle competition, with MAE in seconds and R² added for interpretability. A useful system must avoid post-trip leakage and return a result quickly enough for an interactive interface.

## 2. Data understanding

The competition training set contains roughly 1.45 million 2016 trips. Inputs include pickup time, coordinates, vendor, and passenger count; `trip_duration` is the target. Important risks are impossible coordinates, zero-passenger records, target outliers, GPS error, and strong time/geography effects. `dropoff_datetime` is known only after a trip and is excluded from modeling.

The repository also provides a deterministic synthetic generator with the same core schema. It supports setup and demonstrations but cannot be used to claim real-world accuracy.

## 3. Data preparation

Quality rules retain trips with 1–6 passengers, 60–7,200 second durations, and coordinates inside a broad NYC bounding box. The pipeline then derives:

- Haversine and Manhattan-proxy distances
- Compass bearing encoded as sine and cosine
- Hour and weekday cyclic encodings
- Weekend and rush-hour indicators
- Original coordinates, passenger count, and vendor

The target is transformed with `log1p`, reducing the influence of the long right tail and aligning training with RMSLE. A single `build_features` implementation is imported by both training and inference to limit training-serving skew.

## 4. Modeling

A standardized Ridge regressor is used because its learned parameters can be exported as transparent JSON and evaluated directly in TypeScript without a second runtime. Regularization controls unstable coefficients from correlated geographic features. An 80/20 seeded split measures generalization; a time-based split and nonlinear model are recommended next experiments when a separate model service is acceptable.

## 5. Evaluation

The training command reports RMSLE, MAE, and R² on untouched validation rows and stores them beside the artifact. Acceptance should consider:

- RMSLE improvement over a median-duration baseline
- Error slices for rush hour, weekend, trip-distance bands, airports, and vendor
- Underprediction rates for long trips
- Geographic and temporal drift relative to 2016

Metrics produced from demo data describe the generator—not real taxi performance—and must be labeled accordingly.

## 6. Deployment

Next.js route handlers validate input bounds and expose `/api/predict`, `/api/model`, and `/api/health`. The Python training stage exports coefficients, scaling statistics, metrics, and provenance to a JSON artifact. TypeScript reproduces the feature transforms and evaluates that artifact directly, allowing the entire application to run and deploy as one ordinary Next.js project.

Production monitoring should log anonymous feature summaries, latency, and eventual duration where available. Retraining should be triggered by material RMSLE degradation, coordinate drift, or changes in traffic policy—not merely on a calendar schedule.
