# Model Card

## Overview

- **Task:** predict NYC taxi trip duration in seconds at pickup time
- **Estimator:** standardized Ridge regression, exported to portable JSON
- **Target:** `log1p(trip_duration)`
- **Features:** 16 pickup-time-safe geographic, temporal, and operational signals
- **Evaluation:** seeded 80/20 holdout; RMSLE, MAE, and R²

Exact metrics, data counts, scaling statistics, and coefficients are written to `models/taxi_duration.json` by the training command and evaluated by Next.js.

## Intended use

Suitable for coursework, exploratory journey planning, and demonstrating an end-to-end tabular ML deployment. Predictions are not guaranteed arrival times, taxi-meter quotes, navigation instructions, or safety-critical decisions.

## Limitations

The Kaggle data represents 2016 patterns and does not encode live traffic, weather, road closures, route geometry, events, driver behavior, or contemporary pricing. Bounding-box validation is not proof that a coordinate is road-accessible. Airport and unusually long journeys may have less reliable estimates. Passenger/vendor variables can proxy operational patterns and should not be interpreted causally.

Synthetic demo data exists only to exercise the pipeline; validation results from it are not evidence of performance on real trips.

## Responsible operation

Label the active prediction source, monitor errors by time and distance segments, and retain no unnecessary rider information. Retrain and revalidate before using the service outside the historical competition setting.
