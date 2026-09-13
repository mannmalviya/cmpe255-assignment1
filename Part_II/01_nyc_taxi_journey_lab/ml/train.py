"""Train and persist a leakage-safe NYC taxi duration model."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURES, build_features, clean_training_data

ROOT = Path(__file__).resolve().parents[1]


def train(input_path: Path, output_path: Path, max_rows: int | None = 300_000) -> dict:
    data = pd.read_csv(input_path, nrows=max_rows, parse_dates=["pickup_datetime"])
    data = clean_training_data(data)
    if len(data) < 100:
        raise ValueError("At least 100 valid rows are required for a meaningful split.")

    x = build_features(data)
    y = np.log1p(data["trip_duration"].to_numpy())
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = Ridge(alpha=3.0)
    model.fit(x_train_scaled, y_train)
    truth = np.expm1(y_test)
    predicted = np.maximum(1, np.expm1(model.predict(scaler.transform(x_test))))
    metrics = {
        "rmsle": round(float(np.sqrt(mean_squared_log_error(truth, predicted))), 4),
        "mae_seconds": round(float(mean_absolute_error(truth, predicted)), 1),
        "r2": round(float(r2_score(truth, predicted)), 4),
    }
    metadata = {
        "model": "Standardized Ridge Regression", "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(x_train), "validation_rows": len(x_test),
        "data_source": "synthetic demo" if "demo" in input_path.name.lower() else "Kaggle competition",
        "features": FEATURES, "metrics": metrics,
        "target_transform": "log1p(trip_duration)", "random_state": 42,
        "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist(),
        "feature_means": scaler.mean_.tolist(),
        "feature_scales": scaler.scale_.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models/taxi_duration.json")
    parser.add_argument("--max-rows", type=int, default=300_000)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output, args.max_rows), indent=2))
