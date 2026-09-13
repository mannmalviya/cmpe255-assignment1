"""Shared, vectorized feature engineering for training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_KM = 6371.0088
FEATURES = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude",
    "dropoff_latitude", "passenger_count", "vendor_id", "distance_km",
    "manhattan_km", "bearing_sin", "bearing_cos", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "is_weekend", "is_rush_hour",
]


def _distance(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the exact model matrix used by both batch training and the API."""
    data = frame.copy()
    dt = pd.to_datetime(data["pickup_datetime"], errors="coerce")
    lat1, lon1 = data["pickup_latitude"], data["pickup_longitude"]
    lat2, lon2 = data["dropoff_latitude"], data["dropoff_longitude"]

    data["distance_km"] = _distance(lat1, lon1, lat2, lon2)
    ns = _distance(lat1, lon1, lat2, lon1)
    ew = _distance(lat1, lon1, lat1, lon2)
    data["manhattan_km"] = ns + ew

    y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - (
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))
    )
    bearing = np.arctan2(y, x)
    data["bearing_sin"], data["bearing_cos"] = np.sin(bearing), np.cos(bearing)

    hour = dt.dt.hour + dt.dt.minute / 60
    dow = dt.dt.dayofweek
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    data["is_weekend"] = (dow >= 5).astype(int)
    data["is_rush_hour"] = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))).astype(int)
    return data[FEATURES].astype(float)


def clean_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit, reproducible quality rules before feature creation."""
    required = {"pickup_datetime", "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude", "passenger_count",
                "vendor_id", "trip_duration"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    mask = (
        frame["pickup_latitude"].between(40.55, 40.95)
        & frame["dropoff_latitude"].between(40.55, 40.95)
        & frame["pickup_longitude"].between(-74.15, -73.70)
        & frame["dropoff_longitude"].between(-74.15, -73.70)
        & frame["passenger_count"].between(1, 6)
        & frame["trip_duration"].between(60, 7200)
    )
    return frame.loc[mask].dropna(subset=list(required)).copy()

