"""Generate a deterministic, schema-compatible demo set (not Kaggle data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(rows: int = 5000) -> None:
    rng = np.random.default_rng(42)
    pickup = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 180 * 24 * 60, rows), unit="m")
    plat = rng.normal(40.758, 0.035, rows).clip(40.60, 40.90)
    plon = rng.normal(-73.985, 0.045, rows).clip(-74.10, -73.75)
    dlat = (plat + rng.normal(0, 0.027, rows)).clip(40.58, 40.92)
    dlon = (plon + rng.normal(0, 0.035, rows)).clip(-74.12, -73.72)
    km = np.sqrt(((dlat - plat) * 111) ** 2 + ((dlon - plon) * 84) ** 2)
    rush = np.isin(pickup.hour, [7, 8, 9, 16, 17, 18, 19])
    seconds = 150 + km * np.where(rush, 260, 180) + rng.gamma(2, 55, rows)
    frame = pd.DataFrame({
        "id": [f"demo_{i:06d}" for i in range(rows)], "vendor_id": rng.integers(1, 3, rows),
        "pickup_datetime": pickup, "dropoff_datetime": pickup + pd.to_timedelta(seconds, unit="s"),
        "passenger_count": rng.integers(1, 7, rows), "pickup_longitude": plon,
        "pickup_latitude": plat, "dropoff_longitude": dlon, "dropoff_latitude": dlat,
        "store_and_fwd_flag": "N", "trip_duration": seconds.astype(int),
    })
    target = ROOT / "data/processed/demo_train.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    print(f"Wrote {len(frame):,} rows to {target}")


if __name__ == "__main__":
    main()
