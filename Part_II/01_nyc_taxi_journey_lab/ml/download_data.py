"""Download and unpack the Kaggle NYC Taxi Trip Duration competition data."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/raw"


def main() -> None:
    if not shutil.which("kaggle"):
        raise SystemExit("Install the Kaggle CLI first: pip install kaggle")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "competitions", "download", "-c", "nyc-taxi-trip-duration",
        "-p", str(DESTINATION), "--unzip",
    ], check=True)
    print(f"Dataset unpacked to {DESTINATION}")


if __name__ == "__main__":
    main()

