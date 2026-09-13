"""Fail fast unless all installed source skills have one concrete demonstration."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
EXPECTED = {"param087/agent-ml-skills": 15, "nimrodfisher/data-analytics-skills": 31}


def main() -> None:
    with (ROOT / "skill_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    names = [row["skill"] for row in rows]
    counts = Counter(row["origin"] for row in rows)
    assert len(rows) == 46, f"expected 46 demonstrations, found {len(rows)}"
    assert len(names) == len(set(names)), "skill names must be unique"
    assert counts == Counter(EXPECTED), f"wrong source counts: {dict(counts)}"
    assert all(row["dataset_slug"] and row["evidence"] for row in rows)
    print("PASS: 46 unique skills mapped (15 ML + 31 analytics).")


if __name__ == "__main__":
    main()
