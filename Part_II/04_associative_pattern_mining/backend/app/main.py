"""FastAPI inference service for BasketLens artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "dashboard.json"
app = FastAPI(title="BasketLens API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3004"], allow_methods=["*"], allow_headers=["*"])


class Basket(BaseModel):
    items: list[str] = Field(min_length=1, max_length=30)


def report() -> dict:
    if not ARTIFACT.exists():
        raise HTTPException(503, "No artifact. Run: python -m ml.pipeline --demo")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "healthy", "artifact_ready": ARTIFACT.exists()}


@app.get("/api/dashboard")
def dashboard() -> dict:
    return report()


@app.get("/api/rules")
def rules(limit: int = Query(50, ge=1, le=250), min_lift: float = Query(1, ge=0)) -> dict:
    selected = [rule for rule in report()["rules"] if rule["lift"] >= min_lift][:limit]
    return {"count": len(selected), "rules": selected}


@app.post("/api/recommend")
def recommend(basket: Basket) -> dict:
    started, chosen = time.perf_counter(), set(basket.items)
    candidates: dict[str, dict] = {}
    for rule in report()["rules"]:
        if set(rule["antecedent"]) <= chosen:
            for item in rule["consequent"]:
                if item not in chosen and (item not in candidates or rule["lift"] > candidates[item]["lift"]):
                    candidates[item] = {"item": item, "confidence": rule["confidence"], "lift": rule["lift"]}
    ranked = sorted(candidates.values(), key=lambda row: (row["lift"], row["confidence"]), reverse=True)[:5]
    return {"recommendations": ranked, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
