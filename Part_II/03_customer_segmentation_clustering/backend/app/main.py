"""FastAPI service for trained customer-segmentation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

app = FastAPI(title="SegmentIQ API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3001"], allow_methods=["*"], allow_headers=["*"])


class Customer(BaseModel):
    age: float = Field(40, ge=18, le=100)
    income: float = Field(72000, ge=0)
    customer_tenure_days: float = Field(900, ge=0)
    children: float = Field(1, ge=0, le=10)
    total_spend: float = Field(900, ge=0)
    total_purchases: float = Field(12, ge=0)
    average_basket: float = Field(75, ge=0)
    campaign_acceptance: float = Field(1, ge=0, le=6)
    web_conversion: float = Field(.5, ge=0)


def read_report() -> dict:
    path = ARTIFACTS / "dashboard.json"
    if not path.exists():
        raise HTTPException(503, "Artifacts are not trained. Run: python -m ml.pipeline")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "healthy", "artifact_ready": (ARTIFACTS / "segmentation.joblib").exists()}


@app.get("/api/dashboard")
def dashboard() -> dict:
    return read_report()


@app.post("/api/predict")
def predict(customer: Customer) -> dict:
    path = ARTIFACTS / "segmentation.joblib"
    if not path.exists():
        raise HTTPException(503, "Model artifact is not trained")
    bundle = joblib.load(path)
    frame = pd.DataFrame([customer.model_dump()])[bundle["features"]]
    x = bundle["preprocess"].transform(frame)
    cluster = int(bundle["model"].predict(x)[0])
    distances = bundle["model"].transform(x)[0]
    probabilities = np.exp(-distances) / np.exp(-distances).sum()
    name, strategy = bundle["mapping"][cluster]
    projection = bundle["pca"].transform(x)[0]
    return {
        "cluster_id": cluster, "persona_name": name, "strategy": strategy,
        "assignment_confidence": round(float(probabilities[cluster]) * 100, 1),
        "distance_to_centroid": round(float(distances[cluster]), 3),
        "pca": [round(float(value), 3) for value in projection],
    }
