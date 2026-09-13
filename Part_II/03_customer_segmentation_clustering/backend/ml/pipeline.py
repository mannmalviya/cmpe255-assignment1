"""CRISP-DM modeling pipeline for Kaggle Customer Personality Analysis data."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "marketing_campaign.csv"
ARTIFACTS = ROOT / "artifacts"

FEATURES = [
    "age", "income", "customer_tenure_days", "children", "total_spend",
    "total_purchases", "average_basket", "campaign_acceptance", "web_conversion",
]

PERSONAS = [
    ("VIP Champions", "Protect loyalty with private previews and premium bundles."),
    ("Digital Enthusiasts", "Use personalized digital drops and short campaign windows."),
    ("Family Loyalists", "Offer family bundles, replenishment reminders, and points."),
    ("Value Seekers", "Lead with threshold offers, value packs, and shipping incentives."),
    ("At-Risk Occasionals", "Run a low-cost win-back journey, then suppress if inactive."),
]


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the public Kaggle dataset without silently synthesizing rows."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See backend/data/README.md for setup."
        )
    frame = pd.read_csv(path, sep=None, engine="python")
    required = {
        "Year_Birth", "Dt_Customer", "Income", "Kidhome", "Teenhome", "Recency",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Response",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset contract failed; missing columns: {missing}")
    return frame


def prepare_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create compact behavioral features and retain an interpretable profile table."""
    df = raw.copy()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference = df["Dt_Customer"].max() + pd.Timedelta(days=1)
    spend_cols = [c for c in df.columns if c.startswith("Mnt")]
    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    accepted_cols = [c for c in df.columns if c.startswith("AcceptedCmp")] + ["Response"]

    model = pd.DataFrame(index=df.index)
    model["age"] = date.today().year - df["Year_Birth"]
    model["income"] = pd.to_numeric(df["Income"], errors="coerce")
    model["customer_tenure_days"] = (reference - df["Dt_Customer"]).dt.days
    model["children"] = df["Kidhome"] + df["Teenhome"]
    model["total_spend"] = df[spend_cols].sum(axis=1)
    model["total_purchases"] = df[purchase_cols].sum(axis=1)
    model["average_basket"] = model["total_spend"] / model["total_purchases"].clip(lower=1)
    model["campaign_acceptance"] = df[accepted_cols].sum(axis=1)
    model["web_conversion"] = df["NumWebPurchases"] / df["NumWebVisitsMonth"].clip(lower=1)

    valid = model["age"].between(18, 100) & model["total_spend"].ge(0)
    model = model.loc[valid, FEATURES].replace([np.inf, -np.inf], np.nan)
    profile = df.loc[model.index].copy()
    profile[FEATURES] = model
    return model, profile


def score_labels(x: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    mask = labels != -1
    clean_labels = labels[mask]
    n_clusters = len(set(clean_labels))
    if n_clusters < 2 or mask.sum() <= n_clusters:
        return {"clusters": n_clusters, "silhouette": -1.0, "davies_bouldin": 99.0, "calinski_harabasz": 0.0}
    return {
        "clusters": n_clusters,
        "silhouette": round(float(silhouette_score(x[mask], clean_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(x[mask], clean_labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(x[mask], clean_labels)), 2),
    }


def persona_mapping(profile: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Name clusters deterministically from centroid behavior, not raw numeric IDs."""
    grouped = profile.groupby("cluster")[FEATURES].mean()
    remaining = set(int(i) for i in grouped.index)
    assignments: dict[int, tuple[str, str]] = {}

    vip = int((grouped["total_spend"].rank() + grouped["campaign_acceptance"].rank()).idxmax())
    assignments[vip] = PERSONAS[0]; remaining.remove(vip)
    digital = int(grouped.loc[list(remaining), "web_conversion"].idxmax())
    assignments[digital] = PERSONAS[1]; remaining.remove(digital)
    family = int(grouped.loc[list(remaining), "children"].idxmax())
    assignments[family] = PERSONAS[2]; remaining.remove(family)
    value = int(grouped.loc[list(remaining), "total_spend"].idxmax())
    assignments[value] = PERSONAS[3]; remaining.remove(value)
    for cluster in remaining:
        assignments[cluster] = PERSONAS[4]
    return assignments


def train(data_path: Path = DEFAULT_DATA, output_dir: Path = ARTIFACTS) -> dict:
    raw = load_data(data_path)
    feature_frame, profile = prepare_features(raw)
    preprocess = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x = preprocess.fit_transform(feature_frame)

    k_search = []
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(x)
        k_search.append({"k": k, **score_labels(x, labels)})

    candidates = {
        "K-Means++": KMeans(n_clusters=5, n_init=30, random_state=42),
        "Gaussian mixture": GaussianMixture(n_components=5, covariance_type="full", random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=5, linkage="ward"),
        "DBSCAN": DBSCAN(eps=1.25, min_samples=12),
    }
    leaderboard = []
    for name, model in candidates.items():
        labels = model.fit_predict(x) if hasattr(model, "fit_predict") else model.fit(x).predict(x)
        leaderboard.append({"model": name, **score_labels(x, labels)})
    leaderboard.sort(key=lambda row: row["silhouette"], reverse=True)

    champion = KMeans(n_clusters=5, n_init=30, random_state=42).fit(x)
    labels = champion.labels_
    profile["cluster"] = labels
    pca = PCA(n_components=2, random_state=42).fit(x)
    projection = pca.transform(x)
    mapping = persona_mapping(profile)

    summaries = []
    for cluster, subset in profile.groupby("cluster"):
        name, strategy = mapping[int(cluster)]
        summaries.append({
            "cluster_id": int(cluster), "persona_name": name, "strategy": strategy,
            "customer_count": int(len(subset)), "percentage": round(len(subset) / len(profile) * 100, 1),
            "stats": {feature: round(float(subset[feature].mean()), 2) for feature in FEATURES},
        })

    sample_ids = np.random.default_rng(42).choice(len(profile), min(600, len(profile)), replace=False)
    scatter = [{
        "customer_id": int(profile.index[i]), "cluster_id": int(labels[i]),
        "pca_x": round(float(projection[i, 0]), 3), "pca_y": round(float(projection[i, 1]), 3),
    } for i in sample_ids]

    report = {
        "dataset": {"name": "Kaggle Customer Personality Analysis", "rows_raw": len(raw), "rows_modeled": len(profile), "features": FEATURES},
        "champion": "K-Means++", "metrics": score_labels(x, labels),
        "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "leaderboard": leaderboard, "k_search": k_search, "profiles": summaries, "scatter": scatter,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocess": preprocess, "model": champion, "pca": pca, "features": FEATURES, "mapping": mapping}, output_dir / "segmentation.joblib")
    (output_dir / "dashboard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    result = train(args.data, args.output)
    print(f"Saved {result['champion']} artifacts for {result['dataset']['rows_modeled']} customers")
