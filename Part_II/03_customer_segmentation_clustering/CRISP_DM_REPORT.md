# CRISP-DM Report: Customer Personality Segmentation

## 1. Business understanding

### Objective

Replace one-size-fits-all marketing with a small set of measurable customer audiences. Each segment must support a different campaign, offer, channel, or contact policy.

### Business questions

1. Which customers represent the highest current value?
2. Which customers prefer digital purchasing?
3. Which households are likely to respond to family-oriented value offers?
4. Which low-engagement customers merit a win-back campaign?

### Success criteria

- Produce four to six stable, interpretable segments.
- Improve silhouette score over a simple baseline.
- Give every segment a distinct marketing treatment.
- Serve a new-customer assignment through an API.
- Monitor cluster share and distance drift after deployment.

Clustering finds behavioral similarity; it does not prove that a campaign will cause higher revenue. Campaign recommendations should be validated with randomized holdouts.

## 2. Data understanding

The project expects Kaggle's **Customer Personality Analysis** file, commonly named `marketing_campaign.csv`. It contains 2,240 customer-level rows and demographic, household, product-spend, purchase-channel, campaign, and recency fields.

Quality checks in the pipeline include:

- Required-column contract validation
- Flexible delimiter detection for the tab-separated source file
- Coercion of income and customer-start date
- Rejection of implausible ages outside 18–100
- Missing-value imputation fitted only in the preprocessing artifact
- Infinite-value replacement after ratio construction

Known concerns include missing income, extreme birth years, skewed monetary fields, correlated product spend fields, and the absence of a ground-truth customer segment label.

## 3. Data preparation

The feature matrix is intentionally compact:

| Feature | Business meaning |
|---|---|
| `age` | Customer life stage |
| `income` | Approximate purchasing capacity |
| `customer_tenure_days` | Relationship maturity |
| `children` | Household structure |
| `total_spend` | Realized customer value |
| `total_purchases` | Shopping frequency |
| `average_basket` | Value per purchase occasion |
| `campaign_acceptance` | Historical campaign responsiveness |
| `web_conversion` | Digital purchase efficiency |

Numeric missing values are median-imputed, then standardized with `StandardScaler`. This is necessary because Euclidean-distance models would otherwise be dominated by large-unit features such as income and spend.

## 4. Modeling

The pipeline compares four complementary approaches:

- **K-Means++:** fast, stable centroid partitioning and direct out-of-sample inference
- **Gaussian mixture:** soft probabilistic membership with elliptical clusters
- **Agglomerative clustering:** hierarchy-based grouping without centroid assumptions
- **DBSCAN:** density-based discovery with explicit noise points

For the production K-Means candidate, `k=2...8` is evaluated. The assignment uses `k=5` for reference parity and then verifies that choice against internal metrics and persona interpretability. PCA provides a two-dimensional visualization only; the production model is fitted on the complete standardized matrix.

## 5. Evaluation

No true segment label exists, so multiple internal measures are required:

- **Silhouette:** cohesion versus separation, from -1 to 1; higher is better.
- **Davies–Bouldin:** average worst-case similarity between clusters; lower is better.
- **Calinski–Harabasz:** between-cluster to within-cluster dispersion; higher is better.

Quantitative scores alone do not establish business value. Before promotion, also review:

- Cluster size: avoid tiny audiences unless strategically important.
- Stability: compare assignments across random seeds with adjusted Rand index.
- Interpretability: confirm every profile has a coherent behavior pattern.
- Actionability: document a treatment and a measurable KPI for every profile.
- Sensitivity: inspect whether outliers or imputation choices materially change centroids.

The dashboard values are demo artifacts until the user executes the pipeline on the downloaded dataset. The training script writes its computed scorecard to `backend/artifacts/dashboard.json`.

## 6. Deployment

The deployment unit is `segmentation.joblib`, containing the fitted imputer/scaler, K-Means model, PCA transform, ordered feature contract, and persona mapping. FastAPI exposes:

- `GET /api/health`
- `GET /api/dashboard`
- `POST /api/predict`

Recommended monitoring:

| Signal | Suggested trigger | Response |
|---|---|---|
| Required-field missingness | +5 percentage points | Investigate upstream data |
| Segment population share | PSI > 0.20 | Review drift and campaign changes |
| Mean centroid distance | +20% from training | Evaluate retraining |
| Segment campaign lift | No positive lift in two cycles | Revise persona treatment |

Retrain quarterly or after a material drift trigger. Version the dataset snapshot, feature code, metrics JSON, and model artifact together. Never interpret cluster membership as a protected-class decision or use it for consequential eligibility decisions without a separate fairness and legal review.
