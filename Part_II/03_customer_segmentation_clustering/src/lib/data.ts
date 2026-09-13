export type Persona = {
  id: number;
  name: string;
  kicker: string;
  color: string;
  count: number;
  share: number;
  income: number;
  spend: number;
  recency: number;
  webConversion: number;
  strategy: string;
};

export type Point = { x: number; y: number; cluster: number; customer: number };

export const personas: Persona[] = [
  { id: 0, name: "VIP Champions", kicker: "High value · highly engaged", color: "#6D5DFC", count: 328, share: 14.6, income: 78.4, spend: 1618, recency: 24, webConversion: 0.71, strategy: "Protect loyalty with private previews, concierge outreach, and premium bundles." },
  { id: 1, name: "Digital Enthusiasts", kicker: "Web-first · promotion ready", color: "#23B6A7", count: 451, share: 20.1, income: 56.2, spend: 932, recency: 35, webConversion: 0.62, strategy: "Use app-exclusive drops, personalized recommendations, and short campaign windows." },
  { id: 2, name: "Family Loyalists", kicker: "Steady · store-led households", color: "#F4A340", count: 538, share: 24.0, income: 49.8, spend: 681, recency: 47, webConversion: 0.41, strategy: "Offer family bundles, replenishment reminders, and points-based rewards." },
  { id: 3, name: "Value Seekers", kicker: "Deal responsive · selective", color: "#EC6A8D", count: 497, share: 22.2, income: 35.1, spend: 318, recency: 63, webConversion: 0.34, strategy: "Lead with threshold offers, value packs, and targeted free-shipping incentives." },
  { id: 4, name: "At-Risk Occasionals", kicker: "Low frequency · reactivation", color: "#4A83E8", count: 426, share: 19.0, income: 42.6, spend: 204, recency: 81, webConversion: 0.22, strategy: "Run a low-cost win-back journey with one clear offer and suppress if inactive." },
];

export const modelMetrics = [
  { model: "K-Means++", silhouette: 0.341, db: 1.19, ch: 481.3, status: "Champion" },
  { model: "Gaussian mixture", silhouette: 0.326, db: 1.27, ch: 443.8, status: "Candidate" },
  { model: "Agglomerative", silhouette: 0.304, db: 1.34, ch: 418.5, status: "Candidate" },
  { model: "DBSCAN", silhouette: 0.218, db: 1.71, ch: 205.7, status: "Baseline" },
];

export const kScores = [
  { k: 2, score: 0.277 }, { k: 3, score: 0.302 }, { k: 4, score: 0.329 },
  { k: 5, score: 0.341 }, { k: 6, score: 0.315 }, { k: 7, score: 0.296 }, { k: 8, score: 0.281 },
];

export const crispPhases = [
  { n: "01", name: "Business understanding", note: "Create actionable audiences that improve campaign relevance and reduce blanket discounting." },
  { n: "02", name: "Data understanding", note: "2,240 customer records, 29 source fields, income missingness reviewed, distributions and outliers profiled." },
  { n: "03", name: "Data preparation", note: "Engineer age, tenure, household size, total spend, purchase volume, basket value, and campaign response." },
  { n: "04", name: "Modeling", note: "Standardize features; compare K-Means++, GMM, agglomerative clustering, and DBSCAN across candidate k." },
  { n: "05", name: "Evaluation", note: "Select using silhouette, Davies–Bouldin, Calinski–Harabasz, stability, and business interpretability." },
  { n: "06", name: "Deployment", note: "Persist model artifacts, expose FastAPI inference, monitor cluster share and centroid drift, retrain quarterly." },
];

function noise(seed: number) {
  const value = Math.sin(seed * 9301 + 49297) * 233280;
  return value - Math.floor(value);
}

export const points: Point[] = personas.flatMap((persona, cluster) =>
  Array.from({ length: 38 }, (_, i) => {
    const angles = [3.85, 5.55, 1.05, 2.2, 0.05];
    const radii = [31, 27, 34, 29, 26];
    const centerX = 50 + Math.cos(angles[cluster]) * radii[cluster];
    const centerY = 50 + Math.sin(angles[cluster]) * radii[cluster] * 0.7;
    return {
      x: Math.max(4, Math.min(96, centerX + (noise(cluster * 91 + i) - 0.5) * 18)),
      y: Math.max(6, Math.min(94, centerY + (noise(cluster * 131 + i + 8) - 0.5) * 24)),
      cluster,
      customer: 10000 + cluster * 100 + i,
    };
  }),
);
