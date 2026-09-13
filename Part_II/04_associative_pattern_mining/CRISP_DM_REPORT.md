# CRISP-DM Report: Instacart Association Pattern Mining

## 1. Business understanding

**Question:** Which products co-occur often enough, and with enough excess over
chance, to support an explainable cart add-on recommendation?

**Users:** merchandising analysts and e-commerce product teams. **Primary KPI:**
attach-rate uplift versus a popularity baseline. Offline acceptance requires
support ≥ 3.5%, confidence ≥ 35%, and lift ≥ 1.20. Stock, margin, dietary context,
and customer consent remain deployment guardrails.

## 2. Data understanding

The intended source is Kaggle's *Instacart Market Basket Analysis*. The pipeline
uses `order_products__prior.csv` and `products.csv`; `order_id` is the basket,
and `product_id` joins to the human-readable product name. Important checks are
key uniqueness in products, orphaned IDs, repeated lines, basket-size skew, and
the long tail of low-support items.

The bundled UI values come from a deterministic 5,000-basket preview with 18
products. They demonstrate functionality and are not empirical Kaggle findings.

## 3. Data preparation

Order lines are joined many-to-one to product metadata, grouped by `order_id`,
and converted to sets. Sets intentionally discard quantity and order position:
standard support measures basket presence, not units. Optional order sampling
uses a fixed seed. A real study should record the sampled order IDs and compare
prior/train splits to avoid temporal leakage.

## 4. Modeling

Two exact frequent-itemset methods are compared under identical thresholds:

- **Apriori:** level-wise candidate generation with downward-closure pruning.
- **ECLAT:** vertical transaction-ID intersections.

For disjoint itemsets A and B:

- support(A → B) = P(A ∪ B)
- confidence(A → B) = P(A ∪ B) / P(A)
- lift(A → B) = confidence(A → B) / P(B)
- leverage(A → B) = P(A ∪ B) − P(A)P(B)
- conviction(A → B) = (1 − P(B)) / (1 − confidence)

Preview configuration: maximum itemset length 3, support .035, confidence .35,
and lift 1.20. The preview produced 78 itemsets and 203 rules. These values will
change on the real data.

## 5. Evaluation

Algorithm parity is checked by comparing itemset support values. Lift ranks
rules, while support limits chance discoveries and confidence communicates
conversion potential. Analysts must inspect the top rules for substitutes,
trivial variants, and taxonomy leakage. Production approval additionally needs:

1. a time-based holdout;
2. coverage and catalog-diversity measures;
3. an online A/B test against popularity recommendations;
4. confidence intervals or stability across resamples.

## 6. Deployment

The versioned JSON artifact contains provenance, thresholds, rule metrics,
network data, and benchmark results. FastAPI loads it once and performs an
antecedent subset lookup for each request. Next.js exposes exploration and admin
views. Monitor p95 latency, rule coverage, attach rate, product availability,
catalog drift, and rule age. Retrain after meaningful catalog or behavior shifts,
not merely on a fixed schedule.

## Limitations

Association is not causation. High lift can occur for niche items, popularity can
dominate confidence, and aggregate rules can hide segment differences. The demo
generator deliberately contains archetypes, so it is useful for software tests
but inappropriate for claiming real shopper behavior.
