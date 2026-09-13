export type Rule = { left: string[]; right: string[]; support: number; confidence: number; lift: number; conviction: number };

export const products = [
  "Organic Hass Avocado", "Sourdough Bread", "Whole Milk", "Salsa", "Peanut Butter",
  "Large Eggs", "Strawberry Jam", "Limes", "Tortilla Chips", "Penne Pasta",
  "Marinara Sauce", "Parmesan", "Organic Bananas", "Greek Yogurt", "Cilantro",
];

export const rules: Rule[] = [
  { left: ["Marinara Sauce", "Penne Pasta"], right: ["Parmesan"], support: .1096, confidence: .8341, lift: 4.451, conviction: 4.898 },
  { left: ["Large Eggs", "Organic Bananas"], right: ["Greek Yogurt"], support: .1038, confidence: .8135, lift: 4.392, conviction: 4.368 },
  { left: ["Marinara Sauce", "Parmesan"], right: ["Penne Pasta"], support: .1096, confidence: .8303, lift: 4.379, conviction: 4.776 },
  { left: ["Peanut Butter", "Sourdough Bread"], right: ["Strawberry Jam"], support: .1068, confidence: .8266, lift: 4.318, conviction: 4.663 },
  { left: ["Cilantro", "Limes"], right: ["Organic Hass Avocado"], support: .1072, confidence: .8221, lift: 2.284, conviction: 3.597 },
  { left: ["Limes", "Organic Hass Avocado"], right: ["Tortilla Chips"], support: .1084, confidence: .8114, lift: 4.118, conviction: 4.258 },
  { left: ["Salsa", "Tortilla Chips"], right: ["Limes"], support: .1042, confidence: .8053, lift: 4.091, conviction: 4.124 },
  { left: ["Whole Milk", "Sourdough Bread"], right: ["Peanut Butter"], support: .0718, confidence: .5524, lift: 2.770, conviction: 1.788 },
];

export const topItems = [
  ["Organic Hass Avocado", 1800, 36.0], ["Sourdough Bread", 1760, 35.2], ["Whole Milk", 1720, 34.4],
  ["Salsa", 1007, 20.1], ["Peanut Butter", 997, 19.9], ["Large Eggs", 986, 19.7],
] as const;

export const crisp = [
  { n: "01", name: "Business understanding", note: "Turn product affinity into measurable cross-sell decisions.", decision: "Optimize attach rate without replacing merchandising judgment.", items: ["Primary KPI: recommendation attach rate", "Guardrails: margin, stock, and relevance", "Success gate: lift > 1.20 and confidence > 35%"] },
  { n: "02", name: "Data understanding", note: "Audit order-level Instacart transactions and product metadata.", decision: "The order—not the customer—is the unit of analysis.", items: ["Validate order and product keys", "Profile basket size and item support", "Check duplicates and orphaned product IDs"] },
  { n: "03", name: "Data preparation", note: "Create unique, unordered product sets for each order.", decision: "Deduplicate items inside each basket to preserve binary support semantics.", items: ["Join order lines to product names", "Deterministic order sampling", "Remove empty and single-item baskets"] },
  { n: "04", name: "Modeling", note: "Mine frequent sets, then derive directional association rules.", decision: "Benchmark Apriori against vertical-tidset ECLAT at equal thresholds.", items: ["Minimum support: 3.5%", "Maximum itemset length: 3", "Confidence and lift pruning"] },
  { n: "05", name: "Evaluation", note: "Judge usefulness with statistical strength and business plausibility.", decision: "Lift is the ranker; support protects against brittle coincidences.", items: ["Review top rules for leakage", "Compare runtime and rule parity", "Validate with a temporal holdout before production"] },
  { n: "06", name: "Deployment", note: "Serve precomputed rules through a low-latency basket API.", decision: "Version thresholds, source metadata, and artifacts together.", items: ["FastAPI recommendation endpoint", "Next.js admin dashboard", "Monitor coverage, latency, drift, and attach rate"] },
];

export const experiments = [
  { support: .025, itemsets: 126, rules: 341, runtime: 172 },
  { support: .035, itemsets: 78, rules: 203, runtime: 110 },
  { support: .050, itemsets: 64, rules: 164, runtime: 86 },
  { support: .075, itemsets: 48, rules: 117, runtime: 63 },
  { support: .100, itemsets: 32, rules: 72, runtime: 41 },
];
