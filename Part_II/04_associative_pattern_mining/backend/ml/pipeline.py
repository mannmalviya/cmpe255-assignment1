"""CRISP-DM market-basket pipeline for Kaggle's Instacart dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACT = ROOT / "artifacts" / "dashboard.json"

CATALOG = {
    "Organic Bananas": "Produce", "Organic Hass Avocado": "Produce",
    "Limes": "Produce", "Cilantro": "Produce", "Baby Spinach": "Produce",
    "Roma Tomato": "Produce", "Whole Milk": "Dairy & Eggs",
    "Large Eggs": "Dairy & Eggs", "Greek Yogurt": "Dairy & Eggs",
    "Sourdough Bread": "Bakery", "Peanut Butter": "Pantry",
    "Strawberry Jam": "Pantry", "Penne Pasta": "Pantry",
    "Marinara Sauce": "Pantry", "Parmesan": "Dairy & Eggs",
    "Tortilla Chips": "Snacks", "Salsa": "Pantry", "Sparkling Water": "Beverages",
}

ARCHETYPES = [
    ["Organic Hass Avocado", "Limes", "Cilantro", "Tortilla Chips", "Salsa"],
    ["Penne Pasta", "Marinara Sauce", "Parmesan", "Sourdough Bread"],
    ["Whole Milk", "Large Eggs", "Organic Bananas", "Greek Yogurt"],
    ["Sourdough Bread", "Peanut Butter", "Strawberry Jam", "Whole Milk"],
    ["Baby Spinach", "Roma Tomato", "Organic Hass Avocado", "Sparkling Water"],
]


def demo_baskets(n: int = 5000, seed: int = 42) -> list[set[str]]:
    """Create deterministic preview data; never label this as observed Kaggle data."""
    rng = random.Random(seed)
    products = list(CATALOG)
    baskets: list[set[str]] = []
    for _ in range(n):
        theme = rng.choice(ARCHETYPES)
        basket = {item for item in theme if rng.random() < .82}
        basket.update(item for item in products if rng.random() < .035)
        while len(basket) < 2:
            basket.add(rng.choice(products))
        baskets.append(basket)
    return baskets


def load_instacart(data_dir: Path, max_orders: int | None) -> list[set[str]]:
    """Load the official Kaggle relational CSVs without silently synthesizing rows."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install backend/requirements.txt to load Kaggle CSVs") from exc
    order_path, product_path = data_dir / "order_products__prior.csv", data_dir / "products.csv"
    if not order_path.exists() or not product_path.exists():
        raise FileNotFoundError(f"Missing Kaggle files. Follow {data_dir / 'README.md'}")
    orders = pd.read_csv(order_path, usecols=["order_id", "product_id"])
    if max_orders and orders["order_id"].nunique() > max_orders:
        keep = orders["order_id"].drop_duplicates().sample(max_orders, random_state=42)
        orders = orders[orders["order_id"].isin(keep)]
    products = pd.read_csv(product_path, usecols=["product_id", "product_name"])
    joined = orders.merge(products, on="product_id", validate="many_to_one")
    return [set(items) for items in joined.groupby("order_id")["product_name"]]


def apriori(baskets: list[set[str]], min_support: float, max_len: int = 3) -> dict[tuple[str, ...], float]:
    """Level-wise Apriori with subset pruning."""
    threshold, total = min_support * len(baskets), len(baskets)
    counts = Counter(item for basket in baskets for item in basket)
    level = {tuple([item]) for item, count in counts.items() if count >= threshold}
    frequent = {itemset: counts[itemset[0]] / total for itemset in level}
    for size in range(2, max_len + 1):
        joined = {
            tuple(sorted(set(a) | set(b))) for a in level for b in level
            if len(set(a) | set(b)) == size
        }
        candidates = {c for c in joined if all(tuple(s) in level for s in combinations(c, size - 1))}
        candidate_counts = Counter(c for basket in baskets for c in candidates if set(c) <= basket)
        level = {c for c, count in candidate_counts.items() if count >= threshold}
        frequent.update({c: candidate_counts[c] / total for c in level})
    return frequent


def eclat(baskets: list[set[str]], min_support: float, max_len: int = 3) -> dict[tuple[str, ...], float]:
    """Vertical tidset ECLAT, used as an independent benchmark."""
    threshold, total = min_support * len(baskets), len(baskets)
    tids: dict[str, set[int]] = defaultdict(set)
    for tid, basket in enumerate(baskets):
        for item in basket:
            tids[item].add(tid)
    output: dict[tuple[str, ...], float] = {}

    def visit(prefix: tuple[str, ...], tail: list[tuple[str, set[int]]]) -> None:
        for idx, (item, item_tids) in enumerate(tail):
            group = prefix + (item,)
            output[tuple(sorted(group))] = len(item_tids) / total
            if len(group) < max_len:
                rest = [(other, item_tids & other_tids) for other, other_tids in tail[idx + 1:]]
                visit(group, [(name, ids) for name, ids in rest if len(ids) >= threshold])

    visit((), sorted((item, ids) for item, ids in tids.items() if len(ids) >= threshold))
    return output


def rules_from(itemsets: dict[tuple[str, ...], float], min_confidence: float, min_lift: float) -> list[dict]:
    rules = []
    for itemset, support in itemsets.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for left in combinations(itemset, size):
                right = tuple(sorted(set(itemset) - set(left)))
                left_support, right_support = itemsets.get(tuple(sorted(left)), 0), itemsets.get(right, 0)
                if not left_support or not right_support:
                    continue
                confidence = support / left_support
                lift = confidence / right_support
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent": list(left), "consequent": list(right),
                        "support": round(support, 4), "confidence": round(confidence, 4),
                        "lift": round(lift, 3), "leverage": round(support - left_support * right_support, 4),
                        "conviction": round((1 - right_support) / max(1e-6, 1 - confidence), 3),
                    })
    return sorted(rules, key=lambda rule: (rule["lift"], rule["confidence"]), reverse=True)


def timed(method, baskets: list[set[str]], support: float) -> tuple[dict, float]:
    start = time.perf_counter()
    result = method(baskets, support)
    return result, round((time.perf_counter() - start) * 1000, 2)


def build_report(baskets: list[set[str]], source: str, support: float = .035,
                 confidence: float = .35, lift: float = 1.2) -> dict:
    apriori_sets, apriori_ms = timed(apriori, baskets, support)
    eclat_sets, eclat_ms = timed(eclat, baskets, support)
    rules = rules_from(eclat_sets, confidence, lift)
    item_counts = Counter(item for basket in baskets for item in basket)
    top_items = [{"name": name, "count": count, "support": round(count / len(baskets), 4)}
                 for name, count in item_counts.most_common(10)]
    nodes = [{"id": item, "group": CATALOG.get(item, "Other"), "value": count}
             for item, count in item_counts.most_common(16)]
    allowed = {node["id"] for node in nodes}
    links = [{"source": r["antecedent"][0], "target": r["consequent"][0], "lift": r["lift"]}
             for r in rules if len(r["antecedent"]) == len(r["consequent"]) == 1
             and r["antecedent"][0] in allowed and r["consequent"][0] in allowed][:24]
    return {
        "generated_at": "pipeline run", "source": source, "transactions": len(baskets),
        "unique_items": len(item_counts), "average_basket": round(sum(map(len, baskets)) / len(baskets), 2),
        "thresholds": {"support": support, "confidence": confidence, "lift": lift},
        "frequent_itemsets": len(eclat_sets), "rule_count": len(rules), "top_items": top_items,
        "rules": rules[:250], "graph": {"nodes": nodes, "links": links},
        "benchmarks": [
            {"algorithm": "Apriori", "runtime_ms": apriori_ms, "itemsets": len(apriori_sets), "status": "Baseline"},
            {"algorithm": "ECLAT", "runtime_ms": eclat_ms, "itemsets": len(eclat_sets), "status": "Champion" if eclat_ms <= apriori_ms else "Challenger"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="use deterministic preview baskets")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--max-orders", type=int, default=10000)
    parser.add_argument("--support", type=float, default=.035)
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--lift", type=float, default=1.2)
    args = parser.parse_args()
    baskets = demo_baskets(args.max_orders) if args.demo else load_instacart(args.data, args.max_orders)
    source = "Deterministic demo preview" if args.demo else "Kaggle Instacart Market Basket Analysis"
    report = build_report(baskets, source, args.support, args.confidence, args.lift)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {report['rule_count']} rules from {report['transactions']:,} baskets to {ARTIFACT}")


if __name__ == "__main__":
    main()
