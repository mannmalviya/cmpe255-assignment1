# Kaggle Instacart data

Download **Instacart Market Basket Analysis** from Kaggle and place these files here:

- `order_products__prior.csv`
- `products.csv`

The pipeline joins `product_id` to `product_name`, groups rows by `order_id`, removes
duplicate products inside a basket, and samples orders deterministically when
`--max-orders` is used. Raw Kaggle data is intentionally gitignored.

For a no-download smoke test, run `python -m ml.pipeline --demo`. Demo artifacts are
clearly labeled and must not be presented as Kaggle-derived results.
