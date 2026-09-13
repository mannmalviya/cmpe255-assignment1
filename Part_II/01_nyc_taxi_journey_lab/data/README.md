# Data

`raw/` is the landing zone for Kaggle `train.csv` and `test.csv`; these licensed competition files are intentionally not committed. Run `python -m ml.download_data` after configuring the Kaggle CLI.

`processed/` holds locally generated or cleaned data. Run `python -m ml.generate_demo_data` to create 5,000 deterministic, schema-compatible rows for an offline demonstration. They are synthetic and must not be presented as Kaggle observations.

Expected training columns: `vendor_id`, `pickup_datetime`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, and `trip_duration`.

