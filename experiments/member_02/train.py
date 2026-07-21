"""Legacy multi-output Random Forest adapted to the team evaluation contract."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.data_loader import load_taxi_pandas
from src.common.evaluation import evaluate_fare_bands, evaluate_regression
from src.common.results import save_result
from src.common.split import RANDOM_STATE, SPLIT_ID, make_common_split


AUTHOR = "member_02"
EXPERIMENT_ID = "member_02_multioutput_random_forest_v1"
TARGET_COLUMN = "total_amount"
COMPONENT_TARGETS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]
NUMERIC_FEATURES = [
    "passenger_count",
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "trip_duration_minutes",
]
CATEGORICAL_FEATURES = [
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
]
DATETIME_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
DEFAULT_MAX_TRAIN_ROWS = 200_000


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the time features used by the legacy experiment."""
    featured = dataframe.copy()
    pickup = pd.to_datetime(featured["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(featured["tpep_dropoff_datetime"], errors="coerce")
    featured["pickup_hour"] = pickup.dt.hour
    featured["pickup_day_of_week"] = pickup.dt.dayofweek
    featured["is_weekend"] = (featured["pickup_day_of_week"] >= 5).astype("int8")
    featured["trip_duration_minutes"] = (dropoff - pickup).dt.total_seconds() / 60
    return featured


def select_training_rows(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply legacy quality filters to training rows only."""
    featured = add_features(dataframe)
    before = len(featured)
    valid = (
        featured[DATETIME_COLUMNS].notna().all(axis=1)
        & featured["trip_distance"].gt(0)
        & featured["trip_duration_minutes"].gt(0)
        & featured[COMPONENT_TARGETS].notna().all(axis=1)
    )
    selected = featured.loc[valid].copy()
    duration_cap = selected["trip_duration_minutes"].quantile(0.995)
    selected = selected.loc[selected["trip_duration_minutes"].le(duration_cap)].copy()
    return selected, {
        "common_train_rows": before,
        "quality_eligible_rows": int(valid.sum()),
        "rows_after_duration_cap": len(selected),
    }


def build_pipeline() -> Pipeline:
    """Build the preprocessing and multi-output Random Forest pipeline."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def run_experiment(max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS) -> dict[str, Any]:
    """Train on the common split, predict every test row, and save metrics."""
    source = load_taxi_pandas()
    common_train, common_test = make_common_split(source)
    train, filtering = select_training_rows(common_train)
    if max_train_rows > 0 and len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=RANDOM_STATE)

    test = add_features(common_test)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    pipeline = build_pipeline()
    started = time.perf_counter()
    pipeline.fit(train[feature_columns], train[COMPONENT_TARGETS])
    component_predictions = pipeline.predict(test[feature_columns])
    predictions = np.asarray(component_predictions).sum(axis=1)
    training_seconds = time.perf_counter() - started

    metrics = evaluate_regression(test[TARGET_COLUMN], predictions)
    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "author": AUTHOR,
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "fare_band_metrics": evaluate_fare_bands(test[TARGET_COLUMN], predictions),
        "data": {
            "split_id": SPLIT_ID,
            "common_train_rows": len(common_train),
            "train_rows_used": len(train),
            "test_rows": len(test),
            "predicted_test_rows": len(predictions),
            "training_filter": filtering,
        },
        "model": {
            "type": "multi-output RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": 15,
            "random_state": RANDOM_STATE,
            "training_seconds": training_seconds,
        },
        "leakage_check": {
            "target_used_as_feature": False,
            "fare_components_used_as_features": False,
            "fare_components_used_as_auxiliary_targets": True,
        },
    }
    destination = save_result(AUTHOR, payload)
    print(f"metrics: {metrics}")
    print(f"saved: {destination}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    args = parser.parse_args()
    run_experiment(args.max_train_rows)


if __name__ == "__main__":
    main()
