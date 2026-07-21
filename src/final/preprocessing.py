"""최종 모델 입력용 NYC Yellow Taxi 데이터를 전처리한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "yellow_tripdata_2026-05_processed.parquet"
)
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_summary.json"

FARE_COMPONENT_COLUMNS = (
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
)
ONE_HOT_COLUMNS = ("RatecodeID", "PULocationID", "DOLocationID", "payment_type")
REQUIRED_COLUMNS = {
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "total_amount",
    *FARE_COMPONENT_COLUMNS,
}
MAX_TRIP_DISTANCE_MILES = 100.0
MIN_TRIP_DURATION_MINUTES = 1.0
MAX_TRIP_DURATION_MINUTES = 180.0
UNKNOWN_RATE_CODE = 99
ANALYSIS_START = pl.datetime(2026, 5, 1)
ANALYSIS_END = pl.datetime(2026, 6, 1)


def validate_columns(dataframe: pl.DataFrame) -> None:
    """필수 원본 컬럼이 모두 있는지 확인한다."""
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"전처리 필수 컬럼이 없습니다: {sorted(missing)}")


def preprocess_taxi_data(dataframe: pl.DataFrame) -> pl.DataFrame:
    """누수 요금 컬럼을 제거하고 모델 입력 피처를 생성한다."""
    validate_columns(dataframe)

    processed = (
        dataframe.with_row_index("source_row_id")
        .filter(
            pl.col("tpep_pickup_datetime").is_between(
                ANALYSIS_START, ANALYSIS_END, closed="left"
            )
            & (pl.col("total_amount") > 0)
        )
        .with_columns(
            pl.col("passenger_count").fill_null(0).cast(pl.Int64),
            pl.col("RatecodeID").fill_null(UNKNOWN_RATE_CODE).cast(pl.Int64),
            (
                (
                    pl.col("tpep_dropoff_datetime")
                    - pl.col("tpep_pickup_datetime")
                ).dt.total_seconds()
                / 60
            ).alias("trip_duration_minutes"),
            (pl.col("tpep_pickup_datetime").dt.weekday() - 1)
            .cast(pl.Int8)
            .alias("pickup_day_of_week"),
            (pl.col("trip_distance") > MAX_TRIP_DISTANCE_MILES)
            .cast(pl.Int8)
            .alias("trip_distance_was_capped"),
            (pl.col("trip_distance") == 0)
            .cast(pl.Int8)
            .alias("trip_distance_is_zero"),
            pl.col("trip_distance").clip(0, MAX_TRIP_DISTANCE_MILES),
            (pl.col("store_and_fwd_flag") == "Y")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("store_and_fwd_flag_Y"),
        )
        .with_columns(
            (
                (pl.col("trip_duration_minutes") < MIN_TRIP_DURATION_MINUTES)
                | (pl.col("trip_duration_minutes") > MAX_TRIP_DURATION_MINUTES)
            )
            .cast(pl.Int8)
            .alias("trip_duration_was_adjusted"),
            pl.col("trip_duration_minutes").clip(
                MIN_TRIP_DURATION_MINUTES, MAX_TRIP_DURATION_MINUTES
            ),
        )
        .drop(
            "VendorID",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "store_and_fwd_flag",
            *FARE_COMPONENT_COLUMNS,
        )
    )

    # 요금제·위치·결제 코드는 수치의 크기에 의미가 없는 명목형 변수다.
    return processed.to_dummies(columns=ONE_HOT_COLUMNS, separator="_")


def build_summary(raw: pl.DataFrame, processed: pl.DataFrame) -> dict[str, Any]:
    """전처리 결과 검증에 필요한 실제 수치를 반환한다."""
    distance = raw.get_column("trip_distance")
    eligible = raw.filter(
        pl.col("tpep_pickup_datetime").is_between(
            ANALYSIS_START, ANALYSIS_END, closed="left"
        )
        & (pl.col("total_amount") > 0)
    )
    eligible_duration = (
        eligible.get_column("tpep_dropoff_datetime")
        - eligible.get_column("tpep_pickup_datetime")
    ).dt.total_seconds() / 60
    valid_distance_rows = eligible.filter(
        (pl.col("total_amount") > 0)
        & (pl.col("trip_distance") > 0)
        & (pl.col("trip_distance") <= MAX_TRIP_DISTANCE_MILES)
    )
    correlation = raw.select(
        pl.corr("total_amount", "trip_distance").alias("raw"),
        pl.corr(
            "total_amount",
            pl.col("trip_distance").clip(0, MAX_TRIP_DISTANCE_MILES),
        ).alias("capped"),
    ).row(0, named=True)
    valid_correlation = valid_distance_rows.select(
        pl.corr("total_amount", "trip_distance").alias("pearson"),
        pl.corr(
            "total_amount", "trip_distance", method="spearman"
        ).alias("spearman"),
    ).row(0, named=True)
    return {
        "input_path": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
        "output_path": str(PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT)),
        "input_rows": raw.height,
        "input_columns": raw.width,
        "output_rows": processed.height,
        "output_columns": processed.width,
        "target": "total_amount",
        "model_excluded_columns": ["source_row_id", "total_amount"],
        "excluded_rows": raw.height - processed.height,
        "pickup_outside_analysis_period_rows": int(
            (~raw.get_column("tpep_pickup_datetime").is_between(
                ANALYSIS_START, ANALYSIS_END, closed="left"
            )).sum()
        ),
        "nonpositive_target_rows": int(
            (raw.get_column("total_amount") <= 0).sum()
        ),
        "passenger_count_nulls_filled": eligible.get_column(
            "passenger_count"
        ).null_count(),
        "ratecode_nulls_filled_with_99": eligible.get_column(
            "RatecodeID"
        ).null_count(),
        "duration_below_minimum_rows": int(
            (eligible_duration < MIN_TRIP_DURATION_MINUTES).sum()
        ),
        "duration_above_maximum_rows": int(
            (eligible_duration > MAX_TRIP_DURATION_MINUTES).sum()
        ),
        "duration_minimum_minutes": MIN_TRIP_DURATION_MINUTES,
        "duration_maximum_minutes": MAX_TRIP_DURATION_MINUTES,
        "trip_distance_zero_rows": int(
            (eligible.get_column("trip_distance") == 0).sum()
        ),
        "trip_distance_capped_rows": int(
            (eligible.get_column("trip_distance") > MAX_TRIP_DISTANCE_MILES).sum()
        ),
        "trip_distance_cap_miles": MAX_TRIP_DISTANCE_MILES,
        "trip_distance_max_before_processing": float(distance.max()),
        "trip_distance_total_amount_correlation": {
            "pearson_raw_all_rows": correlation["raw"],
            "pearson_capped_all_rows": correlation["capped"],
            "valid_rows": valid_distance_rows.height,
            "valid_rule": "total_amount > 0 and 0 < trip_distance <= 100",
            "pearson_valid_rows": valid_correlation["pearson"],
            "spearman_valid_rows": valid_correlation["spearman"],
        },
        "pickup_day_of_week_mapping": {
            "0": "Monday",
            "1": "Tuesday",
            "2": "Wednesday",
            "3": "Thursday",
            "4": "Friday",
            "5": "Saturday",
            "6": "Sunday",
        },
        "store_and_fwd_encoding": {
            "column": "store_and_fwd_flag_Y",
            "Y": 1,
            "N_or_unknown": 0,
            "unknown_is_represented_by": "payment_type_0",
        },
        "removed_columns": [
            "VendorID",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "store_and_fwd_flag",
            *FARE_COMPONENT_COLUMNS,
        ],
    }


def main() -> None:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_DATA_PATH}")

    raw = pl.read_parquet(RAW_DATA_PATH)
    processed = preprocess_taxi_data(raw)
    summary = build_summary(raw, processed)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.write_parquet(PROCESSED_DATA_PATH, compression="zstd")
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"전처리 데이터: {PROCESSED_DATA_PATH}")
    print(f"shape: {processed.shape}")
    print(f"전처리 요약: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
