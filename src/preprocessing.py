"""NYC Taxi 모델링 데이터의 품질 규칙과 피처 엔지니어링을 관리한다."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "high_fare"
HIGH_FARE_THRESHOLD = 30.0
HOLDOUT_START = pd.Timestamp("2026-05-25")

NUMERIC_FEATURES = [
    "passenger_count",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_dayofweek_sin",
    "pickup_dayofweek_cos",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["VendorID", "PULocationID", "DOLocationID"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = {
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
}


def validate_columns(dataframe: pd.DataFrame) -> None:
    """전처리에 필요한 컬럼이 모두 존재하는지 확인한다."""
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"전처리 필수 컬럼이 없습니다: {sorted(missing)}")


def prepare_modeling_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    """단계별 품질 규칙을 적용하고 전처리 감사 결과를 반환한다."""
    validate_columns(dataframe)
    audit_rows: list[dict[str, int | str]] = []

    duplicate_mask = dataframe.duplicated()
    selected_columns = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]
    working = dataframe.loc[~duplicate_mask, selected_columns].copy()
    audit_rows.append(
        {
            "step": "Duplicate removal",
            "rule": "전체 컬럼이 동일한 행 제거",
            "before_rows": len(dataframe),
            "removed_rows": int(duplicate_mask.sum()),
            "after_rows": len(working),
        }
    )

    def apply_filter(step: str, rule: str, mask: pd.Series) -> None:
        nonlocal working
        before = len(working)
        working = working.loc[mask].copy()
        audit_rows.append(
            {
                "step": step,
                "rule": rule,
                "before_rows": before,
                "removed_rows": before - len(working),
                "after_rows": len(working),
            }
        )

    apply_filter(
        "Analysis period",
        "pickup이 2026-05-01 이상, 2026-06-01 미만",
        working["tpep_pickup_datetime"].between(
            "2026-05-01", "2026-06-01", inclusive="left"
        ),
    )

    working["duration_minutes"] = (
        working["tpep_dropoff_datetime"] - working["tpep_pickup_datetime"]
    ).dt.total_seconds().div(60)
    apply_filter(
        "Trip duration",
        "1분 이상 180분 이하",
        working["duration_minutes"].between(1, 180, inclusive="both"),
    )
    apply_filter(
        "Trip distance",
        "0.1마일 이상 100마일 이하",
        working["trip_distance"].between(0.1, 100, inclusive="both"),
    )

    working["average_speed_mph"] = working["trip_distance"].div(
        working["duration_minutes"].div(60)
    )
    apply_filter(
        "Average speed",
        "계산 평균속도가 80mph 이하",
        working["average_speed_mph"] <= 80,
    )
    apply_filter(
        "Total amount",
        "total_amount가 0보다 큼",
        working["total_amount"] > 0,
    )
    apply_filter(
        "Fare amount",
        "fare_amount가 0 이상",
        working["fare_amount"] >= 0,
    )
    apply_filter(
        "Location code",
        "승하차 LocationID가 1~265",
        working["PULocationID"].between(1, 265, inclusive="both")
        & working["DOLocationID"].between(1, 265, inclusive="both"),
    )

    invalid_passenger = working["passenger_count"].notna() & ~working[
        "passenger_count"
    ].between(1, 6, inclusive="both")
    original_missing_passenger = int(working["passenger_count"].isna().sum())
    working.loc[invalid_passenger, "passenger_count"] = pd.NA

    pickup_hour = working["tpep_pickup_datetime"].dt.hour
    pickup_dayofweek = working["tpep_pickup_datetime"].dt.dayofweek
    working["pickup_hour_sin"] = np.sin(2 * np.pi * pickup_hour / 24)
    working["pickup_hour_cos"] = np.cos(2 * np.pi * pickup_hour / 24)
    working["pickup_dayofweek_sin"] = np.sin(2 * np.pi * pickup_dayofweek / 7)
    working["pickup_dayofweek_cos"] = np.cos(2 * np.pi * pickup_dayofweek / 7)
    working["is_weekend"] = pickup_dayofweek.isin([5, 6]).astype("int8")
    working["pickup_period"] = pd.cut(
        pickup_hour,
        bins=[-1, 5, 9, 15, 19, 23],
        labels=["night", "morning_peak", "daytime", "evening_peak", "late_evening"],
    ).astype("string")
    working["route_id"] = (
        working["PULocationID"].astype("string")
        + "_"
        + working["DOLocationID"].astype("string")
    )
    working["route_period_id"] = working["route_id"] + "_" + working["pickup_period"]
    working[TARGET_COLUMN] = (
        working["total_amount"] >= HIGH_FARE_THRESHOLD
    ).astype("int8")

    audit = pd.DataFrame(audit_rows)
    summary: dict[str, int | float | str] = {
        "raw_rows": len(dataframe),
        "valid_rows": len(working),
        "total_removed_rows": len(dataframe) - len(working),
        "retention_rate": len(working) / len(dataframe),
        "original_missing_passenger_rows": original_missing_passenger,
        "invalid_passenger_rows_reclassified": int(invalid_passenger.sum()),
        "passenger_rows_for_imputation": int(working["passenger_count"].isna().sum()),
        "positive_label_rows": int(working[TARGET_COLUMN].sum()),
        "positive_label_rate": float(working[TARGET_COLUMN].mean()),
        "holdout_start": HOLDOUT_START.strftime("%Y-%m-%d"),
    }
    return working, audit, summary
