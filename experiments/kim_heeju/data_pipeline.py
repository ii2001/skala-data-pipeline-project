"""NYC Taxi 데이터를 내려받고 Pandas·Polars로 동일하게 정제한다."""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import polars as pl

from config import (
    DATA_END_DATETIME,
    DATA_PATH,
    DATA_START_DATETIME,
    DATA_URL,
    DROPOFF_COLUMN,
    FARE_COMPONENT_COLUMNS,
    FARE_UPPER_LIMITS,
    LOAD_COLUMNS,
    MAX_DURATION_MINUTES,
    MAX_TRIP_DISTANCE_MILES,
    MIN_DURATION_MINUTES,
    PICKUP_COLUMN,
    SOURCE_COLUMNS,
    TARGET_COLUMN,
)


def ensure_directories(data_path: Path = DATA_PATH) -> None:
    """실행에 필요한 데이터·산출물 디렉터리를 생성한다."""
    data_path.parent.mkdir(parents=True, exist_ok=True)


def download_data(url: str = DATA_URL, destination: Path = DATA_PATH) -> Path:
    """원본 Parquet를 캐시하며, 실패 시 원인이 포함된 오류를 발생시킨다."""
    ensure_directories(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    temporary = destination.with_suffix(".download")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        temporary.replace(destination)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"NYC Taxi 데이터 다운로드에 실패했습니다: {error}") from error

    return destination


def validate_schema(columns: list[str]) -> None:
    """분석에 필요한 컬럼이 원본 파일에 모두 있는지 확인한다."""
    missing = sorted(set(LOAD_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"원본 데이터에 필요한 컬럼이 없습니다: {missing}")


def load_with_pandas(path: Path) -> pd.DataFrame:
    """필요한 열만 Pandas DataFrame으로 읽는다."""
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
    try:
        frame = pd.read_parquet(path, columns=LOAD_COLUMNS)
    except Exception as error:
        raise RuntimeError(f"Pandas로 Parquet를 읽지 못했습니다: {error}") from error
    validate_schema(frame.columns.tolist())
    return frame


def load_with_polars(path: Path) -> pl.DataFrame:
    """필요한 열만 Polars DataFrame으로 읽는다."""
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
    try:
        frame = pl.read_parquet(path, columns=LOAD_COLUMNS)
    except Exception as error:
        raise RuntimeError(f"Polars로 Parquet를 읽지 못했습니다: {error}") from error
    validate_schema(frame.columns)
    return frame


def summarize_pandas(frame: pd.DataFrame) -> dict:
    """Pandas 원본의 크기, 자료형, 결측치와 중복 수를 요약한다."""
    visible = frame[SOURCE_COLUMNS]
    return {
        "shape": visible.shape,
        "dtypes": {column: str(dtype) for column, dtype in visible.dtypes.items()},
        "missing": {key: int(value) for key, value in visible.isna().sum().items()},
        "duplicates": int(visible.duplicated().sum()),
    }


def summarize_polars(frame: pl.DataFrame) -> dict:
    """Polars 원본의 크기, 자료형, 결측치와 중복 수를 요약한다."""
    visible = frame.select(SOURCE_COLUMNS)
    return {
        "shape": visible.shape,
        "dtypes": dict(zip(visible.columns, map(str, visible.dtypes), strict=True)),
        "missing": visible.null_count().row(0, named=True),
        "duplicates": visible.height - visible.unique().height,
    }


def summarize_fare_exclusions(frame: pd.DataFrame) -> dict[str, int]:
    """음수 요금과 취소 운행의 제외 대상 수를 중복 여부와 함께 계산한다."""
    negative_mask = frame[FARE_COMPONENT_COLUMNS].lt(0).any(axis=1)
    voided_mask = frame["payment_type"].eq(6)
    upper_limit_mask = pd.Series(False, index=frame.index)
    for column, upper_limit in FARE_UPPER_LIMITS.items():
        upper_limit_mask |= frame[column].gt(upper_limit)
    outside_month_mask = ~frame[PICKUP_COLUMN].between(
        DATA_START_DATETIME, DATA_END_DATETIME, inclusive="left"
    )
    return {
        "outside_month_rows": int(outside_month_mask.sum()),
        "negative_fare_rows": int(negative_mask.sum()),
        "voided_trip_rows": int(voided_mask.sum()),
        "overlap_rows": int((negative_mask & voided_mask).sum()),
        "upper_limit_rows": int(upper_limit_mask.sum()),
        "unique_excluded_rows": int((negative_mask | voided_mask | upper_limit_mask).sum()),
    }


def summarize_total_consistency(frame: pd.DataFrame) -> dict[str, float | int]:
    """결측치가 없는 행에서 9개 요금 합계와 total_amount의 일치 정도를 계산한다."""
    complete = frame.dropna(subset=[*FARE_COMPONENT_COLUMNS, "total_amount"])
    component_sum = complete[FARE_COMPONENT_COLUMNS].sum(axis=1)
    absolute_difference = (complete["total_amount"] - component_sum).abs()
    return {
        "rows": len(complete),
        "mean_absolute_difference": float(absolute_difference.mean()),
        "within_one_cent_ratio": float(absolute_difference.le(0.01).mean()),
    }


def clean_pandas(frame: pd.DataFrame) -> pd.DataFrame:
    """시간 파생변수를 만들고 비정상 운행 및 중복 행을 제거한다."""
    cleaned = frame.copy()
    cleaned[PICKUP_COLUMN] = pd.to_datetime(cleaned[PICKUP_COLUMN], errors="coerce")
    cleaned[DROPOFF_COLUMN] = pd.to_datetime(cleaned[DROPOFF_COLUMN], errors="coerce")
    cleaned[TARGET_COLUMN] = (
        cleaned[DROPOFF_COLUMN] - cleaned[PICKUP_COLUMN]
    ).dt.total_seconds() / 60
    cleaned["store_and_fwd_flag"] = cleaned["store_and_fwd_flag"].astype("string").str.strip()
    cleaned = cleaned.drop_duplicates(subset=SOURCE_COLUMNS)
    within_data_month = cleaned[PICKUP_COLUMN].between(
        DATA_START_DATETIME, DATA_END_DATETIME, inclusive="left"
    )
    nonnegative_fares = (
        cleaned[FARE_COMPONENT_COLUMNS].ge(0) | cleaned[FARE_COMPONENT_COLUMNS].isna()
    ).all(axis=1)
    within_fare_limits = pd.Series(True, index=cleaned.index)
    for column, upper_limit in FARE_UPPER_LIMITS.items():
        within_fare_limits &= cleaned[column].le(upper_limit) | cleaned[column].isna()
    not_voided = cleaned["payment_type"].ne(6) | cleaned["payment_type"].isna()
    cleaned = cleaned[
        cleaned[TARGET_COLUMN].between(MIN_DURATION_MINUTES, MAX_DURATION_MINUTES)
        & within_data_month
        & cleaned["trip_distance"].between(0.01, MAX_TRIP_DISTANCE_MILES)
        & nonnegative_fares
        & within_fare_limits
        & not_voided
    ].copy()
    cleaned["pickup_hour"] = cleaned[PICKUP_COLUMN].dt.hour
    cleaned["pickup_weekday"] = cleaned[PICKUP_COLUMN].dt.dayofweek
    cleaned["is_weekend"] = (cleaned["pickup_weekday"] >= 5).astype("int8")
    return cleaned.reset_index(drop=True)


def clean_polars(frame: pl.DataFrame) -> pl.DataFrame:
    """Pandas와 같은 조건으로 Polars 데이터를 정제한다."""
    return (
        frame.with_columns(
            (
                (pl.col(DROPOFF_COLUMN) - pl.col(PICKUP_COLUMN)).dt.total_seconds() / 60
            ).alias(TARGET_COLUMN),
            pl.col("store_and_fwd_flag").cast(pl.String).str.strip_chars(),
        )
        .unique(subset=SOURCE_COLUMNS, maintain_order=True)
        .filter(
            pl.col(TARGET_COLUMN).is_between(
                MIN_DURATION_MINUTES, MAX_DURATION_MINUTES, closed="both"
            )
            & pl.col(PICKUP_COLUMN).is_between(
                DATA_START_DATETIME, DATA_END_DATETIME, closed="left"
            )
            & pl.col("trip_distance").is_between(
                0.01, MAX_TRIP_DISTANCE_MILES, closed="both"
            )
            & pl.all_horizontal(
                [
                    (pl.col(column) >= 0) | pl.col(column).is_null()
                    for column in FARE_COMPONENT_COLUMNS
                ]
            )
            & pl.all_horizontal(
                [
                    (pl.col(column) <= upper_limit) | pl.col(column).is_null()
                    for column, upper_limit in FARE_UPPER_LIMITS.items()
                ]
            )
            & ((pl.col("payment_type") != 6) | pl.col("payment_type").is_null())
        )
        .with_columns(
            pl.col(PICKUP_COLUMN).dt.hour().alias("pickup_hour"),
            pl.col(PICKUP_COLUMN).dt.weekday().sub(1).alias("pickup_weekday"),
        )
        .with_columns((pl.col("pickup_weekday") >= 5).cast(pl.Int8).alias("is_weekend"))
    )


def validate_cleaned_frames(pandas_frame: pd.DataFrame, polars_frame: pl.DataFrame) -> None:
    """두 도구의 핵심 정제 결과가 같은지 검증한다."""
    if len(pandas_frame) != polars_frame.height:
        raise ValueError(
            "Pandas와 Polars의 정제 후 행 수가 다릅니다: "
            f"{len(pandas_frame):,} != {polars_frame.height:,}"
        )
    for column in [TARGET_COLUMN, "pickup_hour", "pickup_weekday", "is_weekend"]:
        pandas_missing = int(pandas_frame[column].isna().sum())
        polars_missing = polars_frame[column].null_count()
        if pandas_missing != polars_missing:
            raise ValueError(f"{column} 결측치 수가 두 도구에서 다릅니다.")
