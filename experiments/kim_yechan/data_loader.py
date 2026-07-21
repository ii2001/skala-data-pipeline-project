"""NYC Yellow Taxi 2026년 5월 데이터를 다운로드하고 기본 정보를 출력한다."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TAXI_FILENAME = "yellow_tripdata_2026-05.parquet"
TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2026-05.parquet"
)


def download_taxi_data(refresh: bool = False) -> Path:
    """원본 Parquet을 캐시하고 로컬 경로를 반환한다."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DATA_DIR / TAXI_FILENAME

    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        print(f"캐시 파일 사용: {destination}")
        return destination

    temporary = destination.with_suffix(f"{destination.suffix}.part")
    print(f"데이터 다운로드: {TAXI_URL}")

    try:
        with urlopen(TAXI_URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def load_taxi_pandas(refresh: bool = False) -> pd.DataFrame:
    """NYC Taxi 원본을 Pandas DataFrame으로 로드한다."""
    source = download_taxi_data(refresh)
    return pd.read_parquet(source)


def load_taxi_polars(refresh: bool = False) -> pl.DataFrame:
    """NYC Taxi 원본을 Polars DataFrame으로 로드한다."""
    source = download_taxi_data(refresh)
    return pl.read_parquet(source)


def load_taxi_data(refresh: bool = False) -> pd.DataFrame:
    """기존 실행 방식과 호환되는 Pandas 로더다."""
    return load_taxi_pandas(refresh)


def show_data_info(refresh: bool = False) -> None:
    """데이터 크기와 컬럼별 타입·결측 개수를 출력한다."""
    dataframe = load_taxi_data(refresh)
    print(f"shape: {dataframe.shape}")
    dataframe.info(show_counts=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Yellow Taxi 데이터 로드")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="캐시가 있어도 원본 데이터를 다시 다운로드한다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_data_info(args.refresh)


if __name__ == "__main__":
    main()
