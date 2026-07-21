"""NYC Yellow Taxi 원본 데이터를 내려받고 로드한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TAXI_FILENAME = "yellow_tripdata_2026-05.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"
TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2026-05.parquet"
)


def verify_taxi_data(path: Path) -> None:
    """manifest의 크기와 SHA-256으로 원본 버전을 검증한다."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if path.stat().st_size != manifest["bytes"]:
        raise ValueError(f"원본 데이터 크기가 manifest와 다릅니다: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError(f"원본 데이터 SHA-256이 manifest와 다릅니다: {path}")


def download_taxi_data(refresh: bool = False) -> Path:
    """원본 Parquet을 캐시하고 로컬 경로를 반환한다."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DATA_DIR / TAXI_FILENAME
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        verify_taxi_data(destination)
        return destination

    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with urlopen(TAXI_URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify_taxi_data(temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_taxi_pandas(refresh: bool = False) -> pd.DataFrame:
    """원본을 Pandas DataFrame으로 로드하고 안정적인 행 ID를 추가한다."""
    dataframe = pd.read_parquet(download_taxi_data(refresh))
    dataframe.insert(0, "source_row_id", dataframe.index.to_numpy())
    return dataframe


def load_taxi_polars(refresh: bool = False) -> pl.DataFrame:
    """원본을 Polars DataFrame으로 로드하고 안정적인 행 ID를 추가한다."""
    return pl.read_parquet(download_taxi_data(refresh)).with_row_index("source_row_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC Yellow Taxi 데이터 확인")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    dataframe = load_taxi_pandas(args.refresh)
    print(f"shape: {dataframe.shape}")
    dataframe.info(show_counts=True)


if __name__ == "__main__":
    main()
