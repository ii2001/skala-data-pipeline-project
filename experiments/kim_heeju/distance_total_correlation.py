"""운행 거리와 최종 총요금의 Pearson 상관관계를 확인한다."""

import pandas as pd

from config import DATA_PATH


def main() -> None:
    """필요한 두 컬럼만 읽어 결측치를 제거하고 상관계수를 출력한다."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")

    data = pd.read_parquet(
        DATA_PATH,
        columns=["trip_distance", "total_amount"],
    ).dropna()

    correlation = data["trip_distance"].corr(data["total_amount"])

    print("[trip_distance와 total_amount의 상관관계]")
    print(f"사용 데이터: {len(data):,}건")
    print(f"Pearson 상관계수: {correlation:.4f}")


if __name__ == "__main__":
    main()
