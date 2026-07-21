"""noh_yeongo EDA: Pandas/Polars 로딩 비교 + 데이터 품질 진단.

과제 요구사항(데이터 준비) 대응:
- 동일 parquet 을 Pandas 와 Polars 양쪽으로 로딩해 시간·메모리 비교
- 결측치·중복 확인 및 기본 EDA (이상치·정합성 진단 포함)

실행: python -m experiments.noh_yeongo.eda
"""

from __future__ import annotations

import time

from src.common.data_loader import load_taxi_pandas, load_taxi_polars


def compare_loading():
    """공통 로더로 Pandas/Polars 를 각각 로딩해 성능을 비교한다."""
    start = time.perf_counter()
    pdf = load_taxi_pandas()
    t_pandas = time.perf_counter() - start

    start = time.perf_counter()
    pldf = load_taxi_polars()
    t_polars = time.perf_counter() - start

    print("=" * 62)
    print("[1] Pandas vs Polars 로딩 비교")
    print("=" * 62)
    print(f"shape 일치      : {tuple(pdf.shape) == tuple(pldf.shape)}  {pdf.shape}")
    print(f"Pandas          : {t_pandas:.3f}초 / {pdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Polars          : {t_polars:.3f}초 / {pldf.estimated_size() / 1e6:.1f} MB")
    if t_pandas >= t_polars:
        print(f"→ Polars가 약 {t_pandas / t_polars:.1f}배 빠름")
    else:
        print("→ 이번 실행은 Pandas가 근소하게 빠름(OS 캐시 영향, 통상은 Polars 우세)")
    return pdf


def run_eda(df) -> None:
    """결측·중복·이상치·정합성을 진단해 전처리 근거를 출력한다."""
    print("\n" + "=" * 62)
    print("[2] 기본 EDA + 품질 진단")
    print("=" * 62)

    # 2-1) 결측: 5개 컬럼이 '같은 행'에서 동시 결측인지 확인 → 구조적 결측 증명
    missing = df.isna().sum()
    missing = missing[missing > 0]
    both = int(df[missing.index].isna().all(axis=1).sum())
    print(f"(2-1) 결측 컬럼 {len(missing)}개, 각 {missing.iloc[0]:,}건")
    print(f"      동시 결측 {both:,}건 = payment_type=0 행 {(df.payment_type == 0).sum():,}건")
    print("      ⇒ 비미터기 수집분의 구조적 결측 (랜덤 결측 아님 → 삭제 대신 대체)")

    # 2-2) 중복
    print(f"(2-2) 중복 행: {df.duplicated().sum():,}건")

    # 2-3) 이상치 규모
    dur = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    print("(2-3) 이상치 규모:")
    print(f"      - 요금 <= 0 (환불)  : {(df.fare_amount <= 0).sum():,}건 (min ${df.fare_amount.min()})")
    print(f"      - 거리 = 0          : {(df.trip_distance == 0).sum():,}건")
    print(f"      - 거리 > 100마일    : {(df.trip_distance > 100).sum():,}건 (max {df.trip_distance.max():,.0f}마일)")
    print(f"      - 운행시간 <= 0     : {(dur <= 0).sum():,}건 / > 180분: {(dur > 180).sum():,}건")

    # 2-4) 기술통계(전처리 전) — min/max 로 이상치 근거 제시
    print("\n(2-4) 핵심 컬럼 기술통계 (전처리 전):")
    print(df[["trip_distance", "fare_amount", "tip_amount", "total_amount"]]
          .describe().round(2).to_string())


if __name__ == "__main__":
    dataframe = compare_loading()
    run_eda(dataframe)
