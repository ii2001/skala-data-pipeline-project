"""noh_yeongo 실험 전처리: 피처 생성과 학습용 품질 필터.

원칙
- 피처는 승차 시점에 알 수 있는 정보만 사용한다(요금 구성요소·운행시간 제외 = 누수 방지).
- 공항 여부는 요금 컬럼(Airport_fee)이 아니라 승·하차 구역 ID로 판정한다.
  (Airport_fee 는 타깃(total_amount)의 구성요소라서 피처 재료로 쓰면 누수)
- 품질 필터는 '학습 데이터에만' 적용한다. 공통 테스트 행은 리더보드 비교를 위해
  전부 유지하고 예측한다.
"""

from __future__ import annotations

import pandas as pd

# 승차 시점 피처 정의
NUM_FEATURES = ["trip_distance", "passenger_count"]
CAT_FEATURES = ["pickup_hour", "pickup_weekday", "RatecodeID"]
BIN_FEATURES = ["is_airport", "is_weekend"]
FEATURES = NUM_FEATURES + CAT_FEATURES + BIN_FEATURES

# 공항 구역 ID: 1=Newark(EWR), 132=JFK, 138=LaGuardia
AIRPORT_ZONES = {1, 132, 138}

# 학습용 품질 필터 기준(원본 EDA에서 확인한 이상치 경계)
MAX_DISTANCE_MILES = 100     # 거리 0 초과 ~ 100마일 이하 (원본 max 307,491마일 오류값 존재)
MAX_DURATION_MIN = 180       # 운행시간 0 초과 ~ 3시간 이하 (음수 5.2만 건 존재)


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """결측 대체와 승차 시점 파생 피처를 추가한다(행 삭제 없음)."""
    df = dataframe.copy()

    # 구조적 결측(비미터기 수집분 95.5만 행) 대체 — 의미 기반 값 사용
    df["passenger_count"] = df["passenger_count"].fillna(1).astype("int8")   # 최빈값 1명
    df["RatecodeID"] = df["RatecodeID"].fillna(99).astype("int8")            # 99 = 미상 코드

    # 거리 피처 클리핑: 미터기 오류(최대 307,491마일)가 테스트에 남아 있어도
    # 예측이 폭주하지 않도록 고정 상한 적용. 학습·예측에 동일하게 적용하므로 누수 아님.
    df["trip_distance"] = df["trip_distance"].clip(0, MAX_DISTANCE_MILES)

    # 승차 시각 기반 파생
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype("int8")
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["pickup_weekday"] >= 5).astype("int8")

    # 공항 여부: 구역 ID 기반(타깃과 무관 → 누수 없음, 결측 행에서도 계산 가능)
    df["is_airport"] = (
        df["PULocationID"].isin(AIRPORT_ZONES) | df["DOLocationID"].isin(AIRPORT_ZONES)
    ).astype("int8")

    return df


def filter_train_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """학습 데이터에서 명백한 오류 행을 제외한다(테스트에는 적용하지 않음)."""
    duration_min = (
        dataframe["tpep_dropoff_datetime"] - dataframe["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    mask = (
        (dataframe["fare_amount"] > 0)                                  # 환불(음수) 제외
        & (dataframe["trip_distance"] > 0)
        & (dataframe["trip_distance"] <= MAX_DISTANCE_MILES)            # 거리 오류 제외
        & (duration_min > 0)
        & (duration_min <= MAX_DURATION_MIN)                            # 시간 오류 제외
    )
    return dataframe.loc[mask]


def prepare_analysis_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """시각화·통계 분석용 정제 프레임을 만든다.

    학습 계약(train.py)과는 별개로, 전체 데이터에서 오류 행을 제외하고
    분석에 필요한 파생 컬럼(운행시간·세금할증 합계 등)을 추가한다.
    """
    df = add_features(dataframe)
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # 요금 분석용: 결측 할증(비미터기 수집분)은 0으로 두고 세금·할증 합계 산출
    df["surcharge_total"] = (
        df["extra"] + df["mta_tax"] + df["improvement_surcharge"]
        + df["congestion_surcharge"].fillna(0) + df["Airport_fee"].fillna(0)
        + df["cbd_congestion_fee"]
    )

    mask = (
        (df["total_amount"] > 0) & (df["fare_amount"] > 0)              # 환불 제외
        & (df["trip_distance"] > 0) & (df["trip_distance"] <= MAX_DISTANCE_MILES)
        & (df["duration_min"] > 0) & (df["duration_min"] <= MAX_DURATION_MIN)
    )
    return df.loc[mask].reset_index(drop=True)
