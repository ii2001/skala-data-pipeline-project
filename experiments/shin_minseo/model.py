"""
================================================================================
NYC Yellow Taxi — total_amount 구성요소 9종 다중출력(Multi-output) 회귀 모델
--------------------------------------------------------------------------------
[프로그램 전체 설명 / 머리말]
  · 목적 : 뉴욕 옐로우택시 주행 정보(거리·시간·요금제 등)만으로 요금 청구서를
           구성하는 9개 항목을 한 번에 예측한다.
  · 정답레이블(9개) : total_amount = 아래 9개 컬럼의 '합'
        fare_amount + extra + mta_tax + tip_amount + tolls_amount
      + improvement_surcharge + congestion_surcharge + Airport_fee
      + cbd_congestion_fee
  · 핵심 주의(데이터 누수 방지) :
        total_amount 및 9개 타깃 컬럼은 '절대' 피처로 쓰지 않는다.
        (타깃의 합이 total_amount 이므로 넣으면 정답을 보고 푸는 셈)
  · 파이프라인 : sklearn.pipeline.Pipeline 으로 [전처리 → 모델] 을 하나로 묶고,
        RandomForestRegressor(다중출력 native 지원)로 9개를 동시에 학습한다.
  · 성능측정 : 5-fold 교차검증(각 타깃 R²·MAE·RMSE 의 평균±표준편차)으로 추정하고,
        배포용 모델은 전체 데이터로 다시 학습해 저장한다.
  · 산출물 : ① 타깃별 평가지표 표(R²·MAE·RMSE)  ② total_amount 복원 정확도
             ③ joblib 로 학습된 파이프라인(.joblib) 저장

[변경 내역]
  · v1.0  최초 작성 — 로딩/결측치·이상치 처리/피처엔지니어링/Pipeline/평가/저장
  · v1.1  단일 hold-out → 5-fold 교차검증으로 성능 측정 방식 변경(안정성 확보),
          최종 모델은 전체 데이터로 학습해 저장하도록 분리

[실행 방법]
  # 원본 데이터(URL 자동 다운로드, 20만건 샘플)
  python src/total_amount_model.py
  # 로컬 parquet 파일 지정 + 샘플 수 조절
  python src/total_amount_model.py --data ./data/yellow_tripdata_2026-05.parquet --sample 300000
================================================================================
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------------------------
# 0) 전역 설정
# ------------------------------------------------------------------------------
# 강의록 제공 데이터셋(NYC Yellow Taxi 2026-05). 로컬 파일이 없으면 여기서 받는다.
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
)

# 예측 대상 9종 (total_amount 를 이루는 청구 항목들)
TARGET_COLS = [
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

RANDOM_STATE = 42  # 재현성(누구나 같은 결과) 확보용 고정 시드

# 로깅: 각 단계를 눈으로 추적(파이프라인 어느 단계에서 실패했는지 즉시 파악)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("taxi")


# ------------------------------------------------------------------------------
# 1) 데이터 로딩
# ------------------------------------------------------------------------------
def load_data(source: str, sample_size: int | None) -> pd.DataFrame:
    """parquet(로컬 경로 또는 URL)을 읽어 DataFrame 으로 반환한다.

    - 대용량(수백만 건)이므로 sample_size 가 지정되면 무작위 샘플만 사용해
      학습 시간을 단축한다.
    - 네트워크/파일 오류는 예외로 잡아 원인을 명확히 알린다(예외 처리).
    """
    log.info("데이터 로딩 시작: %s", source)
    try:
        df = pd.read_parquet(source)
    except FileNotFoundError:
        raise SystemExit(f"[오류] 파일을 찾을 수 없습니다: {source}")
    except Exception as e:  # 네트워크 실패, pyarrow 미설치 등
        raise SystemExit(f"[오류] 데이터를 읽지 못했습니다 ({type(e).__name__}): {e}")

    log.info("원본 shape: %s", df.shape)

    if sample_size and len(df) > sample_size:
        # random_state 고정 → 매번 동일한 샘플(재현성)
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        log.info("샘플링 후 shape: %s (n=%d)", df.shape, sample_size)

    return df


def resolve_targets(df: pd.DataFrame) -> list[str]:
    """9개 타깃 컬럼명을 실제 데이터의 컬럼명과 대소문자 무시로 매칭한다.

    (연도별로 Airport_fee/airport_fee 처럼 표기가 달라 방어적으로 처리)
    데이터에 없는 타깃은 경고 후 제외해, 스크립트가 멈추지 않게 한다.
    """
    lower_map = {c.lower(): c for c in df.columns}
    resolved: list[str] = []
    for t in TARGET_COLS:
        if t.lower() in lower_map:
            resolved.append(lower_map[t.lower()])
        else:
            log.warning("타깃 컬럼 '%s' 이(가) 데이터에 없어 제외합니다.", t)
    if not resolved:
        raise SystemExit("[오류] 예측 대상(타깃) 컬럼을 하나도 찾지 못했습니다.")
    log.info("최종 타깃 %d개: %s", len(resolved), resolved)
    return resolved


# ------------------------------------------------------------------------------
# 2) 결측치 처리 + 이상치 정제 + 피처 엔지니어링
# ------------------------------------------------------------------------------
def clean_and_engineer(
    df: pd.DataFrame, targets: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """데이터 특성에 맞춘 결측치 처리·이상치 제거·파생 피처 생성.

    반환: (X 피처 DataFrame, y 타깃 DataFrame)
    """
    df = df.copy()

    # (a) 승하차 시각 → 주행 시간(분) 파생. 잘못된 시각은 결측(NaT)이 되어 이후 제거됨.
    pu = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    do = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df["trip_duration_min"] = (do - pu).dt.total_seconds() / 60.0
    df["pickup_hour"] = pu.dt.hour        # 0~23 시간대
    df["pickup_weekday"] = pu.dt.weekday  # 0(월)~6(일)

    # (b) 타깃(요금 항목) 결측치 → 0 으로 대체.
    #     Airport_fee·congestion_surcharge·cbd_congestion_fee 는 해당 상황이
    #     아니면 값이 비어 있는데, 이는 '요금 없음(0원)'을 의미하므로 0이 타당.
    for t in targets:
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0.0)

    # (c) 이상치·비정상 레코드 제거(도메인 규칙 기반).
    before = len(df)
    total = df[targets].sum(axis=1)  # 9개 합 = 이론상 total_amount
    mask = (
        df["trip_distance"].between(0.1, 100)        # 0.1~100 마일만 유효
        & df["trip_duration_min"].between(1, 180)    # 1분~3시간
        & (df["fare_amount"] >= 0)                   # 음수 기본요금 제거
        & (total > 0)                                # 합계가 양수인 정상 거래
        & pu.notna()                                 # 시각 파싱 실패행 제거
    )
    df = df.loc[mask].reset_index(drop=True)
    log.info("이상치 제거: %d → %d (%d건 제거)", before, len(df), before - len(df))

    # (d) 피처 선택. total_amount 와 9개 타깃은 '누수'라서 절대 포함하지 않는다.
    num_features = [
        "trip_distance",
        "trip_duration_min",
        "passenger_count",
        "pickup_hour",
        "pickup_weekday",
    ]
    cat_features = ["VendorID", "RatecodeID", "payment_type", "store_and_fwd_flag"]

    # 데이터에 실제 존재하는 컬럼만 사용(버전별 스키마 차이 방어)
    num_features = [c for c in num_features if c in df.columns]
    cat_features = [c for c in cat_features if c in df.columns]

    X = df[num_features + cat_features].copy()
    # 범주형은 object(결측=np.nan)로 통일한다.
    # nullable 'string'(pd.NA)은 일부 sklearn 버전의 결측 판정과 충돌하므로 피함.
    for c in cat_features:
        X[c] = X[c].astype(object).where(pd.notna(X[c]), np.nan)

    y = df[targets].copy()

    # [결측치 분석] 범주형 결측 현황을 실제로 확인해 로그에 남긴다.
    #   관찰: RatecodeID 와 store_and_fwd_flag 가 '동일 행'에서 함께 결측(약 22%).
    #        → 특정 벤더/집계경로가 두 항목을 아예 기록하지 않는다는 뜻으로,
    #          결측 자체가 하나의 신호(정보성 결측)이다.
    #   대체 전략 비교(most_frequent vs constant"MISSING")를 실측한 결과 평균 R²
    #   차이는 ±0.002 수준으로 무의미했고, 정작 목표인 total_amount 복원은
    #   most_frequent 가 근소 우위여서 기본값(most_frequent)을 채택했다.
    na_cat = {c: int(X[c].isna().sum()) for c in cat_features}
    log.info("범주형 결측 건수(전체 %d건 중): %s", len(X), na_cat)

    log.info("피처: 수치형 %s | 범주형 %s", num_features, cat_features)
    # 파이프라인 구성 단계에서 재사용하기 위해 속성으로 매달아 반환
    X.attrs["num_features"] = num_features
    X.attrs["cat_features"] = cat_features
    return X, y


# ------------------------------------------------------------------------------
# 3) 전처리 + 모델을 하나의 Pipeline 으로 구성
# ------------------------------------------------------------------------------
def build_pipeline(num_features: list[str], cat_features: list[str]) -> Pipeline:
    """ColumnTransformer(결측치 대체 + 인코딩) → RandomForest 를 Pipeline 으로 결합."""
    # 수치형: 중앙값으로 결측 대체(이상치에 강건)
    numeric = SimpleImputer(strategy="median")
    # 범주형: 최빈값 대체 후 One-Hot 인코딩(모르는 값이 나와도 무시).
    #  RatecodeID 는 ~95%가 1(표준요금)이라 결측을 최빈값으로 보는 것이 합리적.
    #  (constant"MISSING" 대안과 실측 비교했으나 성능 차이는 오차 범위 → 기본값 채택)
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric, num_features),
            ("cat", categorical, cat_features),
        ]
    )

    # RandomForestRegressor 는 다중출력(9개 동시 예측)을 native 로 지원한다.
    #  min_samples_leaf=50 : 전체 388만 행 학습 시 트리가 과도하게 깊어져
    #    모델이 수십 GB로 불어나는 것을 막는 규제(메모리 안전 + 과적합 완화).
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=20,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


# ------------------------------------------------------------------------------
# 4) 평가지표 출력
# ------------------------------------------------------------------------------
def evaluate_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """5-fold 교차검증으로 9개 타깃 성능을 측정한다.

    - 단일 분할(hold-out)은 우연히 쉬운/어려운 평가셋을 뽑을 위험이 있어,
      데이터를 5등분해 '각 조각을 한 번씩 평가셋으로' 돌려가며 5번 평가한다.
    - 매 fold 마다 전처리(결측치 대체 등)를 그 fold의 학습셋에서만 학습하므로
      데이터 누수가 없다(Pipeline 을 fold 안에서 새로 fit).
    - 지표별 평균±표준편차를 함께 출력해 성능의 '안정성'까지 보여준다.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    per = {t: {"R2": [], "MAE": [], "RMSE": []} for t in y.columns}
    total_r2: list[float] = []

    for i, (tr, te) in enumerate(kf.split(X), start=1):
        pipe = build_pipeline(num_features, cat_features)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pd.DataFrame(
            pipe.predict(X.iloc[te]), columns=y.columns, index=y.iloc[te].index
        )
        y_te = y.iloc[te]
        for t in y.columns:
            per[t]["R2"].append(r2_score(y_te[t], pred[t]))
            per[t]["MAE"].append(mean_absolute_error(y_te[t], pred[t]))
            per[t]["RMSE"].append(mean_squared_error(y_te[t], pred[t]) ** 0.5)
        # total_amount 복원: 예측한 9개를 더해 실제 합과 비교
        total_r2.append(r2_score(y_te.sum(axis=1), pred.sum(axis=1)))
        log.info("교차검증 fold %d/%d 완료", i, n_splits)

    rows = []
    for t in y.columns:
        rows.append(
            {
                "target": t,
                "R2_mean": np.mean(per[t]["R2"]),
                "R2_std": np.std(per[t]["R2"]),
                "MAE_mean": np.mean(per[t]["MAE"]),
                "RMSE_mean": np.mean(per[t]["RMSE"]),
            }
        )
    metrics = pd.DataFrame(rows).set_index("target")

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n" + "=" * 74)
    print(f" 5-fold 교차검증 평가지표 (단위: 달러, R²는 1에 가까울수록 좋음)")
    print("=" * 74)
    print(metrics.to_string())
    print("-" * 74)
    print(f" 타깃 평균 R² : {metrics['R2_mean'].mean():.4f}")
    print(
        f" total_amount 복원 R² : mean={np.mean(total_r2):.4f} "
        f"std={np.std(total_r2):.4f}"
    )
    print("=" * 74 + "\n")
    return metrics


# ------------------------------------------------------------------------------
# 5) 메인 파이프라인 (E-T-L 흐름: 로딩 → 정제 → 학습 → 평가 → 저장)
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="NYC Taxi total_amount 9종 다중출력 회귀")
    parser.add_argument(
        "--data", default=DATA_URL, help="parquet 경로 또는 URL (기본: 강의록 제공 URL)"
    )
    parser.add_argument(
        "--sample", type=int, default=200_000, help="학습에 사용할 샘플 수(0이면 전체)"
    )
    parser.add_argument(
        "--out", default="taxi_total_amount_model.joblib", help="모델 저장 경로"
    )
    parser.add_argument("--folds", type=int, default=5, help="교차검증 fold 수(기본 5)")
    args = parser.parse_args()

    # 1) 로딩
    df = load_data(args.data, args.sample if args.sample > 0 else None)

    # 2) 타깃 확정 + 정제 + 피처 생성
    targets = resolve_targets(df)
    X, y = clean_and_engineer(df, targets)
    if len(X) < 100:
        raise SystemExit("[오류] 정제 후 데이터가 너무 적어 학습할 수 없습니다.")

    num_features = X.attrs["num_features"]
    cat_features = X.attrs["cat_features"]

    # 3) 5-fold 교차검증으로 성능 측정 (fold 마다 전처리를 새로 학습 → 누수 없음)
    log.info("5-fold 교차검증 시작... (RandomForest, 다중출력 %d종)", len(targets))
    evaluate_cv(X, y, num_features, cat_features, n_splits=args.folds)

    # 4) 최종 모델은 '전체 데이터'로 다시 학습해 성능을 최대한 끌어올린 뒤 저장한다.
    #    (교차검증은 성능 '추정'용, 배포 모델은 가진 데이터를 모두 사용)
    log.info("최종 모델 학습 중 (전체 %d건 사용)...", len(X))
    final_pipe = build_pipeline(num_features, cat_features)
    final_pipe.fit(X, y)

    # 5) 모델 저장 (joblib)
    out_path = Path(args.out)
    joblib.dump(final_pipe, out_path)
    log.info("모델 저장 완료 → %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


if __name__ == "__main__":
    # 최상위 예외 처리: 예기치 못한 오류도 사용자 친화적으로 알린다.
    try:
        main()
    except SystemExit:
        raise  # 우리가 의도한 종료 메시지는 그대로 전달
    except Exception as e:  # noqa: BLE001
        log.exception("예상치 못한 오류로 중단되었습니다: %s", e)
        sys.exit(1)
