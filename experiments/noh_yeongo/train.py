"""noh_yeongo 실험 실행: 공통 80:20 분할 + 승차 시점 선형 회귀.

실행: python -m experiments.noh_yeongo.train
결과: reports/experiments/noh_yeongo/metrics.json (공통 형식)
      experiments/noh_yeongo/artifacts/models/pickup_time_linear.joblib
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiments.noh_yeongo.preprocessing import (
    BIN_FEATURES, CAT_FEATURES, FEATURES, NUM_FEATURES,
    add_features, filter_train_rows,
)
from src.common.data_loader import load_taxi_pandas
from src.common.evaluation import evaluate_fare_bands, evaluate_regression
from src.common.results import save_result
from src.common.split import SPLIT_ID, TARGET_COLUMN, make_common_split

AUTHOR = "noh_yeongo"
EXPERIMENT_ID = "noh_yeongo_pickup_time_linear_v1"
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "models" / "pickup_time_linear.joblib"


def build_pipeline() -> Pipeline:
    """전처리(표준화+원핫)와 선형 회귀를 하나의 Pipeline 으로 구성한다."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("bin", "passthrough", BIN_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])


def main() -> None:
    # 1) 공통 로더(원본 검증 + source_row_id 부여) → 공통 80:20 분할
    dataframe = load_taxi_pandas()
    train_df, test_df = make_common_split(dataframe)

    # 2) 피처 생성(양쪽 동일) + 품질 필터(학습에만 적용, 테스트는 전 행 예측)
    train_df = filter_train_rows(add_features(train_df))
    test_df = add_features(test_df)
    print(f"학습 {len(train_df):,}행 / 테스트 {len(test_df):,}행 (테스트 제외 0행)")

    # 3) 학습 및 예측
    pipeline = build_pipeline()
    pipeline.fit(train_df[FEATURES], train_df[TARGET_COLUMN])
    predictions = pipeline.predict(test_df[FEATURES])

    # 4) 공통 평가 함수로 지표 산출(팀 규칙)
    metrics = evaluate_regression(test_df[TARGET_COLUMN], predictions)
    fare_bands = evaluate_fare_bands(test_df[TARGET_COLUMN], predictions)
    print("공통 지표:", {k: round(v, 4) for k, v in metrics.items()})

    # 5) 모델 저장 + 공통 형식 결과 저장(reports/experiments/noh_yeongo/metrics.json)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "author": AUTHOR,
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "data": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "excluded_test_rows": 0,
            "split_id": SPLIT_ID,
            # 피처에 요금 구성요소·사후 정보 미포함(승차 시점 정보만) 확인 완료
            "leakage_check": True,
        },
        "fare_bands": fare_bands,
        "artifacts": {
            "report": "reports/experiments/noh_yeongo/report.md",
            "model": str(MODEL_PATH.relative_to(MODEL_PATH.parents[4])),
        },
    }
    destination = save_result(AUTHOR, payload)
    print(f"✅ 결과 저장: {destination}")


if __name__ == "__main__":
    main()
