"""noh_yeongo 요금 구성 항목별 회귀 성능 분석.

total_amount 를 구성하는 9개 항목 각각을 같은 승차 시점 피처·공통 분할로
예측해 "어떤 요금은 규칙이 정하고, 어떤 요금은 사람이 정하는지"를 비교한다.

누수 방지 규칙:
- 모든 요금 구성 컬럼은 입력에서 제외 (train.py 와 동일)
- Airport_fee 예측 시 is_airport(공항 구역 파생) 피처 제외 — 구역 기반이지만
  보수적으로 제외해 간접 누수 가능성 차단
- 결측 정답(congestion_surcharge·Airport_fee 의 비미터기 수집분)은 0으로
  채우지 않고 해당 항목의 학습·평가에서 제외

실행: python -m experiments.noh_yeongo.item_analysis
출력: reports/experiments/noh_yeongo/item_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiments.noh_yeongo.preprocessing import (
    BIN_FEATURES, CAT_FEATURES, NUM_FEATURES, add_features, filter_train_rows,
)
from src.common.data_loader import load_taxi_pandas
from src.common.split import make_common_split

RESULT_PATH = (Path(__file__).resolve().parents[2]
               / "reports" / "experiments" / "noh_yeongo" / "item_metrics.json")

COMPONENTS = ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
              "improvement_surcharge", "congestion_surcharge", "Airport_fee",
              "cbd_congestion_fee"]


def build_pipeline(bin_feats: list[str]) -> Pipeline:
    """train.py 와 동일한 전처리 + 선형 회귀 Pipeline."""
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("bin", "passthrough", bin_feats),
        ])),
        ("model", LinearRegression()),
    ])


def main() -> None:
    dataframe = load_taxi_pandas()
    train_df, test_df = make_common_split(dataframe)
    train_df = filter_train_rows(add_features(train_df))
    test_df = add_features(test_df)

    results = {}
    print("=" * 50)
    print("        [ 요금 항목별 성능 평가 (공통 분할) ]")
    print("=" * 50)
    for target in COMPONENTS:
        # 결측 정답은 학습·평가에서 제외 (0 대체 금지 — 미측정을 가짜 0으로 만들지 않음)
        tr = train_df[train_df[target].notna()]
        te = test_df[test_df[target].notna()]
        # Airport_fee 는 공항 파생 피처를 보수적으로 제외
        bin_feats = [f for f in BIN_FEATURES
                     if not (target == "Airport_fee" and f == "is_airport")]

        pipe = build_pipeline(bin_feats)
        pipe.fit(tr[NUM_FEATURES + CAT_FEATURES + bin_feats], tr[target])
        pred = pipe.predict(te[NUM_FEATURES + CAT_FEATURES + bin_feats])

        results[target] = {
            "r2": round(float(r2_score(te[target], pred)), 4),
            "mae": round(float(mean_absolute_error(te[target], pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(te[target], pred))), 4),
            "nonzero_ratio": round(float((te[target] != 0).mean()), 4),
            "test_rows": int(len(te)),
        }
        m = results[target]
        print(f"🔹 {target}")
        print(f"   - R2 Score : {m['r2']:.4f}")
        print(f"   - MAE      : ${m['mae']:.4f}")
        print(f"   - RMSE     : ${m['rmse']:.4f}")
        print("-" * 50)

    avg = {k: round(float(np.mean([m[k] for m in results.values()])), 4)
           for k in ["r2", "mae", "rmse"]}
    print(f"\n🏆 전체 항목 평균 R2 Score : {avg['r2']:.4f}")
    print(f"🏆 전체 항목 평균 MAE      : ${avg['mae']:.4f}")
    print(f"🏆 전체 항목 평균 RMSE     : ${avg['rmse']:.4f}")

    RESULT_PATH.write_text(json.dumps({"items": results, "average": avg},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
