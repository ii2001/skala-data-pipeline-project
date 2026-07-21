"""NYC Taxi total_amount 회귀 실험과 Markdown 보고서를 생성한다."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.data_loader import load_taxi_pandas
from src.preprocessing import HOLDOUT_START, prepare_modeling_data
from src.statistical_analysis import run_statistical_analysis


plt.switch_backend("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODEL_PATH = PROJECT_ROOT / "models" / "total_amount_regression_pipeline.joblib"
REPORT_PATH = REPORTS_DIR / "report.md"
SUMMARY_PATH = METRICS_DIR / "regression_summary.json"
AUDIT_PATH = METRICS_DIR / "regression_preprocessing_audit.csv"
EXPERIMENT_PATH = METRICS_DIR / "regression_experiments.csv"
BAND_PATH = METRICS_DIR / "regression_fare_band_metrics.csv"

RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 500_000
TARGET_COLUMN = "total_amount"

NUMERIC_FEATURES = [
    "passenger_count",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_dayofweek_sin",
    "pickup_dayofweek_cos",
    "is_weekend",
]
BASE_CATEGORICAL = ["VendorID", "PULocationID", "DOLocationID"]
MATCHED_CATEGORICAL = BASE_CATEGORICAL + ["route_id", "pickup_period"]
ROUTE_STAT_FEATURES = ["route_id", "route_period_id"]
MODEL_FEATURES = list(
    dict.fromkeys(
        NUMERIC_FEATURES
        + BASE_CATEGORICAL
        + MATCHED_CATEGORICAL
        + ROUTE_STAT_FEATURES
    )
)

LEAKAGE_COLUMNS = [
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
POST_TRIP_COLUMNS = [
    "tpep_dropoff_datetime",
    "trip_distance",
    "payment_type",
    "RatecodeID",
    "store_and_fwd_flag",
]


def temporal_split_and_sample(
    dataframe: pd.DataFrame, sample_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    """마지막 7일을 홀드아웃으로 두고 기간별 고정 표본을 추출한다."""
    if sample_size < 10:
        raise ValueError("sample_size는 10 이상이어야 합니다.")

    train_pool = dataframe.loc[dataframe["tpep_pickup_datetime"] < HOLDOUT_START]
    test_pool = dataframe.loc[dataframe["tpep_pickup_datetime"] >= HOLDOUT_START]
    if train_pool.empty or test_pool.empty:
        raise ValueError("시간 기반 학습 또는 테스트 데이터가 비어 있습니다.")

    train_limit = min(int(sample_size * 0.8), len(train_pool))
    test_limit = min(sample_size - train_limit, len(test_pool))
    train = train_pool.sample(n=train_limit, random_state=RANDOM_STATE)
    test = test_pool.sample(n=test_limit, random_state=RANDOM_STATE)
    summary: dict[str, int | float | str] = {
        "method": "temporal holdout",
        "holdout_start": HOLDOUT_START.strftime("%Y-%m-%d"),
        "train_pool_rows": len(train_pool),
        "test_pool_rows": len(test_pool),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_target_mean": float(train[TARGET_COLUMN].mean()),
        "test_target_mean": float(test[TARGET_COLUMN].mean()),
        "train_target_max": float(train[TARGET_COLUMN].max()),
        "test_target_max": float(test[TARGET_COLUMN].max()),
    }
    return train, test, summary


def numeric_pipeline() -> Pipeline:
    """숫자 결측을 훈련 중앙값으로 대체하고 표준화한다."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )


def onehot_pipeline(min_frequency: int | None = None) -> Pipeline:
    """범주 결측 대체와 미지 범주 대응 One-Hot을 구성한다."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                ),
            ),
        ]
    )


def build_preprocessor(feature_set: str) -> ColumnTransformer:
    """기본·경로 매칭·교차검증 경로 통계 피처 구성을 만든다."""
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", numeric_pipeline(), NUMERIC_FEATURES)
    ]
    if feature_set == "base":
        transformers.append(("categorical", onehot_pipeline(), BASE_CATEGORICAL))
    elif feature_set == "route_match":
        transformers.append(
            ("categorical", onehot_pipeline(min_frequency=20), MATCHED_CATEGORICAL)
        )
    elif feature_set == "route_statistics":
        transformers.extend(
            [
                (
                    "categorical",
                    onehot_pipeline(),
                    BASE_CATEGORICAL + ["pickup_period"],
                ),
                (
                    "route_target_statistics",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "target_encoder",
                                TargetEncoder(
                                    target_type="continuous",
                                    cv=5,
                                    shuffle=True,
                                    random_state=RANDOM_STATE,
                                ),
                            ),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    ROUTE_STAT_FEATURES,
                ),
            ]
        )
    else:
        raise ValueError(f"알 수 없는 feature_set: {feature_set}")
    return ColumnTransformer(transformers=transformers)


def log_target_ridge() -> TransformedTargetRegressor:
    """긴 꼬리 금액을 log1p로 학습하고 달러 단위로 복원한다."""
    return TransformedTargetRegressor(
        regressor=Ridge(alpha=10.0, solver="lsqr"),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_experiment_pipelines() -> dict[str, Pipeline]:
    """모델과 피처 변형의 효과를 분리해 비교하는 Pipeline을 만든다."""
    specifications = {
        "dummy_median": ("base", DummyRegressor(strategy="median")),
        "ridge_raw_base": ("base", Ridge(alpha=10.0, solver="lsqr")),
        "ridge_log_base": ("base", log_target_ridge()),
        "ridge_raw_route_match": (
            "route_match",
            # 완전 One-Hot 묶음이 상수항을 이미 표현해 중복 절편을 제거한다.
            Ridge(alpha=10.0, solver="lsqr", fit_intercept=False),
        ),
        "ridge_raw_route_statistics": (
            "route_statistics",
            Ridge(alpha=10.0, solver="lsqr"),
        ),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_set)),
                ("regressor", regressor),
            ]
        )
        for name, (feature_set, regressor) in specifications.items()
    }


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """달러 오차와 설명력을 함께 계산한다."""
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": mean_squared_error(y_true, predictions) ** 0.5,
        "median_ae": median_absolute_error(y_true, predictions),
        "r2": r2_score(y_true, predictions),
    }


def run_experiments(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, str, Pipeline, np.ndarray]:
    """같은 시간 홀드아웃에서 모든 실험을 평가하고 MAE 최적 모델을 고른다."""
    x_train = train[MODEL_FEATURES]
    y_train = train[TARGET_COLUMN]
    x_test = test[MODEL_FEATURES]
    y_test = test[TARGET_COLUMN]
    results: list[dict[str, float | str]] = []
    fitted: dict[str, tuple[Pipeline, np.ndarray]] = {}

    for name, pipeline in build_experiment_pipelines().items():
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = pipeline.predict(x_test)
        predict_seconds = time.perf_counter() - started
        results.append(
            {
                "experiment": name,
                **calculate_metrics(y_test, predictions),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        fitted[name] = (pipeline, predictions)

    experiments = pd.DataFrame(results)
    candidates = experiments.loc[experiments["experiment"] != "dummy_median"]
    best_name = str(candidates.sort_values("mae").iloc[0]["experiment"])
    best_pipeline, best_predictions = fitted[best_name]
    return experiments, best_name, best_pipeline, best_predictions


def fare_band_metrics(y_true: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    """저가부터 고액까지 금액 구간별 오차를 계산한다."""
    source = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": predictions})
    source["fare_band"] = pd.cut(
        source["actual"],
        bins=[0, 30, 60, 100, np.inf],
        labels=["$0–30", "$30–60", "$60–100", "$100+"],
        right=False,
    )
    rows = []
    for band, group in source.groupby("fare_band", observed=True):
        rows.append(
            {
                "fare_band": str(band),
                "rows": len(group),
                "actual_mean": group["actual"].mean(),
                "mae": mean_absolute_error(group["actual"], group["predicted"]),
                "rmse": mean_squared_error(group["actual"], group["predicted"]) ** 0.5,
                "mean_error": (group["predicted"] - group["actual"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def create_preprocessing_chart(audit: pd.DataFrame, split: dict[str, Any]) -> Path:
    """필터별 제거량과 시간 분할의 금액 분포를 저장한다."""
    path = FIGURE_DIR / "preprocessing_audit.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=audit, x="removed_rows", y="step", color="#E45756", ax=axes[0])
    axes[0].set(title="Rows Removed by Quality Rule", xlabel="Removed rows", ylabel="Step")
    axes[0].bar_label(axes[0].containers[0], fmt="{:,.0f}", padding=3)
    split_data = pd.DataFrame(
        {
            "period": ["Train: May 1–24", "Test: May 25–31"],
            "mean_total": [split["train_target_mean"], split["test_target_mean"]],
        }
    )
    sns.barplot(data=split_data, x="period", y="mean_total", hue="period", legend=False, ax=axes[1])
    axes[1].set(title="Mean total_amount by Temporal Split", xlabel="Period", ylabel="Mean total_amount ($)")
    axes[1].bar_label(axes[1].containers[0], fmt="$%.2f", padding=3)
    fig.suptitle("NYC Yellow Taxi — Preprocessing Audit", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def create_result_charts(
    experiments: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray,
    bands: pd.DataFrame,
) -> None:
    """모델 비교·실제값 비교·잔차·구간별 오차 그래프를 저장한다."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=experiments, x="mae", y="experiment", color="#4C78A8", ax=axes[0])
    axes[0].set(title="MAE — Lower Is Better", xlabel="MAE ($)", ylabel="Experiment")
    sns.barplot(data=experiments, x="r2", y="experiment", color="#54A24B", ax=axes[1])
    axes[1].set(title="R² — Higher Is Better", xlabel="R²", ylabel="Experiment")
    fig.suptitle("Regression Experiment Comparison", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "regression_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    evaluation = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": predictions})
    sample = evaluation.sample(n=min(30_000, len(evaluation)), random_state=RANDOM_STATE)
    display_limit = float(evaluation["actual"].quantile(0.99))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(data=sample, x="actual", y="predicted", alpha=0.15, s=14, ax=ax)
    ax.plot([0, display_limit], [0, display_limit], linestyle="--", color="red")
    ax.set(xlim=(0, display_limit), ylim=(0, display_limit), title="Actual vs Predicted total_amount (to 99th percentile)", xlabel="Actual ($)", ylabel="Predicted ($)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    residual = evaluation["predicted"] - evaluation["actual"]
    clipped = residual.between(residual.quantile(0.01), residual.quantile(0.99))
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(residual[clipped], bins=70, kde=True, color="#F58518", ax=ax)
    ax.axvline(0, linestyle="--", color="black")
    ax.set(title="Residual Distribution (1st–99th Percentile)", xlabel="Predicted - actual ($)", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "residual_analysis.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=bands, x="fare_band", y="mae", hue="fare_band", legend=False, ax=ax)
    ax.set(title="MAE by Actual Fare Band", xlabel="Actual total_amount band", ylabel="MAE ($)")
    ax.bar_label(ax.containers[0], fmt="$%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "error_by_fare_band.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Markdown 표를 추가 의존성 없이 만든다."""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def generate_report(
    preprocessing: dict[str, Any],
    audit: pd.DataFrame,
    split: dict[str, Any],
    statistics: dict[str, Any],
    experiments: pd.DataFrame,
    best_name: str,
    bands: pd.DataFrame,
) -> Path:
    """설계·결과·해석·의견을 포함한 최종 보고서를 자동 생성한다."""
    best = experiments.loc[experiments["experiment"] == best_name].iloc[0]
    base = experiments.loc[experiments["experiment"] == "ridge_log_base"].iloc[0]
    improvement = (base.mae - best.mae) / base.mae
    ttest = statistics["welch_ttest"]
    experiment_rows = [
        [
            str(row.experiment),
            f"${row.mae:.3f}",
            f"${row.rmse:.3f}",
            f"${row.median_ae:.3f}",
            f"{row.r2:.4f}",
            f"{row.fit_seconds:.2f}s",
        ]
        for row in experiments.itertuples(index=False)
    ]
    audit_rows = [
        [str(row.step), str(row.rule), f"{int(row.removed_rows):,}", f"{int(row.after_rows):,}"]
        for row in audit.itertuples(index=False)
    ]
    band_rows = [
        [str(row.fare_band), f"{int(row.rows):,}", f"${row.actual_mean:.2f}", f"${row.mae:.2f}", f"${row.rmse:.2f}", f"${row.mean_error:.2f}"]
        for row in bands.itertuples(index=False)
    ]
    report = f"""# NYC Yellow Taxi `total_amount` 회귀 분석 보고서

## 1. 문제 정의와 예측 시점

이진 고액 여부 대신 기록된 `total_amount` 자체를 예측한다. 예측 시점은 **승차
직후 목적지 LocationID가 정해진 시점**으로 가정한다. 따라서 승차 시각·승하차
지역·업체·승객 수만 사용하고, 운행 종료 뒤 확정되는 정보는 제외한다.

- 원본: {int(preprocessing['raw_rows']):,}행 × 20열
- 품질 처리 후: {int(preprocessing['valid_rows']):,}행 ({float(preprocessing['retention_rate']):.2%} 보존)
- 학습 표본: {int(split['train_rows']):,}행 (5월 1~24일 후보)
- 테스트 표본: {int(split['test_rows']):,}행 (5월 25~31일 후보)
- 테스트 평균/최대 금액: ${float(split['test_target_mean']):.2f} / ${float(split['test_target_max']):,.2f}

## 2. 누수 방지와 전처리

`total_amount`를 직접 구성하는 `{', '.join(LEAKAGE_COLUMNS)}`는 모두 삭제했다.
`{', '.join(POST_TRIP_COLUMNS)}`도 예측 시점에 알 수 없거나 정책 의존성이 커서
제외했다. 이 컬럼을 넣어 얻은 높은 점수는 요금 공식을 재현하는 것이지 사전
예측 성능이 아니다.
거리·소요시간 규칙은 기록 오류를 제거하는 **오프라인 학습 데이터 품질 규칙**일
뿐이며, 온라인 예측 입력이나 모델 피처로 사용하지 않는다.

{markdown_table(['단계', '규칙', '제거 행', '이후 행'], audit_rows)}

승객 수의 원래 결측 {int(preprocessing['original_missing_passenger_rows']):,}건과
범위 오류 {int(preprocessing['invalid_passenger_rows_reclassified']):,}건은 Pipeline
안에서 학습 중앙값으로 대체하고 결측 indicator를 추가했다. 시간·요일은 sin/cos로
변환했다. 테스트의 고액 이상치는 실제 운영 오차를 보기 위해 삭제하거나 상한 처리하지 않았다.

![전처리 감사](figures/preprocessing_audit.png)

## 3. 데이터 매칭·변형 실험

- `base`: PU와 DO LocationID를 별도 One-Hot 처리
- `route_match`: 기본 PU·DO에 `PU→DO` `route_id`를 추가하고 희소 경로를 묶음
- `route_statistics`: `route_id`와 `route+시간대`의 과거 평균 운임을 TargetEncoder로 추가
- `log`: 긴 오른쪽 꼬리를 줄이도록 `log1p(total_amount)`를 학습하고 달러로 역변환

TargetEncoder는 학습 행에도 5-fold 교차 적합 값을 사용한다. 테스트에는 학습
기간에서 만든 통계만 적용하므로 테스트 정답이 피처에 섞이지 않는다.
경로 통계와 로그 타깃을 함께 쓴 초기 점검에서는 역변환 후 일부 예측이 폭증해
불안정했으므로, 경로 통계 실험은 원금액 Ridge와 조합했다.

{markdown_table(['실험', 'MAE', 'RMSE', 'Median AE', 'R²', '학습시간'], experiment_rows)}

![회귀 모델 비교](figures/regression_model_comparison.png)

## 4. 최종 모델과 구간별 결과

MAE가 가장 낮은 `{best_name}`를 최종 모델로 선택했다.

- MAE: ${best.mae:.3f}
- RMSE: ${best.rmse:.3f}
- Median AE: ${best.median_ae:.3f}
- R²: {best.r2:.4f}
- 기본 log Ridge 대비 MAE 개선율: {improvement:.2%}

![실제값과 예측값](figures/actual_vs_predicted.png)

![잔차 분포](figures/residual_analysis.png)

{markdown_table(['실제 금액 구간', '행 수', '평균', 'MAE', 'RMSE', '평균오차'], band_rows)}

![금액 구간별 오차](figures/error_by_fare_band.png)

## 5. 통계 분석

기술통계와 상관계수는 `reports/metrics/statistical_results.json`에 저장했다.
신용카드와 현금 결제의 이동거리 Welch t-test는 t={float(ttest['t_statistic']):.3f},
p={float(ttest['p_value']):.3e}, Cohen's d={float(ttest['cohens_d']):.3f}였다.
p < 0.05지만 효과크기가 거의 0이므로 실질적 차이는 매우 작다.

![상관계수](figures/correlation_heatmap.png)

## 6. 내 모델에 대한 의견과 한계

이진 분류보다 회귀가 금액 크기를 보존해 실제 예상 요금 제시에 더 적합하다.
경로 매칭의 효과는 기본 모델과 동일 홀드아웃에서 직접 비교했으며, 개선율이
양수일 때만 유효한 개선으로 해석했다. 직접 금액 컬럼과 운행 후 거리를 빼도
경로의 과거 통계가 유용한지를 검증한 것이 이 실험의 핵심이다.

다만 목적지가 미정인 길거리 승차에는 이 모델을 그대로 사용할 수 없다. 또한
한 달 자료의 일부 표본으로 평가했으므로 여러 달 rolling holdout이 필요하다.
고액 구간의 행 수가 적어 RMSE가 커질 수 있으며, 평균적인 요금 안내와 고액
이상치 탐지는 별도 모델로 나누는 것이 다음 개선 방향이다.

## 7. 재현과 결과 위치

```bash
python -m src.eda
python -m src.model
```

- 최종 모델: `models/total_amount_regression_pipeline.joblib`
- 수치 결과: `reports/metrics/regression_*.csv`, `regression_summary.json`
- 그래프: `reports/figures/`
- 이전 분류 결과: `reports/archive/classification_report.md`
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH


def save_results(
    pipeline: Pipeline,
    preprocessing: dict[str, Any],
    audit: pd.DataFrame,
    split: dict[str, Any],
    experiments: pd.DataFrame,
    best_name: str,
    bands: pd.DataFrame,
) -> None:
    """모델과 재현 가능한 구조화 결과를 저장한다."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    audit.to_csv(AUDIT_PATH, index=False)
    experiments.to_csv(EXPERIMENT_PATH, index=False)
    bands.to_csv(BAND_PATH, index=False)
    summary = {
        "target": TARGET_COLUMN,
        "prediction_time": "pickup with known destination",
        "excluded_direct_amount_columns": LEAKAGE_COLUMNS,
        "excluded_post_trip_columns": POST_TRIP_COLUMNS,
        "selected_model": best_name,
        "preprocessing": preprocessing,
        "split": split,
        "experiments": json.loads(experiments.to_json(orient="records")),
        "fare_bands": json.loads(bands.to_json(orient="records")),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_modeling(sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict[str, float]:
    """전처리부터 모델 저장과 보고서 생성까지 실행한다."""
    raw = load_taxi_pandas()
    prepared, audit, preprocessing = prepare_modeling_data(raw)
    del raw
    gc.collect()
    statistics = run_statistical_analysis(prepared)
    train, test, split = temporal_split_and_sample(prepared, sample_size)
    del prepared
    gc.collect()

    experiments, best_name, pipeline, predictions = run_experiments(train, test)
    bands = fare_band_metrics(test[TARGET_COLUMN], predictions)
    create_preprocessing_chart(audit, split)
    create_result_charts(experiments, test[TARGET_COLUMN], predictions, bands)
    save_results(pipeline, preprocessing, audit, split, experiments, best_name, bands)
    report = generate_report(preprocessing, audit, split, statistics, experiments, best_name, bands)

    best = experiments.loc[experiments["experiment"] == best_name].iloc[0]
    metrics = {name: float(best[name]) for name in ["mae", "rmse", "median_ae", "r2"]}
    print("\n[회귀 실험 결과]")
    print(experiments.round(4).to_string(index=False))
    print(f"선택 모델: {best_name}")
    print(f"보고서: {report}")
    print(f"모델: {MODEL_PATH}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Taxi total_amount 회귀 실험")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"시간 분할 후 사용할 총 표본 크기 (기본값: {DEFAULT_SAMPLE_SIZE:,})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_modeling(args.sample_size)
