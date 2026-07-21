"""Day 2 종합 실습의 분석, 학습, 평가, 보고서 생성을 한 번에 수행한다."""

from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.data_loader import download_taxi_data
from src.common.evaluation import evaluate_fare_bands, evaluate_regression
from src.common.results import validate_result
from src.common.split import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    RANDOM_STATE,
    SPLIT_ID,
    make_common_split,
)


matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message=(
        "X does not have valid feature names, but LGBMRegressor was fitted with "
        "feature names"
    ),
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "final"
FIGURE_DIR = REPORT_DIR / "figures"
INTERACTIVE_DIR = REPORT_DIR / "interactive"
METRICS_PATH = REPORT_DIR / "metrics.json"
DESCRIPTIVE_PATH = REPORT_DIR / "descriptive_statistics.csv"
CORRELATION_PATH = REPORT_DIR / "correlation.csv"
STATISTICS_PATH = REPORT_DIR / "statistical_results.json"
REPORT_PATH = REPORT_DIR / "report.md"
STATIC_CHART_PATH = FIGURE_DIR / "eda_overview.png"
INTERACTIVE_CHART_PATH = INTERACTIVE_DIR / "hourly_total_amount.html"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"

TARGET_COLUMN = "total_amount"
ANALYSIS_START = pd.Timestamp("2026-05-01")
ANALYSIS_END = pd.Timestamp("2026-06-01")
MAX_TRIP_DISTANCE_MILES = 100.0
MIN_TRIP_DURATION_MINUTES = 1.0
MAX_TRIP_DURATION_MINUTES = 180.0
EDA_SAMPLE_SIZE = 100_000

NUMERIC_FEATURES = [
    "passenger_count",
    "trip_distance",
    "trip_duration_minutes",
    "pickup_hour",
    "pickup_day_of_week",
    "trip_distance_was_capped",
    "trip_distance_is_zero",
    "trip_duration_was_adjusted",
]
CODE_FEATURES = ["RatecodeID", "PULocationID", "DOLocationID", "payment_type"]
STORE_FEATURES = ["store_and_fwd_flag"]
MODEL_FEATURES = NUMERIC_FEATURES + CODE_FEATURES + STORE_FEATURES
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


def ensure_output_directories() -> None:
    """최종 산출물 디렉터리를 준비한다."""
    for directory in (REPORT_DIR, FIGURE_DIR, INTERACTIVE_DIR, MODEL_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)


def load_with_pandas_and_polars() -> tuple[pd.DataFrame, dict[str, Any]]:
    """동일 Parquet을 두 라이브러리로 각각 읽고 결과 일치 여부를 검증한다."""
    data_path = download_taxi_data()

    started = time.perf_counter()
    pandas_data = pd.read_parquet(data_path)
    pandas_seconds = time.perf_counter() - started

    started = time.perf_counter()
    polars_data = pl.read_parquet(data_path)
    polars_seconds = time.perf_counter() - started

    pandas_nulls = {name: int(value) for name, value in pandas_data.isna().sum().items()}
    polars_nulls = {
        name: int(value) for name, value in polars_data.null_count().row(0, named=True).items()
    }
    same_shape = pandas_data.shape == polars_data.shape
    same_columns = list(pandas_data.columns) == polars_data.columns
    same_null_counts = pandas_nulls == polars_nulls
    if not (same_shape and same_columns and same_null_counts):
        raise ValueError("Pandas와 Polars 로딩 결과가 일치하지 않습니다.")

    comparison = {
        "pandas_shape": list(pandas_data.shape),
        "polars_shape": list(polars_data.shape),
        "pandas_seconds": pandas_seconds,
        "polars_seconds": polars_seconds,
        "same_shape": same_shape,
        "same_columns": same_columns,
        "same_null_counts": same_null_counts,
        "null_counts": {name: count for name, count in pandas_nulls.items() if count},
    }
    del polars_data
    gc.collect()

    pandas_data.insert(0, "source_row_id", pandas_data.index.to_numpy())
    return pandas_data, comparison


def eligible_mask(dataframe: pd.DataFrame) -> pd.Series:
    """공통 평가 모집단과 같은 2026년 5월 양수 타깃 행을 선택한다."""
    return dataframe["tpep_pickup_datetime"].between(
        ANALYSIS_START, ANALYSIS_END, inclusive="left"
    ) & dataframe[TARGET_COLUMN].gt(0)


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """원본 시각과 거리에서 누수 없는 모델 피처를 생성한다."""
    duration = (
        dataframe["tpep_dropoff_datetime"] - dataframe["tpep_pickup_datetime"]
    ).dt.total_seconds().div(60)
    distance = dataframe["trip_distance"]
    return pd.DataFrame(
        {
            "passenger_count": dataframe["passenger_count"],
            "trip_distance": distance.clip(0, MAX_TRIP_DISTANCE_MILES),
            "trip_duration_minutes": duration.clip(
                MIN_TRIP_DURATION_MINUTES, MAX_TRIP_DURATION_MINUTES
            ),
            "pickup_hour": dataframe["tpep_pickup_datetime"].dt.hour,
            "pickup_day_of_week": dataframe["tpep_pickup_datetime"].dt.dayofweek,
            "trip_distance_was_capped": (
                distance > MAX_TRIP_DISTANCE_MILES
            ).astype("int8"),
            "trip_distance_is_zero": (distance == 0).astype("int8"),
            "trip_duration_was_adjusted": (
                (duration < MIN_TRIP_DURATION_MINUTES)
                | (duration > MAX_TRIP_DURATION_MINUTES)
            ).astype("int8"),
            "RatecodeID": dataframe["RatecodeID"],
            "PULocationID": dataframe["PULocationID"],
            "DOLocationID": dataframe["DOLocationID"],
            "payment_type": dataframe["payment_type"],
            "store_and_fwd_flag": dataframe["store_and_fwd_flag"],
        },
        index=dataframe.index,
    )


def create_visualizations(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Seaborn 정적 차트와 Plotly 인터랙티브 차트를 생성한다."""
    sample_size = min(EDA_SAMPLE_SIZE, len(dataframe))
    sample = dataframe.sample(sample_size, random_state=RANDOM_STATE).copy()
    duration = (
        sample["tpep_dropoff_datetime"] - sample["tpep_pickup_datetime"]
    ).dt.total_seconds().div(60)
    chart_data = pd.DataFrame(
        {
            "trip_distance": sample["trip_distance"].clip(
                0, MAX_TRIP_DISTANCE_MILES
            ),
            "trip_duration_minutes": duration.clip(
                MIN_TRIP_DURATION_MINUTES, MAX_TRIP_DURATION_MINUTES
            ),
            "passenger_count": sample["passenger_count"].fillna(0),
            TARGET_COLUMN: sample[TARGET_COLUMN],
        }
    )

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        data=chart_data.loc[chart_data["trip_distance"] <= 30],
        x="trip_distance",
        bins=60,
        color="#f4b400",
        ax=axes[0],
    )
    axes[0].set_title("Trip Distance Distribution (0–30 miles)")
    axes[0].set_xlabel("Trip distance (miles)")
    axes[0].set_ylabel("Trip count")

    correlation = chart_data.corr(numeric_only=True)
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=axes[1],
    )
    axes[1].set_title("Feature Correlation (sample)")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(STATIC_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)

    hourly = (
        dataframe.assign(pickup_hour=dataframe["tpep_pickup_datetime"].dt.hour)
        .groupby("pickup_hour", as_index=False)
        .agg(
            mean_total_amount=(TARGET_COLUMN, "mean"),
            median_total_amount=(TARGET_COLUMN, "median"),
            trip_count=(TARGET_COLUMN, "size"),
        )
        .sort_values("pickup_hour")
    )
    interactive = px.line(
        hourly,
        x="pickup_hour",
        y=["mean_total_amount", "median_total_amount"],
        markers=True,
        hover_data=["trip_count"],
        title="Hourly Mean and Median Total Amount",
        labels={
            "pickup_hour": "Pickup hour",
            "value": "Total amount (USD)",
            "variable": "Statistic",
            "trip_count": "Trip count",
        },
    )
    interactive.write_html(INTERACTIVE_CHART_PATH, include_plotlyjs=True)
    return {
        "sample_rows": sample_size,
        "sample_trip_distance_total_correlation": float(
            correlation.loc["trip_distance", TARGET_COLUMN]
        ),
        "hourly_rows": len(hourly),
    }


def run_statistical_analysis(dataframe: pd.DataFrame) -> dict[str, Any]:
    """기술통계, 상관계수, 결제수단별 Welch t-test를 계산한다."""
    duration = (
        dataframe["tpep_dropoff_datetime"] - dataframe["tpep_pickup_datetime"]
    ).dt.total_seconds().div(60).clip(
        MIN_TRIP_DURATION_MINUTES, MAX_TRIP_DURATION_MINUTES
    )
    statistics_data = pd.DataFrame(
        {
            "passenger_count": dataframe["passenger_count"].fillna(0),
            "trip_distance": dataframe["trip_distance"].clip(
                0, MAX_TRIP_DISTANCE_MILES
            ),
            "trip_duration_minutes": duration,
            TARGET_COLUMN: dataframe[TARGET_COLUMN],
        }
    )
    descriptive = statistics_data.describe(
        percentiles=[0.25, 0.5, 0.75]
    ).transpose()
    correlation = statistics_data.corr(method="pearson")
    descriptive.to_csv(DESCRIPTIVE_PATH)
    correlation.to_csv(CORRELATION_PATH)

    credit = dataframe.loc[dataframe["payment_type"] == 1, TARGET_COLUMN].to_numpy()
    cash = dataframe.loc[dataframe["payment_type"] == 2, TARGET_COLUMN].to_numpy()
    test = ttest_ind(credit, cash, equal_var=False, nan_policy="omit")
    pooled_standard_deviation = np.sqrt(
        ((len(credit) - 1) * credit.var(ddof=1) + (len(cash) - 1) * cash.var(ddof=1))
        / (len(credit) + len(cash) - 2)
    )
    cohens_d = (credit.mean() - cash.mean()) / pooled_standard_deviation
    if test.pvalue < 0.05:
        p_value_interpretation = (
            "p-value가 0.05보다 작아 신용카드와 현금 결제의 평균 total_amount가 "
            "같다는 귀무가설을 기각한다. 표본이 매우 크므로 효과크기도 함께 본다."
        )
    else:
        p_value_interpretation = (
            "p-value가 0.05 이상이므로 두 결제수단의 평균 total_amount 차이가 "
            "통계적으로 유의하다고 보기 어렵다."
        )

    result = {
        "descriptive": {
            column: {name: float(value) for name, value in values.items()}
            for column, values in descriptive.to_dict(orient="index").items()
        },
        "correlation": {
            column: {name: float(value) for name, value in values.items()}
            for column, values in correlation.to_dict().items()
        },
        "ttest": {
            "comparison": "credit_card_vs_cash_total_amount",
            "credit_rows": len(credit),
            "cash_rows": len(cash),
            "credit_mean": float(credit.mean()),
            "cash_mean": float(cash.mean()),
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "cohens_d": float(cohens_d),
            "interpretation": p_value_interpretation,
        },
    }
    STATISTICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_model_pipeline() -> Pipeline:
    """결측 처리, 원핫 인코딩, LightGBM 회귀를 하나의 Pipeline으로 구성한다."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )
    code_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=99)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=True, dtype=np.float32
                ),
            ),
        ]
    )
    store_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=None, strategy="constant", fill_value="Unknown"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("codes", code_transformer, CODE_FEATURES),
            ("store", store_transformer, STORE_FEATURES),
        ],
        sparse_threshold=0.3,
    )
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def markdown_descriptive_table(descriptive: dict[str, Any]) -> str:
    """보고서용 핵심 기술통계 표를 만든다."""
    rows = [
        "| 변수 | 평균 | 표준편차 | 25% | 중앙값 | 75% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for column, values in descriptive.items():
        rows.append(
            f"| {column} | {values['mean']:.4f} | {values['std']:.4f} | "
            f"{values['25%']:.4f} | {values['50%']:.4f} | {values['75%']:.4f} |"
        )
    return "\n".join(rows)


def markdown_fare_band_table(fare_bands: list[dict[str, Any]]) -> str:
    """보고서용 금액 구간별 평가 표를 만든다."""
    rows = [
        "| 실제 금액 구간 | 행 수 | MAE | RMSE | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in fare_bands:
        rows.append(
            f"| {result['fare_band']} | {result['rows']:,} | "
            f"{result['mae']:.4f} | {result['rmse']:.4f} | {result['r2']:.4f} |"
        )
    return "\n".join(rows)


def write_report(
    load_comparison: dict[str, Any],
    duplicate_rows: int,
    visualization: dict[str, Any],
    statistics: dict[str, Any],
    metrics_payload: dict[str, Any],
    model_reload_verified: bool,
) -> None:
    """실제 분석 및 모델 결과를 최종 Markdown 보고서로 생성한다."""
    metrics = metrics_payload["metrics"]
    data = metrics_payload["data"]
    ttest = statistics["ttest"]
    correlation = statistics["correlation"]
    report = f"""# NYC Yellow Taxi total_amount 예측 최종 보고서

## 1. 프로젝트 개요

NYC Yellow Taxi 2026년 5월 운행 데이터를 이용해 기본 EDA, 시각화, 통계 분석,
회귀 Pipeline 학습을 하나의 실행 코드로 자동화했다. 팀 비교 후
`experiments/Lee_hyeonjun/model.py`의 LightGBM 접근을 최종 모델로 선정하고,
절대경로·누수·Pipeline·저장 문제를 최종 코드에서 보완했다.

## 2. 데이터 준비와 Pandas·Polars 비교

- 원본: {load_comparison['pandas_shape'][0]:,}행 × {load_comparison['pandas_shape'][1]}열
- Pandas 로딩: {load_comparison['pandas_seconds']:.4f}초
- Polars 로딩: {load_comparison['polars_seconds']:.4f}초
- shape·컬럼·결측 건수 일치: Y
- 전체 중복 행: {duplicate_rows:,}건
- 원본 결측: {json.dumps(load_comparison['null_counts'], ensure_ascii=False)}

단일 장비에서 한 번 측정한 로딩 시간은 환경과 캐시의 영향을 받으므로 특정
라이브러리가 항상 빠르다고 일반화하지 않는다. 두 구현은 각각 동일 원본을 읽고
shape, 컬럼 순서, 결측 건수를 독립적으로 계산해 결과가 같은지 검증했다.

결측치는 Pipeline 안에서 `passenger_count=0`, `RatecodeID=99`,
`store_and_fwd_flag=Unknown`으로 처리한다. 직접 요금 구성 9개 컬럼과
`VendorID`는 모델 입력에서 제외했다.

## 3. EDA와 시각화

- Seaborn 정적 차트: `figures/eda_overview.png`
- Plotly 인터랙티브 차트: `interactive/hourly_total_amount.html`
- 정적 차트 표본: {visualization['sample_rows']:,}행(고정 시드 42)
- 표본의 trip_distance–total_amount 상관: {visualization['sample_trip_distance_total_correlation']:.4f}

거리 분포는 오른쪽 꼬리가 길며, 잘못 기록된 극단값의 영향을 줄이기 위해
100마일 상한과 별도 indicator를 적용했다. 인터랙티브 차트에서는 시간대별 평균과
중앙값을 함께 표시해 일부 고액 운행이 평균에 미치는 영향을 비교할 수 있다.

## 4. 기술통계와 상관 분석

{markdown_descriptive_table(statistics['descriptive'])}

전처리 범위에서 trip_distance와 total_amount의 Pearson 상관계수는
{correlation[TARGET_COLUMN]['trip_distance']:.4f}이다. 상관관계는 선형 연관성을
보여줄 뿐 인과관계를 의미하지 않는다. 전체 상관행렬은 `correlation.csv`에 저장했다.

## 5. Welch t-test

- 비교: 신용카드({ttest['credit_rows']:,}행) vs 현금({ttest['cash_rows']:,}행)
- 평균 total_amount: ${ttest['credit_mean']:.4f} vs ${ttest['cash_mean']:.4f}
- t 통계량: {ttest['t_statistic']:.4f}
- p-value: {ttest['p_value']:.6e}
- Cohen's d: {ttest['cohens_d']:.4f}

{ttest['interpretation']}

## 6. ML Pipeline과 평가

선정 모델은 `ColumnTransformer → LightGBMRegressor`를 하나의 sklearn
`Pipeline`으로 구성했다. 숫자형 결측 처리, 명목형 코드 원핫 인코딩,
`store_and_fwd_flag` 축약 원핫 인코딩을 학습 데이터에만 fit한다.

- 학습: {data['train_rows']:,}행
- 테스트: {data['test_rows']:,}행
- MAE: {metrics['mae']:.4f}
- RMSE: {metrics['rmse']:.4f}
- Median AE: {metrics['median_ae']:.4f}
- R²: {metrics['r2']:.4f}
- 테스트 제외 행: {data['excluded_test_rows']:,}행
- 저장 모델 재로딩·재예측 검증: {'성공' if model_reload_verified else '실패'}

### 실제 금액 구간별 평가

{markdown_fare_band_table(metrics_payload['fare_bands'])}

모델 파일은 `models/final_model.joblib`, 전체 지표와 금액 구간별 결과는
`reports/final/metrics.json`에 저장했다.

## 7. 결과에 대한 의견과 개선 사항

LightGBM은 거리, 실제 탑승 시간, 시간대, 요일, 승하차 위치처럼 비선형 관계와
상호작용이 예상되는 피처에 적합하다. 다만 실제 탑승 시간과 결제수단은 운행 종료
후 알 수 있으므로 이 모델의 예측 시점은 **운행 종료 직후**로 한정한다.

향후에는 시간 순서 기반 외부 검증, 희귀 LocationID 묶기, LightGBM 파라미터
교차검증, 고액 운행 별도 분석을 수행할 수 있다. 특히 60~100달러 구간의 R²가
음수이고 100달러 이상 MAE가 전체보다 크므로 고액 운행 개선이 우선 과제다.
현재 지표는 한 달 데이터와 한 번의 고정 분할 결과이므로 다른 기간에도 동일한
성능을 보장하지 않는다.

## 8. 실행 방법

```bash
python -m src.final.train
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_output_directories()
    print("[1/6] Pandas·Polars 데이터 로딩 및 비교")
    raw, load_comparison = load_with_pandas_and_polars()
    raw_columns = [column for column in raw.columns if column != "source_row_id"]
    duplicate_rows = int(raw.duplicated(subset=raw_columns).sum())
    print(
        f"shape={tuple(load_comparison['pandas_shape'])}, "
        f"duplicates={duplicate_rows:,}"
    )
    print(
        f"load_seconds: pandas={load_comparison['pandas_seconds']:.4f}, "
        f"polars={load_comparison['polars_seconds']:.4f}"
    )
    print(f"missing_values={load_comparison['null_counts']}")

    print("[2/6] EDA 시각화 및 통계 분석")
    analysis_data = raw.loc[
        eligible_mask(raw),
        [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "payment_type",
            TARGET_COLUMN,
        ],
    ].copy()
    visualization = create_visualizations(analysis_data)
    statistics = run_statistical_analysis(analysis_data)
    descriptive_output = pd.DataFrame(statistics["descriptive"]).transpose()[
        ["mean", "std", "25%", "50%", "75%"]
    ]
    correlation_output = pd.DataFrame(statistics["correlation"])
    print("기술통계:")
    print(descriptive_output.to_string(float_format=lambda value: f"{value:.4f}"))
    print("상관계수:")
    print(correlation_output.to_string(float_format=lambda value: f"{value:.4f}"))
    print(
        "Welch t-test: "
        f"t={statistics['ttest']['t_statistic']:.4f}, "
        f"p={statistics['ttest']['p_value']:.6e}, "
        f"Cohen's d={statistics['ttest']['cohens_d']:.4f}"
    )
    del analysis_data
    gc.collect()

    print("[3/6] 공통 80:20 분할 및 피처 생성")
    train_raw, test_raw = make_common_split(raw)
    train_duplicate_mask = train_raw.duplicated(subset=raw_columns)
    removed_train_duplicates = int(train_duplicate_mask.sum())
    if removed_train_duplicates:
        train_raw = train_raw.loc[~train_duplicate_mask].copy()
    y_train = train_raw[TARGET_COLUMN].to_numpy()
    y_test = test_raw[TARGET_COLUMN].to_numpy()
    x_train = engineer_features(train_raw)
    x_test = engineer_features(test_raw)
    del raw, train_raw, test_raw
    gc.collect()

    if list(x_train.columns) != MODEL_FEATURES:
        raise ValueError("최종 모델 피처 순서가 정의와 다릅니다.")

    print(f"[4/6] LightGBM Pipeline 학습: {len(x_train):,}행")
    pipeline = build_model_pipeline()
    pipeline.fit(x_train, y_train)

    print(f"[5/6] 테스트 평가: {len(x_test):,}행")
    predictions = pipeline.predict(x_test)
    metrics = evaluate_regression(y_test, predictions)
    fare_bands = evaluate_fare_bands(y_test, predictions)
    print(
        f"평가지표: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"Median AE={metrics['median_ae']:.4f}, R²={metrics['r2']:.4f}"
    )
    metrics_payload = {
        "experiment_id": "final_lee_hyeonjun_lightgbm_v1",
        "author": "Lee_hyeonjun",
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "data": {
            "raw_rows": load_comparison["pandas_shape"][0],
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "excluded_test_rows": 0,
            "removed_train_duplicates": removed_train_duplicates,
            "split_id": SPLIT_ID,
            "leakage_check": True,
        },
        "fare_bands": fare_bands,
        "features": MODEL_FEATURES,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "model": {
            "name": "LGBMRegressor",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": RANDOM_STATE,
        },
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    validate_result(metrics_payload)
    METRICS_PATH.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[6/6] 모델·보고서 저장 및 재로딩 검증")
    joblib.dump(pipeline, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    verification_rows = min(1_000, len(x_test))
    reloaded_predictions = reloaded.predict(x_test.iloc[:verification_rows])
    model_reload_verified = bool(
        np.allclose(predictions[:verification_rows], reloaded_predictions)
    )
    if not model_reload_verified:
        raise ValueError("저장한 모델의 재로딩 예측값이 일치하지 않습니다.")
    write_report(
        load_comparison,
        duplicate_rows,
        visualization,
        statistics,
        metrics_payload,
        model_reload_verified,
    )

    if len(x_train) != EXPECTED_TRAIN_ROWS or len(x_test) != EXPECTED_TEST_ROWS:
        raise ValueError("최종 학습·테스트 행 수가 공통 계약과 다릅니다.")
    print(
        "완료: "
        f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"R²={metrics['r2']:.4f}"
    )
    print(f"보고서: {REPORT_PATH}")
    print(f"모델: {MODEL_PATH}")


if __name__ == "__main__":
    main()
