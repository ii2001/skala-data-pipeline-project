"""NYC Taxi 요금 구성을 검증하고 고액 운행 분류 Pipeline을 학습한다."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_taxi_pandas


plt.switch_backend("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "high_fare_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
EVALUATION_CHART_PATH = PROJECT_ROOT / "reports" / "figures" / "model_evaluation.png"
FARE_CHART_PATH = PROJECT_ROOT / "reports" / "figures" / "fare_composition_analysis.png"

TARGET_COLUMN = "high_fare"
HIGH_FARE_THRESHOLD = 30.0
RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 500_000

NUMERIC_FEATURES = ["passenger_count", "pickup_hour", "pickup_dayofweek"]
CATEGORICAL_FEATURES = ["VendorID", "PULocationID", "DOLocationID"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FARE_COMPONENTS = [
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

PAYMENT_LABELS = {
    0: "Flex Fare",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
}

MODEL_REQUIRED_COLUMNS = {
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "total_amount",
}


def validate_columns(dataframe: pd.DataFrame, required: set[str]) -> None:
    """분석에 필요한 컬럼 누락 여부를 확인한다."""
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")


def analyze_fare_composition(
    dataframe: pd.DataFrame,
) -> tuple[Path, dict[str, float]]:
    """구성요소 합계와 기록된 total_amount의 오차를 분석·시각화한다."""
    validate_columns(dataframe, {*FARE_COMPONENTS, "total_amount", "payment_type"})

    component_missing = dataframe[FARE_COMPONENTS].isna().any(axis=1)
    component_sum = dataframe[FARE_COMPONENTS].fillna(0.0).sum(axis=1)
    residual = dataframe["total_amount"] - component_sum
    exact_match = residual.abs() <= 0.01

    payment_summary = pd.DataFrame(
        {
            "payment_type": dataframe["payment_type"],
            "exact_match": exact_match,
            "absolute_error": residual.abs(),
        }
    )
    payment_summary = (
        payment_summary.groupby("payment_type", as_index=False)
        .agg(
            rows=("payment_type", "size"),
            exact_match_rate=("exact_match", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .sort_values("rows", ascending=False)
    )
    payment_summary["payment_method"] = (
        payment_summary["payment_type"].map(PAYMENT_LABELS).fillna("Other")
    )
    payment_summary["exact_match_rate"] *= 100

    plot_limit = dataframe["total_amount"].quantile(0.99)
    scatter_mask = dataframe["total_amount"].between(0, plot_limit) & component_sum.between(
        0, plot_limit
    )
    scatter_source = pd.DataFrame(
        {
            "recorded_total": dataframe.loc[scatter_mask, "total_amount"],
            "component_sum": component_sum.loc[scatter_mask],
        }
    )
    scatter_sample = scatter_source.sample(
        n=min(50_000, len(scatter_source)),
        random_state=RANDOM_STATE,
    )

    residual_lower = residual.quantile(0.01)
    residual_upper = residual.quantile(0.99)
    residual_in_range = residual[residual.between(residual_lower, residual_upper)]
    residual_sample = residual_in_range.sample(
        n=min(100_000, len(residual_in_range)),
        random_state=RANDOM_STATE,
    )
    common_residuals = (
        residual.round(2)
        .value_counts()
        .head(8)
        .rename_axis("residual")
        .reset_index(name="row_count")
    )
    common_residuals["residual_label"] = common_residuals["residual"].map(
        lambda value: f"${value:.2f}"
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    sns.scatterplot(
        data=scatter_sample,
        x="component_sum",
        y="recorded_total",
        alpha=0.15,
        s=12,
        ax=axes[0, 0],
    )
    axes[0, 0].plot([0, plot_limit], [0, plot_limit], linestyle="--", color="red")
    axes[0, 0].set(
        title="Recorded Total vs Component Sum",
        xlabel="Component sum ($, nulls treated as 0)",
        ylabel="Recorded total_amount ($)",
    )

    sns.histplot(residual_sample, bins=60, kde=True, color="#F58518", ax=axes[0, 1])
    axes[0, 1].axvline(0, linestyle="--", color="black")
    axes[0, 1].set(
        title="Residual Distribution (1st–99th Percentile)",
        xlabel="total_amount - component sum ($)",
        ylabel="Sample count",
    )

    sns.barplot(
        data=payment_summary,
        x="exact_match_rate",
        y="payment_method",
        hue="payment_method",
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set(
        title="Exact Match Rate by Payment Method",
        xlabel="Match within 1 cent (%)",
        ylabel="Payment method",
        xlim=(0, 100),
    )
    axes[1, 0].bar_label(axes[1, 0].containers[0], fmt="%.1f%%", padding=3)

    sns.barplot(
        data=common_residuals,
        x="row_count",
        y="residual_label",
        color="#54A24B",
        ax=axes[1, 1],
    )
    axes[1, 1].set(
        title="Most Common Residual Values",
        xlabel="Number of rows",
        ylabel="Residual",
    )

    summary = {
        "component_missing_rows": int(component_missing.sum()),
        "exact_match_rate": float(exact_match.mean()),
        "mean_absolute_error": float(residual.abs().mean()),
        "median_residual": float(residual.median()),
    }
    fig.suptitle(
        "NYC Yellow Taxi — Fare Composition Validation",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FARE_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FARE_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n[요금 구성요소 합계 검증]")
    print(f"구성요소 결측 행: {summary['component_missing_rows']:,}")
    print(f"1센트 이내 일치율: {summary['exact_match_rate']:.2%}")
    print(f"평균 절대 오차: ${summary['mean_absolute_error']:.2f}")
    print(f"잔차 중앙값: ${summary['median_residual']:.2f}")
    return FARE_CHART_PATH, summary


def prepare_modeling_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """업무 규칙으로 이상치를 제외하고 고액 운행 라벨을 생성한다."""
    validate_columns(dataframe, MODEL_REQUIRED_COLUMNS)

    duration_minutes = (
        dataframe["tpep_dropoff_datetime"] - dataframe["tpep_pickup_datetime"]
    ).dt.total_seconds().div(60)
    valid_mask = (
        dataframe["tpep_pickup_datetime"].between(
            "2026-05-01", "2026-06-01", inclusive="left"
        )
        & duration_minutes.between(1, 180, inclusive="both")
        & dataframe["trip_distance"].between(0.1, 100, inclusive="both")
        & (dataframe["total_amount"] > 0)
    )

    selected_columns = [
        "VendorID",
        "tpep_pickup_datetime",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "total_amount",
    ]
    modeling = dataframe.loc[valid_mask, selected_columns].copy()
    if modeling.empty:
        raise ValueError("업무 규칙을 만족하는 모델링 데이터가 없습니다.")

    modeling["pickup_hour"] = modeling["tpep_pickup_datetime"].dt.hour
    modeling["pickup_dayofweek"] = modeling["tpep_pickup_datetime"].dt.dayofweek
    modeling[TARGET_COLUMN] = (
        modeling["total_amount"] >= HIGH_FARE_THRESHOLD
    ).astype("int8")
    return modeling.loc[:, MODEL_FEATURES + [TARGET_COLUMN]]


def stratified_sample(dataframe: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """대용량 학습 시간을 제한하면서 라벨 비율을 보존한다."""
    if sample_size <= 0:
        raise ValueError("sample_size는 1 이상이어야 합니다.")
    if len(dataframe) <= sample_size:
        return dataframe

    sampled, _ = train_test_split(
        dataframe,
        train_size=sample_size,
        stratify=dataframe[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return sampled


def build_pipeline() -> Pipeline:
    """결측 처리·인코딩·표준화·분류기를 하나의 Pipeline으로 구성한다."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        random_state=RANDOM_STATE,
        solver="lbfgs",
        tol=1e-3,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    y_true: pd.Series, predictions: pd.Series, probabilities: pd.Series
) -> dict[str, float]:
    """분류 모델의 주요 평가 지표를 계산한다."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def create_evaluation_chart(
    y_true: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
    metrics: dict[str, float],
) -> Path:
    """평가지표·혼동행렬·ROC·PR 곡선을 2×2 그래프로 저장한다."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_values = [metrics[name] for name in metric_names]
    sns.barplot(x=metric_names, y=metric_values, color="#4C78A8", ax=axes[0, 0])
    axes[0, 0].set(
        title="Classification Metrics",
        xlabel="Metric",
        ylabel="Score",
        ylim=(0, 1),
    )
    axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%.3f", padding=3)

    matrix = confusion_matrix(y_true, predictions)
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0, 1])
    axes[0, 1].set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    axes[1, 0].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {metrics['roc_auc']:.3f}",
    )
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 0].set(
        title="ROC Curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    axes[1, 0].legend()

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    axes[1, 1].plot(recall, precision, color="#F58518")
    axes[1, 1].set(
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    fig.suptitle(
        "NYC Yellow Taxi — High Fare Classification ($30 or More)",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    EVALUATION_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EVALUATION_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return EVALUATION_CHART_PATH


def save_results(
    pipeline: Pipeline,
    metrics: dict[str, float],
    fare_composition: dict[str, float],
    valid_rows: int,
    train_rows: int,
    test_rows: int,
) -> None:
    """학습 Pipeline과 평가 결과를 파일로 저장한다."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    result = {
        "target": f"total_amount >= {HIGH_FARE_THRESHOLD}",
        "valid_rows": valid_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "fare_composition": fare_composition,
        **metrics,
    }
    METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_modeling(sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict[str, float]:
    """요금 검증부터 학습·평가·저장·그래프 생성을 실행한다."""
    raw = load_taxi_pandas()
    fare_chart_path, fare_composition = analyze_fare_composition(raw)
    modeling = prepare_modeling_data(raw)
    del raw
    gc.collect()

    valid_rows = len(modeling)
    sampled = stratified_sample(modeling, sample_size)
    del modeling
    gc.collect()

    features = sampled[MODEL_FEATURES]
    target = sampled[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(y_test, predictions, probabilities)

    save_results(
        pipeline,
        metrics,
        fare_composition,
        valid_rows,
        len(x_train),
        len(x_test),
    )
    evaluation_path = create_evaluation_chart(
        y_test,
        predictions,
        probabilities,
        metrics,
    )

    print("\n[고액 운행 분류 결과]")
    print(f"유효 데이터: {valid_rows:,}행")
    print(f"모델링 표본: {len(sampled):,}행")
    print(f"고액 운행 비율: {target.mean():.2%}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"모델 저장: {MODEL_PATH}")
    print(f"요금 구성 검증 그래프 저장: {fare_chart_path}")
    print(f"모델 평가 그래프 저장: {evaluation_path}")
    print(f"평가 지표 저장: {METRICS_PATH}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Taxi 고액 운행 분류 모델 학습")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"층화 표본 크기 (기본값: {DEFAULT_SAMPLE_SIZE:,})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_modeling(args.sample_size)
