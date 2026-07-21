"""전처리·통계·모델 비교를 실행하고 최종 Markdown 보고서를 생성한다."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
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
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    HIGH_FARE_THRESHOLD,
    HOLDOUT_START,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    prepare_modeling_data,
)
from src.statistical_analysis import run_statistical_analysis


plt.switch_backend("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODEL_PATH = PROJECT_ROOT / "models" / "high_fare_pipeline.joblib"
REPORT_PATH = REPORTS_DIR / "report.md"
SUMMARY_PATH = METRICS_DIR / "model_summary.json"
AUDIT_PATH = METRICS_DIR / "preprocessing_audit.csv"
EXPERIMENT_PATH = METRICS_DIR / "experiment_results.csv"
FARE_CHART_PATH = FIGURE_DIR / "fare_composition_analysis.png"
PREPROCESSING_CHART_PATH = FIGURE_DIR / "preprocessing_audit.png"
EXPERIMENT_CHART_PATH = FIGURE_DIR / "experiment_comparison.png"
EVALUATION_CHART_PATH = FIGURE_DIR / "model_evaluation.png"

RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 500_000

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


def analyze_fare_composition(
    dataframe: pd.DataFrame,
) -> tuple[Path, dict[str, float | int]]:
    """구성요소 합계와 기록된 total_amount의 오차를 분석·시각화한다."""
    required = {*FARE_COMPONENTS, "total_amount", "payment_type"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"요금 검증 필수 컬럼이 없습니다: {sorted(missing)}")

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
    residual_range = residual.between(residual.quantile(0.01), residual.quantile(0.99))
    residual_source = residual[residual_range]
    residual_sample = residual_source.sample(
        n=min(100_000, len(residual_source)),
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

    summary: dict[str, float | int] = {
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
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FARE_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return FARE_CHART_PATH, summary


def stratified_limit(dataframe: pd.DataFrame, row_limit: int) -> pd.DataFrame:
    """라벨 비율을 유지하면서 지정 행 수로 제한한다."""
    if len(dataframe) <= row_limit:
        return dataframe
    sampled, _ = train_test_split(
        dataframe,
        train_size=row_limit,
        stratify=dataframe[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return sampled


def temporal_split_and_sample(
    dataframe: pd.DataFrame, sample_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    """5월 마지막 7일을 홀드아웃으로 두고 각 기간을 층화 표본 추출한다."""
    if sample_size < 10:
        raise ValueError("sample_size는 10 이상이어야 합니다.")

    train_pool = dataframe.loc[dataframe["tpep_pickup_datetime"] < HOLDOUT_START]
    test_pool = dataframe.loc[dataframe["tpep_pickup_datetime"] >= HOLDOUT_START]
    if train_pool.empty or test_pool.empty:
        raise ValueError("시간 기반 학습 또는 테스트 데이터가 비어 있습니다.")

    train_limit = int(sample_size * 0.8)
    test_limit = sample_size - train_limit
    train = stratified_limit(train_pool, train_limit)
    test = stratified_limit(test_pool, test_limit)
    split_summary: dict[str, int | float | str] = {
        "method": "temporal holdout",
        "holdout_start": HOLDOUT_START.strftime("%Y-%m-%d"),
        "train_pool_rows": len(train_pool),
        "test_pool_rows": len(test_pool),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rate": float(train[TARGET_COLUMN].mean()),
        "test_positive_rate": float(test[TARGET_COLUMN].mean()),
    }
    return train, test, split_summary


def build_preprocessor() -> ColumnTransformer:
    """훈련 데이터에서만 결측 대체·표준화·One-Hot을 학습한다."""
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
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_experiment_pipelines() -> dict[str, Pipeline]:
    """동일 전처리로 비교할 기준·일반·불균형 보정 모델을 만든다."""
    common_logistic = {
        "max_iter": 300,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
        "tol": 1e-3,
    }
    classifiers = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_unbalanced": LogisticRegression(**common_logistic),
        "logistic_balanced": LogisticRegression(
            class_weight="balanced",
            **common_logistic,
        ),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def calculate_metrics(
    y_true: pd.Series, predictions: Any, probabilities: Any
) -> dict[str, float]:
    """분류 모델의 평가 지표를 계산한다."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def run_experiments(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, str, Pipeline, Any, Any]:
    """세 모델을 같은 시간 홀드아웃에서 평가하고 F1 최적 모델을 선택한다."""
    x_train = train[MODEL_FEATURES]
    y_train = train[TARGET_COLUMN]
    x_test = test[MODEL_FEATURES]
    y_test = test[TARGET_COLUMN]
    results: list[dict[str, float | str]] = []
    fitted: dict[str, tuple[Pipeline, Any, Any]] = {}

    for name, pipeline in build_experiment_pipelines().items():
        fit_started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predict_seconds = time.perf_counter() - predict_started
        metrics = calculate_metrics(y_test, predictions, probabilities)
        results.append(
            {
                "experiment": name,
                **metrics,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        fitted[name] = (pipeline, predictions, probabilities)

    experiment_results = pd.DataFrame(results)
    candidates = experiment_results[
        experiment_results["experiment"].str.startswith("logistic")
    ]
    best_name = str(candidates.sort_values("f1", ascending=False).iloc[0]["experiment"])
    best_pipeline, predictions, probabilities = fitted[best_name]
    return experiment_results, best_name, best_pipeline, predictions, probabilities


def create_preprocessing_chart(
    audit: pd.DataFrame,
    split_summary: dict[str, int | float | str],
) -> Path:
    """단계별 제거 행과 시간 분할 라벨 비율을 시각화한다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(
        data=audit,
        x="removed_rows",
        y="step",
        color="#E45756",
        ax=axes[0],
    )
    axes[0].set(
        title="Rows Removed by Preprocessing Step",
        xlabel="Removed rows",
        ylabel="Step",
    )
    axes[0].bar_label(axes[0].containers[0], fmt="{:,.0f}", padding=3)

    split_rates = pd.DataFrame(
        {
            "period": ["Train: May 1–24", "Test: May 25–31"],
            "high_fare_rate": [
                float(split_summary["train_positive_rate"]) * 100,
                float(split_summary["test_positive_rate"]) * 100,
            ],
        }
    )
    sns.barplot(
        data=split_rates,
        x="period",
        y="high_fare_rate",
        hue="period",
        legend=False,
        ax=axes[1],
    )
    axes[1].set(
        title="High-Fare Rate by Temporal Split",
        xlabel="Period",
        ylabel="High-fare rate (%)",
        ylim=(0, split_rates["high_fare_rate"].max() * 1.25),
    )
    axes[1].bar_label(axes[1].containers[0], fmt="%.1f%%", padding=3)
    fig.suptitle("NYC Yellow Taxi — Preprocessing Audit", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREPROCESSING_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return PREPROCESSING_CHART_PATH


def create_experiment_chart(experiments: pd.DataFrame) -> Path:
    """모델별 주요 지표를 비교하는 그래프를 저장한다."""
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_data = experiments.melt(
        id_vars="experiment",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=plot_data, x="metric", y="score", hue="experiment", ax=ax)
    ax.set(
        title="Model Experiment Comparison — Temporal Holdout",
        xlabel="Metric",
        ylabel="Score",
        ylim=(0, 1),
    )
    ax.legend(title="Experiment", loc="lower right")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXPERIMENT_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return EXPERIMENT_CHART_PATH


def create_evaluation_chart(
    y_true: pd.Series,
    predictions: Any,
    probabilities: Any,
    metrics: dict[str, float],
    model_name: str,
) -> Path:
    """최종 모델의 지표·혼동행렬·ROC·PR 곡선을 저장한다."""
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
        f"NYC Yellow Taxi — Final Model ({model_name})",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EVALUATION_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return EVALUATION_CHART_PATH


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """추가 의존성 없이 Markdown 표 문자열을 만든다."""
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def generate_report(
    preprocessing_summary: dict[str, int | float | str],
    audit: pd.DataFrame,
    split_summary: dict[str, int | float | str],
    fare_summary: dict[str, float | int],
    statistics: dict[str, object],
    experiments: pd.DataFrame,
    best_name: str,
) -> Path:
    """실험 방법·결과·해석·의견을 담은 report.md를 자동 생성한다."""
    best = experiments.loc[experiments["experiment"] == best_name].iloc[0]
    audit_rows = [
        [
            str(row.step),
            str(row.rule),
            f"{int(row.before_rows):,}",
            f"{int(row.removed_rows):,}",
            f"{int(row.after_rows):,}",
        ]
        for row in audit.itertuples(index=False)
    ]
    experiment_rows = [
        [
            str(row.experiment),
            f"{row.accuracy:.4f}",
            f"{row.precision:.4f}",
            f"{row.recall:.4f}",
            f"{row.f1:.4f}",
            f"{row.roc_auc:.4f}",
            f"{row.fit_seconds:.2f}s",
        ]
        for row in experiments.itertuples(index=False)
    ]
    ttest = statistics["welch_ttest"]
    if not isinstance(ttest, dict):
        raise TypeError("t-test 결과 형식이 올바르지 않습니다.")
    p_value = float(ttest["p_value"])
    p_display = f"{p_value:.3e}" if p_value > 0 else "< 1e-300"

    report = f"""# NYC Yellow Taxi 고액 운행 예측 분석 보고서

## 1. 분석 목적

2026년 5월 NYC Yellow Taxi 기록에서 `total_amount >= $30`을 고액 운행으로
정의하고, 승차 시점에 알 수 있는 시간·지역·공급업체·승객 수 정보로 이를
분류한다. Accuracy뿐 아니라 불균형에 민감한 F1과 ROC-AUC를 함께 평가한다.

## 2. 데이터와 라벨

- 원본: {int(preprocessing_summary['raw_rows']):,}행 × 20열
- 전처리 후: {int(preprocessing_summary['valid_rows']):,}행
- 보존율: {float(preprocessing_summary['retention_rate']):.2%}
- 고액 운행 비율: {float(preprocessing_summary['positive_label_rate']):.2%}
- 라벨: `high_fare = 1 if total_amount >= 30 else 0`

요금 구성요소를 결측 0으로 합산했을 때 기록된 총액과 1센트 이내 일치율은
{float(fare_summary['exact_match_rate']):.2%}, 평균 절대 오차는
${float(fare_summary['mean_absolute_error']):.2f}였다. 따라서 구성요소 합계로
라벨을 재생성하지 않고 기록된 `total_amount`를 사용했다.

![요금 구성 검증](figures/fare_composition_analysis.png)

## 3. 전처리 실험

{markdown_table(['단계', '규칙', '이전 행', '제거 행', '이후 행'], audit_rows)}

승객 수는 품질 필터 후 기존 결측 {int(preprocessing_summary['original_missing_passenger_rows']):,}건에
0명 또는 7명 이상인 {int(preprocessing_summary['invalid_passenger_rows_reclassified']):,}건을
추가로 결측 처리했다. Pipeline 내부에서 **훈련 데이터 중앙값**으로 대체하고
결측 indicator를 추가해 값 대체와 결측 패턴을 구분했다.

시간과 요일은 경계가 이어지는 주기형 특성을 반영하기 위해 sin/cos로 변환했다.
`total_amount`와 9개 요금 구성요소, 거리, 하차시간, 결제수단은 정답 누수를
막기 위해 모델 입력에서 제외했다.

![전처리 감사](figures/preprocessing_audit.png)

## 4. 통계 분석

Welch t-test로 신용카드와 현금 결제의 평균 이동거리를 비교했다.

- 신용카드 평균: {float(ttest['group_1_mean_miles']):.3f}마일
- 현금 평균: {float(ttest['group_2_mean_miles']):.3f}마일
- t 통계량: {float(ttest['t_statistic']):.3f}
- p-value: {p_display}
- Cohen's d: {float(ttest['cohens_d']):.3f}

p < 0.05이므로 평균 차이는 통계적으로 유의하지만, 대규모 표본에서는 작은
차이도 유의해질 수 있으므로 효과크기와 인과관계를 별도로 고려해야 한다.

![상관계수](figures/correlation_heatmap.png)

## 5. 실험 설계

랜덤 분할 대신 {split_summary['holdout_start']} 이전을 학습 후보, 이후를
테스트 후보로 분리했다. 시간 순서를 보존한 상태에서 라벨 비율을 유지해
학습 {int(split_summary['train_rows']):,}행, 테스트
{int(split_summary['test_rows']):,}행을 표본 추출했다.

비교 모델은 다수 클래스 기준선, 일반 Logistic Regression, `class_weight`로
불균형을 보정한 Logistic Regression이다. 모든 Logistic 모델은 동일한
ColumnTransformer와 시간 홀드아웃을 사용했다.

{markdown_table(['실험', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', '학습시간'], experiment_rows)}

![모델 비교](figures/experiment_comparison.png)

## 6. 최종 모델과 결과

F1이 가장 높은 `{best_name}`를 최종 모델로 선택했다.

- Accuracy: {best.accuracy:.4f}
- Precision: {best.precision:.4f}
- Recall: {best.recall:.4f}
- F1: {best.f1:.4f}
- ROC-AUC: {best.roc_auc:.4f}

![최종 모델 평가](figures/model_evaluation.png)

## 7. 모델에 대한 의견과 한계

기준 모델과 비교해 실제 분류 신호가 있음을 확인했으며, 지역과 시간 정보만으로
고액 운행 가능성을 어느 정도 구분할 수 있었다. 구성요소 합계를 입력하면 높은
점수를 쉽게 얻을 수 있지만 이는 예측이 아니라 정답 공식의 재현이므로 제외한
현재 결과가 더 정직한 성능이라고 판단한다.

다만 한 달 데이터의 50만 표본만 학습했고, 목적지가 승차 시점에 알려진다는
가정으로 `DOLocationID`를 사용했다. 계절 변화와 목적지 미확정 상황에는 성능이
달라질 수 있다. 또한 공식 최대요금 근거가 없어 500달러 초과 28행을 임의로
삭제하지 않았으며, 이진 라벨 특성상 금액 크기는 학습 입력에 포함되지 않는다.
다음 개선은 여러 달 시간 홀드아웃, 임계값 튜닝, 트리 기반 모델 비교 순으로
진행하는 것이 적절하다.

## 8. 재현 방법

```bash
python -m src.eda
python -m src.model
```

수치 결과는 `reports/metrics/`, 그림은 `reports/figures/`, 학습 모델은
`models/high_fare_pipeline.joblib`에 저장된다.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH


def save_structured_results(
    pipeline: Pipeline,
    preprocessing_summary: dict[str, int | float | str],
    audit: pd.DataFrame,
    split_summary: dict[str, int | float | str],
    fare_summary: dict[str, float | int],
    experiments: pd.DataFrame,
    best_name: str,
) -> None:
    """모델·감사표·실험표·요약 JSON을 지정 디렉터리에 저장한다."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    audit.to_csv(AUDIT_PATH, index=False)
    experiments.to_csv(EXPERIMENT_PATH, index=False)

    summary = {
        "target": f"total_amount >= {HIGH_FARE_THRESHOLD}",
        "selected_model": best_name,
        "preprocessing": preprocessing_summary,
        "split": split_summary,
        "fare_composition": fare_summary,
        "experiments": json.loads(experiments.to_json(orient="records")),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_modeling(sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict[str, float]:
    """전체 분석을 실행하고 모든 결과와 report.md를 생성한다."""
    raw = load_taxi_pandas()
    _, fare_summary = analyze_fare_composition(raw)
    prepared, audit, preprocessing_summary = prepare_modeling_data(raw)
    del raw
    gc.collect()

    statistics = run_statistical_analysis(prepared)
    train, test, split_summary = temporal_split_and_sample(prepared, sample_size)
    del prepared
    gc.collect()

    experiments, best_name, pipeline, predictions, probabilities = run_experiments(
        train,
        test,
    )
    best_row = experiments.loc[experiments["experiment"] == best_name].iloc[0]
    best_metrics = {
        name: float(best_row[name])
        for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    create_preprocessing_chart(audit, split_summary)
    create_experiment_chart(experiments)
    create_evaluation_chart(
        test[TARGET_COLUMN],
        predictions,
        probabilities,
        best_metrics,
        best_name,
    )
    save_structured_results(
        pipeline,
        preprocessing_summary,
        audit,
        split_summary,
        fare_summary,
        experiments,
        best_name,
    )
    report_path = generate_report(
        preprocessing_summary,
        audit,
        split_summary,
        fare_summary,
        statistics,
        experiments,
        best_name,
    )

    print("\n[최종 실험 결과]")
    print(experiments.round(4).to_string(index=False))
    print(f"선택 모델: {best_name}")
    print(f"보고서: {report_path}")
    print(f"수치 결과 디렉터리: {METRICS_DIR}")
    print(f"그래프 디렉터리: {FIGURE_DIR}")
    print(f"모델: {MODEL_PATH}")
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Taxi 전체 모델 실험 및 보고서 생성")
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
