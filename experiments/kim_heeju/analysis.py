"""기본 EDA 통계와 시각화 산출물을 생성한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import ttest_ind

from config import (
    FARE_COMPONENT_COLUMNS,
    OUTPUT_DIR,
    PLOT_SAMPLE_SIZE,
    RANDOM_STATE,
    SOURCE_COLUMNS,
    TARGET_COLUMN,
)


def calculate_statistics(frame: pd.DataFrame) -> dict:
    """기술통계·요금 분포와 평일/주말 기본요금 검정을 계산한다."""
    eda_columns = [
        *SOURCE_COLUMNS,
        TARGET_COLUMN,
        "pickup_hour",
        "pickup_weekday",
        "is_weekend",
    ]
    numeric = frame[eda_columns].select_dtypes(include="number")
    descriptive = numeric.describe(percentiles=[0.25, 0.5, 0.75]).round(3)
    correlation = numeric.corr(numeric_only=True).round(3)
    weekday = frame.loc[frame["is_weekend"] == 0, "fare_amount"]
    weekend = frame.loc[frame["is_weekend"] == 1, "fare_amount"]
    test = ttest_ind(weekday, weekend, equal_var=False, nan_policy="omit")
    zero_ratio = frame[FARE_COMPONENT_COLUMNS].eq(0).mean().sort_values(ascending=False)
    return {
        "descriptive": descriptive,
        "correlation": correlation,
        "fare_correlation": correlation["fare_amount"].sort_values(ascending=False),
        "zero_ratio": zero_ratio,
        "component_mean": frame[FARE_COMPONENT_COLUMNS].mean().sort_values(ascending=False),
        "ttest": {
            "weekday_count": len(weekday),
            "weekend_count": len(weekend),
            "weekday_mean": float(weekday.mean()),
            "weekend_mean": float(weekend.mean()),
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
        },
    }


def create_static_visualization(
    frame: pd.DataFrame, output_path: Path = OUTPUT_DIR / "taxi_eda.png"
) -> Path:
    """요금 분포·거리 관계·구성요소·상관관계를 한 Figure에 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = frame.sample(min(len(frame), PLOT_SAMPLE_SIZE), random_state=RANDOM_STATE)
    numeric = sample.select_dtypes(include="number")

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    sns.histplot(sample, x="fare_amount", bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set(title="Fare Amount Distribution", xlabel="Fare amount ($)")

    sns.scatterplot(
        sample, x="trip_distance", y="fare_amount", alpha=0.15, s=10, ax=axes[0, 1]
    )
    axes[0, 1].set(
        title="Trip Distance vs Fare Amount",
        xlabel="Trip distance (miles)",
        ylabel="Fare amount ($)",
    )

    component_mean = frame[FARE_COMPONENT_COLUMNS].mean().sort_values(ascending=False)
    sns.barplot(x=component_mean.values, y=component_mean.index, ax=axes[1, 0])
    axes[1, 0].set(title="Average Fare Components", xlabel="Average amount ($)", ylabel="")

    sns.heatmap(numeric.corr(numeric_only=True), cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Numeric Correlation Heatmap")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_interactive_visualization(
    frame: pd.DataFrame, output_path: Path = OUTPUT_DIR / "taxi_fare_by_hour.html"
) -> Path:
    """요일·시간대별 평균 요금 구성요소 합계를 Plotly HTML로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = frame.assign(
        predicted_total_base=frame[FARE_COMPONENT_COLUMNS].sum(axis=1, min_count=1)
    )
    grouped = (
        plot_frame.groupby(["pickup_weekday", "pickup_hour"], as_index=False)[
            "predicted_total_base"
        ]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped["weekday"] = grouped["pickup_weekday"].map(
        dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
    )
    figure = px.line(
        grouped,
        x="pickup_hour",
        y="mean",
        color="weekday",
        hover_data=["count"],
        markers=True,
        title="Average NYC Yellow Taxi Fare Components by Day and Hour",
        labels={"pickup_hour": "Pickup hour", "mean": "Average component sum ($)"},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path
