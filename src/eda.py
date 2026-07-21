"""NYC Yellow Taxi 데이터의 기본 EDA와 시각화 결과를 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.data_loader import load_taxi_pandas, load_taxi_polars


plt.switch_backend("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
INTERACTIVE_DIR = PROJECT_ROOT / "reports" / "interactive"
STATIC_CHART_PATH = FIGURE_DIR / "eda_overview.png"
INTERACTIVE_CHART_PATH = INTERACTIVE_DIR / "hourly_trips.html"
INTERACTIVE_CHART_TITLE = "NYC Yellow Taxi Trips by Pickup Hour — May 2026"
TRIP_COUNT_LABEL = "Number of trips"

REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
}

PAYMENT_LABELS = {
    0: "Flex Fare",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}


def validate_columns(dataframe: pd.DataFrame) -> None:
    """분석에 필요한 컬럼이 모두 존재하는지 확인한다."""
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")


def load_and_compare(refresh: bool = False) -> pd.DataFrame:
    """동일 원본을 Pandas와 Polars로 읽고 구조가 같은지 검증한다."""
    pandas_df = load_taxi_pandas(refresh)
    polars_df = load_taxi_polars()

    if pandas_df.shape != polars_df.shape:
        raise ValueError(
            f"라이브러리별 shape 불일치: {pandas_df.shape} != {polars_df.shape}"
        )
    if pandas_df.columns.tolist() != polars_df.columns:
        raise ValueError("Pandas와 Polars의 컬럼 순서가 일치하지 않습니다.")

    pandas_mb = pandas_df.memory_usage(deep=True).sum() / 1024**2
    polars_mb = polars_df.estimated_size("mb")
    schema = pd.DataFrame(
        {
            "Pandas dtype": pandas_df.dtypes.astype(str),
            "Polars dtype": [str(dtype) for dtype in polars_df.dtypes],
        },
        index=pandas_df.columns,
    )
    print("\n[Pandas · Polars 로드 결과 비교]")
    print(f"공통 shape: {pandas_df.shape}")
    print(f"Pandas 메모리 사용량: {pandas_mb:,.1f} MiB")
    print(f"Polars 추정 메모리 사용량: {polars_mb:,.1f} MiB")
    print("\n[라이브러리별 컬럼 타입]")
    print(schema.to_string())
    return pandas_df


def clean_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """중복을 제거하고 결측값을 명시적인 미상 값으로 변환한다."""
    validate_columns(dataframe)
    duplicate_count = int(dataframe.duplicated().sum())
    cleaned = dataframe.drop_duplicates().copy()

    # 대체한 값이 실제 관측값과 구분되도록 결측 여부를 별도 보존한다.
    missing_indicator_columns = (
        "passenger_count",
        "RatecodeID",
        "congestion_surcharge",
        "Airport_fee",
    )
    for column in missing_indicator_columns:
        cleaned[f"{column}_was_missing"] = cleaned[column].isna()

    cleaned = cleaned.fillna(
        {
            "passenger_count": -1,
            "RatecodeID": 99,
            "store_and_fwd_flag": "Unknown",
            "congestion_surcharge": 0.0,
            "Airport_fee": 0.0,
        }
    )

    remaining_nulls = int(cleaned.isna().sum().sum())
    if remaining_nulls:
        raise ValueError(f"처리되지 않은 결측값이 {remaining_nulls:,}개 있습니다.")

    stats = {
        "rows_before": len(dataframe),
        "rows_after": len(cleaned),
        "duplicates_removed": duplicate_count,
    }
    return cleaned, stats


def select_analysis_period(dataframe: pd.DataFrame) -> pd.DataFrame:
    """파일명과 일치하는 2026년 5월 승차 데이터만 선택한다."""
    pickup = dataframe["tpep_pickup_datetime"]
    mask = pickup.between("2026-05-01", "2026-06-01", inclusive="left")
    selected = dataframe.loc[mask].copy()
    if selected.empty:
        raise ValueError("2026년 5월 승차 데이터가 없습니다.")
    return selected


def build_eda_tables(
    dataframe: pd.DataFrame, missing_rate: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """그래프에 사용할 시간대·결제수단·결측률 집계표를 만든다."""
    dataframe["pickup_hour"] = dataframe["tpep_pickup_datetime"].dt.hour

    hourly = (
        dataframe.groupby("pickup_hour", as_index=False)
        .agg(
            trip_count=("pickup_hour", "size"),
            average_total_amount=("total_amount", "mean"),
            average_trip_distance=("trip_distance", "mean"),
        )
        .sort_values("pickup_hour")
    )

    payment = (
        dataframe["payment_type"]
        .value_counts()
        .rename_axis("payment_type")
        .reset_index(name="trip_count")
    )
    payment["payment_method"] = payment["payment_type"].map(PAYMENT_LABELS).fillna(
        "Other"
    )

    missing = (
        missing_rate[missing_rate > 0]
        .sort_values(ascending=False)
        .rename("missing_rate")
        .rename_axis("column")
        .reset_index()
    )
    return hourly, payment, missing


def create_static_chart(
    dataframe: pd.DataFrame,
    hourly: pd.DataFrame,
    payment: pd.DataFrame,
    missing: pd.DataFrame,
) -> Path:
    """Seaborn을 이용해 2×2 정적 EDA 대시보드를 저장한다."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.lineplot(data=hourly, x="pickup_hour", y="trip_count", marker="o", ax=axes[0, 0])
    axes[0, 0].set(
        title="Trips by Pickup Hour",
        xlabel="Pickup hour",
        ylabel=TRIP_COUNT_LABEL,
    )
    axes[0, 0].set_xticks(range(0, 24, 2))

    positive_distance = dataframe.loc[dataframe["trip_distance"] > 0, "trip_distance"]
    upper_limit = positive_distance.quantile(0.99)
    distance_sample = positive_distance[positive_distance <= upper_limit].sample(
        n=min(100_000, len(positive_distance)), random_state=42
    )
    sns.histplot(distance_sample, bins=50, kde=True, ax=axes[0, 1], color="#F58518")
    axes[0, 1].set(
        title="Trip Distance Distribution (up to 99th percentile)",
        xlabel="Trip distance (miles)",
        ylabel="Sample count",
    )

    sns.barplot(
        data=payment,
        x="trip_count",
        y="payment_method",
        hue="payment_method",
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set(
        title="Trips by Payment Method",
        xlabel=TRIP_COUNT_LABEL,
        ylabel="Payment method",
    )

    sns.barplot(
        data=missing,
        x="missing_rate",
        y="column",
        color="#E45756",
        ax=axes[1, 1],
    )
    axes[1, 1].set(
        title="Missing Values Before Cleaning",
        xlabel="Missing rate (%)",
        ylabel="Column",
    )
    axes[1, 1].bar_label(axes[1, 1].containers[0], fmt="%.1f%%", padding=3)
    axes[1, 1].set_xlim(0, missing["missing_rate"].max() * 1.15)

    fig.suptitle("NYC Yellow Taxi — May 2026 EDA Overview", fontsize=18, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(STATIC_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return STATIC_CHART_PATH


def create_interactive_chart(hourly: pd.DataFrame) -> Path:
    """Plotly로 시간대별 운행량 인터랙티브 차트를 저장한다."""
    fig = px.bar(
        hourly,
        x="pickup_hour",
        y="trip_count",
        color="average_total_amount",
        custom_data=["average_trip_distance"],
        title=INTERACTIVE_CHART_TITLE,
        labels={
            "pickup_hour": "Pickup hour",
            "trip_count": TRIP_COUNT_LABEL,
            "average_total_amount": "Average total amount ($)",
        },
        color_continuous_scale="Viridis",
    )
    fig.update_traces(
        hovertemplate=(
            "Pickup hour: %{x}<br>Trips: %{y:,}<br>"
            "Average distance: %{customdata[0]:.2f} miles<extra></extra>"
        )
    )
    fig.update_layout(xaxis={"dtick": 1})
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    chart_fragment = fig.to_html(full_html=False, include_plotlyjs=True)
    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{INTERACTIVE_CHART_TITLE}</title>
</head>
<body>
{chart_fragment}
</body>
</html>
"""
    INTERACTIVE_CHART_PATH.write_text(html_document, encoding="utf-8")
    return INTERACTIVE_CHART_PATH


def print_eda_summary(
    dataframe: pd.DataFrame, stats: dict[str, int], analysis_rows: int
) -> None:
    """기본 기술 요약과 정제 결과를 콘솔에 출력한다."""
    numeric_columns = [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]
    summary = dataframe[numeric_columns].copy()
    summary["passenger_count"] = summary["passenger_count"].mask(
        summary["passenger_count"] == -1
    )

    print("\n[결측치 · 중복 처리]")
    print(f"처리 전 행 수: {stats['rows_before']:,}")
    print(f"중복 제거: {stats['duplicates_removed']:,}")
    print(f"처리 후 행 수: {stats['rows_after']:,}")
    print(f"2026년 5월 분석 행 수: {analysis_rows:,}")
    print(f"처리 후 전체 결측값: {int(dataframe.isna().sum().sum()):,}")
    print("\n[주요 수치형 컬럼 기술통계]")
    print(summary.describe().round(2).to_string())


def run_eda(refresh: bool = False) -> None:
    raw = load_and_compare(refresh)
    missing_rate = raw.isna().mean().mul(100)
    cleaned, stats = clean_data(raw)
    analysis = select_analysis_period(cleaned)
    hourly, payment, missing = build_eda_tables(analysis, missing_rate)

    static_path = create_static_chart(analysis, hourly, payment, missing)
    interactive_path = create_interactive_chart(hourly)
    print_eda_summary(cleaned, stats, len(analysis))
    print(f"\nSeaborn 차트 저장: {static_path}")
    print(f"Plotly 차트 저장: {interactive_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Yellow Taxi 기본 EDA 실행")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="캐시가 있어도 원본 데이터를 다시 다운로드한다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(args.refresh)
