"""전처리된 NYC Taxi 데이터의 기술통계·상관·t-test를 저장한다."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

from src.preprocessing import TARGET_COLUMN


plt.switch_backend("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATISTICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "statistical_results.json"
CORRELATION_CHART_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "correlation_heatmap.png"
)

STATISTICAL_COLUMNS = [
    "duration_minutes",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    TARGET_COLUMN,
]


def run_statistical_analysis(dataframe: pd.DataFrame) -> dict[str, object]:
    """기술통계·상관계수와 신용카드/현금 거리 차이 검정을 수행한다."""
    missing = set(STATISTICAL_COLUMNS + ["payment_type"]).difference(dataframe.columns)
    if missing:
        raise ValueError(f"통계분석 필수 컬럼이 없습니다: {sorted(missing)}")

    statistical_data = dataframe[STATISTICAL_COLUMNS]
    descriptive = statistical_data.describe().round(4)
    correlation = statistical_data.corr().round(4)

    credit_distance = dataframe.loc[dataframe["payment_type"] == 1, "trip_distance"]
    cash_distance = dataframe.loc[dataframe["payment_type"] == 2, "trip_distance"]
    if credit_distance.empty or cash_distance.empty:
        raise ValueError("t-test에 필요한 신용카드 또는 현금 결제 표본이 없습니다.")

    t_statistic, p_value = ttest_ind(
        credit_distance,
        cash_distance,
        equal_var=False,
        nan_policy="omit",
    )
    pooled_variance = (
        (len(credit_distance) - 1) * credit_distance.var(ddof=1)
        + (len(cash_distance) - 1) * cash_distance.var(ddof=1)
    ) / (len(credit_distance) + len(cash_distance) - 2)
    cohens_d = (credit_distance.mean() - cash_distance.mean()) / pooled_variance**0.5

    result: dict[str, object] = {
        "descriptive_statistics": descriptive.to_dict(),
        "correlation": correlation.to_dict(),
        "welch_ttest": {
            "question": "신용카드와 현금 결제의 평균 이동거리가 같은가?",
            "group_1": "Credit card",
            "group_2": "Cash",
            "group_1_rows": len(credit_distance),
            "group_2_rows": len(cash_distance),
            "group_1_mean_miles": float(credit_distance.mean()),
            "group_2_mean_miles": float(cash_distance.mean()),
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant_at_0_05": bool(p_value < 0.05),
            "interpretation": (
                "p < 0.05이므로 두 결제 그룹의 평균 이동거리 차이는 "
                "통계적으로 유의하다. 다만 효과크기와 인과관계는 별도로 해석한다."
            ),
        },
    }

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("NYC Yellow Taxi — Correlation Heatmap")
    fig.tight_layout()
    CORRELATION_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CORRELATION_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    STATISTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATISTICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
