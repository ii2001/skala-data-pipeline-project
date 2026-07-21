"""noh_yeongo 통계 분석: 기술통계 + 상관계수 + t-test.

과제 요구사항(통계 분석) 대응:
- 기술통계(평균·표준편차·분위수) 산출
- 변수 간 상관계수 계산(히트맵 저장)
- scipy.stats.ttest_ind 로 t-test 수행 및 p-value 해석

실행: python -m experiments.noh_yeongo.statistical_analysis
출력: reports/experiments/noh_yeongo/figures/corr_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from experiments.noh_yeongo.preprocessing import prepare_analysis_frame
from src.common.data_loader import load_taxi_pandas

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "experiments" / "noh_yeongo" / "figures"

# 통계 분석 대상 핵심 수치형 컬럼
NUM_COLS = ["trip_distance", "duration_min", "fare_amount", "tip_amount",
            "surcharge_total", "tolls_amount", "total_amount"]


def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """기술통계: 평균·표준편차·사분위수."""
    desc = df[NUM_COLS].describe().T[["mean", "std", "25%", "50%", "75%"]].round(2)
    print("=" * 62)
    print("[1] 기술통계 (평균 / 표준편차 / 분위수)")
    print("=" * 62)
    print(desc.to_string())
    return desc


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """피어슨 상관계수 행렬 + 히트맵 저장."""
    corr = df[NUM_COLS].corr().round(3)
    print("\n" + "=" * 62)
    print("[2] 상관계수 (Pearson)")
    print("=" * 62)
    print(corr.to_string())

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("요금 관련 변수 간 상관계수 히트맵")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "corr_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return corr


def ttest(df: pd.DataFrame, mask_a, mask_b, name_a: str, name_b: str,
          col: str = "total_amount") -> None:
    """두 집단의 col 평균 차이를 Welch t-test(등분산 미가정)로 검정한다."""
    a, b = df.loc[mask_a, col], df.loc[mask_b, col]
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    print(f"\n  {name_a} 평균 ${a.mean():.2f} (n={len(a):,}) vs "
          f"{name_b} 평균 ${b.mean():.2f} (n={len(b):,})")
    print(f"  t = {t_stat:.2f}, p-value = {p_value:.3e}")
    # p-value 해석: p < 0.05 → 귀무가설(두 평균이 같다) 기각
    print("  해석: p < 0.05 → 두 집단의 평균 차이는 통계적으로 유의함"
          if p_value < 0.05 else
          "  해석: p >= 0.05 → 평균 차이가 유의하다고 볼 수 없음")


if __name__ == "__main__":
    frame = prepare_analysis_frame(load_taxi_pandas())

    descriptive(frame)
    correlation(frame)

    print("\n" + "=" * 62)
    print("[3] t-test (독립표본, Welch)")
    print("=" * 62)
    print("\n(3-1) 주말 vs 평일 총요금")
    ttest(frame, frame.is_weekend == 0, frame.is_weekend == 1, "평일", "주말")
    print("\n(3-2) 공항 운행 vs 일반 운행 총요금 (구역 ID 기반)")
    ttest(frame, frame.is_airport == 1, frame.is_airport == 0, "공항", "일반")
