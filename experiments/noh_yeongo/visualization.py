"""noh_yeongo 시각화: Seaborn 정적 2x2 + Plotly 인터랙티브.

과제 요구사항(시각화) 대응:
- Seaborn 정적 차트(분포·그룹 비교), Plotly 인터랙티브 차트 각 1개 이상
- 모든 차트에 제목·축 레이블 포함

실행: python -m experiments.noh_yeongo.visualization
출력: reports/experiments/noh_yeongo/figures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # 창 없이 파일 저장 전용
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from experiments.noh_yeongo.preprocessing import prepare_analysis_frame
from src.common.data_loader import load_taxi_pandas

# macOS 한글 폰트 설정(제목·레이블 깨짐 방지)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "experiments" / "noh_yeongo" / "figures"


def seaborn_2x2(df: pd.DataFrame) -> Path:
    """총요금 분포·시간대 수요·요금 구성·공항 비교를 한 장(2x2)으로 요약한다."""
    sns.set_theme(style="whitegrid", font="AppleGothic")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"NYC Yellow Taxi 요금 분석 (2026년 5월, 정제 후 {len(df):,}건)", fontsize=15)

    # ① 총요금 분포: 중앙값 부근에 몰린 우측 꼬리 분포
    sns.histplot(df["total_amount"].clip(upper=100), bins=60, ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].axvline(df["total_amount"].median(), color="red", ls="--",
                       label=f"중앙값 ${df['total_amount'].median():.2f}")
    axes[0, 0].legend()
    axes[0, 0].set_title("① 총요금 분포")
    axes[0, 0].set_xlabel("총요금 (USD, 100 초과는 100으로 표시)")
    axes[0, 0].set_ylabel("운행 건수")

    # ② 시간대별 운행 건수: 저녁 18시 피크의 수요 곡선
    hourly = df.groupby("pickup_hour").size().reset_index(name="cnt")
    sns.barplot(data=hourly, x="pickup_hour", y="cnt", ax=axes[0, 1], color="#55A868")
    axes[0, 1].set_title("② 시간대별 운행 건수")
    axes[0, 1].set_xlabel("승차 시각 (시)")
    axes[0, 1].set_ylabel("운행 건수")

    # ③ 평균 요금 구성 해부: 승객이 낸 돈이 어디로 가는가
    comp = {"기본요금": df.fare_amount.mean(), "팁": df.tip_amount.mean(),
            "세금·할증": df.surcharge_total.mean(), "통행료": df.tolls_amount.mean()}
    comp_df = pd.DataFrame({"항목": comp.keys(), "평균 금액": comp.values()})
    sns.barplot(data=comp_df, x="항목", y="평균 금액", ax=axes[1, 0],
                hue="항목", legend=False, palette="muted")
    for i, v in enumerate(comp.values()):
        axes[1, 0].text(i, v + 0.3, f"${v:.2f}", ha="center")
    axes[1, 0].set_title(f"③ 평균 총요금 ${df.total_amount.mean():.2f}의 구성")
    axes[1, 0].set_xlabel("요금 항목")
    axes[1, 0].set_ylabel("평균 금액 (USD)")

    # ④ 그룹 비교: 공항(구역 ID 기반) vs 일반 운행 평균 총요금
    grp = df.groupby("is_airport")["total_amount"].mean().reset_index()
    grp["구분"] = grp["is_airport"].map({0: "일반", 1: "공항(EWR/JFK/LGA)"})
    sns.barplot(data=grp, x="구분", y="total_amount", ax=axes[1, 1],
                hue="구분", legend=False, palette=["#4C72B0", "#C44E52"])
    for i, v in enumerate(grp["total_amount"]):
        axes[1, 1].text(i, v + 1, f"${v:.2f}", ha="center")
    axes[1, 1].set_title("④ 공항 운행 vs 일반 운행 평균 총요금")
    axes[1, 1].set_xlabel("운행 구분")
    axes[1, 1].set_ylabel("평균 총요금 (USD)")

    plt.tight_layout()
    out = FIG_DIR / "seaborn_2x2.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plotly_demand_fare(df: pd.DataFrame) -> Path:
    """시간대별 운행 건수(막대)와 평균 총요금(선)을 이중 축으로 겹친 인터랙티브 차트."""
    cnt = df.groupby("pickup_hour").size()
    avg = df.groupby("pickup_hour")["total_amount"].mean().round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cnt.index, y=cnt.values, name="운행 건수",
                         marker_color="#55A868", opacity=0.7))
    fig.add_trace(go.Scatter(x=avg.index, y=avg.values, name="평균 총요금($)",
                             yaxis="y2", mode="lines+markers",
                             line=dict(color="#C44E52", width=3)))
    fig.update_layout(
        title="시간대별 수요와 평균 총요금 (2026년 5월) — 새벽 5시에 요금이 가장 비싸다",
        xaxis=dict(title="승차 시각 (시)", dtick=1),
        yaxis=dict(title="운행 건수"),
        yaxis2=dict(title="평균 총요금 (USD)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",              # 마우스 오버 시 두 값 동시 표시
    )
    out = FIG_DIR / "plotly_demand_fare.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare_analysis_frame(load_taxi_pandas())
    print(f"Seaborn 정적 차트 저장 : {seaborn_2x2(frame)}")
    print(f"Plotly 인터랙티브 저장 : {plotly_demand_fare(frame)}")
