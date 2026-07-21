# Noh Yeongo — Pickup-Time Linear Regression

승차 시점에 알 수 있는 정보만으로 `total_amount`를 예측하는 선형 회귀 실험입니다.
공통 규칙(`make_common_split` + `src/common/evaluation.py`)을 사용합니다.

## 실행 방법

```bash
python -m experiments.noh_yeongo.eda                   # Pandas/Polars 비교 + 품질 진단
python -m experiments.noh_yeongo.visualization         # Seaborn 2x2 + Plotly 차트
python -m experiments.noh_yeongo.statistical_analysis  # 기술통계·상관·t-test
python -m experiments.noh_yeongo.train                 # 공통 분할 학습·평가·결과 저장
python -m experiments.noh_yeongo.item_analysis         # 요금 항목별 성능 분석
```

- 결과: `reports/experiments/noh_yeongo/metrics.json` (공통 형식)
- 차트: `reports/experiments/noh_yeongo/figures/` (seaborn_2x2.png, corr_heatmap.png, plotly_demand_fare.html)
- 모델: `experiments/noh_yeongo/artifacts/models/pickup_time_linear.joblib`

## 결과 (공통 80:20 분할, 테스트 815,056행 전체 예측)

| 지표 | 값 |
|---|---|
| MAE | $6.4650 |
| RMSE | $11.2331 |
| Median AE | $4.3036 |
| R² | 0.7403 |

- 학습 3,128,626행(품질 필터 후) / 테스트 815,056행 / 테스트 제외 0행
- 테스트에 남아 있는 미터기 오류 행(거리 30만 마일 등)도 거리 클리핑(상한 100마일)으로
  예측 폭주 없이 전 행 예측

## 전처리 기준

| 구분 | 처리 | 근거 |
|---|---|---|
| 구조적 결측 (95.5만 행) | `passenger_count`→1(최빈값), `RatecodeID`→99(미상 코드) 대체 | 결측 행 = `payment_type=0` 행과 일치하는 비미터기 수집분. 삭제 시 데이터 23.4% 손실이라 대체 선택 |
| 공항 여부 | 승·하차 구역 ID(1/132/138) 기반 플래그 | `Airport_fee`는 타깃(total)의 구성요소라 피처 재료로 쓰면 누수 → 구역 ID로 대체 |
| 품질 필터 (학습만) | fare>0, 0<거리≤100마일, 0<운행시간≤180분 | 환불(음수)·미터기 오류(거리 30만 마일, 음수 운행시간) 제외. 테스트는 공통 비교를 위해 전 행 유지 |

## 피처 (승차 시점 정보 7개)

`trip_distance`, `passenger_count`, `pickup_hour`, `pickup_weekday`,
`RatecodeID`, `is_airport`, `is_weekend`

제외: 요금 구성요소(누수), `duration_min`(하차 후 확정되는 사후 정보), `payment_type`

## 모델 선택 이유

- `LinearRegression`: 가장 단순한 기준 모델로 시작해 계수 해석이 가능하고,
  이후 실험(트리 모델·경로 피처)의 개선 폭을 측정하는 기준선 역할
- 전처리(StandardScaler + OneHotEncoder)와 모델을 단일 `Pipeline`으로 구성해
  joblib 하나로 재사용 가능

## 보고서

- 공통 형식 결과: `reports/experiments/noh_yeongo/metrics.json`
- 상세 보고서: `reports/experiments/noh_yeongo/report.md`
  (전처리 근거, 요금 구간별 성능, 누수 점검 내역, 다음 실험 계획)
