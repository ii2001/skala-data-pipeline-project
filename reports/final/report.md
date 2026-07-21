# NYC Yellow Taxi total_amount 예측 최종 보고서

## 1. 프로젝트 개요

NYC Yellow Taxi 2026년 5월 운행 데이터를 이용해 기본 EDA, 시각화, 통계 분석,
회귀 Pipeline 학습을 하나의 실행 코드로 자동화했다. 팀 비교 후
`experiments/Lee_hyeonjun/model.py`의 LightGBM 접근을 최종 모델로 선정하고,
절대경로·누수·Pipeline·저장 문제를 최종 코드에서 보완했다.

## 2. 데이터 준비와 Pandas·Polars 비교

- 원본: 4,090,836행 × 20열
- Pandas 로딩: 0.2089초
- Polars 로딩: 0.1205초
- shape·컬럼·결측 건수 일치: Y
- 전체 중복 행: 0건
- 원본 결측: {"passenger_count": 955371, "RatecodeID": 955371, "store_and_fwd_flag": 955371, "congestion_surcharge": 955371, "Airport_fee": 955371}

단일 장비에서 한 번 측정한 로딩 시간은 환경과 캐시의 영향을 받으므로 특정
라이브러리가 항상 빠르다고 일반화하지 않는다. 두 구현은 각각 동일 원본을 읽고
shape, 컬럼 순서, 결측 건수를 독립적으로 계산해 결과가 같은지 검증했다.

결측치는 Pipeline 안에서 `passenger_count=0`, `RatecodeID=99`,
`store_and_fwd_flag=Unknown`으로 처리한다. 직접 요금 구성 9개 컬럼과
`VendorID`는 모델 입력에서 제외했다.

## 3. EDA와 시각화

- Seaborn 정적 차트: `figures/eda_overview.png`
- Plotly 인터랙티브 차트: `interactive/hourly_total_amount.html`
- 정적 차트 표본: 100,000행(고정 시드 42)
- 표본의 trip_distance–total_amount 상관: 0.8012

거리 분포는 오른쪽 꼬리가 길며, 잘못 기록된 극단값의 영향을 줄이기 위해
100마일 상한과 별도 indicator를 적용했다. 인터랙티브 차트에서는 시간대별 평균과
중앙값을 함께 표시해 일부 고액 운행이 평균에 미치는 영향을 비교할 수 있다.

## 4. 기술통계와 상관 분석

| 변수 | 평균 | 표준편차 | 25% | 중앙값 | 75% |
|---|---:|---:|---:|---:|---:|
| passenger_count | 0.9516 | 0.7663 | 1.0000 | 1.0000 | 1.0000 |
| trip_distance | 3.4171 | 4.2624 | 1.0400 | 1.8800 | 3.8200 |
| trip_duration_minutes | 18.5469 | 15.7068 | 8.6000 | 14.5000 | 23.2833 |
| total_amount | 30.7040 | 22.6465 | 17.7000 | 23.9400 | 34.9800 |

전처리 범위에서 trip_distance와 total_amount의 Pearson 상관계수는
0.7845이다. 상관관계는 선형 연관성을
보여줄 뿐 인과관계를 의미하지 않는다. 전체 상관행렬은 `correlation.csv`에 저장했다.

## 5. Welch t-test

- 비교: 신용카드(2,727,210행) vs 현금(368,965행)
- 평균 total_amount: $30.6586 vs $26.1149
- t 통계량: 108.7091
- p-value: 0.000000e+00
- Cohen's d: 0.1948

p-value가 0.05보다 작아 신용카드와 현금 결제의 평균 total_amount가 같다는 귀무가설을 기각한다. 표본이 매우 크므로 효과크기도 함께 본다.

## 6. ML Pipeline과 평가

선정 모델은 `ColumnTransformer → LightGBMRegressor`를 하나의 sklearn
`Pipeline`으로 구성했다. 숫자형 결측 처리, 명목형 코드 원핫 인코딩,
`store_and_fwd_flag` 축약 원핫 인코딩을 학습 데이터에만 fit한다.

- 학습: 3,260,221행
- 테스트: 815,056행
- MAE: 3.4002
- RMSE: 7.7800
- Median AE: 1.5451
- R²: 0.8754
- 테스트 제외 행: 0행
- 저장 모델 재로딩·재예측 검증: 성공

### 실제 금액 구간별 평가

| 실제 금액 구간 | 행 수 | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| under_30 | 541,434 | 2.0137 | 3.8190 | 0.4764 |
| 30_to_60 | 201,498 | 4.4965 | 6.8343 | 0.2131 |
| 60_to_100 | 56,518 | 8.6498 | 13.2332 | -0.2851 |
| 100_plus | 15,606 | 18.3387 | 37.6560 | 0.2050 |

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
