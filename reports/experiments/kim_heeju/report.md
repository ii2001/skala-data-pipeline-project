# NYC Yellow Taxi 요금 구성요소 예측 보고서

## 1. 프로젝트 개요

이 프로젝트는 2026년 5월 NYC Yellow Taxi 운행 기록을 사용해
`total_amount`를 구성하는 9개 요금 항목을 각각 예측한 뒤 합산하는 방식으로
최종 요금을 예측한다. 단일 `total_amount` 회귀 모델 대신 구성 항목별 특성에
맞는 모델을 사용해 각 요금이 최종 결과에 어떻게 반영되는지 확인할 수 있게 구성했다.

- 데이터 출처: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet
- 원본 형식: Parquet
- 정제 후 데이터: 3,872,300건
- 학습 기간: 5월 1~24일, 3,049,445건
- 테스트 기간: 5월 25~31일, 822,855건
- 공통 알고리즘: LightGBM
- 모델 저장 파일: `outputs/fare_component_models.joblib`

## 2. 예측 대상

| 컬럼 | 의미 | 예측 방식 |
|---|---|---|
| `fare_amount` | 기본 운임 | 연속값 회귀 |
| `extra` | 시간대 등에 따른 추가 요금 | 연속값 회귀 |
| `mta_tax` | MTA 세금 | 정액 금액 클래스 분류 |
| `tip_amount` | 팁 | 부과 여부 분류 + 양수 금액 회귀 |
| `tolls_amount` | 통행료 | 부과 여부 분류 + 양수 금액 회귀 |
| `improvement_surcharge` | 운송 서비스 개선 할증료 | 정액 금액 클래스 분류 |
| `congestion_surcharge` | 혼잡 할증료 | 정액 금액 클래스 분류 |
| `Airport_fee` | 공항 이용 요금 | 정액 금액 클래스 분류 |
| `cbd_congestion_fee` | 중심업무지구(CBD) 혼잡 요금 | 정액 금액 클래스 분류 |

최종 예측값은 위 9개 모델의 달러 단위 예측값을 행별로 더해 구했다.
`total_amount`는 개별 모델의 입력으로 사용하지 않고 최종 합산 성능 평가에만 사용했다.

## 3. 데이터 로딩과 검증

- 메모리 사용을 줄이기 위해 분석·학습에 필요한 컬럼만 선택해 로딩했다.
- 같은 Parquet 파일을 Pandas와 Polars로 각각 읽어 행·열 크기, 자료형,
  컬럼별 결측치, 19개 분석 컬럼 기준 중복 행 수를 비교했다.
- 양쪽에 동일한 정제 규칙을 적용한 후 행 수와 파생변수 결측치 수가
  일치하는지 검증했다.
- `store_and_fwd_flag`는 문자열 앞뒤 공백을 제거했다.

## 4. 데이터 정제·이상치 처리

다음 기준을 통과한 행만 모델링에 사용했다.

1. 19개 분석 컬럼의 값이 모두 같은 중복 행을 제거했다.
2. 승차 시각이 2026년 5월 1일 00:00 이상, 6월 1일 00:00 미만인 행만 남겼다.
3. 하차 시각에서 승차 시각을 빼 `trip_duration_minutes`를 계산하고 1~120분만 사용했다.
4. `trip_distance`는 0.01~100mile 범위만 사용했다.
5. 9개 요금 중 하나라도 음수인 행은 취소·환불 가능성이 있어 제외했다.
6. `payment_type == 6`인 voided trip을 취소 운행으로 보고 제외했다.
7. 각 요금의 99.9% 분위수 또는 공식 정액 요금을 기준으로 다음 상한을 적용했다.

| 요금 컬럼 | 사용 상한 |
|---|---:|
| `fare_amount` | $148.46 |
| `extra` | $10.75 |
| `mta_tax` | $0.50 |
| `tip_amount` | $30.00 |
| `tolls_amount` | $22.25 |
| `improvement_surcharge` | $1.00 |
| `congestion_surcharge` | $2.50 |
| `Airport_fee` | $2.00 |
| `cbd_congestion_fee` | $0.75 |

요금 결측치를 0으로 대체하면 '미기록'을 '미부과'로 왜곡할 수 있으므로 정제 단계에서
유지했다. 각 모델을 학습·평가할 때만 해당 목표값이 결측인 행을 제외했다.

## 5. EDA와 통계 검정

- 수치형 컬럼의 평균, 표준편차, 최솟값, 25%·50%·75% 분위수, 최댓값을 계산했다.
- 수치형 상관계수 행렬과 `fare_amount`와의 개별 상관계수를 확인했다.
- 거리·기본요금 관계, 요금 분포, 평균 요금 구성, 상관 히트맵을
  `outputs/taxi_eda.png`로 저장했다.
- 요일·승차 시간대별 평균 요금 합계를 `outputs/taxi_fare_by_hour.html`로 저장했다.

### 5.1 핵심 EDA 결과

- `trip_distance`와 `fare_amount`의 상관계수: 0.876

| 요금 컬럼 | 0원 비율 |
|---|---:|
| `tolls_amount` | 93.41% |
| `Airport_fee` | 71.50% |
| `extra` | 55.89% |
| `tip_amount` | 36.29% |
| `cbd_congestion_fee` | 33.05% |
| `congestion_surcharge` | 8.06% |
| `improvement_surcharge` | 3.55% |
| `mta_tax` | 0.73% |
| `fare_amount` | 0.04% |

`tip_amount`와 `tolls_amount`는 0원 비율이 높고, 0이 아닌 경우에는 금액이 연속적으로
변하므로 분류와 회귀를 결합한 2단계 모델을 선택했다.

### 5.2 Welch t-test

평일과 주말의 평균 기본 요금이 같다는 귀무가설을 검정했다. 두 그룹의 분산이
같다고 가정하지 않는 Welch t-test를 사용했다.

- 평일: 2,641,733건, 평균 $21.33
- 주말: 1,230,567건, 평균 $20.61
- t 통계량: 42.0102
- p-value: < 1e-300
- 해석: p-value가 0.05보다 작으므로 귀무가설을 기각한다. 이 데이터에서
  평일과 주말의 평균 기본 요금 차이는 통계적으로 유의하다.

## 6. 입력 컬럼과 파생변수

### 6.1 수치형 입력

`passenger_count`, `trip_distance`, `trip_duration_minutes`, `pickup_hour`, `pickup_weekday`

- `passenger_count`: 승객 수
- `trip_distance`: 운행 거리
- `trip_duration_minutes`: 승·하차 시각으로 계산한 운행 시간
- `pickup_hour`: 승차 시각에서 추출한 0~23시
- `pickup_weekday`: 승차 시각에서 추출한 요일(0=월요일, 6=일요일)

### 6.2 범주형 입력

`VendorID`, `RatecodeID`, `PULocationID`, `DOLocationID`, `payment_type`, `is_weekend`

- `VendorID`: 기록 제공 업체
- `RatecodeID`: 최종 적용 요금 코드
- `PULocationID`, `DOLocationID`: 승차·하차 Taxi Zone
- `payment_type`: 결제 방식
- `is_weekend`: `pickup_weekday >= 5`로 만든 주말 여부(0/1)

개별 요금 컬럼과 `total_amount`는 다른 요금 모델의 입력에 넣지 않아 목표값
정보가 유출되지 않도록 했다.

## 7. 전처리 Pipeline

모든 목표 모델에 같은 `ColumnTransformer`를 사용했고, 전처리기와 LightGBM을
하나의 sklearn `Pipeline`으로 묶었다.

1. **수치형**: `SimpleImputer(strategy="median", add_indicator=True)`로 결측치를
   학습 데이터의 중앙값으로 대체하고, 결측 여부 지시 컬럼을 추가했다.
2. **수치형 스케일**: `StandardScaler()`로 평균 0, 표준편차 1에 가깝게 변환했다.
3. **범주형**: `SimpleImputer(strategy="most_frequent")`로 결측치를 최빈값으로 대체했다.
4. **범주형 인코딩**: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`로
   범주를 숫자로 변환했다. 학습 때 없던 테스트 범주는 -1로 처리했다.
5. Pandas의 `pd.NA`는 sklearn이 안정적으로 처리하도록 `np.nan`으로 변환했다.

LightGBM은 트리 기반 모델이라 표준화가 필수는 아니지만, 공통 전처리
Pipeline을 명확하게 구성하고 모델 교체 가능성을 유지하기 위해 적용했다.

## 8. 모델 구성과 학습 전략

### 8.1 연속 요금

`fare_amount`와 `extra`는 0 이상의 연속 금액으로 보고 `LGBMRegressor`로 직접
회귀했다. 음수 예측은 요금으로 해석할 수 없으므로 0으로 클리핑했다.

### 8.2 0 폭증 요금

`tip_amount`와 `tolls_amount`는 다음 두 단계로 예측했다.

1. `LGBMClassifier`로 해당 요금이 0보다 큰지 판별했다.
2. 학습 데이터 중 실제 금액이 0보다 큰 행만 사용해 `LGBMRegressor`를 학습했다.
3. 1단계가 미부과로 판단하면 0달러, 부과로 판단하면 2단계 회귀 금액을 최종값으로 사용했다.

### 8.3 정액 수수료

`mta_tax`, `improvement_surcharge`, `congestion_surcharge`, `Airport_fee`, `cbd_congestion_fee`는 연속적인 임의 금액이 아니라 몇 가지 정해진 금액이
반복되므로 회귀 대신 다중 분류를 사용했다. `LabelEncoder`로 실제 금액
클래스를 정수 라벨로 변환해 `LGBMClassifier`를 학습하고, 예측 후 다시 달러
금액으로 역변환했다.

### 8.4 LightGBM 파라미터

| 모델 | 명시 파라미터 |
|---|---|
| `LGBMRegressor` | `n_estimators=250`, `random_state=42`, `n_jobs=-1`, `verbosity=-1` |
| `LGBMClassifier` | `n_estimators=200`, `random_state=42`, `n_jobs=-1`, `verbosity=-1` |

- `n_estimators`는 생성할 boosting tree 수이다.
- `random_state=42`로 재현 가능성을 높였다.
- `n_jobs=-1`로 사용 가능한 CPU 코어를 활용했다. GPU 학습은 사용하지 않았다.
- `verbosity=-1`로 학습 로그를 줄였다.
- 표에 없는 파라미터는 LightGBM 기본값을 사용했으며, 별도의
  하이퍼파라미터 탐색은 수행하지 않았다.

## 9. 학습·테스트 분할과 재사용

랜덤 분할 대신 시간 순서를 보존했다. 5월 1~24일을 학습에, 5월 25~31일을
테스트에 사용해 과거 기록으로 이후 기간을 예측하는 상황을 모사했다.

- 각 목표의 학습 데이터에서는 해당 목표 결측 행만 제외했다.
- 테스트 예측은 정제 후 테스트 행 전체에 대해 생성했다.
- 학습된 전처리기·모델·라벨 인코더를 하나의 joblib bundle로 저장했다.
- 저장 파일이 있고 입력 컬럼·목표 구성이 현재 설정과 같으면 재학습하지 않고
  모델을 불러와 평가했다.

## 10. 평가 방법

- **MAE**: 실제와 예측 금액의 절대 차이 평균으로, 달러 단위의 일반적인
  오차 크기를 직관적으로 보여준다.
- **RMSE**: 큰 오차에 더 큰 패널티를 주므로 큰 예측 실패를 확인한다.
- **R²**: 모델이 목표 금액의 변동을 설명하는 정도를 보조적으로 확인한다.
- **Accuracy·weighted F1**: 정액 요금 클래스 분류의 전체 정답률과 클래스별
  표본 수를 반영한 F1을 평가한다.
- **부과 F1**: 2단계 모델의 1단계가 0원·양수 요금을 잘 구분하는지 평가한다.
- **기준 모델**: 회귀는 학습 목표의 중앙값, 분류는 최빈 금액 클래스로
  항상 예측한 결과와 비교했다.

## 11. 개별 모델 결과

### 11.1 회귀·2단계 금액 예측

| 목표 | 평가 행 | MAE | RMSE | R² | 기준 MAE |
|---|---:|---:|---:|---:|---:|
| `fare_amount` | 822,855 | 1.684 | 3.896 | 0.942 | 10.891 |
| `extra` | 822,855 | 0.075 | 0.329 | 0.964 | 1.157 |
| `tip_amount` | 822,855 | 0.861 | 1.915 | 0.727 | 2.487 |
| `tolls_amount` | 822,855 | 0.227 | 1.316 | 0.573 | 0.527 |

2단계 모델의 부과 여부 성능은 다음과 같다.

| 목표 | 부과 F1 |
|---|---:|
| `tip_amount` | 0.962 |
| `tolls_amount` | 0.780 |

### 11.2 정액 수수료 클래스 분류

| 목표 | 평가 행 | Accuracy | weighted F1 | 기준 Accuracy | 금액 MAE |
|---|---:|---:|---:|---:|---:|
| `mta_tax` | 822,855 | 0.999 | 0.999 | 0.993 | $0.000 |
| `improvement_surcharge` | 822,855 | 0.966 | 0.950 | 0.966 | $0.034 |
| `congestion_surcharge` | 624,178 | 0.994 | 0.994 | 0.891 | $0.015 |
| `Airport_fee` | 624,178 | 0.998 | 0.998 | 0.912 | $0.005 |
| `cbd_congestion_fee` | 822,855 | 0.910 | 0.909 | 0.657 | $0.068 |

분류 정확도가 높더라도 클래스 비율이 불균형하면 다수 클래스 예측으로도 높은
값을 얻을 수 있으므로 기준 정확도와 weighted F1, 달러 단위 MAE를 함께 확인했다.

## 12. 최종 합산 평가

- 평가 행: 822,855건
- MAE: **$2.832**
- RMSE: **$5.102**
- R²: **0.934**
- 실제 9개 구성요소 합계와 `total_amount` 자체의 MAE: **$0.544**

최종 기준은 9개 예측값의 합계와 실제 `total_amount`의 MAE다. 현재 모델은
최종 요금을 평균적으로 약 $2.83 차이로 예측했다. 실제 9개 요금을
그대로 더해도 `total_amount`와 평균 $0.544의 차이가 있어, 모델
오차에는 예측 오차 외에 원본 요금 구조의 불일치도 포함된다.

## 13. 산출물

- 정적 EDA: `outputs/taxi_eda.png`
- 인터랙티브 시각화: `outputs/taxi_fare_by_hour.html`
- 학습된 모델: `outputs/fare_component_models.joblib`
- 자동 생성 보고서: `report.md`

## 14. 한계와 개선 방향

- 실제 운행거리·운행시간·결제 방식을 입력으로 사용하므로 현재 모델은
  **운행 완료 후 요금 검증**에 더 적합하다. 승차 전 견적 모델이 필요하면 실제
  운행 시간·거리를 예상 경로 기반 값으로 교체해야 한다.
- `RatecodeID`는 최종 요금 코드이므로 승차 전 예측 시점에 알 수 있는지 재검토가 필요하다.
- 신용카드 팁은 데이터에 기록되지만 현금 팁은 누락될 수 있어 `tip_amount`
  모델의 목표에 관측 편향이 존재할 수 있다.
- 모델 선택·파라미터 튜닝을 위한 별도의 검증 기간이 없어 테스트 결과를
  반복적으로 모델 선택에 사용하면 성능이 과대 평가될 수 있다.
- 추후에는 학습·검증·테스트 3개 기간으로 나누고, 팀원별 모델을 검증 기간의
  최종 합산 MAE로 비교한 뒤 선택된 모델을 테스트 기간에 한 번만 평가하는
  방식이 바람직하다.
- Taxi Zone의 borough·공항 여부, 날씨, 공휴일, 시간대별 교통량을 추가하고
  불균형 클래스에 대한 macro F1·balanced accuracy를 보조 지표로 사용할 수 있다.
