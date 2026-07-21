# NYC Yellow Taxi `total_amount` 예측

> SKALA Day 2 종합 실습 — 데이터 준비부터 EDA, 시각화, 통계 검정,
> ML Pipeline, 자동 보고서까지 한 번에 재현하는 End-to-End 프로젝트

NYC Yellow Taxi 2026년 5월 실제 운행 4,090,836건을 분석하고, 운행 종료 시점의
정보로 승객이 지불한 총액 `total_amount`를 예측합니다. 팀원별 실험을 검토한 뒤
Lee Hyeonjun의 LightGBM 접근을 최종 모델로 선정했으며, 제출용 구현은
[`src/final/train.py`](src/final/train.py) 한 곳에서 관리합니다.

## 최종 결과 한눈에 보기

| 항목         |            최종 결과 |
|------------|-----------------:|
| 원본 데이터     | 4,090,836행 × 20열 |
| 공통 학습 데이터  |       3,260,221행 |
| 공통 테스트 데이터 |         815,056행 |
| 테스트 제외 행   |               0행 |
| MAE        |           3.4002 |
| RMSE       |           7.7800 |
| Median AE  |           1.5451 |
| R²         |           0.8754 |

- 최종 자동 보고서: [`reports/final/report.md`](reports/final/report.md)
- 전체 평가 결과: [`reports/final/metrics.json`](reports/final/metrics.json)
- 저장된 Pipeline: [`models/final_model.joblib`](models/final_model.joblib)
- 인터랙티브 차트: [`hourly_total_amount.html`](reports/final/interactive/hourly_total_amount.html)

## EDA 결과

아래 정적 차트는 고정 시드로 선택한 100,000행을 사용합니다. 왼쪽은 이동거리
분포, 오른쪽은 주요 수치 변수의 Pearson 상관계수입니다. 전체 데이터로 계산한
`trip_distance`와 `total_amount`의 상관계수는 0.7845였습니다.

![Trip distance distribution and correlation heatmap](reports/final/figures/eda_overview.png)

시간대별 평균·중앙값과 운행량은 Plotly 파일에서 직접 확인할 수 있습니다.
HTML 파일을 내려받아 브라우저로 열면 마우스 hover와 확대가 가능합니다.

아래 차트는 최종 테스트 815,056행의 예측 결과를 사용합니다. 실제값–예측값,
잔차 분포, 금액 구간별 MAE를 함께 확인하면 전체 지표만으로 가려지는 고액 운행의
오차 증가를 볼 수 있습니다.

![Model evaluation with actual vs predicted, residuals, and fare-band MAE](reports/final/figures/model_evaluation.png)

## 왜 이 데이터를 선택했는가

NYC TLC Yellow Taxi 데이터는 이번 실습의 모든 요구사항을 한 데이터셋 안에서
검증하기에 적합했습니다.

- **충분한 규모**: 409만 건으로 Pandas와 Polars의 로딩 결과 및 실행 특성을
  실제 대용량 데이터에서 비교할 수 있습니다.
- **다양한 데이터 타입**: 시각, 연속형 거리·금액, 승객 수, 명목형 위치·결제
  코드가 함께 있어 EDA와 Pipeline 구성이 가능합니다.
- **실제 결측 문제**: `passenger_count`, `RatecodeID`, `store_and_fwd_flag`에 각각
  955,371건의 결측이 있어 명시적인 결측 처리 기준을 설명할 수 있습니다.
- **통계 분석 가능성**: 거리·시간·금액의 상관관계와 결제수단별 평균 차이를
  기술통계, 상관계수, t-test로 분석할 수 있습니다.
- **재현 가능성**: 원본 URL, 파일 크기, SHA-256, 행·열 수를
  [`data/dataset_manifest.json`](data/dataset_manifest.json)에 고정했습니다.

원본은 NYC TLC가 제공하는 2026년 5월 Yellow Taxi Trip Records Parquet입니다.
분석 대상은 2026년 5월이면서 기록된 `total_amount > 0`인 4,075,277행입니다.

## 문제 정의와 예측 시점

- **타깃**: 승객에게 청구된 총액 `total_amount`
- **문제 유형**: 회귀
- **예측 시점**: 운행 종료 직후

실제 탑승 시간, 최종 `RatecodeID`, 결제수단을 사용하므로 승차 직전 가격 안내
모델로 해석하면 안 됩니다. 이 모델은 운행이 끝난 뒤 금액을 추정하거나 기록 오류를
점검하는 용도입니다.

## 팀이 합의한 전처리 기준

| 대상                   | 처리 기준                     | 결정 이유                                 |
|----------------------|---------------------------|---------------------------------------|
| 요금 구성 9개 컬럼          | 모델 입력에서 제거                | 합계인 `total_amount`를 직접 구성하므로 타깃 누수 발생 |
| `VendorID`           | 제거                        | 공급자 식별자보다 운행 자체 특성에 집중                |
| 승·하차 시각              | duration, 시간, 요일 파생 후 제거  | 원본 시각의 고카디널리티를 줄이고 해석 가능한 피처 생성       |
| `passenger_count`    | 결측을 0으로 대체                | 원본에서 미기록된 승객 수를 별도 상태로 보존             |
| `trip_distance`      | 0~100마일 상한 처리 + indicator | 최대 307,491.47마일 오류값이 원본 상관을 왜곡        |
| duration             | 1~180분 제한 + indicator     | 0분·장시간 기록 오류의 영향을 완화                  |
| `RatecodeID`         | 결측 99 후 원핫 인코딩            | 5는 협의 요금이며 공식 unknown 코드는 99          |
| `store_and_fwd_flag` | Unknown 대체 후 축약 원핫        | N/Y/결측 정보를 적은 차원으로 표현                 |
| PU·DO Location, 결제수단 | 원핫 인코딩                    | 코드 숫자의 크기에는 순서 의미가 없음                 |

모든 imputation과 원핫 인코딩은 sklearn `Pipeline` 안에서 학습 데이터에만
fit합니다. `source_row_id`, `total_amount`, 직접 요금 구성 컬럼은 모델 입력에
포함되지 않습니다.

## 데이터 분석 결과

### 기술통계

| 변수                    |      평균 |    표준편차 |     25% |     중앙값 |     75% |
|-----------------------|--------:|--------:|--------:|--------:|--------:|
| passenger_count       |  0.9516 |  0.7663 |  1.0000 |  1.0000 |  1.0000 |
| trip_distance         |  3.4171 |  4.2624 |  1.0400 |  1.8800 |  3.8200 |
| trip_duration_minutes | 18.5469 | 15.7068 |  8.6000 | 14.5000 | 23.2833 |
| total_amount          | 30.7040 | 22.6465 | 17.7000 | 23.9400 | 34.9800 |

전체 수치는 [`descriptive_statistics.csv`](reports/final/descriptive_statistics.csv),
상관행렬은 [`correlation.csv`](reports/final/correlation.csv)에 저장했습니다.

### 상관관계 해석

- `trip_distance`–`total_amount`: **0.7845**
- `trip_duration_minutes`–`total_amount`: **0.6632**
- `passenger_count`–`total_amount`: **0.0034**

거리와 실제 운행 시간은 총액과 강한 양의 선형 관계를 보였지만, 승객 수는 거의
관계가 없었습니다. 상관계수는 인과관계를 의미하지 않으므로 모델에서는 위치·시간대
등의 비선형 상호작용도 함께 학습했습니다.

### Welch t-test

신용카드와 현금 결제의 평균 `total_amount`를 `scipy.stats.ttest_ind`의 Welch
t-test로 비교했습니다.

| 항목        |             결과 |
|-----------|---------------:|
| 신용카드 평균   |       $30.6586 |
| 현금 평균     |       $26.1149 |
| t 통계량     |       108.7091 |
| p-value   | 계산 정밀도에서 0에 수렴 |
| Cohen's d |         0.1948 |

p-value가 0.05보다 작아 평균이 같다는 귀무가설은 기각합니다. 다만 표본이 매우
크고 Cohen's d가 약 0.19이므로, 통계적으로 유의하다는 사실과 실제 차이의 크기를
구분해 해석했습니다. 상세 결과는
[`statistical_results.json`](reports/final/statistical_results.json)에 있습니다.

## 최종 모델 결정 과정

팀원별 실험은 서로 다른 표본과 기존 분할로 작성된 결과도 있어 과거 점수를 그대로
순위화하지 않았습니다. 다음 기준으로 Lee Hyeonjun의 LightGBM 접근을 선택한 뒤,
최종 코드에서 전체 공통 분할로 다시 학습·평가했습니다.

1. 거리·시간·위치 사이의 비선형 관계를 학습할 수 있는가
2. 326만 학습 행을 현실적인 시간과 메모리로 처리할 수 있는가
3. 전처리와 모델이 하나의 sklearn `Pipeline`으로 저장되는가
4. 직접 요금 구성 컬럼을 사용하지 않아 누수 기준을 만족하는가
5. 고정 분할과 모델 재로딩으로 결과를 재현할 수 있는가

최종 구조는 `ColumnTransformer → LGBMRegressor`입니다. 선택 기록은
[`reports/comparison/selection_report.md`](reports/comparison/selection_report.md)에
정리했습니다.

### 금액 구간별 성능

| 실제 금액 구간 |     행 수 |     MAE |    RMSE |      R² |
|----------|--------:|--------:|--------:|--------:|
| $30 미만   | 541,434 |  2.0137 |  3.8190 |  0.4764 |
| $30~60   | 201,498 |  4.4965 |  6.8343 |  0.2131 |
| $60~100  |  56,518 |  8.6498 | 13.2332 | -0.2851 |
| $100 이상  |  15,606 | 18.3387 | 37.6560 |  0.2050 |

전체 R²는 0.8754지만 고액 구간의 오차가 큽니다. 따라서 “전체 성능이 높으니 모든
금액대에서 정확하다”고 해석하지 않았으며, 고액 운행 개선을 후속 과제로 남겼습니다.

## 과제 요구사항 확인

| 채점 항목                   | 구현 위치                                                                                                                        | 상태 |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|:--:|
| Pandas·Polars 로딩 비교     | [`train.py`](src/final/train.py)                                                                                             | 완료 |
| 결측·중복 처리와 기본 EDA        | [`train.py`](src/final/train.py), [`report.md`](reports/final/report.md)                                                     | 완료 |
| Seaborn 정적 차트           | [`eda_overview.png`](reports/final/figures/eda_overview.png)                                                                 | 완료 |
| 모델 평가 정적 차트             | [`model_evaluation.png`](reports/final/figures/model_evaluation.png)                                                         | 완료 |
| Plotly 인터랙티브 차트         | [`hourly_total_amount.html`](reports/final/interactive/hourly_total_amount.html)                                             | 완료 |
| 기술통계·상관계수               | [`descriptive_statistics.csv`](reports/final/descriptive_statistics.csv), [`correlation.csv`](reports/final/correlation.csv) | 완료 |
| `ttest_ind`와 p-value 해석 | [`statistical_results.json`](reports/final/statistical_results.json)                                                         | 완료 |
| sklearn Pipeline        | [`train.py`](src/final/train.py)                                                                                             | 완료 |
| 평가 지표                   | [`metrics.json`](reports/final/metrics.json)                                                                                 | 완료 |
| joblib 모델 저장            | [`final_model.joblib`](models/final_model.joblib)                                                                            | 완료 |
| `report.md` 자동 생성       | [`report.md`](reports/final/report.md)                                                                                       | 완료 |

## 실행 방법

Python 3.9 이상 환경에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.final.train
```

macOS에서 LightGBM의 OpenMP 오류가 발생하면 최초 한 번 실행합니다.

```bash
brew install libomp
```

전처리된 원핫 데이터만 별도로 만들려면 다음 명령을 사용합니다.

```bash
python -m src.final.preprocessing
```

## 저장소 구조

```text
.
├── data/
│   ├── dataset_manifest.json       # 원본 버전·SHA-256
│   ├── raw/                        # 다운로드 원본(Git 제외)
│   └── processed/                  # 재생성 전처리 데이터(Git 제외)
├── experiments/                    # 팀원별 실험과 선택 모델의 출처
├── models/
│   └── final_model.joblib          # 최종 sklearn Pipeline
├── reports/
│   ├── comparison/                 # 모델 선택 근거
│   └── final/                      # 최종 보고서·수치·차트
├── src/
│   ├── common/                     # 공통 로더·분할·평가
│   └── final/
│       ├── preprocessing.py        # 전처리 데이터 export
│       └── train.py                # 최종 End-to-End 실행 기준
└── requirements.txt
```

## 한계와 개선 방향

- 실제 탑승 시간과 결제수단을 사용하므로 운행 시작 전 예측에는 사용할 수 없습니다.
- 2026년 5월 한 달 자료와 한 번의 고정 분할 결과이므로 다른 기간 성능을 보장하지
  않습니다.
- 고액 운행의 오차가 크므로 시간 순서 외부 검증, 희귀 위치 통합, 고액 구간 가중치,
  LightGBM 파라미터 검증이 후속 개선 대상입니다.

제출 전 실행 화면 캡처, 캠퍼스·반·이름 기입, PDF 변환과 5분 발표는
[`reports/final/README.md`](reports/final/README.md)의 체크리스트를 따릅니다.
