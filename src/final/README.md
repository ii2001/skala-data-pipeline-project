# Final Source of Truth

이 디렉터리가 최종 제출 코드의 기준입니다. 팀원별 `experiments/`는 비교와 개발
이력이며, 교수님이 재현할 최종 실행 파일은 `train.py`입니다.

## `train.py` — 최종 End-to-End 실행

```bash
python -m src.final.train
```

한 번의 실행으로 다음 작업을 순서대로 수행합니다.

1. 동일 Parquet을 Pandas와 Polars로 각각 로딩
2. shape·컬럼·결측 건수 일치 검증 및 중복 검사
3. Seaborn 정적 EDA와 Plotly 인터랙티브 차트 생성
4. 평균·표준편차·분위수, Pearson 상관계수 계산
5. 신용카드·현금 결제 금액의 Welch t-test와 p-value 해석
6. 공통 80:20 분할과 모델 피처 생성
7. sklearn Pipeline으로 전처리와 LightGBM 학습
8. 전체·금액 구간별 평가, joblib 모델 저장
9. 저장 모델 재로딩·재예측 검증과 `report.md` 자동 생성

## 최종 Pipeline

```text
원본 운행 데이터
  → 시간·거리 피처 생성
  → ColumnTransformer
      ├─ 숫자형: 결측 처리
      ├─ 코드형: 결측 처리 + OneHotEncoder
      └─ store flag: Unknown 처리 + 축약 OneHotEncoder
  → LGBMRegressor
  → total_amount 예측
```

숫자형 피처는 승객 수, 거리, duration, pickup 시간·요일과 이상값 indicator입니다.
범주형 피처는 `RatecodeID`, 승·하차 LocationID, 결제수단,
`store_and_fwd_flag`입니다. 직접 요금 구성 9개 컬럼과 `VendorID`는 입력에서
제외합니다.

## `preprocessing.py` — 전처리 데이터 export

```bash
python -m src.final.preprocessing
```

다음 파일을 재생성합니다.

- `data/processed/yellow_tripdata_2026-05_processed.parquet`
- `data/processed/preprocessing_summary.json`

이 파일은 전처리 결과를 열 단위로 확인하거나 다른 분석에 재사용하기 위한
보조 산출물입니다. 최종 모델 학습의 기준은 `train.py` 내부 Pipeline이며,
imputation과 원핫 인코딩은 학습 데이터에만 fit합니다.

## 주요 품질 규칙

- 평가 모집단: 2026년 5월, `total_amount > 0`
- 거리: 0~100마일 제한, 0·상한 처리 indicator 보존
- duration: 1~180분 제한, 보정 indicator 보존
- `passenger_count`: 결측 0
- `RatecodeID`: 결측 99
- pickup 요일: 월요일 0~일요일 6
- 모델 제외: `source_row_id`, `total_amount`, 직접 요금 구성 컬럼

macOS에서 LightGBM OpenMP 오류가 발생하면 `brew install libomp`를 먼저
실행합니다.
