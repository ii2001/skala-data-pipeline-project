# NYC Yellow Taxi Data Pipeline

SKALA Day 2 종합 실습을 위한 NYC Yellow Taxi 2026년 5월 데이터 분석
프로젝트입니다. 원본 Parquet을 Pandas와 Polars로 로드해 구조를 비교하고,
결측치·중복 처리와 기본 EDA 시각화를 수행합니다.

## 프로젝트 구조

```text
.
├── data/
│   ├── raw/          # 다운로드한 원본 데이터
│   ├── processed/    # 전처리 결과
│   └── external/     # 외부 참조 데이터
├── models/           # 학습 모델
├── notebooks/        # EDA 및 실험
├── reports/          # report.md, 정적 차트, Plotly HTML
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── statistical_analysis.py
│   └── model.py
├── .gitignore
├── README.md
└── requirements.txt
```

데이터 파일과 모델 바이너리는 루트 `.gitignore`에서 제외합니다. 필요한
데이터 디렉터리는 실행 코드가 자동으로 생성합니다.

## 환경 설정

Python 3.9 이상 환경에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 로드

```bash
python -m src.data_loader
```

최초 실행 시 약 66 MiB의 Parquet 파일을 `data/raw/`에 다운로드합니다.
이미 받은 파일을 무시하고 다시 다운로드하려면 다음과 같이 실행합니다.

```bash
python -m src.data_loader --refresh
```

현재 확인된 데이터 구조는 4,090,836행 × 20열입니다.

## 기본 EDA와 시각화

```bash
python -m src.eda
```

실행 시 다음 결과를 생성합니다.

- `reports/figures/eda_overview.png`: Seaborn 2×2 정적 EDA 차트
- `reports/figures/missing_values_analysis.png`: 결측 패턴과 결제유형별 결측률
- `reports/interactive/hourly_trips.html`: Plotly 시간대별 인터랙티브 차트

정적 차트는 시간대별 운행량, 이동거리 분포, 결제수단 분포, 원본 결측률을
보여줍니다. 이동거리 분포는 409만 행을 모두 집계하는 대신 99분위수 이하에서
고정 시드로 최대 10만 행을 표본 추출해 재현 가능하게 그립니다.

## 고액 운행 분류 모델

기록된 `total_amount`가 30달러 이상인 운행을 `high_fare=1`로 정의합니다.
요금 구성 컬럼은 정답을 직접 구성하는 누수 변수이므로 모델 입력에서
제외하고, 승차 시간·요일, 승하차 위치, 공급업체, 승객 수만 사용합니다.

```bash
python -m src.model
```

5월 1~24일을 학습 후보, 25~31일을 테스트 후보로 분리한 뒤 라벨 비율을
유지한 총 50만 행을 표본 추출합니다. Dummy 기준 모델, 일반 Logistic,
불균형 보정 Logistic을 동일 조건에서 비교하고 F1이 가장 높은 모델을
저장합니다.

- `models/high_fare_pipeline.joblib`: 전처리와 Logistic Regression Pipeline
- `reports/report.md`: 실험 방법·결과·해석·한계를 정리한 자동 보고서
- `reports/metrics/`: 전처리 감사·통계·모델 비교 수치 결과
- `reports/figures/model_evaluation.png`: 혼동행렬·ROC·PR 곡선
- `reports/figures/experiment_comparison.png`: 모델 실험 비교
- `reports/figures/preprocessing_audit.png`: 단계별 전처리 결과
- `reports/figures/fare_composition_analysis.png`: 요금 구성 합계 검증

구성요소 결측을 0으로 처리한 단순 합계는 기록된 `total_amount`와 항상
일치하지 않으므로, 구성요소 합계는 데이터 품질 진단에만 사용합니다.
