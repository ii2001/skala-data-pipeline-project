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

원본·가공 데이터와 임시 모델은 루트 `.gitignore`에서 제외하고, 최종 제출
모델만 추적할 수 있게 예외 처리합니다. 필요한 디렉터리는 코드가 생성합니다.

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

## total_amount 회귀 모델

기록된 `total_amount` 자체를 예측합니다. 요금 구성 컬럼은 정답을 직접
구성하는 누수 변수이므로 제외하고, 목적지가 정해진 승차 시점을 기준으로
시간·요일, 승하차 위치, 공급업체, 승객 수만 사용합니다.

```bash
python -m src.model
```

5월 1~24일을 학습 후보, 25~31일을 테스트 후보로 분리해 총 50만 행을
표본 추출합니다. Dummy, 원금액/로그 Ridge, `PU→DO` 경로 매칭, 학습 기간의
경로 운임 통계를 같은 홀드아웃에서 비교하고 MAE가 가장 낮은 모델을 저장합니다.

- `models/total_amount_regression_pipeline.joblib`: 전처리와 회귀 Pipeline
- `reports/report.md`: 실험 방법·결과·해석·한계를 정리한 자동 보고서
- `reports/metrics/`: 전처리 감사·통계·회귀 비교·금액 구간별 수치 결과
- `reports/figures/regression_model_comparison.png`: 회귀 실험 비교
- `reports/figures/actual_vs_predicted.png`: 실제 금액과 예측 금액
- `reports/figures/error_by_fare_band.png`: 실제 금액 구간별 오차
- `reports/figures/preprocessing_audit.png`: 단계별 전처리 결과

이전 고액 운행 이진 분류 결과는
`reports/archive/classification_report.md`에 보관합니다.
