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
│   └── eda.py
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
