# NYC Yellow Taxi Data Pipeline

SKALA Day 2 종합 실습을 위한 NYC Yellow Taxi 2026년 5월 데이터 분석
프로젝트입니다. 현재 단계에서는 원본 Parquet 다운로드와 Pandas 로드,
기본 데이터 구조 확인을 제공합니다.

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
│   └── data_loader.py
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
