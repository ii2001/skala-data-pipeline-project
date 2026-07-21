# Dataset

## 선택 데이터

NYC TLC Yellow Taxi Trip Records의 2026년 5월 Parquet을 사용합니다.

- 원본 shape: 4,090,836행 × 20열
- 파일 크기: 69,699,174바이트
- 분석 타깃: `total_amount`
- 분석 모집단: 2026년 5월이며 `total_amount > 0`인 4,075,277행

대용량 데이터 로딩 비교, 시각·수치·범주형 피처 처리, 실제 결측 처리, 통계 분석,
회귀 모델링을 한 데이터셋에서 수행할 수 있어 Day 2 종합 실습 데이터로
선택했습니다.

## 디렉터리

- `raw/`: 다운로드한 원본 Parquet
- `processed/`: 코드로 재생성하는 전처리 데이터
- `external/`: 외부 참조 데이터가 필요한 경우 사용
- `dataset_manifest.json`: URL, 파일 크기, SHA-256, 행·열 수

원본과 전처리 데이터는 용량 때문에 Git에 커밋하지 않습니다. 다음 명령으로 같은
원본을 다운로드하고 manifest와 일치하는지 검증합니다.

```bash
python -m src.common.data_loader
```

전처리 데이터를 재생성하려면 다음 명령을 사용합니다.

```bash
python -m src.final.preprocessing
```

원본 데이터는 수정하지 않으며 모든 정제와 파생 변수 생성은 코드에서 수행합니다.
