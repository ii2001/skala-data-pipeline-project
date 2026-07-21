# NYC Yellow Taxi End-to-End 분석

2026년 5월 NYC Yellow Taxi의 9개 요금 구성요소를 각각 예측하고 합산 금액을
평가하는 Day 2 종합 실습이다. Pandas·Polars 비교, EDA, 통계 검정, 시각화,
sklearn Pipeline, 기준 모델 비교, 모델 저장과 자동 보고서 생성을 수행한다.

## 실행

```bash
uv sync
uv run python main.py
```

첫 실행은 공식 Parquet 파일을 `data/`에 다운로드하므로 시간이 걸릴 수 있다.
분석 결과는 터미널, `outputs/`, `report.md`에 저장된다.

## 검사

```bash
uv run ruff check .
uv run pytest
```
