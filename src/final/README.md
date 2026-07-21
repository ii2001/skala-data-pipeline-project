# Final Preprocessing

팀에서 합의한 최종 모델 입력 데이터를 생성합니다. 현재 단계에서는 모델을
학습하지 않고 전처리 결과만 `data/processed/`에 저장합니다.

```bash
python -m src.final.preprocessing
```

생성 파일은 다음과 같습니다.

- `data/processed/yellow_tripdata_2026-05_processed.parquet`
- `data/processed/preprocessing_summary.json`

`pickup_day_of_week`은 월요일 0부터 일요일 6까지 사용합니다. `RatecodeID`의
결측치는 TLC 데이터 사전의 unknown 코드인 99로 채웁니다. `trip_distance`는
100마일을 초과하는 오류 후보만 상한 처리하고, 0마일과 상한 처리 여부를 별도
indicator 컬럼으로 보존합니다.
