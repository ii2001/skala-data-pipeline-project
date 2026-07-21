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
결측치는 TLC 데이터 사전의 unknown 코드인 99로 채운 뒤 원핫 인코딩합니다.
`trip_distance`는 100마일을 초과하는 오류 후보만 상한 처리하고, 0마일과 상한
처리 여부를 별도 indicator 컬럼으로 보존합니다.

`trip_duration_minutes`는 1~180분으로 제한하고 보정 여부를 별도 컬럼에
기록합니다. 파생이 끝난 원본 승·하차 시각은 제거합니다. 2026년 5월이 아니거나
`total_amount <= 0`인 행은 공통 평가 모집단과 맞추기 위해 제외합니다.

`source_row_id`는 행 추적용이고 `total_amount`는 정답이므로 두 컬럼 모두 모델의
입력 피처에서는 제외합니다. `store_and_fwd_flag`의 결측 indicator는
`payment_type_0`과 모든 행에서 동일하므로 중복 생성하지 않습니다.
