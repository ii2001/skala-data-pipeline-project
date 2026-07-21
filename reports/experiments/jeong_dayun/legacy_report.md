# Jeong Dayun 기존 실험 결과

이 문서는 `data_pipeline_project/report.md`의 모델 결과를 팀 저장소로 옮겨
요약한 기록입니다. 새 공통 분할 평가 결과가 아니므로 리더보드 입력에는
사용하지 않습니다.

## 기존 실험 설정

- 원본: NYC Yellow Taxi 2026년 5월, 4,090,836행 × 20열
- 전처리 후: 3,015,437행
- 모델링 표본: 200,000행
- 자체 분할: 학습 160,000행, 테스트 40,000행, 시드 42
- 모델: `RandomForestRegressor(n_estimators=100, max_depth=15)`
- 방식: 9개 요금 구성요소 다중 출력 예측 후 합산

## 기존 결과

| 지표   |        값 |
|------|---------:|
| MAE  | 2.187189 |
| RMSE | 5.257450 |
| R²   | 0.941963 |

기존 실행에서는 Median AE를 기록하지 않았습니다. 공통 평가 결과는
`python -m experiments.jeong_dayun.train` 실행 후 생성되는 `metrics.json`을
사용해야 합니다.
