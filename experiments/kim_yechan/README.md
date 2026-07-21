# Kim Yechan — Route Statistics Ridge

팀 저장소 구조 변경 전에 완료한 개인 실험을 코드·모델·결과까지 보존한
디렉터리입니다.

- 정답: `total_amount`
- 기존 분할: 2026-05-01~24 학습 후보, 25~31 테스트 후보
- 기존 표본: 학습 400,000행, 테스트 100,000행
- 최종 모델: 경로 Target Encoding + Ridge
- 기존 결과: MAE 5.6470, RMSE 9.6127, R² 0.7881
- 보고서: `reports/experiments/kim_yechan/report.md`
- 기존 모델: `artifacts/models/total_amount_regression_pipeline.joblib`

```bash
python -m experiments.kim_yechan.eda
python -m experiments.kim_yechan.model
```

이 결과는 기존 시간 홀드아웃 기준이므로 새 공통 80:20 전체 데이터 분할
리더보드에는 그대로 제출하지 않습니다. 코드를 유지한 상태에서 새 공통 분할로
추가 실험 버전을 만들 수 있습니다.
