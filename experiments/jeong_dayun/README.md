# Jeong Dayun — Multi-output Random Forest

기존 `data_pipeline_project`의 실험을 팀 저장소 규칙에 맞게 이전한 버전입니다.
비금액 운행 정보로 9개 요금 구성요소를 각각 예측한 뒤, 예측값의 합으로
`total_amount`를 계산합니다. 요금 구성 컬럼과 `total_amount`는 입력 피처로
사용하지 않습니다.

## 공통 계약 적용

- 데이터 로드: `src.common.data_loader.load_taxi_pandas`
- 분할: `src.common.split.make_common_split`의 고정 80:20 분할
- 평가: MAE, RMSE, Median AE, R²와 실제 금액 구간별 지표
- 테스트: 공통 테스트 815,056행을 삭제하지 않고 모두 예측
- 학습: 기존 실험과 같이 최대 200,000행을 시드 42로 표본 추출

학습 데이터에서만 운행거리와 운행시간이 양수인지 확인하고, 운행시간 99.5%
분위수 상한을 적용합니다. 결측 수치형은 중앙값, 결측 범주형은 `unknown`으로
Pipeline 안에서 처리합니다.

## 실행

```bash
python -m experiments.jeong_dayun.train
```

전체 적격 학습 행을 사용하려면 `--max-train-rows 0`을 지정합니다. 실행 결과는
`reports/experiments/jeong_dayun/metrics.json`에 저장됩니다. 개인 모델 파일은
저장하거나 커밋하지 않습니다.

이전 프로젝트에서 기록한 결과는
`reports/experiments/jeong_dayun/legacy_report.md`에 보존했습니다. 해당 결과는
자체 20만 행 표본 분할을 사용했으므로 공통 리더보드 결과로 간주하지 않습니다.
