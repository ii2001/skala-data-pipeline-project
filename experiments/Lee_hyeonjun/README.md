# Lee Hyeonjun — Selected LightGBM Approach

팀원별 모델 검토 후 최종 채택한 LightGBM 회귀 접근의 출처입니다.

## 선택 이유

- 거리·duration·시간대·위치 사이의 비선형 관계를 학습할 수 있습니다.
- 326만 학습 행을 전체 사용하면서도 학습과 예측이 효율적입니다.
- 명목형 위치·결제 코드를 원핫 인코딩한 희소 행렬을 처리할 수 있습니다.
- MAE, RMSE, Median AE, R²를 공통 테스트 815,056행 전체에서 계산할 수 있습니다.

초기 실험 코드의 절대경로, 별도 전처리 객체, 결과 미저장 문제는 최종 구현에서
제거했습니다. 최종 버전은 `ColumnTransformer`와 `LGBMRegressor`를 하나의
sklearn `Pipeline`으로 저장합니다.

## 최종 구현 위치

- 실행 코드: [`../../src/final/train.py`](../../src/final/train.py)
- 최종 보고서: [`../../reports/final/report.md`](../../reports/final/report.md)
- 선택 근거: [`../../reports/comparison/selection_report.md`](../../reports/comparison/selection_report.md)

다음 두 명령은 같은 최종 End-to-End 분석을 실행합니다.

```bash
python -m experiments.Lee_hyeonjun.model
python -m src.final.train
```

`experiments/Lee_hyeonjun/model.py`는 중복 구현을 두지 않고 최종 실행 함수로
연결되는 진입점입니다.
