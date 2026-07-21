# Final Model

`final_model.joblib`은 팀에서 최종 선정한 전체 sklearn `Pipeline`입니다.

- 전처리: `ColumnTransformer`, 결측 처리, 원핫 인코딩
- 모델: `LGBMRegressor(n_estimators=100, learning_rate=0.05)`
- 타깃: `total_amount`
- 테스트: 815,056행 전체
- 성능: MAE 3.4002, RMSE 7.7800, Median AE 1.5451, R² 0.8754

`python -m src.final.train`이 모델을 생성한 뒤 다시 로딩하여 같은 입력의 예측값이
일치하는지 검증합니다. 모델만 단독으로 재학습하거나 수동 교체하지 않습니다.

개인 실험 모델은 제출 대상이 아니며 최종 `final_model.joblib`만 Git에서
관리합니다. 모델의 입력 피처와 한계는 `reports/final/report.md`를 확인합니다.
