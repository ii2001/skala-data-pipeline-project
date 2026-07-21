# Lee Hyeonjun — Selected LightGBM Model

팀 모델 비교 후 최종 선정된 LightGBM 회귀 접근입니다. 절대경로 제거, 공통 분할,
EDA·통계 분석, sklearn Pipeline, 모델·보고서 저장을 포함한 최종 구현은
`src/final/train.py`에서 관리합니다.

다음 두 명령은 같은 최종 End-to-End 분석을 실행합니다.

```bash
python -m experiments.Lee_hyeonjun.model
python -m src.final.train
```
