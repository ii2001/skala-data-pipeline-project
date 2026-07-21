# Team Experiments

> 이 디렉터리는 팀원별 개발·비교 이력입니다. 최종 제출 실행 기준은
> `src/final/train.py`, 최종 결과는 `reports/final/`입니다.

각 팀원은 자신의 디렉터리 안에서만 전처리와 모델 코드를 관리합니다.

## 필수 제출 파일

- `config.json`: 작성자·실험 ID·타깃·모델 설정
- `train.py`: 공통 분할을 사용해 학습하고 예측하는 실행 코드
- `README.md`: 전처리 기준, 피처, 모델 선택 이유와 실행 방법
- `reports/experiments/{작성자}/metrics.json`: 공통 형식의 실행 결과

`src.common.split.make_common_split`과 `src/common/evaluation.py`의 평가 함수를
반드시 사용합니다. 다른 팀원의 디렉터리는 수정하지 않습니다.

`kim_yechan`과 `jeong_dayun`에는 기존 분할·표본 기반 실험이 보관되어 있습니다.
`Lee_hyeonjun`은 최종 선택 모델의 출처이며 실행은 `src/final/train.py`로
연결됩니다. 평가 조건이 다른 과거 수치는 최종 공통 지표와 직접 비교하지 않습니다.
