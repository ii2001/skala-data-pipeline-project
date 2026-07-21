# Team Experiments

각 팀원은 자신의 디렉터리 안에서만 전처리와 모델 코드를 관리합니다.

## 필수 제출 파일

- `config.json`: 작성자·실험 ID·타깃·모델 설정
- `train.py`: 공통 분할을 사용해 학습하고 예측하는 실행 코드
- `README.md`: 전처리 기준, 피처, 모델 선택 이유와 실행 방법
- `reports/experiments/{작성자}/metrics.json`: 공통 형식의 실행 결과

`src.common.split.make_common_split`과 `src/common/evaluation.py`의 평가 함수를
반드시 사용합니다. 다른 팀원의 디렉터리는 수정하지 않습니다.

`kim_yechan`에는 복구한 기존 실험이 보관되어 있습니다. `member_02`~`member_06`
디렉터리 이름과 각 `config.json`의 `author`는 담당자의 GitHub ID로 변경합니다.
