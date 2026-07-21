# Model Selection

여섯 팀원의 실험과 최종 모델 선택 근거를 보관합니다.

과거 개인 실험에는 시간 홀드아웃, 20만 행 표본, 다중 출력 등 서로 다른 평가
조건이 포함되어 있으므로 기록된 점수를 직접 순위화하지 않습니다. 팀 논의로
LightGBM 접근을 선택한 뒤 공통 80:20 분할과 테스트 815,056행 전체를 사용해
최종 평가했습니다.

- 최종 선택 기록: [`selection_report.md`](selection_report.md)
- 최종 실제 지표: [`../final/metrics.json`](../final/metrics.json)
- 자동 보고서: [`../final/report.md`](../final/report.md)

`scripts.compare_experiments`는 공통 형식의 팀원별 `metrics.json`이 모두 준비된
경우 리더보드를 만드는 보조 도구입니다. 최종 제출 수치는 `reports/final/`을
기준으로 합니다.
