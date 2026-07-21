# 팀 작업 규칙

## 브랜치와 담당 범위

- 브랜치: `feat/{github-id}/{experiment-name}`
- 각 팀원은 `experiments/{github-id}/`와 자신의 결과 디렉터리만 수정합니다.
- `src/common/` 변경은 별도 PR로 올리고 팀 검토를 받습니다.
- 하나의 PR에는 하나의 실험 또는 하나의 공통 변경만 포함합니다.

## 실험 제출 기준

1. `src.common.data_loader`로 원본을 로드합니다.
2. `src.common.split.make_common_split`로 전체 데이터를 80:20으로 나눕니다.
3. 팀원별 전처리는 학습 데이터에서만 학습합니다.
4. `src.common.evaluation`으로 테스트 결과를 계산합니다.
5. `src.common.results.save_result`로 `metrics.json`을 저장합니다.
6. 전처리 기준, 제외 행 수, 피처와 모델 파라미터를 README에 설명합니다.

학습 데이터의 이상치·결측 처리 방식은 자유지만, 공통 테스트 815,056행은
삭제하지 않고 모두 예측해야 최종 선정 대상이 됩니다.

데이터, 가상환경, 개인 모델 파일은 커밋하지 않습니다. 최종 모델 선정 전에는
`models/final_model.joblib`을 만들거나 교체하지 않습니다.

## 커밋 예시

```text
✨ feat: member01 경로 기반 Ridge 실험 추가
📊 data: 공통 평가 결과와 그래프 추가
📝 docs: 전처리 기준과 모델 해석 정리
🐛 fix: 테스트 데이터 누수 제거
```
