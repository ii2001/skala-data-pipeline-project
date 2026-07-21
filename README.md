# NYC Yellow Taxi Team Model Lab

SKALA Day 2 종합 실습을 위한 6인 팀 실험 저장소입니다. 각 팀원이 서로 다른
전처리와 모델을 독립적으로 실험하고, 동일한 시간 분할과 평가 함수로 비교해
최종 모델 한 개를 선정합니다.

## 저장소 구조

```text
.
├── data/                       # Git에 올리지 않는 원본·가공 데이터
├── experiments/
│   ├── kim_yechan/            # 기존 개인 실험 코드와 모델
│   ├── member_02/
│   ├── member_03/
│   ├── member_04/
│   ├── member_05/
│   └── member_06/
├── models/                     # 최종 선정 모델만 추적
├── reports/
│   ├── experiments/           # 개인별 수치·그래프·보고서
│   ├── comparison/            # 리더보드와 선정 근거
│   └── final/                 # 최종 제출 보고서
├── scripts/
│   └── compare_experiments.py
└── src/common/                # 공통 로더·분할·평가·결과 스키마
```

## 최초 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.common.data_loader
```

원본은 NYC Yellow Taxi 2026년 5월 Parquet이며 로컬 `data/raw/`에 캐시됩니다.
Pandas 기준 원본은 4,090,836행 × 20열이고, 공통 로더가 비교용
`source_row_id`를 추가합니다.
원본 버전과 SHA-256은 `data/dataset_manifest.json`에 고정되어 있습니다.

## 팀원 작업 순서

1. `member_02`~`member_06`을 나머지 팀원의 GitHub ID로 배정합니다.
2. `feat/{github-id}/{experiment}` 브랜치를 만듭니다.
3. 자기 `experiments/{github-id}/config.json`을 작성합니다.
4. `src.common.split.make_common_split`로 학습·테스트 행을 고정합니다.
5. 자기 전처리와 Pipeline을 구현합니다.
6. `src.common.evaluation`으로 공통 지표를 계산합니다.
7. `src.common.results.save_result`로 결과를 저장합니다.
8. 코드·설정·결과를 한 PR로 올립니다.

자세한 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 확인합니다.

## 공통 평가 계약

- 정답: `total_amount`
- 분할: 전체 유효 데이터 고정 80:20 랜덤 분할
- 학습: 3,260,221행
- 테스트: 815,056행
- 공통 평가 모집단: 기록된 `total_amount > 0`
- 고정 시드: 42
- 필수 지표: MAE, RMSE, Median AE, R²
- 보조 지표: 실제 금액 구간별 MAE/RMSE/R²
- 정답 또는 직접 요금 구성 컬럼을 입력으로 사용하는 누수 금지

학습 행의 품질 필터는 자유지만 고정 테스트 815,056행은 모두 예측해야 합니다.
테스트 행을 제외한 실험은 결과를 기록할 수 있지만 최종 선정 적격에서 제외됩니다.

## 전체 실험 비교

모든 팀원의 `reports/experiments/{author}/metrics.json`이 준비된 후 실행합니다.

```bash
python -m scripts.compare_experiments
```

결과는 `reports/comparison/leaderboard.csv`와 `selection_report.md`에 저장됩니다.
최종 선정은 MAE 순위만 보지 않고 누수 여부, 고액 구간 오차, 재현성, 코드
완성도를 함께 검토합니다.

## 최종 제출

선정된 실험만 `src/final/`로 정리하고 전체 데이터로 다시 실행합니다. 개인
모델 파일은 커밋하지 않으며, 최종 `models/final_model.joblib`과
`reports/final/report.md`만 제출 대상으로 관리합니다.

최종 End-to-End 분석과 모델 학습은 다음 명령으로 실행합니다.

```bash
python -m src.final.train
```

## 최종 데이터 전처리

팀에서 합의한 피처만 사용한 전처리 데이터는 다음 명령으로 생성합니다.

```bash
python -m src.final.preprocessing
```

결과는 `data/processed/yellow_tripdata_2026-05_processed.parquet`에 저장됩니다.
요금 구성 9개 컬럼과 `VendorID`는 제거하고 `total_amount`만 정답으로 유지합니다.
탑승 시간·요일 피처, 결측값 처리, 이동거리 이상치 처리 및 범주형 인코딩 기준은
`src/final/README.md`를 따릅니다.
