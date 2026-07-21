# Data

- `raw/`: 다운로드한 원본 Parquet
- `processed/`: 재생성 가능한 전처리 데이터
- `external/`: 외부 참조 데이터

데이터 파일은 Git에 커밋하지 않습니다. 공통 로더가 원본을 내려받습니다.
`dataset_manifest.json`의 URL·크기·SHA-256으로 팀원이 같은 원본을 사용하는지
확인합니다.
