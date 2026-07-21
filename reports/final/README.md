# Final Submission

6개 실험을 비교해 선정한 모델의 자동 생성 `report.md`, 최종 그래프와 발표용
자료만 보관합니다. 개인 실험 결과를 직접 복사하지 않고 최종 코드를 재실행해
생성합니다.

`python -m src.final.train` 실행 시 다음 최종 산출물이 생성됩니다.

- `report.md`: 자동 생성 최종 보고서
- `metrics.json`: 전체 및 금액 구간별 회귀 평가 지표
- `descriptive_statistics.csv`: 평균·표준편차·분위수
- `correlation.csv`: Pearson 상관계수
- `statistical_results.json`: Welch t-test와 p-value 해석
- `figures/eda_overview.png`: Seaborn 정적 차트
- `interactive/hourly_total_amount.html`: Plotly 인터랙티브 차트

## 수동 제출 체크리스트

1. `python -m src.final.train`의 전체 실행 화면을 캡처합니다.
2. `report.md`에 캠퍼스명·반·이름과 본인 의견을 최종 확인합니다.
3. 보고서와 실행 화면을 PDF로 변환합니다.
4. 5분 발표에서는 데이터·EDA·통계·모델·한계 순서로 설명합니다.
5. 저장소 전체를 `{캠퍼스}_{반}_{이름}_day2종합실습.zip`으로 압축합니다.
