"""NYC Yellow Taxi 데이터 준비와 기본 EDA를 실행한다."""

from __future__ import annotations

import gc
import sys

from analysis import (
    calculate_statistics,
    create_interactive_visualization,
    create_static_visualization,
)
from config import (
    CONTINUOUS_TARGETS,
    DATA_PATH,
    FIXED_FEE_TARGETS,
    OUTPUT_DIR,
    ZERO_INFLATED_TARGETS,
)
from data_pipeline import (
    clean_pandas,
    clean_polars,
    download_data,
    load_with_pandas,
    load_with_polars,
    summarize_fare_exclusions,
    summarize_pandas,
    summarize_polars,
    summarize_total_consistency,
    validate_cleaned_frames,
)
from modeling import train_all_fee_models
from reporting import generate_report


def print_section(title: str) -> None:
    """실행 단계가 한눈에 구분되도록 제목을 출력한다."""
    print(f"\n{'=' * 72}")
    print(f" {title}")
    print("=" * 72)


def print_data_summary(name: str, summary: dict) -> None:
    """Pandas·Polars 로딩 결과를 동일한 형식으로 출력한다."""
    rows, columns = summary["shape"]
    missing = summary["missing"]
    print(f"\n[{name}]")
    print(f"  크기       : {rows:,}행 × {columns}열")
    print(f"  중복 행    : {summary['duplicates']:,}건")
    print(f"  결측치 합계: {sum(missing.values()):,}건")
    print("  컬럼별 자료형 / 결측치")
    print(f"    {'컬럼':<28} {'자료형':<38} {'결측치':>12}")
    print(f"    {'-' * 28} {'-' * 38} {'-' * 12}")
    for column, dtype in summary["dtypes"].items():
        print(f"    {column:<28} {dtype:<38} {missing[column]:>12,}")


def print_statistics(statistics: dict) -> None:
    """기술통계와 상관관계를 읽기 쉬운 형식으로 출력한다."""
    print_section("3. 기본 EDA")
    print("\n[수치형 기술통계]")
    print(statistics["descriptive"].T.to_string())

    print("\n[기본요금과의 상관관계]")
    for column, value in statistics["fare_correlation"].items():
        print(f"  {column:<28}: {value:>7.3f}")

    print("\n[요금 구성요소별 0원 비율]")
    for column, value in statistics["zero_ratio"].items():
        print(f"  {column:<28}: {value:>7.2%}")

    test = statistics["ttest"]
    p_value = "< 1e-300" if test["p_value"] == 0 else f"{test['p_value']:.6g}"
    print("\n[평일·주말 기본요금 Welch t-test]")
    print(f"  평일 평균: ${test['weekday_mean']:.2f} ({test['weekday_count']:,}건)")
    print(f"  주말 평균: ${test['weekend_mean']:.2f} ({test['weekend_count']:,}건)")
    print(f"  t통계량  : {test['statistic']:.4f}")
    print(f"  p-value  : {p_value}")


def print_model_results(model_result: dict) -> None:
    """9개 개별 모델과 합산 금액의 평가 결과를 출력한다."""
    print_section("5. 요금 구성요소별 모델 평가")
    print("  학습 모델: LightGBM")
    source_text = "저장 모델 불러오기" if model_result["model_source"] == "loaded" else "신규 학습"
    print(f"  모델 준비: {source_text}")
    print(
        f"  시간 분할 전체 데이터: 학습 {model_result['train_rows']:,}건 / "
        f"테스트 {model_result['test_rows']:,}건"
    )
    print("\n[연속 요금 회귀 — 달러 금액 예측]")
    print(f"  {'목표':<26} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'기준 MAE':>10}")
    print(f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
    for target in CONTINUOUS_TARGETS:
        metrics = model_result["metrics"][target]
        print(
            f"  {target:<26} {metrics['mae']:>8.3f} {metrics['rmse']:>8.3f} "
            f"{metrics['r2']:>8.3f} {metrics['baseline_mae']:>10.3f}"
        )

    print("\n[0원 여부 + 양수 금액 2단계 모델]")
    print(f"  {'목표':<26} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'부과 F1':>10}")
    print(f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
    for target in ZERO_INFLATED_TARGETS:
        metrics = model_result["metrics"][target]
        print(
            f"  {target:<26} {metrics['mae']:>8.3f} {metrics['rmse']:>8.3f} "
            f"{metrics['r2']:>8.3f} {metrics['charge_f1']:>10.3f}"
        )

    print("\n[정액 수수료 금액 클래스 분류]")
    print(f"  {'목표':<26} {'Accuracy':>10} {'weighted F1':>12} {'기준 정확도':>12}")
    print(f"  {'-' * 26} {'-' * 10} {'-' * 12} {'-' * 12}")
    for target in FIXED_FEE_TARGETS:
        metrics = model_result["metrics"][target]
        print(
            f"  {target:<26} {metrics['accuracy']:>10.3f} "
            f"{metrics['weighted_f1']:>12.3f} {metrics['baseline_accuracy']:>12.3f}"
        )

    total = model_result["total_metrics"]
    print("\n[9개 예측값 합산 vs 실제 total_amount]")
    print(f"  MAE                    : ${total['mae']:.3f}")
    print(f"  RMSE                   : ${total['rmse']:.3f}")
    print(f"  R²                     : {total['r2']:.3f}")
    print(f"  실제 구성요소 합산 MAE: ${total['component_sum_mae']:.3f}")
    print(f"  모델 재로딩 검증       : {'성공' if model_result['reload_verified'] else '실패'}")


def run() -> dict:
    """데이터 준비·정제·기본 EDA를 실행하고 결과 객체를 반환한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_section("1. 데이터 준비")
    path = download_data(destination=DATA_PATH)
    print(f"  데이터 파일: {path}")

    pandas_raw = load_with_pandas(path)
    polars_raw = load_with_polars(path)
    pandas_summary = summarize_pandas(pandas_raw)
    polars_summary = summarize_polars(polars_raw)
    fare_exclusions = summarize_fare_exclusions(pandas_raw)
    total_consistency = summarize_total_consistency(pandas_raw)
    print_section("2. Pandas · Polars 원본 비교")
    print_data_summary("Pandas", pandas_summary)
    print_data_summary("Polars", polars_summary)

    pandas_clean = clean_pandas(pandas_raw)
    polars_clean = clean_polars(polars_raw)
    validate_cleaned_frames(pandas_clean, polars_clean)
    polars_clean_rows = polars_clean.height
    removed_rows = len(pandas_raw) - len(pandas_clean)
    print("\n[정제 결과]")
    print(f"  5월 외 운행   : {fare_exclusions['outside_month_rows']:>12,}건")
    print(f"  음수 요금 행  : {fare_exclusions['negative_fare_rows']:>12,}건")
    print(f"  취소 운행 행  : {fare_exclusions['voided_trip_rows']:>12,}건")
    print(f"  두 조건 중복  : {fare_exclusions['overlap_rows']:>12,}건")
    print(f"  요금 상한 초과: {fare_exclusions['upper_limit_rows']:>12,}건")
    print(f"  요금 기준 제외: {fare_exclusions['unique_excluded_rows']:>12,}건")
    print(f"  정제 전 행 수: {len(pandas_raw):>12,}건")
    print(f"  제거된 행 수: {removed_rows:>12,}건")
    print(f"  정제 후 행 수: {len(pandas_clean):>12,}건")
    print("  도구 간 검증 : Pandas와 Polars 결과 일치")
    print("\n[요금 합계 검증 — 정제 전 결측 없는 행]")
    print(f"  검증 행 수   : {total_consistency['rows']:>12,}건")
    print(
        "  평균 절대 차이: "
        f"${total_consistency['mean_absolute_difference']:>11.4f}"
    )
    print(
        "  1센트 내 일치 : "
        f"{total_consistency['within_one_cent_ratio']:>11.2%}"
    )

    # 검증이 끝난 대용량 원본·Polars 객체를 해제해 시각화 메모리를 확보한다.
    del pandas_raw, polars_raw, polars_clean
    gc.collect()

    statistics = calculate_statistics(pandas_clean)
    print_statistics(statistics)

    print_section("4. EDA 시각화 생성")
    static_plot = create_static_visualization(pandas_clean)
    print(f"  Seaborn PNG : {static_plot}")
    interactive_plot = create_interactive_visualization(pandas_clean)
    print(f"  Plotly HTML : {interactive_plot}")

    results = {
        "comparison": {
            "pandas_shape": pandas_summary["shape"],
            "polars_shape": polars_summary["shape"],
            "pandas_duplicates": pandas_summary["duplicates"],
            "polars_duplicates": polars_summary["duplicates"],
        },
        "cleaning": {
            "pandas_rows": len(pandas_clean),
            "polars_rows": polars_clean_rows,
        },
        "statistics": statistics,
        "artifacts": {
            "static_plot": static_plot,
            "interactive_plot": interactive_plot,
        },
    }

    model_result = train_all_fee_models(pandas_clean)
    print_model_results(model_result)
    results["model"] = model_result
    results["artifacts"]["model"] = model_result["model_path"]
    report_path = generate_report(results)
    results["artifacts"]["report"] = report_path

    print_section("6. 최종 산출물")
    for name, artifact in results["artifacts"].items():
        status = "정상 생성" if artifact.exists() else "파일 없음"
        print(f"  {name:<16}: [{status}] {artifact}")
    print("\nEDA와 9개 요금 모델 평가가 완료되었습니다.\n")
    return results


def main() -> None:
    """예외를 사용자 친화적인 메시지로 출력하고 종료 코드를 설정한다."""
    try:
        run()
    except Exception as error:
        print(f"[오류] 실행 중 문제가 발생했습니다: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
