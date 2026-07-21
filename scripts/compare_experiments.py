"""팀원별 metrics.json을 모아 공통 리더보드와 선정 문서를 만든다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.results import validate_result
from src.common.split import EXPECTED_TEST_ROWS, SPLIT_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "reports" / "experiments"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "comparison"


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    """팀원별 결과 파일을 읽고 공통 스키마를 검증한다."""
    results = []
    for path in sorted(input_dir.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_result(payload)
        if payload["author"] != path.parent.name:
            raise ValueError(
                f"결과 author와 디렉터리가 다릅니다: {path.parent.name} != "
                f"{payload['author']}"
            )
        try:
            payload["source"] = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            payload["source"] = str(path)
        results.append(payload)
    if not results:
        raise FileNotFoundError(f"비교할 metrics.json이 없습니다: {input_dir}")
    return results


def build_leaderboard(results: list[dict[str, Any]]) -> pd.DataFrame:
    """MAE 오름차순의 공통 비교표를 만든다."""
    rows = []
    for result in results:
        metrics = result["metrics"]
        data = result["data"]
        rows.append(
            {
                "author": result["author"],
                "experiment_id": result["experiment_id"],
                "target": result["target"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "median_ae": metrics["median_ae"],
                "r2": metrics["r2"],
                "train_rows": data.get("train_rows"),
                "test_rows": data.get("test_rows"),
                "excluded_test_rows": data.get("excluded_test_rows"),
                "leakage_check": data.get("leakage_check", False),
                "eligible": (
                    result["target"] == "total_amount"
                    and data.get("leakage_check", False)
                    and data.get("split_id") == SPLIT_ID
                    and data.get("test_rows") == EXPECTED_TEST_ROWS
                    and data.get("excluded_test_rows") == 0
                ),
                "source": result["source"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eligible", "mae"],
        ascending=[False, True],
    )


def write_selection_report(leaderboard: pd.DataFrame, output_dir: Path) -> Path:
    """자동 순위와 수동 검토 항목이 있는 선정 문서를 작성한다."""
    rows = []
    for rank, row in enumerate(leaderboard.itertuples(index=False), start=1):
        rows.append(
            f"| {rank} | {row.author} | {row.experiment_id} | "
            f"{row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} | "
            f"{'Y' if row.leakage_check else 'N'} | "
            f"{'Y' if row.eligible else 'N'} |"
        )
    report = "\n".join(
        [
            "# 팀 모델 비교 및 최종 선정",
            "",
            "MAE가 낮은 순서를 기본으로 표시하되 누수 검증을 통과한 실험을 우선한다.",
            "최종 선정 전에는 고액 구간 오차, 재현성, 코드 품질을 수동 검토한다.",
            "",
            "| 순위 | 작성자 | 실험 ID | MAE | RMSE | R² | 누수 검증 | 적격 |",
            "|---:|---|---|---:|---:|---:|:---:|:---:|",
            *rows,
            "",
            "## 최종 선정 체크리스트",
            "",
            "- [ ] 공통 원본 행과 시간 홀드아웃 사용",
            f"- [ ] 고정 테스트 {EXPECTED_TEST_ROWS:,}행 전체 예측",
            "- [ ] 정답 및 직접 요금 구성 컬럼 누수 없음",
            "- [ ] $100 이상 구간 오차 확인",
            "- [ ] Pipeline 재로딩 및 재예측 성공",
            "- [ ] 전처리 기준과 제외 행 수 설명 가능",
            "- [ ] 최종 선정 사유 작성",
            "",
        ]
    )
    destination = output_dir / "selection_report.md"
    destination.write_text(report, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="팀 실험 결과 비교")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = load_results(args.input_dir)
    leaderboard = build_leaderboard(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    report_path = write_selection_report(leaderboard, args.output_dir)
    print(leaderboard.to_string(index=False))
    print(f"선정 문서: {report_path}")


if __name__ == "__main__":
    main()
