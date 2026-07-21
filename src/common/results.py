"""팀원별 결과 JSON 형식을 검증하고 저장한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.evaluation import REQUIRED_METRICS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_REPORTS_DIR = PROJECT_ROOT / "reports" / "experiments"
REQUIRED_TOP_LEVEL = {"experiment_id", "author", "target", "metrics", "data"}


def validate_result(payload: dict[str, Any]) -> None:
    """리더보드에 필요한 필수 필드와 지표를 검사한다."""
    missing = REQUIRED_TOP_LEVEL.difference(payload)
    if missing:
        raise ValueError(f"결과 JSON 필수 항목이 없습니다: {sorted(missing)}")
    missing_metrics = set(REQUIRED_METRICS).difference(payload["metrics"])
    if missing_metrics:
        raise ValueError(f"공통 평가 지표가 없습니다: {sorted(missing_metrics)}")
    if not payload["author"] or not payload["experiment_id"]:
        raise ValueError("author와 experiment_id는 비워둘 수 없습니다.")


def save_result(author: str, payload: dict[str, Any]) -> Path:
    """검증한 결과를 팀원별 reports 디렉터리에 저장한다."""
    validate_result(payload)
    if payload["author"] != author:
        raise ValueError("저장 경로의 author와 결과의 author가 다릅니다.")
    destination = EXPERIMENT_REPORTS_DIR / author / "metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination
