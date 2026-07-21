"""팀 모델을 같은 방식으로 비교하기 위한 회귀 평가 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


REQUIRED_METRICS = ("mae", "rmse", "median_ae", "r2")


def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray,
) -> dict[str, float]:
    """결측·무한값을 허용하지 않고 공통 회귀 지표를 계산한다."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("정답과 예측값의 shape이 다릅니다.")
    if actual.size < 2:
        raise ValueError("R² 계산에는 평가 행이 2개 이상 필요합니다.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("평가값에 결측 또는 무한값이 있습니다.")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "median_ae": float(median_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_fare_bands(
    y_true: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray,
) -> list[dict[str, float | int | str]]:
    """실제 total_amount 구간별 공통 오차를 계산한다."""
    source = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(predictions, dtype=float),
        }
    )
    source["fare_band"] = pd.cut(
        source["actual"],
        bins=[-np.inf, 30, 60, 100, np.inf],
        labels=["under_30", "30_to_60", "60_to_100", "100_plus"],
        right=False,
    )
    results = []
    for band, group in source.groupby("fare_band", observed=True):
        metrics = evaluate_regression(group["actual"], group["predicted"])
        results.append({"fare_band": str(band), "rows": len(group), **metrics})
    return results
