"""모든 팀원이 동일하게 사용하는 전체 데이터 80:20 분할."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


PICKUP_COLUMN = "tpep_pickup_datetime"
SOURCE_ROW_ID = "source_row_id"
TARGET_COLUMN = "total_amount"
ANALYSIS_END = pd.Timestamp("2026-06-01")
RANDOM_STATE = 42
TEST_SIZE = 0.2
SPLIT_ID = "random_80_20_full_positive_total_v1"
EXPECTED_TRAIN_ROWS = 3_260_221
EXPECTED_TEST_ROWS = 815_056


def make_common_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전체 유효 행을 고정 시드로 학습 80%, 테스트 20%로 분할한다."""
    required = {PICKUP_COLUMN, SOURCE_ROW_ID, TARGET_COLUMN}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"공통 분할 필수 컬럼이 없습니다: {sorted(missing)}")

    in_period = dataframe[PICKUP_COLUMN].between(
        "2026-05-01", ANALYSIS_END, inclusive="left"
    )
    valid_target = dataframe[TARGET_COLUMN].notna() & dataframe[TARGET_COLUMN].gt(0)
    source = dataframe.loc[in_period & valid_target]
    train, test = train_test_split(
        source,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(
            "manifest 기준 공통 분할 행 수와 다릅니다: "
            f"train={len(train):,}, test={len(test):,}"
        )
    return train.sort_values(SOURCE_ROW_ID), test.sort_values(SOURCE_ROW_ID)

