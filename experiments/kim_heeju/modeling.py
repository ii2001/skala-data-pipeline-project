"""9개 요금 구성요소를 각각 예측하고 합산 금액을 평가한다."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_TARGETS,
    FARE_COMPONENT_COLUMNS,
    FIXED_FEE_TARGETS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    RANDOM_STATE,
    TRAIN_END_DATETIME,
    ZERO_INFLATED_TARGETS,
)


def build_preprocessor() -> ColumnTransformer:
    """수치형 결측치와 범주형 미지값을 처리하는 공통 전처리기를 만든다."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def build_regression_pipeline() -> Pipeline:
    """공통 전처리와 LightGBM 회귀 모델을 하나의 Pipeline으로 묶는다."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LGBMRegressor(
                    n_estimators=250,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_classification_pipeline() -> Pipeline:
    """공통 전처리와 LightGBM 분류 모델을 하나의 Pipeline으로 묶는다."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Pandas의 pd.NA를 sklearn이 안정적으로 처리할 수 있는 np.nan으로 바꾼다."""
    features = frame[MODEL_FEATURES].copy()
    features[CATEGORICAL_FEATURES] = features[CATEGORICAL_FEATURES].astype(object)
    features[CATEGORICAL_FEATURES] = features[CATEGORICAL_FEATURES].where(
        features[CATEGORICAL_FEATURES].notna(), np.nan
    )
    return features


def split_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5월 1~24일 전체를 학습, 25~31일 전체를 테스트 데이터로 분리한다."""
    train = frame.loc[frame["tpep_pickup_datetime"] < TRAIN_END_DATETIME]
    test = frame.loc[frame["tpep_pickup_datetime"] >= TRAIN_END_DATETIME]
    if train.empty or test.empty:
        raise ValueError("시간 기준 학습 또는 테스트 데이터가 비어 있습니다.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """달러 단위 회귀 지표를 계산한다."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def fit_continuous_target(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[dict, np.ndarray, dict]:
    """연속 요금을 회귀하고 중앙값 기준 모델과 비교한다."""
    train_valid = train.dropna(subset=[target])
    test_valid_mask = test[target].notna()
    pipeline = build_regression_pipeline()
    pipeline.fit(prepare_features(train_valid), train_valid[target])
    predictions = np.clip(pipeline.predict(prepare_features(test)), 0, None)
    valid_predictions = predictions[test_valid_mask.to_numpy()]
    actual = test.loc[test_valid_mask, target]
    baseline = np.full(len(actual), train_valid[target].median())
    result = regression_metrics(actual, valid_predictions)
    result["baseline_mae"] = float(mean_absolute_error(actual, baseline))
    result["test_rows"] = int(test_valid_mask.sum())
    return result, predictions, {"kind": "regression", "pipeline": pipeline}


def fit_zero_inflated_target(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[dict, np.ndarray, dict]:
    """0이 많은 요금을 부과 여부 분류와 양수 금액 회귀의 두 단계로 예측한다."""
    train_valid = train.dropna(subset=[target])
    test_valid_mask = test[target].notna()
    classifier = build_classification_pipeline()
    classifier.fit(prepare_features(train_valid), (train_valid[target] > 0).astype(int))

    positive_train = train_valid.loc[train_valid[target] > 0]
    regressor = build_regression_pipeline()
    regressor.fit(prepare_features(positive_train), positive_train[target])

    features = prepare_features(test)
    has_charge = classifier.predict(features).astype(bool)
    positive_amount = np.clip(regressor.predict(features), 0, None)
    predictions = np.where(has_charge, positive_amount, 0.0)
    valid = test_valid_mask.to_numpy()
    actual = test.loc[test_valid_mask, target]
    result = regression_metrics(actual, predictions[valid])
    result["charge_f1"] = float(
        f1_score((actual > 0).astype(int), has_charge[valid], zero_division=0)
    )
    baseline = np.full(len(actual), train_valid[target].median())
    result["baseline_mae"] = float(mean_absolute_error(actual, baseline))
    result["test_rows"] = int(test_valid_mask.sum())
    bundle = {"kind": "two_stage", "classifier": classifier, "regressor": regressor}
    return result, predictions, bundle


def fit_fixed_fee_target(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[dict, np.ndarray, dict]:
    """정액 수수료의 금액 종류를 다중 분류로 예측한다."""
    train_valid = train.dropna(subset=[target])
    test_valid_mask = test[target].notna()
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(train_valid[target])
    classifier = build_classification_pipeline()
    classifier.fit(prepare_features(train_valid), encoded_target)
    encoded_predictions = classifier.predict(prepare_features(test)).astype(int)
    predictions = encoder.inverse_transform(encoded_predictions).astype(float)

    valid = test_valid_mask.to_numpy()
    actual = test.loc[test_valid_mask, target]
    actual_encoded = encoder.transform(actual)
    result = regression_metrics(actual, predictions[valid])
    result["accuracy"] = float(accuracy_score(actual_encoded, encoded_predictions[valid]))
    result["weighted_f1"] = float(
        f1_score(actual_encoded, encoded_predictions[valid], average="weighted", zero_division=0)
    )
    most_frequent = float(train_valid[target].mode().iloc[0])
    baseline_predictions = np.full(len(actual), most_frequent)
    result["baseline_mae"] = float(
        mean_absolute_error(actual, baseline_predictions)
    )
    encoded_baseline = encoder.transform(baseline_predictions)
    result["baseline_accuracy"] = float(accuracy_score(actual_encoded, encoded_baseline))
    result["test_rows"] = int(test_valid_mask.sum())
    bundle = {"kind": "classification", "classifier": classifier, "encoder": encoder}
    return result, predictions, bundle


def predict_from_bundle(model_bundle: dict, features: pd.DataFrame) -> np.ndarray:
    """저장된 목표별 모델 종류에 맞춰 달러 단위 예측값을 반환한다."""
    prepared = prepare_features(features)
    kind = model_bundle["kind"]
    if kind == "regression":
        return np.clip(model_bundle["pipeline"].predict(prepared), 0, None)
    if kind == "two_stage":
        has_charge = model_bundle["classifier"].predict(prepared).astype(bool)
        amount = np.clip(model_bundle["regressor"].predict(prepared), 0, None)
        return np.where(has_charge, amount, 0.0)
    if kind == "classification":
        encoded = model_bundle["classifier"].predict(prepared).astype(int)
        return model_bundle["encoder"].inverse_transform(encoded).astype(float)
    raise ValueError(f"지원하지 않는 저장 모델 종류입니다: {kind}")


def evaluate_model_bundles(
    train: pd.DataFrame, test: pd.DataFrame, models: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, float]]:
    """저장 또는 신규 모델의 개별 예측과 최종 합산 평가지표를 계산한다."""
    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for target in FARE_COMPONENT_COLUMNS:
        bundle = models[target]
        target_predictions = predict_from_bundle(bundle, test)
        predictions[target] = target_predictions
        train_valid = train.dropna(subset=[target])
        test_valid_mask = test[target].notna()
        valid = test_valid_mask.to_numpy()
        actual = test.loc[test_valid_mask, target]
        result = regression_metrics(actual, target_predictions[valid])
        if bundle["kind"] == "classification":
            encoded_actual = bundle["encoder"].transform(actual)
            encoded_predicted = bundle["encoder"].transform(target_predictions[valid])
            result["accuracy"] = float(accuracy_score(encoded_actual, encoded_predicted))
            result["weighted_f1"] = float(
                f1_score(encoded_actual, encoded_predicted, average="weighted", zero_division=0)
            )
            baseline_value = float(train_valid[target].mode().iloc[0])
            encoded_baseline = bundle["encoder"].transform(
                np.full(len(actual), baseline_value)
            )
            result["baseline_accuracy"] = float(
                accuracy_score(encoded_actual, encoded_baseline)
            )
        else:
            baseline_value = float(train_valid[target].median())
            if bundle["kind"] == "two_stage":
                predicted_charge = target_predictions[valid] > 0
                result["charge_f1"] = float(
                    f1_score((actual > 0).astype(int), predicted_charge, zero_division=0)
                )
        result["baseline_mae"] = float(
            mean_absolute_error(actual, np.full(len(actual), baseline_value))
        )
        result["test_rows"] = int(test_valid_mask.sum())
        metrics[target] = result

    predicted_total = np.column_stack(
        [predictions[target] for target in FARE_COMPONENT_COLUMNS]
    ).sum(axis=1)
    total_mask = test["total_amount"].notna()
    total_metrics = regression_metrics(
        test.loc[total_mask, "total_amount"], predicted_total[total_mask.to_numpy()]
    )
    oracle_sum = test[FARE_COMPONENT_COLUMNS].sum(axis=1, min_count=len(FARE_COMPONENT_COLUMNS))
    oracle_mask = total_mask & oracle_sum.notna()
    total_metrics["component_sum_mae"] = float(
        mean_absolute_error(test.loc[oracle_mask, "total_amount"], oracle_sum.loc[oracle_mask])
    )
    total_metrics["test_rows"] = int(total_mask.sum())
    return metrics, total_metrics


def train_all_fee_models(
    frame: pd.DataFrame, output_path: Path = OUTPUT_DIR / "fare_component_models.joblib"
) -> dict:
    """저장 모델을 우선 사용하고, 없을 때만 9개 모델을 학습·저장한다."""
    train, test = split_by_time(frame)
    if output_path.exists():
        loaded = joblib.load(output_path)
        if set(loaded.get("models", {})) != set(FARE_COMPONENT_COLUMNS):
            raise ValueError(
                "저장 모델의 목표 구성이 현재 설정과 다릅니다. "
                "모델 파일을 삭제한 뒤 다시 실행해 주세요."
            )
        if loaded.get("features") != MODEL_FEATURES:
            raise ValueError(
                "저장 모델의 입력 컬럼이 현재 설정과 다릅니다. "
                "모델 파일을 삭제한 뒤 다시 실행해 주세요."
            )
        metrics, total_metrics = evaluate_model_bundles(train, test, loaded["models"])
        return {
            "metrics": metrics,
            "total_metrics": total_metrics,
            "model_path": output_path,
            "train_rows": len(train),
            "test_rows": len(test),
            "reload_verified": True,
            "model_source": "loaded",
        }

    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, dict] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for target in CONTINUOUS_TARGETS:
            metrics[target], predictions[target], models[target] = fit_continuous_target(
                train, test, target
            )
        for target in ZERO_INFLATED_TARGETS:
            metrics[target], predictions[target], models[target] = fit_zero_inflated_target(
                train, test, target
            )
        for target in FIXED_FEE_TARGETS:
            metrics[target], predictions[target], models[target] = fit_fixed_fee_target(
                train, test, target
            )

    predicted_total = np.column_stack(
        [predictions[target] for target in FARE_COMPONENT_COLUMNS]
    ).sum(axis=1)
    total_mask = test["total_amount"].notna()
    total_metrics = regression_metrics(
        test.loc[total_mask, "total_amount"], predicted_total[total_mask.to_numpy()]
    )
    oracle_sum = test[FARE_COMPONENT_COLUMNS].sum(axis=1, min_count=len(FARE_COMPONENT_COLUMNS))
    oracle_mask = total_mask & oracle_sum.notna()
    total_metrics["component_sum_mae"] = float(
        mean_absolute_error(test.loc[oracle_mask, "total_amount"], oracle_sum.loc[oracle_mask])
    )
    total_metrics["test_rows"] = int(total_mask.sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "features": MODEL_FEATURES,
        "targets": FARE_COMPONENT_COLUMNS,
    }
    joblib.dump(bundle, output_path)
    loaded = joblib.load(output_path)
    if set(loaded["models"]) != set(FARE_COMPONENT_COLUMNS):
        raise RuntimeError("저장한 9개 요금 모델을 정상적으로 다시 불러오지 못했습니다.")
    return {
        "metrics": metrics,
        "total_metrics": total_metrics,
        "model_path": output_path,
        "train_rows": len(train),
        "test_rows": len(test),
        "reload_verified": True,
        "model_source": "trained",
    }
