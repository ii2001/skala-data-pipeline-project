"""프로젝트 전반에서 공유하는 경로와 분석 기준을 정의한다."""

from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = DATA_DIR / "yellow_tripdata_2026-05.parquet"
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2026-05.parquet"
)

PICKUP_COLUMN = "tpep_pickup_datetime"
DROPOFF_COLUMN = "tpep_dropoff_datetime"
TARGET_COLUMN = "trip_duration_minutes"

FARE_COMPONENT_COLUMNS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]

# 정상 요금 모델을 위한 목표별 상한이다. 실제 데이터의 99.9% 분위수와 공식 정액 요금을
# 기준으로 하며, 결측치는 이상치가 아니므로 별도로 유지한다.
FARE_UPPER_LIMITS = {
    "fare_amount": 148.46,
    "extra": 10.75,
    "mta_tax": 0.50,
    "tip_amount": 30.00,
    "tolls_amount": 22.25,
    "improvement_surcharge": 1.00,
    "congestion_surcharge": 2.50,
    "Airport_fee": 2.00,
    "cbd_congestion_fee": 0.75,
}

# 원본 전체를 읽지 않고 과제에 필요한 열만 선택해 메모리 사용량을 줄인다.
SOURCE_COLUMNS = [
    "VendorID",
    PICKUP_COLUMN,
    DROPOFF_COLUMN,
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    *FARE_COMPONENT_COLUMNS,
]

# total_amount는 9개 예측값의 합계 평가에만 사용하며 원본 EDA 컬럼 표에서는 숨긴다.
LOAD_COLUMNS = [*SOURCE_COLUMNS, "total_amount"]

RANDOM_STATE = 42
PLOT_SAMPLE_SIZE = 50_000
TRAIN_END_DATETIME = datetime(2026, 5, 25)
MIN_DURATION_MINUTES = 1.0
MAX_DURATION_MINUTES = 120.0
MAX_TRIP_DISTANCE_MILES = 100.0
DATA_START_DATETIME = datetime(2026, 5, 1)
DATA_END_DATETIME = datetime(2026, 6, 1)

NUMERIC_FEATURES = [
    "passenger_count",
    "trip_distance",
    TARGET_COLUMN,
    "pickup_hour",
    "pickup_weekday",
]
CATEGORICAL_FEATURES = [
    "VendorID",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "is_weekend",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CONTINUOUS_TARGETS = ["fare_amount", "extra"]
ZERO_INFLATED_TARGETS = ["tip_amount", "tolls_amount"]
FIXED_FEE_TARGETS = [
    "mta_tax",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]
