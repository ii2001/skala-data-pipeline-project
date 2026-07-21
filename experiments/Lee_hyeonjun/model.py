import os
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import lightgbm as lgb

# 공통 분할 및 저장 스키마 모듈 가정 (프로젝트 구조 규칙 반영)
# from src.common.split import make_common_split
# from src.common.results import save_result

# 1. 전체 데이터 로드 (data/raw/ 경로 규칙 반영)
print("전체 데이터 로딩 중...")
data_path = '/Users/home/test/yellow_tripdata_2026-05.parquet'
if not os.path.exists(data_path):
    data_path = '../yellow_tripdata_2026-05.parquet' # 백업 경로
df = pd.read_parquet(data_path)

# 2. 파생 변수 생성 및 전처리
print("데이터 전처리 중...")
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# 실제 탑승한 시간(duration) 및 요일 추가
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_dow'] = df['tpep_pickup_datetime'].dt.dayofweek

# 비정상 데이터 필터링 (거리 및 시간 0 이하 제거)
df_clean = df[(df['trip_distance'] > 0) & (df['trip_duration'] > 0)].copy()

# passenger_count 결측치 0으로 채우기
df_clean['passenger_count'] = df_clean['passenger_count'].fillna(0)

# 타겟 및 피처 분리 (9개 요금 컬럼 및 VendorID 제거 후 total_amount만 타겟으로 지정)
target_col = 'total_amount'
drop_cols = [
    'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 
    'improvement_surcharge', 'congestion_surcharge', 'Airport_fee', 'cbd_congestion_fee',
    'total_amount', 'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag'
]

df_clean = df_clean.dropna(subset=[target_col])

# 남은 컬럼 중 특성과 타겟 정의
feature_cols = [col for col in df_clean.columns if col not in drop_cols]

X = df_clean[feature_cols]
y = df_clean[target_col]

# 공통 스플릿 적용 (프로젝트 규칙 연동: src.common.split.make_common_split 활용 권장)
# X_train, X_test, y_train, y_test = make_common_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 전처리 파이프라인 구축
# - passenger_count 결측치는 이미 0으로 채웠으나 파이프라인 안정성을 위해 medain 임퓨터 유지
# - PULocationID, DOLocationID, payment_type, RatecodeID 등 범주형 컬럼 원핫 인코딩
numeric_features = ['passenger_count', 'trip_distance', 'trip_duration', 'pickup_hour', 'pickup_dow']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_features = ['RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# 4. 모델 학습 및 평가
print(f"모델 학습 중... (총 {len(X_train):,} 건)")

X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, device='cpu')
model.fit(X_train_processed, y_train)

pred = model.predict(X_test_processed)

# 지표 계산
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)

print("\n" + "="*50)
print("             [ total_amount 예측 성능 결과 ]")
print("="*50)
print(f"🔹 R2 Score : {r2:.4f}")
print(f"🔹 MAE      : ${mae:.4f}")
print(f"🔹 RMSE     : ${rmse:.4f}")
print("="*50)

# 5. 결과 저장 (reports/experiments/{author}/metrics.json 규격 준수용 출력)
metrics_result = {
    "r2_score": float(r2),
    "mae": float(mae),
    "rmse": float(rmse)
}