"""
Create metrics using evidently to be added to Grafana
"""
import datetime
import time
import random
import logging
import uuid
import pytz
import pandas as pd
import io
import psycopg
import os
import json
import joblib
from dotenv import load_dotenv

from evidently import Dataset, DataDefinition, Report, Regression
from evidently.metrics import *
from evidently.presets import *

load_dotenv()

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
DROP TABLE IF EXISTS evident_metrics;
CREATE TABLE evident_metrics (
  timestamp timestamp,
  prediction_drift FLOAT,
  num_drifted_columns INTEGER,
  share_missing_values FLOAT
);
"""

# Load data
ref = pd.read_parquet("../data/clean/reference.parquet")
tar = pd.read_parquet("../data/raw/green_tripdata_2025-03.parquet")
tar = tar.drop("ehail_fee", axis=1)

# Load model
with open("models/lin_reg.bin", 'rb') as f_in:
  model = joblib.load(f_in)

# Define date start point and columns
begin = datetime.datetime(2025, 3, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']

# Define report metrics
report = Report([
  DataDriftPreset(),
  DatasetMissingValueCount(),
  ValueDrift(column="prediction")
])

def prep_db():
  with psycopg.connect(f"host=localhost port=5434 user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASS")}", autocommit=True) as conn:
    res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    if len(res.fetchall()) == 0:
      conn.execute("CREATE DATABASE test;")
    
    with psycopg.connect(f"host=localhost port=5434 dbname=test user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASS")}", autocommit=True) as conn1:
      conn1.execute(create_table_statement)


def calculate_metrics(curr, ref, tar, i):
  # Narrow down the month range
  current_data = tar[(tar['lpep_pickup_datetime'] >= (begin + datetime.timedelta(i)))
  & (tar['lpep_pickup_datetime'] < (begin + datetime.timedelta(i + 1)))]

  # Make prediction with model
  current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

  # Define the data schema
  schema = DataDefinition(
    regression=[Regression(target="duration_min", prediction="prediction")],
    numerical_columns=num_features,
    categorical_columns=cat_features
  )

  ref_ds = Dataset.from_pandas(
    ref,
    data_definition=schema
  )

  curr_ds = Dataset.from_pandas(
    current_data,
    data_definition=schema
  )

  # Create evidently report
  out = report.run(reference_data=ref_ds, current_data=current_data)
  result = json.loads(out.json())

  prediction_drift = result['metrics'][17]['value']
  num_drifted_cols = result['metrics'][0]['value']['count']
  share_missing_values = result['metrics'][19]['value']['share']

  # Push to db
  curr.execute(
    "INSERT INTO evident_metrics (timestamp, prediction_drift, num_drifted_columns, share_missing_values) VALUES (%s, %s, %s, %s)",
    (begin + datetime.timedelta(i), prediction_drift, num_drifted_cols, share_missing_values)
  )

def main():
  prep_db()
  last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
  with psycopg.connect(f"host=localhost port=5434 dbname=test user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASS")}", autocommit=True) as conn:
    for i in range(0, 30):
      with conn.cursor() as curr:
        calculate_metrics(curr, ref, tar, i)
      
      new_send = datetime.datetime.now()
      seconds_elapsed = (new_send - last_send).total_seconds()
      if seconds_elapsed < SEND_TIMEOUT:
        time.sleep(SEND_TIMEOUT - seconds_elapsed)
      while last_send < new_send:
        last_send += datetime.timedelta(seconds=10)
      logging.info("data sent")

if __name__ == "__main__":
  main()
