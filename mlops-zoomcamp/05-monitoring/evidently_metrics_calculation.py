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
from dotenv import load_dotenv

from evidently import Dataset, DataDefinition, Report
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
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics (
  timestamp timestamp,
  value1 INTEGER,
  value2 VARCHAR,
  value3 FLOAT
);
"""

reference_data = pd.read_parquet("../data/clean/reference.parquet")
with open("models/lin_reg.bin", 'rb') as f_in:
  model = joblib.load(f_in)

raw_data = pd.read_parquet("../data/raw/green_tripdata_2025-03.parquet")
begin = datetime.datetime(2025, 3, 1, 0, 0)

def prep_db():
  with psycopg.connect(f"host=localhost port=5434 user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASS")}", autocommit=True) as conn:
    res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    if len(res.fetchall()) == 0:
      conn.execute("CREATE DATABASE test;")
    
    with psycopg.connect(f"host=localhost port=5434 dbname=test user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASS")}", autocommit=True) as conn1:
      conn1.execute(create_table_statement)


def calculate_dummy_metrics_postgresql(curr):
  value1 = rand.randint(0, 1000)
  value2 = str(uuid.uuid4())
  value3 = rand.random()

  curr.execute(
    "INSERT INTO dummy_metrics (timestamp, value1, value2, value3) VALUES (%s, %s, %s, %s)",
    (datetime.datetime.now(pytz.timezone("Singapore")), value1, value2, value3)
  )

def main():
  prep_db()
  last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
  with psycopg.connect(f"host=localhost port=5434 dbname=test user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASS")}", autocommit=True) as conn:
    for i in range(0, 100):
      with conn.cursor() as curr:
        calculate_dummy_metrics_postgresql(curr)
      
      new_send = datetime.datetime.now()
      seconds_elapsed = (new_send - last_send).total_seconds()
      if seconds_elapsed < SEND_TIMEOUT:
        time.sleep(SEND_TIMEOUT - seconds_elapsed)
      while last_send < new_send:
        last_send += datetime.timedelta(seconds=10)
      logging.info("data sent")

if __name__ == "__main__":
  main()
