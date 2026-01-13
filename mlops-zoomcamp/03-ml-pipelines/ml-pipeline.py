import os
import pickle
import mlflow
import pandas as pd
import xgboost as xgb

from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_square_error

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment()

# Create the folders to store models
models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

# Read and download data
def read_dataframe(year: int, month: int) -> pd.DataFrame:
  """
  Read the NYC taxi data into a pandas dataframe

  Args:
      year (int): year to download
      month (int): month to download

  Returns:
      pd.DataFrame: NYC taxi data
  """
  url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
  df = pd.read_parquet(url)

  # Create target duration column
  df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
  df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

  # Filter duration columns
  df = df[(df.duration >= 1) & (df.duration <= 60)]

  # Specify categorical variables
  categorical = ['PULocationID', 'DOLocationID']
  df[categorical] = df[categorical].astype(str)

  df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

  return df


# Preprocess data
def create_X(df: pd.DataFrame, dv: DictVectorizer=None) -> (pd.DataFrame, DictVectorizer):
  """
  Preprocesses data by passing it through a dictionary vectorizer

  Args:
      df (pd.DataFrame): Original data in pandas dataframe
      dv (DictVectorizer, optional): sklearn dictionary vectorizer

  Returns:
    (pd.DataFrame, DictVectorizer)
  """
  categorical = ['PU_DO']
  numerical = ['trip_distance']
  dicts = df[categorical + numerical].to_dict(orient='records')

  if dv is None:
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)
  else:
    X = dv.transform(dicts)

  return X, dv


def train_model(
  X_train: pd.DataFrame,
  y_train: pd.DataFrame,
  X_val: pd.DataFrame,
  y_val: pd.DataFrame,
  dv: DictVectorizer
) -> str:
  """
  Train and evaluate an XGBoost regression model

  Args:
    X_train (pd.DataFrame): Training data
    y_train (pd.DataFrame): Training targets
    X_val (pd.DataFrame): Validation data
    y_val (pd.DataFrame): Validation targets
    dv (DictVectorizer): Trained dictionary vectorizer

  Returns:
    str: Mlflow run id
  """
  with mlflow.start_run() as run:
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
      'learning_rate': 0.09585355369315604,
      'max_depth': 30,
      'min_child_weight': 1.060597050922164,
      'objective': 'reg:linear',
      'reg_alpha': 0.018060244040060163,
      'reg_lambda': 0.011658731377413597,
      'seed': 42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
      params=best_params,
      dtrain=train,
      num_boost_round=50,
      evals=[(valid, 'validation')],
      early_stopping_rounds=10
    )

    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
      pickle.dump(dv, f_out)

    mlflow.log_artifact(
      "models/preprocessor.b",
      artifact_path="preprocessor"
    )

    mlflow.xgboost.log_model(
      booster,
      name="models_mlflow",
      input_example=X_train
    )

    return run.info.run_id


def run(year: int, month: int) -> str:
  """
  Main execution loop

  Args:
      year (int): year to train on
      month (int): month to train on

  Returns:
      str: Mlflow run id
  """
  df_train = read_dataframe(year=year, month=month)

  next_year = year if month < 12 else year + 1
  next_month = month + 1 if month < 12 else 1
  df_val = read_dataframe(year=next_year, month=next_month)

  X_train, dv = create_X(df_train)
  X_val, _ = create_X(df_val, dv)

  target = 'duration'
  y_train = df_train[target].values
  y_val = df_val[target].values

  run_id = train_model(X_train, y_train, X_val, y_val, dv)
  print(f"MLflow run_id: {run_id}")

  return run_id

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
  parser.add_argument('--year', type=int, required=True, help="Year (YYYY) of data to be trained on")
  parser.add_argument('--month', type=int, required=True, help="Month (month number) of data to be trained on")
  args = parser.parse_args()

  run_id = run(year=args.year, month=args.month)

  with open("run_id.txt", "w") as f:
    f.write(run_id)