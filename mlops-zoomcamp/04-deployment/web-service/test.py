import requests

# # For default deployment
# ride = {
#   "PULocationID": 130,
#   "DOLocationID": 20,
#   "trip_distance": 3.
# }

# url = "http://localhost:9696/predict"
# response = requests.post(url, json=ride)
# print(response.json())

# For Mlflow deployment
ride = {
  "dataframe_split": {
    "columns": ["PU_DO", "trip_distance"],
    "data": [["166_41", 0.65]]
  }
}

url = "http://localhost:9696/invocations"
response = requests.post(url, json=ride)
print(response.json())