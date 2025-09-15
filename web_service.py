from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
import pandas as pd
import os

import random


app = FastAPI()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

current_model = mlflow.pyfunc.load_model("models:/tracking-quickstart/2")
next_model = current_model
p = 0.8


class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
async def predict(iris: Iris):
    X = pd.DataFrame(
        [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    )

    if random.random() <= p:
        return {f"predicted: {current_model.predict(X)[0]}"}
    else:
        return {f"predicted: {next_model.predict(X)[0]}"}


@app.post("/update-model")
async def update_model(version: int):
    global next_model
    next_model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{version}")


@app.post("accept-next-model")
async def accept_next_model():
    global current_model
    current_model = next_model
