from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

loaded_model = mlflow.pyfunc.load_model("models:/tracking-quickstart/2")


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
    return {f"predicted: {loaded_model.predict(X)[0]}"}


@app.patch("/update-model")
async def update_model(version: int):
    global loaded_model
    loaded_model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{version}")
