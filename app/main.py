
from fastapi import FastAPI
import pandas as pd
import joblib

from app.schemas import EmployeeData

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Machine Learning API for employee attrition prediction",
    version="1.0.0"
)

# Load trained model
model = joblib.load("models/attrition_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Employee Attrition Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: EmployeeData):

    employee = pd.DataFrame([data.model_dump()])

    prediction = int(model.predict(employee)[0])

    probability = float(
        model.predict_proba(employee)[0][1]
    )

    return {
        "attrition_prediction": (
            "Yes" if prediction == 1 else "No"
        ),
        "prediction_value": prediction,
        "attrition_probability": round(probability, 4)
    }
