
# Employee Attrition Prediction System

An End-to-End Machine Learning and MLOps
project that predicts employee attrition using
employee-related information.

## Project Overview
This project predicts whether an employee is
likely to leave an organization using Machine
Learning classification algorithms.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Docker
- Pytest
- GitHub Actions
- Render

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest

## Project Architecture
Dataset
   |
Data Preprocessing
   |
Model Training
   |
Model Evaluation
   |
Saved ML Pipeline
   |
FastAPI REST API
   |
Docker Container
   |
GitHub Actions
   |
Render Deployment

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Home |
| GET | /health | Health check |
| POST | /predict | Attrition prediction |

## Live Deployment

API:
https://employee-attrition-api-s0v0.onrender.com

Swagger:
https://employee-attrition-api-s0v0.onrender.com/docs

## Run Using Docker

docker build -t employee-attrition-api .

docker run -d -p 8000:8000 \
--name attrition-api employee-attrition-api

## Run Tests

docker run --rm employee-attrition-api python -m pytest -v

## Dataset
IBM HR Analytics Employee Attrition Dataset.

## Testing
Automated tests using Pytest.
CI pipeline using GitHub Actions.

## Deployment
Deployed using Render.

## Disclaimer
This project is for educational and research
purposes. Predictions should not be used as
the sole basis for employment decisions.
