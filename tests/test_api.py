
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():

    data = {
        "Age": 30,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1100,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 70,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert result["attrition_prediction"] in ["Yes", "No"]
    assert result["prediction_value"] in [0, 1]
    assert 0 <= result["attrition_probability"] <= 1


def test_invalid_age():
    response = client.post("/predict", json={"Age": 10})
    assert response.status_code == 422
