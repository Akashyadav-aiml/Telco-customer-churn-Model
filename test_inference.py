import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import app
from predict import ChurnModel

SAMPLE = json.loads((Path(__file__).parent / "samples" / "customer.json").read_text())


@pytest.fixture(scope="module")
def model():
    return ChurnModel()


def test_sample(model):
    out = model.predict(SAMPLE)
    assert out["customerID"] == "7590-VHVEG"
    assert 0 <= out["churn_probability"] <= 1
    assert out["churn_prediction"] == int(out["churn_probability"] >= model.threshold)


def test_total_charges_string_or_number(model):
    assert model.predict(SAMPLE) == model.predict({**SAMPLE, "TotalCharges": 29.85})


def test_blank_total_charges(model):
    assert 0 <= model.predict({**SAMPLE, "tenure": 0, "TotalCharges": " "})["churn_probability"] <= 1


def test_risky_customer_scores_higher(model):
    risky = {**SAMPLE, "InternetService": "Fiber optic", "tenure": 2, "MonthlyCharges": 95.0,
             "TotalCharges": "190"}
    loyal = {**SAMPLE, "Contract": "Two year", "tenure": 60, "MonthlyCharges": 55.0,
             "TotalCharges": "3300", "PaymentMethod": "Credit card (automatic)"}
    assert model.predict(risky)["churn_probability"] > model.predict(loyal)["churn_probability"]


@pytest.mark.parametrize("field, value", [("Contract", "Weekly"), ("tenure", -1), ("MonthlyCharges", "abc")])
def test_bad_input(model, field, value):
    with pytest.raises(ValidationError):
        model.predict({**SAMPLE, field: value})


def test_api():
    client = TestClient(app)
    assert client.post("/predict", json=SAMPLE).status_code == 200
    assert client.post("/predict", json={**SAMPLE, "Contract": "Weekly"}).status_code == 422
