"""Score a single customer with the saved churn model.

    python predict.py samples/customer.json
    python predict.py - < samples/customer.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    customerID: str | None = None
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def blank_is_none(cls, v):
        # raw data has " " for customers that haven't been billed yet
        return None if isinstance(v, str) and not v.strip() else v


class ChurnModel:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["threshold"]
        self.metadata = bundle["metadata"]

    def predict(self, data):
        customer = data if isinstance(data, Customer) else Customer.model_validate(data)
        row = pd.DataFrame([customer.model_dump()])
        row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")  # same as training
        proba = float(self.pipeline.predict_proba(row[self.pipeline.feature_names_in_])[0, 1])
        return {
            "customerID": customer.customerID,
            "churn_prediction": int(proba >= self.threshold),
            "churn_probability": round(proba, 4),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="customer JSON file, or - for stdin")
    args = parser.parse_args()
    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
    print(json.dumps(ChurnModel().predict(json.loads(raw)), indent=2))
