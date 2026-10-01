"""uvicorn app:app --port 8000"""
from fastapi import FastAPI

from predict import ChurnModel, Customer

app = FastAPI(title="Churn prediction")
model = ChurnModel()


@app.get("/health")
def health():
    return {"status": "ok", "model": model.metadata["model"], "trained_at": model.metadata["trained_at"]}


@app.post("/predict")
def predict(customer: Customer):
    return model.predict(customer)
