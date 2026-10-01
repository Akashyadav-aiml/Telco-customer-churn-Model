# Telco customer churn

Take-home for the ML Engineer role. Predicts churn on the Kaggle Telco dataset and serves the model through a small script and a FastAPI endpoint.

All the work is in `churn_model.ipynb` (EDA, training, evaluation). The last section of the notebook writes `predict.py`, `app.py` and `test_inference.py` and saves the model to `models/`.

## Setup

Needs Python 3.11+ (I used 3.13).

```
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

The CSV isn't committed. To re-run the notebook, download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) into `data/`. The trained model is already in `models/`, so prediction works without it.

## Usage

```
python predict.py samples/customer.json
```

```
{
  "customerID": "7590-VHVEG",
  "churn_prediction": 1,
  "churn_probability": 0.7093
}
```

API:

```
uvicorn app:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @samples/customer.json

# PowerShell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -InFile samples\customer.json
```

Bad input (unknown category, negative tenure, missing field) gets a 422. `TotalCharges` can be a number or a string like in the raw data.

Retrain: run all cells in the notebook (about 1.5 min), or `jupyter execute --inplace churn_model.ipynb`.

Tests: `pytest`

## Approach

Data
- `TotalCharges` is read as text. 11 rows are blank and all of them have tenure 0 (new customers, nothing billed yet), so I fill them with 0 rather than a median.
- Dropped `customerID`, one-hot encoded the categoricals, scaled the numerics.
- Churn is 26.5%. I used a stratified split and stratified 5-fold CV, PR-AUC as the main metric, and a tuned decision threshold. I didn't use class weights because they distort the probabilities, and the API returns a probability.

Leakage
- 80/20 split first. The test set is only used for the final numbers.
- All fitted preprocessing is inside the sklearn Pipeline, so CV refits it on each fold.
- The model is chosen on CV PR-AUC and the threshold on out-of-fold predictions from the train set (max F1).

Models
- Dummy (majority class) as a reference, logistic regression as the baseline, then random forest and XGBoost.
- RandomizedSearchCV for all three: 40 iterations for XGBoost, 15 for the others.

## Results

Test set is 1,409 customers. F1, precision, recall and accuracy are at each model's tuned threshold.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | F1 | Precision | Recall | Accuracy | Threshold |
|---|---|---|---|---|---|---|---|---|
| Dummy | 0.265 | 0.265 | 0.500 | 0.000 | 0.000 | 0.000 | 0.735 | 0.50 |
| Logistic Regression | 0.662 ± 0.020 | 0.634 | 0.842 | 0.617 | 0.539 | 0.722 | 0.762 | 0.33 |
| Random Forest | 0.667 ± 0.020 | 0.656 | 0.845 | 0.630 | 0.556 | 0.727 | 0.774 | 0.34 |
| XGBoost | 0.673 ± 0.018 | 0.666 | 0.848 | 0.636 | 0.563 | 0.733 | 0.778 | 0.35 |

XGBoost is the saved model. Its lead over logistic regression is small, within one CV std. The best XGBoost config uses depth-2 trees, so the signal is mostly additive. If explainability mattered more I'd ship the logistic regression.

The threshold made more difference than the choice of model. At 0.5, XGBoost catches 192 of the 374 churners in the test set. At 0.35 it catches 274, and F1 goes from 0.578 to 0.636.

The main drivers are a month-to-month contract, no online security or tech support, fiber optic internet, and paying by electronic check.

## Why not accuracy

Only 26.5% of customers churn, so predicting "no churn" for everyone already gets 73.5% accuracy while catching nobody. XGBoost gets 77.8%, which barely looks better on paper, but it finds 73% of the churners.

The two errors also don't cost the same: a missed churner is lost revenue, a false alarm is one retention offer. And accuracy only describes one threshold, while what the business needs is a ranking of who to contact. So I looked at PR-AUC and ROC-AUC for ranking quality, and at F1, precision and recall at the chosen threshold.

## With 2 more days

1. Move the training code out of the notebook into a small package with a CLI and a config file. Track runs and models in MLflow so retrains are reproducible and comparable.
2. Add a Docker image and CI (tests, lint), plus batch scoring. Churn is usually scored for the whole customer base at once, not one customer at a time.
3. Add monitoring: log predictions, watch input and score drift, and track precision and recall once labels come in. Set the threshold from the actual cost of a retention offer versus a lost customer, instead of F1.
