# Real-Time Fraud Detection System

A machine learning system that detects fraudulent credit card transactions 
in real time, deployed as a REST API ready for production.

## The Problem

Financial institutions lose billions annually to fraudulent transactions. 
The challenge: only 0.17% of transactions are fraud — a severely imbalanced 
problem where a naive model predicting "no fraud" achieves 99.83% accuracy 
but detects zero fraudulent transactions.

## The Solution

An end-to-end ML pipeline that:
- Identifies fraud with 85.7% Recall and 85.7% Precision
- Generates only 14 false alarms per 98 fraud cases detected
- Serves predictions via REST API in milliseconds
- Runs in a Docker container — reproducible anywhere

## Results

| Metric             | Logistic Regression | XGBoost (Final) |
|--------------------|--------------------:|----------------:|
| Recall             | 0.9082              | 0.8571          |
| Precision          | 0.0551              | 0.8571          |
| False Alarms       | 1,527               | 14              |
| Fraud Detected     | 89 / 98             | 84 / 98         |

**Why XGBoost?** Logistic Regression detects 5 more frauds but generates 
1,513 additional false alarms — blocking legitimate customers at an 
unacceptable operational cost.

## Tech Stack

- **Model**: XGBoost optimized with Optuna
- **API**: FastAPI + Uvicorn
- **Container**: Docker
- **Language**: Python 3.10

## Project Structure

```
fraud-detection-api/
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature engineering pipeline
│   └── 03_modeling.ipynb             # Model training and evaluation
├── api/
│   └── main.py                       # FastAPI inference endpoint
├── models/
│   ├── fraud_model.pkl               # Trained XGBoost model
│   └── threshold.pkl                 # Optimized decision threshold
├── Dockerfile
└── requirements.txt
```

## Quick Start

**With Docker:**
```bash
docker build -t fraud-detection-api .
docker run -d -p 8000:8000 fraud-detection-api
```

**Without Docker:**
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

**API Documentation:** http://127.0.0.1:8000/docs

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.27, "V2": 2.46, ..., "Amount": 0.01, "Hour": 15.8}'
```

**Response:**
```json
{
  "is_fraud": true,
  "probability": 0.9991,
  "risk_level": "HIGH",
  "threshold_used": 0.22
}
```

## Key Technical Decisions

**Why log transformation on Amount?**
Amount has extreme outliers ($25,691 max). Log compression improves 
model performance without losing information.

**Why threshold = 0.22?**
Optimized to maximize Recall while maintaining Precision ≥ 0.70 — 
reflecting the business constraint that undetected fraud costs more 
than false alarms.

**Why XGBoost over Logistic Regression?**
Despite slightly lower Recall, XGBoost generates 99% fewer false alarms —
a critical operational advantage in production fraud systems.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
— 284,807 transactions, 0.17% fraud rate.
