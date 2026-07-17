# Main API file for fraud detection

# Imports and model loading
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# App initialization
app = FastAPI(
    title="Fraud Detection API",
    description="API for fraud detection",
    version="1.0.0"
)

# Loading model and threshold
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'fraud_model.pkl'))
threshold = joblib.load(os.path.join(BASE_DIR, 'models', 'threshold.pkl'))

print(f"Model loaded correctly")
print(f"Threshold: {threshold:.2f}")

# Input definition


class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Hour: float

# Prediction endpoint


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": threshold
    }


'''
@app.post("/predict")
def predict(transaction: Transaction):
    # Aplicar log transformation a Amount
    amount_log = np.log1p(transaction.Amount)

    # Construir array en el orden exacto que el modelo espera
    # V1-V28, Hour, Amount_log
    features = np.array([[
        transaction.V1, transaction.V2, transaction.V3,
        transaction.V4, transaction.V5, transaction.V6,
        transaction.V7, transaction.V8, transaction.V9,
        transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18,
        transaction.V19, transaction.V20, transaction.V21,
        transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27,
        transaction.V28, transaction.Hour, amount_log
    ]])

    # Predecir
    probability = model.predict_proba(features)[0][1]
    is_fraud = probability >= threshold

    # Nivel de riesgo
    if probability >= 0.8:
        risk_level = "HIGH"
    elif probability >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "is_fraud": bool(is_fraud),
        "probability": round(float(probability), 4),
        "risk_level": risk_level,
        "threshold_used": round(float(threshold), 4)
    }
'''


@app.post("/predict")
def predict(transaction: Transaction):
    amount_log = np.log1p(transaction.Amount)

    features = np.array([[
        transaction.V1, transaction.V2, transaction.V3,
        transaction.V4, transaction.V5, transaction.V6,
        transaction.V7, transaction.V8, transaction.V9,
        transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18,
        transaction.V19, transaction.V20, transaction.V21,
        transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27,
        transaction.V28, transaction.Hour, amount_log
    ]])

    # Debug
    print(f"Features shape: {features.shape}")
    print(f"Amount original: {transaction.Amount}")
    print(f"Amount_log: {amount_log}")
    print(f"Features: {features}")

    proba = model.predict_proba(features)
    print(f"Probabilidades raw: {proba}")

    probability = proba[0][1]
    is_fraud = probability >= threshold

    if probability >= 0.8:
        risk_level = "HIGH"
    elif probability >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "is_fraud": bool(is_fraud),
        "probability": round(float(probability), 4),
        "risk_level": risk_level,
        "threshold_used": round(float(threshold), 4)
    }
