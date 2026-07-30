import os
import joblib
import pandas as pd
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_artifacts.joblib")

loaded_artifacts: Dict[str, Any] = {}
pipeline = None
threshold: float = 0.5
num_features: List[str] = []
cat_features: List[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, threshold, num_features, cat_features, loaded_artifacts
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
    
    loaded_artifacts = joblib.load(MODEL_PATH)
    pipeline = loaded_artifacts["pipeline"]
    threshold = float(loaded_artifacts.get("best_threshold", 0.5))
    num_features = loaded_artifacts.get("num_features", [])
    cat_features = loaded_artifacts.get("cat_features", [])
    
    yield
    loaded_artifacts.clear()

app = FastAPI(
    title="UrbanVal / Credit Risk API",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionResponse(BaseModel):
    default_probability: float
    is_default: int
    decision: str
    applied_threshold: float

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "online", "model_loaded": pipeline is not None, "threshold": threshold}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(applicant_data: Dict[str, Any]):
    if pipeline is None:
        raise HTTPException(status_code=533, detail="Model uninitialized")

    try:
        input_df = pd.DataFrame([applicant_data])
        for col in num_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        for col in cat_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)

        prob_default = float(pipeline.predict_proba(input_df)[:, 1][0])
        is_default = int(prob_default >= threshold)
        decision = "REJECT / HIGH RISK" if is_default == 1 else "APPROVE / LOW RISK"

        return PredictionResponse(
            default_probability=round(prob_default, 4),
            is_default=is_default,
            decision=decision,
            applied_threshold=round(threshold, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")
