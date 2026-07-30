import os
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, create_model

# Path to your saved joblib file
MODEL_PATH = "/content/drive/MyDrive/linearregression/credit_risk_artifacts.joblib"

# Global artifact containers
loaded_artifacts: Dict[str, Any] = {}
pipeline = None
threshold: float = 0.5
num_features: List[str] = []
cat_features: List[str] = []
CreditApplicantSchema = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager to load model artifacts into memory on server startup.
    """
    global pipeline, threshold, num_features, cat_features, CreditApplicantSchema, loaded_artifacts
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it exists in the root directory.")
    
    # 1. Load joblib dictionary
    loaded_artifacts = joblib.load(MODEL_PATH)
    pipeline = loaded_artifacts["pipeline"]
    threshold = float(loaded_artifacts.get("best_threshold", 0.5))
    num_features = loaded_artifacts.get("num_features", [])
    cat_features = loaded_artifacts.get("cat_features", [])
    
    # 2. Dynamically build Pydantic schema based on dataset feature types
    field_definitions: Dict[str, Any] = {}
    
    for col in num_features:
        field_definitions[col] = (float, ...)  # Float required field
        
    for col in cat_features:
        field_definitions[col] = (str, ...)    # String required field

    CreditApplicantSchema = create_model("CreditApplicantSchema", **field_definitions)
    
    print(f"✓ Model loaded successfully.")
    print(f"✓ Decision Threshold: {threshold:.4f}")
    print(f"✓ Features expected: {len(num_features)} Numerical | {len(cat_features)} Categorical")
    
    yield
    
    # Cleanup on shutdown if necessary
    loaded_artifacts.clear()


app = FastAPI(
    title="UrbanVal / Credit Risk API",
    description="FastAPI service for scoring credit loan default probability using XGBoost.",
    version="1.0.0",
    lifespan=lifespan
)


# Response Schema
class PredictionResponse(BaseModel):
    default_probability: float
    is_default: int
    decision: str
    applied_threshold: float


@app.get("/", tags=["Health"])
def health_check():
    """Simple API status check."""
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "threshold": threshold
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(applicant_data: Dict[str, Any]):
    """
    Predict default risk for a single loan applicant.
    Pass a JSON object with all required feature key-value pairs.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    try:
        # Convert incoming JSON dict to DataFrame format matching pipeline input
        input_df = pd.DataFrame([applicant_data])
        
        # Ensure correct column alignment/types
        for col in num_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
        for col in cat_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)

        # Predict default probability (class 1)
        prob_default = float(pipeline.predict_proba(input_df)[:, 1][0])
        
        # Apply optimal recall decision threshold
        is_default = int(prob_default >= threshold)
        decision = "REJECT / HIGH RISK" if is_default == 1 else "APPROVE / LOW RISK"

        return PredictionResponse(
            default_probability=round(prob_default, 4),
            is_default=is_default,
            decision=decision,
            applied_threshold=round(threshold, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/model-info", tags=["Metadata"])
def get_model_metadata():
    """Retrieve details on expected input features and allowed categories."""
    return {
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "categorical_options": loaded_artifacts.get("cat_options", {}),
        "decision_threshold": threshold
    }
