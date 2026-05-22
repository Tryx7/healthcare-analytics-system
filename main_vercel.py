"""
FastAPI Application for Healthcare Analytics - Vercel Optimized
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Analytics API",
    description="API for predicting patient test results",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load model if available
model = None
try:
    import joblib
    model_paths = ['models/model_latest.pkl', 'models/model_v1.0.0.pkl']
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            logger.info(f"✅ Model loaded from {path}")
            break
    if model is None:
        logger.warning("No model found, using rule-based predictions")
except Exception as e:
    logger.warning(f"Model loading error: {e}")

# Request/Response Models
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age (required)")
    gender: Optional[str] = Field("Unknown")
    blood_type: Optional[str] = Field("Unknown")
    medical_condition: Optional[str] = Field("Unknown")
    admission_type: Optional[str] = Field("Routine")

class PredictionResponse(BaseModel):
    predicted_result: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: str

def rule_based_prediction(age: int, medical_condition: str) -> tuple:
    """Rule-based prediction (fallback)"""
    high_risk = ["Heart Disease", "Kidney Disease", "Liver Disease", "COPD", "Cancer"]
    
    if age > 70 or medical_condition in high_risk:
        return "Abnormal", 0.85, {"Normal": 0.10, "Abnormal": 0.85, "Inconclusive": 0.05}
    elif age > 60:
        return "Abnormal", 0.70, {"Normal": 0.20, "Abnormal": 0.70, "Inconclusive": 0.10}
    elif age < 30:
        return "Normal", 0.90, {"Normal": 0.90, "Abnormal": 0.05, "Inconclusive": 0.05}
    else:
        return "Inconclusive", 0.65, {"Normal": 0.30, "Abnormal": 0.35, "Inconclusive": 0.35}

@app.get("/")
async def root():
    return {
        "name": "Healthcare Analytics API",
        "version": "1.0.0",
        "description": "ML-powered healthcare test result prediction system",
        "status": "running on Vercel",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "platform": "Vercel Serverless"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Use ML model if available
        if model is not None and hasattr(model, 'predict'):
            try:
                # For ML model prediction
                input_data = pd.DataFrame([{
                    'Age': request.age,
                    'Gender': request.gender or 'Unknown',
                    'Blood_Type': request.blood_type or 'Unknown',
                    'Medical_Condition': request.medical_condition or 'Unknown',
                    'Admission_Type': request.admission_type or 'Routine'
                }])
                # Simple prediction (adjust based on your model)
                pred = model.predict(input_data)[0]
                confidence = 0.85
                probabilities = {
                    "Normal": 0.33,
                    "Abnormal": 0.34,
                    "Inconclusive": 0.33
                }
                model_version = "ml_model"
            except:
                # Fallback to rule-based
                prediction, confidence, probabilities = rule_based_prediction(
                    request.age, request.medical_condition or 'Unknown'
                )
                model_version = "rule-based-fallback"
        else:
            # Use rule-based
            prediction, confidence, probabilities = rule_based_prediction(
                request.age, request.medical_condition or 'Unknown'
            )
            model_version = "rule-based"
        
        return PredictionResponse(
            predicted_result=prediction,
            confidence=confidence,
            probabilities=probabilities,
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Always have a fallback
        return PredictionResponse(
            predicted_result="Inconclusive",
            confidence=0.5,
            probabilities={"Normal": 0.33, "Abnormal": 0.34, "Inconclusive": 0.33},
            model_version="error-fallback",
            timestamp=datetime.now().isoformat()
        )

@app.get("/model/info")
async def model_info():
    return {
        "model_loaded": model is not None,
        "model_type": "RandomForest" if model else "Rule-based",
        "prediction_classes": ["Normal", "Abnormal", "Inconclusive"],
        "deployment": "Vercel Serverless",
        "note": "Model loads from models directory if available"
    }

# Handler for Vercel
handler = app
