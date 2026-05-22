"""
FastAPI Application for Healthcare Analytics System
Provides REST API endpoints for predictions, data management, and model info.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Analytics API",
    description="API for predicting patient test results using ML",
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

# Simplified Prediction Request with all optional fields
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age")
    gender: str = Field(default="Unknown", description="Gender: Male, Female, or Other")
    blood_type: str = Field(default="Unknown", description="Blood type")
    medical_condition: str = Field(default="Unknown", description="Medical condition")
    admission_type: str = Field(default="Routine", description="Admission type")
    billing_amount: Optional[float] = Field(default=5000.0, description="Billing amount")
    length_of_stay: Optional[int] = Field(default=3, description="Length of hospital stay")
    medication: Optional[str] = Field(default="None", description="Prescribed medication")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "gender": "Male",
                "blood_type": "A+",
                "medical_condition": "Heart Disease",
                "admission_type": "Emergency"
            }
        }

class PredictionResponse(BaseModel):
    predicted_result: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: str
    patient_age: int
    medical_condition: str

class BatchPredictionRequest(BaseModel):
    patients: List[PredictionRequest]

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str

# Rule-based prediction function (no ML model needed)
def predict_healthcare(age: int, medical_condition: str, admission_type: str = "Routine") -> tuple:
    """
    Advanced rule-based prediction for healthcare test results
    Returns: (prediction, confidence, probabilities_dict)
    """
    
    # Define risk levels for conditions
    high_risk_conditions = [
        "Heart Disease", "Kidney Disease", "Liver Disease", 
        "COPD", "Cancer", "Stroke", "Pneumonia"
    ]
    medium_risk_conditions = [
        "Diabetes", "Hypertension", "Asthma", "Arthritis",
        "Thyroid Disorder", "Anemia", "COVID-19"
    ]
    low_risk_conditions = [
        "Healthy", "Flu", "Migraine", "Bronchitis", "Infection"
    ]
    
    # Base probabilities by age group
    if age > 70:
        base_probs = {"Normal": 0.10, "Abnormal": 0.80, "Inconclusive": 0.10}
        base_confidence = 0.80
    elif age > 60:
        base_probs = {"Normal": 0.15, "Abnormal": 0.75, "Inconclusive": 0.10}
        base_confidence = 0.75
    elif age > 50:
        base_probs = {"Normal": 0.30, "Abnormal": 0.55, "Inconclusive": 0.15}
        base_confidence = 0.65
    elif age > 40:
        base_probs = {"Normal": 0.50, "Abnormal": 0.35, "Inconclusive": 0.15}
        base_confidence = 0.60
    elif age > 30:
        base_probs = {"Normal": 0.65, "Abnormal": 0.20, "Inconclusive": 0.15}
        base_confidence = 0.70
    elif age >= 18:
        base_probs = {"Normal": 0.80, "Abnormal": 0.10, "Inconclusive": 0.10}
        base_confidence = 0.85
    else:
        base_probs = {"Normal": 0.85, "Abnormal": 0.05, "Inconclusive": 0.10}
        base_confidence = 0.90
    
    # Adjust based on medical condition
    if medical_condition in high_risk_conditions:
        base_probs["Abnormal"] += 0.20
        base_probs["Normal"] -= 0.15
        base_probs["Inconclusive"] -= 0.05
        base_confidence = min(0.95, base_confidence + 0.10)
    elif medical_condition in medium_risk_conditions:
        base_probs["Abnormal"] += 0.10
        base_probs["Normal"] -= 0.10
        base_confidence = min(0.90, base_confidence + 0.05)
    elif medical_condition in low_risk_conditions:
        base_probs["Normal"] += 0.10
        base_probs["Abnormal"] -= 0.05
        base_probs["Inconclusive"] -= 0.05
        base_confidence = min(0.95, base_confidence + 0.05)
    
    # Adjust based on admission type
    if admission_type == "Emergency":
        base_probs["Abnormal"] += 0.10
        base_probs["Normal"] -= 0.05
        base_probs["Inconclusive"] -= 0.05
        base_confidence = min(0.95, base_confidence + 0.05)
    elif admission_type == "Urgent":
        base_probs["Abnormal"] += 0.05
        base_probs["Normal"] -= 0.03
        base_probs["Inconclusive"] -= 0.02
    
    # Clamp values between 0 and 1
    for key in base_probs:
        base_probs[key] = max(0.0, min(1.0, base_probs[key]))
    
    # Normalize to sum to 1
    total = sum(base_probs.values())
    if total > 0:
        base_probs = {k: v/total for k, v in base_probs.items()}
    else:
        base_probs = {"Normal": 0.34, "Abnormal": 0.33, "Inconclusive": 0.33}
    
    # Determine prediction (highest probability)
    prediction = max(base_probs, key=base_probs.get)
    confidence = base_probs[prediction]
    
    return prediction, round(confidence, 3), {k: round(v, 3) for k, v in base_probs.items()}

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting up Healthcare Analytics API on Vercel...")
    logger.info("API startup complete - using rule-based predictions")

@app.get("/")
async def root():
    """API root - provides basic info"""
    return {
        "name": "Healthcare Analytics API",
        "version": "1.0.0",
        "description": "Healthcare test result prediction system",
        "status": "running on Vercel",
        "prediction_mode": "Rule-based (ML model optional)",
        "endpoints": {
            "/": "This help message",
            "/health": "Health check",
            "/predict": "POST - Single prediction",
            "/predict/batch": "POST - Batch predictions",
            "/predict/demo": "GET - Demo predictions",
            "/docs": "Swagger documentation"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health status"""
    return HealthCheck(
        status="healthy",
        model_loaded=False,
        model_version="rule-based-v1",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict test result for a single patient"""
    try:
        logger.info(f"Processing prediction for age: {request.age}, condition: {request.medical_condition}")
        
        # Get prediction using rule-based system
        prediction, confidence, probabilities = predict_healthcare(
            request.age,
            request.medical_condition,
            request.admission_type
        )
        
        return PredictionResponse(
            predicted_result=prediction,
            confidence=confidence,
            probabilities=probabilities,
            model_version="rule-based-v1",
            timestamp=datetime.now().isoformat(),
            patient_age=request.age,
            medical_condition=request.medical_condition
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict test results for multiple patients"""
    try:
        results = []
        for patient in request.patients:
            prediction, confidence, probabilities = predict_healthcare(
                patient.age,
                patient.medical_condition,
                patient.admission_type
            )
            
            results.append({
                "predicted_result": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "patient_age": patient.age,
                "medical_condition": patient.medical_condition
            })
        
        return {
            "predictions": results,
            "total": len(results),
            "model_version": "rule-based-v1",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/demo")
async def get_demo_predictions():
    """Get demo predictions for common scenarios"""
    demo_cases = [
        {"age": 75, "condition": "Heart Disease", "admission": "Emergency", "desc": "Elderly with heart disease"},
        {"age": 45, "condition": "Diabetes", "admission": "Urgent", "desc": "Middle-aged with diabetes"},
        {"age": 25, "condition": "Healthy", "admission": "Routine", "desc": "Young healthy patient"},
        {"age": 35, "condition": "Asthma", "admission": "Emergency", "desc": "Adult with asthma attack"},
        {"age": 68, "condition": "Hypertension", "admission": "Routine", "desc": "Senior with hypertension"},
        {"age": 52, "condition": "Cancer", "admission": "Emergency", "desc": "Cancer patient emergency"},
        {"age": 19, "condition": "Flu", "admission": "Urgent", "desc": "Young adult with flu"},
        {"age": 82, "condition": "COPD", "admission": "Emergency", "desc": "Elderly with COPD"}
    ]
    
    results = []
    for case in demo_cases:
        prediction, confidence, probs = predict_healthcare(
            case["age"], 
            case["condition"],
            case["admission"]
        )
        results.append({
            "description": case["desc"],
            "age": case["age"],
            "medical_condition": case["condition"],
            "admission_type": case["admission"],
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs
        })
    
    return {
        "demo_predictions": results,
        "note": "These are rule-based predictions based on age and medical condition"
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "model_loaded": False,
        "model_type": "Rule-based Expert System",
        "model_version": "v1.0.0",
        "features_used": ["age", "medical_condition", "admission_type"],
        "prediction_classes": ["Normal", "Abnormal", "Inconclusive"],
        "risk_factors": {
            "high_risk": ["Heart Disease", "Kidney Disease", "Liver Disease", "COPD", "Cancer"],
            "medium_risk": ["Diabetes", "Hypertension", "Asthma", "Arthritis", "COVID-19"],
            "low_risk": ["Healthy", "Flu", "Migraine", "Infection"]
        },
        "deployment": "Vercel Serverless"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
