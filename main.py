"""
FastAPI Application for Healthcare Analytics System
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
from contextlib import asynccontextmanager

# Import our modules
from data_cleaning import HealthcareDataCleaner
from database import HealthcareDatabase
from ml_model import HealthcareMLModel, retrain_model
from scheduler import start_scheduler_in_background

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
db = None

def get_model():
    """Get or load the ML model"""
    global model
    if model is None:
        model_dir = 'models'
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if model_files:
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(model_dir, latest_model)
                model = HealthcareMLModel()
                model.load_model(model_path)
                logger.info(f"Loaded model: {latest_model}")
            else:
                logger.warning("No trained model found")
    return model

def get_db():
    """Get database instance"""
    global db
    if db is None:
        db = HealthcareDatabase()
    return db

# Pydantic models - ALL FIELDS OPTIONAL EXCEPT AGE
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age (required)")
    gender: Optional[str] = Field("Unknown", description="Gender")
    blood_type: Optional[str] = Field("Unknown", description="Blood type")
    medical_condition: Optional[str] = Field("Unknown", description="Medical condition")
    admission_type: Optional[str] = Field("Routine", description="Admission type")
    billing_amount: Optional[float] = Field(5000.0, description="Billing amount")
    length_of_stay: Optional[int] = Field(3, description="Length of stay")
    age_group: Optional[str] = Field("Adult", description="Age group")
    billing_category: Optional[str] = Field("Medium", description="Billing category")
    medication: Optional[str] = Field("None", description="Medication")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
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

class BatchPredictionRequest(BaseModel):
    patients: List[PredictionRequest]

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Healthcare Analytics API...")
    
    # Initialize database
    database = get_db()
    if database.config:
        try:
            database.create_schema()
            stats = database.get_statistics()
            if stats.get('total_records', 0) == 0:
                logger.info("Loading initial data from CSV to PostgreSQL...")
                cleaner = HealthcareDataCleaner("data/healthcare_dataset.csv")
                cleaned_data = cleaner.clean()
                database.insert_patient_records(cleaned_data)
                logger.info(f"✅ Loaded {len(cleaned_data)} records to database")
        except Exception as e:
            logger.warning(f"Database initialization warning: {e}")
    
    # Load model
    get_model()
    
    # Start scheduler
    start_scheduler_in_background()
    
    logger.info("✅ API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Analytics API",
    description="API for predicting patient test results using ML",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def rule_based_prediction(age: int, medical_condition: str) -> tuple:
    """Fallback prediction if model fails"""
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
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)",
            "model_info": "/model/info",
            "statistics": "/statistics",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    current_model = get_model()
    return HealthCheck(
        status="healthy",
        model_loaded=current_model is not None,
        model_version=current_model.model_version if current_model else None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    current_model = get_model()
    
    try:
        if current_model is not None:
            input_data = pd.DataFrame([{
                'Age': request.age,
                'Gender': request.gender or 'Unknown',
                'Blood_Type': request.blood_type or 'Unknown',
                'Medical_Condition': request.medical_condition or 'Unknown',
                'Admission_Type': request.admission_type or 'Routine',
                'Billing_Amount': request.billing_amount or 5000.0,
                'Length_of_Stay': request.length_of_stay or 3,
                'Age_Group': request.age_group or 'Adult',
                'Billing_Category': request.billing_category or 'Medium',
                'Medication': request.medication or 'None'
            }])
            
            predictions, confidences, probabilities = current_model.predict(input_data)
            
            prob_dict = {}
            for i, cls in enumerate(current_model.target_encoder.classes_):
                prob_dict[cls] = float(probabilities[0][i])
            
            return PredictionResponse(
                predicted_result=predictions[0],
                confidence=float(confidences[0]),
                probabilities=prob_dict,
                model_version=current_model.model_version,
                timestamp=datetime.now().isoformat()
            )
        else:
            prediction, confidence, probabilities = rule_based_prediction(
                request.age,
                request.medical_condition or 'Unknown'
            )
            return PredictionResponse(
                predicted_result=prediction,
                confidence=confidence,
                probabilities=probabilities,
                model_version="rule-based-fallback",
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        prediction, confidence, probabilities = rule_based_prediction(
            request.age,
            request.medical_condition or 'Unknown'
        )
        return PredictionResponse(
            predicted_result=prediction,
            confidence=confidence,
            probabilities=probabilities,
            model_version="rule-based-fallback",
            timestamp=datetime.now().isoformat()
        )

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    current_model = get_model()
    results = []
    
    for patient in request.patients:
        try:
            if current_model is not None:
                input_data = pd.DataFrame([{
                    'Age': patient.age,
                    'Gender': patient.gender or 'Unknown',
                    'Blood_Type': patient.blood_type or 'Unknown',
                    'Medical_Condition': patient.medical_condition or 'Unknown',
                    'Admission_Type': patient.admission_type or 'Routine',
                    'Billing_Amount': patient.billing_amount or 5000.0,
                    'Length_of_Stay': patient.length_of_stay or 3,
                    'Age_Group': patient.age_group or 'Adult',
                    'Billing_Category': patient.billing_category or 'Medium',
                    'Medication': patient.medication or 'None'
                }])
                predictions, confidences, probabilities = current_model.predict(input_data)
                
                prob_dict = {}
                for i, cls in enumerate(current_model.target_encoder.classes_):
                    prob_dict[cls] = float(probabilities[0][i])
                
                results.append({
                    "predicted_result": predictions[0],
                    "confidence": float(confidences[0]),
                    "probabilities": prob_dict
                })
            else:
                prediction, confidence, probabilities = rule_based_prediction(
                    patient.age,
                    patient.medical_condition or 'Unknown'
                )
                results.append({
                    "predicted_result": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities
                })
        except Exception as e:
            results.append({"error": str(e)})
    
    return {
        "predictions": results,
        "model_version": current_model.model_version if current_model else "rule-based-fallback",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    current_model = get_model()
    if current_model is None:
        return {"model_loaded": False, "message": "No model loaded"}
    
    return {
        "version": current_model.model_version,
        "metrics": current_model.metrics,
        "feature_importance": current_model.get_feature_importance(),
        "classes": list(current_model.target_encoder.classes_),
        "feature_columns": current_model.feature_columns
    }

@app.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    def retrain_task():
        try:
            retrain_model()
            global model
            model = None
            get_model()
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    background_tasks.add_task(retrain_task)
    return {"message": "Model retraining started", "timestamp": datetime.now().isoformat()}

@app.get("/statistics")
async def get_statistics():
    database = get_db()
    if not database.config:
        return {"message": "Database not configured"}
    return database.get_statistics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
