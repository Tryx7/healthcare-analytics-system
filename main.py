"""
FastAPI Application for Healthcare Analytics System
Provides REST API endpoints for predictions, data management, and model info.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
<<<<<<< HEAD
import joblib
=======
from contextlib import asynccontextmanager
>>>>>>> bbb4252 (first commit)

# Import our modules
from data_cleaning import HealthcareDataCleaner
from database import HealthcareDatabase
from ml_model import HealthcareMLModel, retrain_model
<<<<<<< HEAD
from scheduler import start_scheduler_in_background, create_apscheduler
=======
from scheduler import start_scheduler_in_background
>>>>>>> bbb4252 (first commit)

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

# Pydantic models for request/response - ALL FIELDS OPTIONAL EXCEPT AGE
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

class PatientData(BaseModel):
    name: str = Field(..., description="Patient name")
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., description="Gender")
    blood_type: str = Field(..., description="Blood type")
    medical_condition: str = Field(..., description="Medical condition")
    date_of_admission: str = Field(..., description="Admission date")
    doctor: str = Field(..., description="Doctor name")
    hospital: str = Field(..., description="Hospital name")
    insurance_provider: str = Field(..., description="Insurance provider")
    billing_amount: float = Field(..., ge=0)
    room_number: int = Field(..., ge=1)
    admission_type: str = Field(..., description="Admission type")
    discharge_date: str = Field(..., description="Discharge date")
    medication: str = Field(..., description="Medication")

class PredictionResponse(BaseModel):
    predicted_result: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    patients: List[PredictionRequest]

class ModelInfo(BaseModel):
    version: str
    metrics: Dict[str, Any]
    feature_importance: Dict[str, float]
    classes: List[str]
    feature_columns: List[str]

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

<<<<<<< HEAD
# Global model instance
model = None
db = None

def get_model():
    """Get or load the ML model"""
    global model
    if model is None:
        # Try to load latest model
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

# Pydantic models for request/response
class PatientData(BaseModel):
    name: str = Field(..., description="Patient name")
    age: int = Field(..., ge=0, le=120, description="Patient age")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    blood_type: str = Field(..., description="Blood type: A+, A-, B+, B-, AB+, AB-, O+, O-")
    medical_condition: str = Field(..., description="Medical condition")
    date_of_admission: str = Field(..., description="Admission date (YYYY-MM-DD)")
    doctor: str = Field(..., description="Doctor name")
    hospital: str = Field(..., description="Hospital name")
    insurance_provider: str = Field(..., description="Insurance provider")
    billing_amount: float = Field(..., ge=0, description="Billing amount")
    room_number: int = Field(..., ge=1, description="Room number")
    admission_type: str = Field(..., description="Admission type: Emergency, Elective, Urgent")
    discharge_date: str = Field(..., description="Discharge date (YYYY-MM-DD)")
    medication: str = Field(..., description="Prescribed medication")

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str
    blood_type: str
    medical_condition: str
    admission_type: str
    billing_amount: float = Field(..., ge=0)
    length_of_stay: int = Field(..., ge=0)
    age_group: str
    billing_category: str
    medication: str

class PredictionResponse(BaseModel):
    predicted_result: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    patients: List[PredictionRequest]

class ModelInfo(BaseModel):
    version: str
    metrics: Dict[str, Any]
    feature_importance: Dict[str, float]
    classes: List[str]
    feature_columns: List[str]

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting up Healthcare Analytics API...")

    # Initialize database
    try:
        database = get_db()
        if database.config:  # Only create schema if config exists
            database.create_schema()
            logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")

    # Load model
    get_model()

    # Start scheduler in background
    try:
        scheduler = create_apscheduler()
        if scheduler:
            scheduler.start()
            logger.info("APScheduler started for weekly retraining")
        else:
            start_scheduler_in_background()
    except Exception as e:
        logger.warning(f"Scheduler startup warning: {e}")

    logger.info("API startup complete")

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health status"""
    current_model = get_model()
    return HealthCheck(
        status="healthy",
        model_loaded=current_model is not None,
        model_version=current_model.model_version if current_model else None,
        timestamp=datetime.now().isoformat()
    )

# Predict endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict test result for a single patient"""
    current_model = get_model()

    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Make prediction
        predictions, confidences, probabilities = current_model.predict(input_data)

        # Build probability dict
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

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch predict endpoint
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict test results for multiple patients"""
    current_model = get_model()

    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_data = pd.DataFrame([p.dict() for p in request.patients])
        predictions, confidences, probabilities = current_model.predict(input_data)

        results = []
        for i in range(len(predictions)):
            prob_dict = {}
            for j, cls in enumerate(current_model.target_encoder.classes_):
                prob_dict[cls] = float(probabilities[i][j])

            results.append({
                "predicted_result": predictions[i],
                "confidence": float(confidences[i]),
                "probabilities": prob_dict
            })

        return {
            "predictions": results,
            "model_version": current_model.model_version,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add patient endpoint
@app.post("/patients")
async def add_patient(patient: PatientData):
    """Add a new patient record to the database"""
    try:
        database = get_db()
        
        if not database.config:
            return {"message": "Patient data received but not stored (database not configured)", "patient": patient.dict()}

        # Convert to DataFrame
        df = pd.DataFrame([patient.dict()])

        # Calculate derived fields
        df['date_of_admission'] = pd.to_datetime(df['date_of_admission'])
        df['discharge_date'] = pd.to_datetime(df['discharge_date'])
        df['length_of_stay'] = (df['discharge_date'] - df['date_of_admission']).dt.days
        df['length_of_stay'] = df['length_of_stay'].clip(lower=0)

        df['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 18, 35, 50, 65, 120], 
                                  labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])

        df['billing_category'] = pd.cut(df['billing_amount'],
                                         bins=[0, 5000, 15000, 30000, float('inf')],
                                         labels=['Low', 'Medium', 'High', 'Very High'])

        database.insert_patient_records(df)

        return {"message": "Patient added successfully", "patient": patient.dict()}

    except Exception as e:
        logger.error(f"Error adding patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get model info endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information and metrics"""
    current_model = get_model()

    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(**current_model.get_model_summary())

# Retrain endpoint
@app.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, model_type: str = "random_forest"):
    """Trigger model retraining (runs in background)"""
    def retrain_task():
        try:
            retrain_model(model_type=model_type)
            # Reload model after retraining
            global model
            model = None
            get_model()
        except Exception as e:
            logger.error(f"Retraining failed: {e}")

    background_tasks.add_task(retrain_task)

    return {
        "message": "Model retraining started in background",
        "model_type": model_type,
        "timestamp": datetime.now().isoformat()
    }

# Statistics endpoint
@app.get("/statistics")
async def get_statistics():
    """Get database statistics"""
    try:
        database = get_db()
        if not database.config:
            return {"message": "Database not configured"}
        stats = database.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get all patients endpoint
@app.get("/patients")
async def get_patients(limit: int = 100, offset: int = 0):
    """Get patient records with pagination"""
    try:
        database = get_db()
        if not database.config:
            return {"message": "Database not configured", "patients": []}
            
        query = f"SELECT * FROM patient_records LIMIT {limit} OFFSET {offset}"

        with database.get_connection() as conn:
            df = pd.read_sql(query, conn)
            return df.to_dict('records')

    except Exception as e:
        logger.error(f"Error getting patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """API root - provides basic info"""
    return {
        "name": "Healthcare Analytics API",
        "version": "1.0.0",
        "description": "ML-powered healthcare test result prediction system",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)",
            "add_patient": "/patients (POST)",
            "get_patients": "/patients (GET)",
            "model_info": "/model/info",
            "retrain": "/model/retrain (POST)",
            "statistics": "/statistics"
        }
    }

=======
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

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health status"""
    current_model = get_model()
    return HealthCheck(
        status="healthy",
        model_loaded=current_model is not None,
        model_version=current_model.model_version if current_model else None,
        timestamp=datetime.now().isoformat()
    )

# Predict endpoint - WITH FALLBACK
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict test result for a single patient"""
    current_model = get_model()

    try:
        if current_model is not None:
            # Prepare data for ML model with defaults
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
            
            # Make prediction using ML model
            predictions, confidences, probabilities = current_model.predict(input_data)
            
            # Build probability dict
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
            # Use rule-based fallback
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
        # Fallback to rule-based on error
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

# Batch predict endpoint
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict test results for multiple patients"""
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
            logger.error(f"Batch prediction error: {e}")
            results.append({
                "predicted_result": "Error",
                "confidence": 0.0,
                "probabilities": {},
                "error": str(e)
            })
    
    return {
        "predictions": results,
        "model_version": current_model.model_version if current_model else "rule-based-fallback",
        "timestamp": datetime.now().isoformat()
    }

# Add patient endpoint
@app.post("/patients")
async def add_patient(patient: PatientData):
    """Add a new patient record to the database"""
    try:
        database = get_db()
        
        if not database.config:
            return {"message": "Patient data received but not stored (database not configured)", "patient": patient.dict()}

        df = pd.DataFrame([patient.dict()])
        df['date_of_admission'] = pd.to_datetime(df['date_of_admission'])
        df['discharge_date'] = pd.to_datetime(df['discharge_date'])
        df['length_of_stay'] = (df['discharge_date'] - df['date_of_admission']).dt.days
        df['length_of_stay'] = df['length_of_stay'].clip(lower=0)
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 120], labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
        df['billing_category'] = pd.cut(df['billing_amount'], bins=[0, 5000, 15000, 30000, float('inf')], labels=['Low', 'Medium', 'High', 'Very High'])

        database.insert_patient_records(df)
        return {"message": "Patient added successfully", "patient": patient.dict()}
    except Exception as e:
        logger.error(f"Error adding patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get model info endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information and metrics"""
    current_model = get_model()
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(**current_model.get_model_summary())

# Retrain endpoint
@app.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, model_type: str = "random_forest"):
    """Trigger model retraining"""
    def retrain_task():
        try:
            retrain_model(model_type=model_type)
            global model
            model = None
            get_model()
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    background_tasks.add_task(retrain_task)
    return {
        "message": "Model retraining started in background",
        "model_type": model_type,
        "timestamp": datetime.now().isoformat()
    }

# Statistics endpoint
@app.get("/statistics")
async def get_statistics():
    """Get database statistics"""
    try:
        database = get_db()
        if not database.config:
            return {"message": "Database not configured"}
        stats = database.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get all patients endpoint
@app.get("/patients")
async def get_patients(limit: int = 100, offset: int = 0):
    """Get patient records"""
    try:
        database = get_db()
        if not database.config:
            return {"message": "Database not configured", "patients": []}
        query = f"SELECT * FROM patient_records LIMIT {limit} OFFSET {offset}"
        with database.get_connection() as conn:
            df = pd.read_sql(query, conn)
            return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Healthcare Analytics API",
        "version": "1.0.0",
        "description": "ML-powered healthcare test result prediction system",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)",
            "add_patient": "/patients (POST)",
            "get_patients": "/patients (GET)",
            "model_info": "/model/info",
            "retrain": "/model/retrain (POST)",
            "statistics": "/statistics"
        }
    }

>>>>>>> bbb4252 (first commit)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
