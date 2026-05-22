"""
Standalone Healthcare Analytics API - Flask version
Compatible with Vercel deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rule-based prediction (no ML model needed for Vercel)
def predict_healthcare(age: int, medical_condition: str, admission_type: str = "Routine") -> tuple:
    """Simple rule-based prediction"""
    
    # Define risk conditions
    high_risk = ["Heart Disease", "Kidney Disease", "Liver Disease", "COPD", "Cancer"]
    medium_risk = ["Diabetes", "Hypertension", "Asthma", "Arthritis"]
    
    # Age-based risk
    if age > 70 or medical_condition in high_risk:
        prediction = "Abnormal"
        confidence = 0.85
        probs = {"Normal": 0.10, "Abnormal": 0.85, "Inconclusive": 0.05}
    elif age > 60 or medical_condition in medium_risk:
        prediction = "Abnormal"
        confidence = 0.70
        probs = {"Normal": 0.20, "Abnormal": 0.70, "Inconclusive": 0.10}
    elif age < 30 and medical_condition == "Healthy":
        prediction = "Normal"
        confidence = 0.90
        probs = {"Normal": 0.90, "Abnormal": 0.05, "Inconclusive": 0.05}
    else:
        prediction = "Inconclusive"
        confidence = 0.65
        probs = {"Normal": 0.30, "Abnormal": 0.30, "Inconclusive": 0.40}
    
    # Adjust for emergency admission
    if admission_type == "Emergency" and prediction != "Normal":
        confidence = min(0.95, confidence + 0.10)
        probs["Abnormal"] = min(0.95, probs.get("Abnormal", 0.5) + 0.10)
    
    return prediction, confidence, probs

# In-memory storage
patients_db = []
predictions_db = []

@app.route('/')
def root():
    """Root endpoint with API information"""
    return jsonify({
        "name": "🏥 Healthcare Analytics API",
        "version": "2.0.0",
        "framework": "Flask",
        "status": "running on Vercel",
        "prediction_mode": "Rule-based",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "POST - Make prediction",
            "/predict/batch": "POST - Batch predictions",
            "/predict/demo": "GET - Demo predictions",
            "/model/info": "GET - Model information"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_version": "rule-based-v2",
        "total_patients": len(patients_db),
        "total_predictions": len(predictions_db),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction for a single patient"""
    try:
        data = request.get_json()
        
        # Extract fields with defaults
        age = data.get('age')
        if age is None:
            return jsonify({"error": "age is required"}), 400
        
        medical_condition = data.get('medical_condition', 'Unknown')
        admission_type = data.get('admission_type', 'Routine')
        
        # Make prediction
        prediction, confidence, probabilities = predict_healthcare(
            age, medical_condition, admission_type
        )
        
        result = {
            "predicted_result": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": "rule-based-v2",
            "timestamp": datetime.now().isoformat(),
            "patient_age": age,
            "medical_condition": medical_condition
        }
        
        # Store prediction
        predictions_db.append(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Make predictions for multiple patients"""
    try:
        data = request.get_json()
        patients = data if isinstance(data, list) else data.get('patients', [])
        
        results = []
        for patient in patients:
            age = patient.get('age')
            if not age:
                continue
                
            prediction, confidence, probabilities = predict_healthcare(
                age,
                patient.get('medical_condition', 'Unknown'),
                patient.get('admission_type', 'Routine')
            )
            
            results.append({
                "predicted_result": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "patient_age": age,
                "medical_condition": patient.get('medical_condition', 'Unknown')
            })
            predictions_db.append(results[-1])
        
        return jsonify({
            "predictions": results,
            "total": len(results),
            "model_version": "rule-based-v2",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/demo')
def demo_predictions():
    """Get demo predictions for common scenarios"""
    demo_cases = [
        {"age": 75, "condition": "Heart Disease", "admission": "Emergency", "desc": "Elderly with heart disease"},
        {"age": 45, "condition": "Diabetes", "admission": "Urgent", "desc": "Middle-aged with diabetes"},
        {"age": 25, "condition": "Healthy", "admission": "Routine", "desc": "Young healthy patient"},
        {"age": 68, "condition": "Hypertension", "admission": "Routine", "desc": "Senior with hypertension"},
        {"age": 35, "condition": "Asthma", "admission": "Emergency", "desc": "Adult with asthma"},
        {"age": 80, "condition": "Cancer", "admission": "Emergency", "desc": "Elderly cancer patient"},
        {"age": 19, "condition": "Flu", "admission": "Urgent", "desc": "Young adult with flu"}
    ]
    
    results = []
    for case in demo_cases:
        prediction, confidence, probs = predict_healthcare(
            case["age"], case["condition"], case["admission"]
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
    
    return jsonify({
        "demo_predictions": results,
        "total": len(results),
        "note": "These are rule-based predictions for demonstration"
    })

@app.route('/model/info')
def model_info():
    """Get model information"""
    return jsonify({
        "model_loaded": True,
        "model_type": "Rule-based Expert System",
        "model_version": "v2.0.0",
        "features_used": ["age", "medical_condition", "admission_type"],
        "prediction_classes": ["Normal", "Abnormal", "Inconclusive"],
        "risk_factors": {
            "high_risk": ["Heart Disease", "Kidney Disease", "Liver Disease", "COPD", "Cancer"],
            "medium_risk": ["Diabetes", "Hypertension", "Asthma", "Arthritis"]
        },
        "deployment": "Vercel",
        "framework": "Flask"
    })

@app.route('/statistics')
def statistics():
    """Get system statistics"""
    result_counts = {}
    for pred in predictions_db:
        result = pred.get('predicted_result', 'Unknown')
        result_counts[result] = result_counts.get(result, 0) + 1
    
    return jsonify({
        "total_patients": len(patients_db),
        "total_predictions": len(predictions_db),
        "prediction_distribution": result_counts,
        "model_version": "rule-based-v2",
        "uptime": datetime.now().isoformat()
    })

# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
