
"""
Flask Application for Healthcare Analytics System
Alternative to FastAPI for easier deployment on some platforms.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model instance
model = None

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
                try:
                    from ml_model import HealthcareMLModel
                    model = HealthcareMLModel()
                    model.load_model(model_path)
                    logger.info(f"Loaded model: {latest_model}")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
    return model

@app.route('/')
def root():
    """API root"""
    return jsonify({
        "name": "Healthcare Analytics API",
        "version": "1.0.0",
        "description": "ML-powered healthcare test result prediction",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "model_info": "/model/info"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    current_model = get_model()
    return jsonify({
        "status": "healthy",
        "model_loaded": current_model is not None,
        "model_version": current_model.model_version if current_model else None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict test result for a single patient"""
    current_model = get_model()

    if current_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Make prediction
        predictions, confidences, probabilities = current_model.predict(input_data)

        # Build probability dict
        prob_dict = {}
        for i, cls in enumerate(current_model.target_encoder.classes_):
            prob_dict[cls] = float(probabilities[0][i])

        return jsonify({
            "predicted_result": predictions[0],
            "confidence": float(confidences[0]),
            "probabilities": prob_dict,
            "model_version": current_model.model_version,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict test results for multiple patients"""
    current_model = get_model()

    if current_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        patients = data.get('patients', [])

        input_data = pd.DataFrame(patients)
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

        return jsonify({
            "predictions": results,
            "model_version": current_model.model_version,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model/info')
def model_info():
    """Get model information"""
    current_model = get_model()

    if current_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify(current_model.get_model_summary())

# For Vercel serverless
# The app object is used directly by Vercel

if __name__ == "__main__":
    # Load model on startup
    get_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
