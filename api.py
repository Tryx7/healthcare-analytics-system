"""
Standalone Healthcare Analytics API
Works without PostgreSQL - uses pre-trained model and in-memory data.
Perfect for quick deployment on free platforms.
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

# Aiven PostgreSQL Configuration - Load from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': os.getenv('DB_SSL_MODE', 'require')
}

# Remove any None values
DB_CONFIG = {k: v for k, v in DB_CONFIG.items() if v is not None}

# Load pre-trained model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_latest.pkl')
model_data = None

try:
    model_data = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully: {model_data.get('version', 'unknown')}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")

# In-memory storage for demo
patients_db = []
predictions_db = []

class HealthcarePredictor:
    def __init__(self, model_data):
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.target_encoder = model_data['target_encoder']
        self.feature_columns = model_data['feature_columns']
        self.version = model_data.get('version', 'unknown')
        self.metrics = model_data.get('metrics', {})

    def preprocess(self, df):
        df = df.copy()
        categorical_cols = ['Gender', 'Blood_Type', 'Medical_Condition', 
                           'Admission_Type', 'Age_Group', 'Billing_Category', 'Medication']

        for col in categorical_cols:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])

        numeric_cols = ['Age', 'Billing_Amount', 'Length_of_Stay']
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df

    def predict(self, df):
        X = self.preprocess(df)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        predicted_labels = self.target_encoder.inverse_transform(predictions)
        confidence_scores = np.max(probabilities, axis=1)
        return predicted_labels, confidence_scores, probabilities

predictor = HealthcarePredictor(model_data) if model_data else None

@app.route('/')
def root():
    return jsonify({
        "name": "🏥 Healthcare Analytics API",
        "version": "1.0.0",
        "description": "ML-powered patient test result prediction system",
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else None,
        "endpoints": {
            "health": {"method": "GET", "path": "/health", "description": "API health check"},
            "predict": {"method": "POST", "path": "/predict", "description": "Single patient prediction"},
            "predict_batch": {"method": "POST", "path": "/predict/batch", "description": "Batch predictions"},
            "model_info": {"method": "GET", "path": "/model/info", "description": "Model metadata"},
            "add_patient": {"method": "POST", "path": "/patients", "description": "Add patient record"},
            "get_patients": {"method": "GET", "path": "/patients", "description": "Get all patients"},
            "statistics": {"method": "GET", "path": "/statistics", "description": "System statistics"}
        }
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else None,
        "total_patients": len(patients_db),
        "total_predictions": len(predictions_db),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        # Map request fields to model features
        feature_map = {
            'age': 'Age',
            'gender': 'Gender',
            'blood_type': 'Blood_Type',
            'medical_condition': 'Medical_Condition',
            'admission_type': 'Admission_Type',
            'billing_amount': 'Billing_Amount',
            'length_of_stay': 'Length_of_Stay',
            'age_group': 'Age_Group',
            'billing_category': 'Billing_Category',
            'medication': 'Medication'
        }

        # Build feature DataFrame
        features = {}
        for req_key, model_key in feature_map.items():
            features[model_key] = data.get(req_key, '')

        input_df = pd.DataFrame([features])
        predictions, confidences, probabilities = predictor.predict(input_df)

        prob_dict = {}
        for i, cls in enumerate(predictor.target_encoder.classes_):
            prob_dict[cls] = float(probabilities[0][i])

        result = {
            "predicted_result": predictions[0],
            "confidence": float(confidences[0]),
            "probabilities": prob_dict,
            "model_version": predictor.version,
            "timestamp": datetime.now().isoformat()
        }

        # Store prediction
        predictions_db.append(result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        patients = data.get('patients', [])

        feature_map = {
            'age': 'Age', 'gender': 'Gender', 'blood_type': 'Blood_Type',
            'medical_condition': 'Medical_Condition', 'admission_type': 'Admission_Type',
            'billing_amount': 'Billing_Amount', 'length_of_stay': 'Length_of_Stay',
            'age_group': 'Age_Group', 'billing_category': 'Billing_Category',
            'medication': 'Medication'
        }

        features_list = []
        for patient in patients:
            features = {model_key: patient.get(req_key, '') for req_key, model_key in feature_map.items()}
            features_list.append(features)

        input_df = pd.DataFrame(features_list)
        predictions, confidences, probabilities = predictor.predict(input_df)

        results = []
        for i in range(len(predictions)):
            prob_dict = {}
            for j, cls in enumerate(predictor.target_encoder.classes_):
                prob_dict[cls] = float(probabilities[i][j])

            result = {
                "predicted_result": predictions[i],
                "confidence": float(confidences[i]),
                "probabilities": prob_dict
            }
            results.append(result)
            predictions_db.append(result)

        return jsonify({
            "predictions": results,
            "model_version": predictor.version,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model/info')
def model_info():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    feature_importance = {}
    if hasattr(predictor.model, 'feature_importances_'):
        feature_importance = dict(zip(predictor.feature_columns, predictor.model.feature_importances_))

    return jsonify({
        "version": predictor.version,
        "metrics": predictor.metrics,
        "feature_importance": feature_importance,
        "classes": list(predictor.target_encoder.classes_),
        "feature_columns": predictor.feature_columns
    })

@app.route('/patients', methods=['POST'])
def add_patient():
    try:
        data = request.get_json()
        patient = {
            "id": len(patients_db) + 1,
            **data,
            "created_at": datetime.now().isoformat()
        }
        patients_db.append(patient)
        return jsonify({"message": "Patient added", "patient": patient})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/patients', methods=['GET'])
def get_patients():
    return jsonify({
        "patients": patients_db,
        "total": len(patients_db)
    })

@app.route('/statistics')
def statistics():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    # Calculate prediction distribution
    result_counts = {}
    for pred in predictions_db:
        result = pred['predicted_result']
        result_counts[result] = result_counts.get(result, 0) + 1

    return jsonify({
        "total_patients": len(patients_db),
        "total_predictions": len(predictions_db),
        "prediction_distribution": result_counts,
        "model_version": predictor.version,
        "model_metrics": predictor.metrics
    })

# For Vercel - export the app
# Vercel looks for 'app' variable

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
