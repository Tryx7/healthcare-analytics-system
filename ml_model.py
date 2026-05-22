
"""
Machine Learning Module for Healthcare Test Result Prediction
Trains and evaluates models to predict patient test results.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareMLModel:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.feature_columns = None
        self.model_version = None
        self.metrics = {}

        os.makedirs(model_dir, exist_ok=True)

    def preprocess_features(self, df, fit=True):
        """Preprocess features for model training/prediction"""
        # Select relevant features
        feature_cols = [
            'age', 'gender', 'blood_type', 'medical_condition', 
            'admission_type', 'billing_amount', 'length_of_stay', 
            'age_group', 'billing_category', 'medication'
        ]

        df_processed = df[feature_cols].copy()

        # Handle categorical variables
        categorical_cols = ['gender', 'blood_type', 'medical_condition', 
                           'admission_type', 'age_group', 'billing_category', 'medication']

        for col in categorical_cols:
            if col in df_processed.columns:
                if fit:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        # Handle unseen categories
                        df_processed[col] = df_processed[col].astype(str).apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        df_processed[col] = le.transform(df_processed[col])

        # Scale numeric features
        numeric_cols = ['age', 'billing_amount', 'length_of_stay']
        if fit:
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        else:
            df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])

        self.feature_columns = feature_cols
        return df_processed

    def preprocess_target(self, y, fit=True):
        """Encode target variable"""
        if fit:
            return self.target_encoder.fit_transform(y)
        return self.target_encoder.transform(y)

    def train(self, df, model_type='random_forest', test_size=0.2, random_state=42):
        """Train the machine learning model"""
        logger.info(f"Starting model training with {model_type}")

        # Prepare features and target
        X = self.preprocess_features(df, fit=True)
        y = self.preprocess_target(df['test_results'], fit=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced',
                multi_class='multinomial'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred, 
                                                           target_names=self.target_encoder.classes_,
                                                           output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        logger.info(f"Model trained successfully")
        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {self.metrics['f1']:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1_weighted')
        self.metrics['cv_f1_mean'] = cv_scores.mean()
        self.metrics['cv_f1_std'] = cv_scores.std()

        return self.metrics

    def predict(self, features_df):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = self.preprocess_features(features_df, fit=False)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Decode predictions
        predicted_labels = self.target_encoder.inverse_transform(predictions)

        # Get confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)

        return predicted_labels, confidence_scores, probabilities

    def save_model(self, version=None):
        """Save model and preprocessing artifacts"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.model_version = version
        model_path = os.path.join(self.model_dir, f'model_{version}.pkl')

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns,
            'version': version,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

        return model_path

    def load_model(self, model_path):
        """Load model from file"""
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.target_encoder = model_data['target_encoder']
        self.feature_columns = model_data['feature_columns']
        self.model_version = model_data['version']
        self.metrics = model_data.get('metrics', {})

        logger.info(f"Model loaded from {model_path}")
        return self

    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_columns, importance))
        return {}

    def get_model_summary(self):
        """Get model summary for API response"""
        return {
            'version': self.model_version,
            'metrics': self.metrics,
            'feature_importance': self.get_feature_importance(),
            'classes': list(self.target_encoder.classes_),
            'feature_columns': self.feature_columns
        }


def retrain_model(data_source='database', model_type='random_forest'):
    """
    Main function to retrain the model
    Can be called from scheduler or API
    """
    from database import HealthcareDatabase

    logger.info("Starting model retraining...")

    # Get data from database
    db = HealthcareDatabase()
    df = db.get_training_data()

    if len(df) < 100:
        logger.warning("Insufficient data for training")
        return None

    # Train model
    model = HealthcareMLModel()
    metrics = model.train(df, model_type=model_type)

    # Save model
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model.save_model(version)

    # Save metadata to database
    db.save_model_metadata(
        version=version,
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        samples=len(df),
        features=model.feature_columns,
        model_path=model_path
    )

    logger.info(f"Model retraining completed. Version: {version}")
    return model


if __name__ == "__main__":
    # Example usage
    print("ML Model Module - Import and use in your application")
