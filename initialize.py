
#!/usr/bin/env python3
"""
Initialization Script for Healthcare Analytics System
Sets up database, loads initial data, and trains the first model.
"""

import os
import sys
import logging
from data_cleaning import HealthcareDataCleaner
from database import HealthcareDatabase
from ml_model import HealthcareMLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize the complete healthcare analytics system"""
    logger.info("="*60)
    logger.info("Healthcare Analytics System - Initialization")
    logger.info("="*60)

    # Step 1: Clean data
    logger.info("\nStep 1: Cleaning healthcare data...")
    cleaner = HealthcareDataCleaner("healthcare_dataset.csv")
    cleaned_data = cleaner.clean()
    cleaner.save_cleaned_data("cleaned_healthcare_data.csv")
    logger.info(f"✓ Cleaned {len(cleaned_data)} records")

    # Step 2: Setup database
    logger.info("\nStep 2: Setting up database...")
    db = HealthcareDatabase()
    try:
        db.create_schema()
        logger.info("✓ Database schema created")
    except Exception as e:
        logger.warning(f"Database schema may already exist: {e}")

    # Step 3: Load data into database
    logger.info("\nStep 3: Loading data into database...")
    try:
        db.insert_patient_records(cleaned_data)
        logger.info("✓ Data loaded into database")
    except Exception as e:
        logger.error(f"Error loading data: {e}")

    # Step 4: Train initial model
    logger.info("\nStep 4: Training initial model...")
    training_data = db.get_training_data()

    if len(training_data) > 0:
        model = HealthcareMLModel()
        metrics = model.train(training_data, model_type='random_forest')

        # Save model
        model_path = model.save_model("v1.0.0")
        logger.info(f"✓ Model trained and saved to {model_path}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")

        # Save metadata
        db.save_model_metadata(
            version="v1.0.0",
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            samples=len(training_data),
            features=model.feature_columns,
            model_path=model_path
        )
        logger.info("✓ Model metadata saved to database")
    else:
        logger.warning("No training data available")

    logger.info("\n" + "="*60)
    logger.info("Initialization Complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Start the API: uvicorn main:app --reload")
    logger.info("2. Access API docs: http://localhost:8000/docs")
    logger.info("3. Check health: http://localhost:8000/health")

if __name__ == "__main__":
    initialize_system()
