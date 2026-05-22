
"""
Scheduler Module for Automated Model Retraining
Runs every Saturday at 12:00 PM to retrain the ML model.
"""

import schedule
import time
import logging
from datetime import datetime
from threading import Thread
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retrain_job():
    """Job to retrain the model"""
    logger.info("="*50)
    logger.info("Starting scheduled model retraining")
    logger.info(f"Current time: {datetime.now()}")

    try:
        from ml_model import retrain_model
        model = retrain_model()

        if model:
            logger.info(f"Model retrained successfully. Version: {model.model_version}")
            logger.info(f"Accuracy: {model.metrics.get('accuracy', 'N/A')}")
        else:
            logger.warning("Model retraining returned None")

    except Exception as e:
        logger.error(f"Error during model retraining: {e}", exc_info=True)

    logger.info("="*50)

def run_scheduler():
    """Run the scheduler in a loop"""
    # Schedule retraining every Saturday at 12:00 PM
    schedule.every().saturday.at("12:00").do(retrain_job)

    logger.info("Scheduler started. Model will be retrained every Saturday at 12:00 PM")
    logger.info("Press Ctrl+C to stop")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def start_scheduler_in_background():
    """Start scheduler in a background thread"""
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler started in background thread")
    return scheduler_thread

# For APScheduler (alternative with more features)
def create_apscheduler():
    """Create APScheduler for more advanced scheduling"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler()

        # Add job: Every Saturday at 12:00 PM
        scheduler.add_job(
            retrain_job,
            trigger=CronTrigger(day_of_week='sat', hour=12, minute=0),
            id='model_retraining',
            name='Weekly Model Retraining',
            replace_existing=True
        )

        return scheduler
    except ImportError:
        logger.warning("APScheduler not installed. Using schedule library instead.")
        return None
    

# Add this to your existing scheduler.py
def retrain_job():
    """Job to retrain the model using your existing ml_model.py"""
    logger.info("="*50)
    logger.info("Starting scheduled model retraining")
    
    try:
        # Use your existing retrain_model function from ml_model.py
        from ml_model import retrain_model
        model = retrain_model(data_source='database', model_type='random_forest')
        
        if model:
            logger.info(f"✅ Model retrained successfully. Version: {model.model_version}")
            logger.info(f"Accuracy: {model.metrics.get('accuracy', 'N/A')}")
        else:
            logger.warning("⚠️ Model retraining returned None")
            
    except Exception as e:
        logger.error(f"❌ Error during model retraining: {e}", exc_info=True)

if __name__ == "__main__":
    # For testing: Run retraining immediately
    # retrain_job()

    # Start scheduler
    run_scheduler()
