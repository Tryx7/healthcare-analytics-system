"""
PostgreSQL Database Module for Healthcare Analytics
Handles all database operations including schema creation, data insertion, and querying.
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import logging
from contextlib import contextmanager
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - Load from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': os.getenv('DB_SSL_MODE', 'require')
}

# Remove None values (for optional configs)
DB_CONFIG = {k: v for k, v in DB_CONFIG.items() if v is not None}

class HealthcareDatabase:
    def __init__(self, config=None):
        self.config = config or DB_CONFIG
        self.connection = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if not self.config:
            raise ValueError("Database configuration is missing. Set environment variables.")
        
        config = self.config.copy()
        # Add SSL CA certificate if provided
        ssl_ca = os.getenv('DB_SSL_CA')
        if ssl_ca and os.path.exists(ssl_ca):
            config['sslrootcert'] = ssl_ca
        conn = psycopg2.connect(**config)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                cursor.close()

    def create_schema(self):
        """Create database schema for healthcare data"""
        schema_sql = """
        -- Drop tables if they exist (use with caution in production)
        DROP TABLE IF EXISTS predictions CASCADE;
        DROP TABLE IF EXISTS patient_records CASCADE;
        DROP TABLE IF EXISTS model_metadata CASCADE;

        -- Main patient records table
        CREATE TABLE patient_records (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            age INTEGER CHECK (age >= 0 AND age <= 120),
            gender VARCHAR(20),
            blood_type VARCHAR(10),
            medical_condition VARCHAR(100),
            date_of_admission DATE,
            doctor VARCHAR(100),
            hospital VARCHAR(100),
            insurance_provider VARCHAR(100),
            billing_amount DECIMAL(12, 2),
            room_number INTEGER,
            admission_type VARCHAR(50),
            discharge_date DATE,
            medication VARCHAR(100),
            test_results VARCHAR(50),
            length_of_stay INTEGER,
            age_group VARCHAR(50),
            billing_category VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Model predictions table
        CREATE TABLE predictions (
            id SERIAL PRIMARY KEY,
            patient_id INTEGER REFERENCES patient_records(id),
            predicted_result VARCHAR(50),
            confidence_score DECIMAL(5, 4),
            model_version VARCHAR(50),
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            features_used JSONB
        );

        -- Model metadata table for tracking retraining
        CREATE TABLE model_metadata (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(50) UNIQUE,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accuracy DECIMAL(5, 4),
            precision_score DECIMAL(5, 4),
            recall_score DECIMAL(5, 4),
            f1_score DECIMAL(5, 4),
            training_samples INTEGER,
            features_used TEXT[],
            model_path VARCHAR(255),
            is_active BOOLEAN DEFAULT FALSE
        );

        -- Create indexes for better query performance
        CREATE INDEX idx_patient_records_test_results ON patient_records(test_results);
        CREATE INDEX idx_patient_records_admission_date ON patient_records(date_of_admission);
        CREATE INDEX idx_patient_records_medical_condition ON patient_records(medical_condition);
        CREATE INDEX idx_predictions_model_version ON predictions(model_version);
        CREATE INDEX idx_model_metadata_active ON model_metadata(is_active);

        -- Create trigger for updated_at
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';

        DROP TRIGGER IF EXISTS update_patient_records_updated_at ON patient_records;
        CREATE TRIGGER update_patient_records_updated_at
            BEFORE UPDATE ON patient_records
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """

        with self.get_cursor() as cursor:
            cursor.execute(schema_sql)
            logger.info("Database schema created successfully")

    def insert_patient_records(self, df):
        """Insert patient records from DataFrame"""
        columns = [
            'name', 'age', 'gender', 'blood_type', 'medical_condition',
            'date_of_admission', 'doctor', 'hospital', 'insurance_provider',
            'billing_amount', 'room_number', 'admission_type', 'discharge_date',
            'medication', 'test_results', 'length_of_stay', 'age_group', 'billing_category'
        ]

        # Prepare data
        data = []
        for _, row in df.iterrows():
            data.append(tuple(row.get(col, None) for col in columns))

        insert_query = """
        INSERT INTO patient_records 
        (name, age, gender, blood_type, medical_condition, date_of_admission, 
         doctor, hospital, insurance_provider, billing_amount, room_number, 
         admission_type, discharge_date, medication, test_results, 
         length_of_stay, age_group, billing_category)
        VALUES %s
        ON CONFLICT DO NOTHING
        """

        with self.get_cursor() as cursor:
            execute_values(cursor, insert_query, data, page_size=1000)
            logger.info(f"Inserted {len(data)} patient records")

    def get_training_data(self):
        """Retrieve data for model training"""
        query = """
        SELECT age, gender, blood_type, medical_condition, admission_type,
               billing_amount, length_of_stay, age_group, billing_category,
               medication, test_results
        FROM patient_records
        WHERE test_results IS NOT NULL
        """

        with self.get_connection() as conn:
            return pd.read_sql(query, conn)

    def get_patient_by_id(self, patient_id):
        """Get single patient record"""
        query = "SELECT * FROM patient_records WHERE id = %s"

        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=(patient_id,))

    def save_prediction(self, patient_id, predicted_result, confidence, model_version, features):
        """Save model prediction"""
        query = """
        INSERT INTO predictions 
        (patient_id, predicted_result, confidence_score, model_version, features_used)
        VALUES (%s, %s, %s, %s, %s)
        """

        import json
        with self.get_cursor() as cursor:
            cursor.execute(query, (patient_id, predicted_result, confidence, model_version, json.dumps(features)))

    def save_model_metadata(self, version, accuracy, precision, recall, f1, samples, features, model_path):
        """Save model training metadata"""
        query = """
        INSERT INTO model_metadata 
        (model_version, accuracy, precision_score, recall_score, f1_score, 
         training_samples, features_used, model_path, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        with self.get_cursor() as cursor:
            cursor.execute(query, (version, accuracy, precision, recall, f1, 
                                   samples, features, model_path, True))

            # Deactivate previous models
            cursor.execute("""
                UPDATE model_metadata 
                SET is_active = FALSE 
                WHERE model_version != %s
            """, (version,))

    def get_active_model(self):
        """Get currently active model metadata"""
        query = "SELECT * FROM model_metadata WHERE is_active = TRUE ORDER BY training_date DESC LIMIT 1"

        with self.get_connection() as conn:
            result = pd.read_sql(query, conn)
            return result.iloc[0] if not result.empty else None

    def get_statistics(self):
        """Get database statistics"""
        stats = {}

        with self.get_connection() as conn:
            # Total records
            stats['total_records'] = pd.read_sql("SELECT COUNT(*) as count FROM patient_records", conn).iloc[0]['count']

            # Records by test result
            stats['by_result'] = pd.read_sql("""
                SELECT test_results, COUNT(*) as count 
                FROM patient_records 
                GROUP BY test_results
            """, conn).to_dict('records')

            # Records by condition
            stats['by_condition'] = pd.read_sql("""
                SELECT medical_condition, COUNT(*) as count 
                FROM patient_records 
                GROUP BY medical_condition
            """, conn).to_dict('records')

            # Model versions
            stats['models'] = pd.read_sql("""
                SELECT model_version, training_date, accuracy, is_active
                FROM model_metadata
                ORDER BY training_date DESC
            """, conn).to_dict('records')

        return stats


if __name__ == "__main__":
    if not DB_CONFIG:
        print("No database configuration found. Set environment variables.")
    else:
        db = HealthcareDatabase()
        db.create_schema()
        print("Database initialized successfully")
