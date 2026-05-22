
"""
Healthcare Data Cleaning Module
Cleans raw healthcare data and prepares it for database storage and ML training.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareDataCleaner:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.cleaned_df = None

    def load_data(self):
        """Load raw healthcare data from CSV"""
        logger.info(f"Loading data from {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df

    def handle_missing_values(self):
        """Handle missing values appropriately"""
        logger.info("Handling missing values...")

        # Drop rows where target variable is missing
        self.df = self.df.dropna(subset=['Test_Results'])

        # Fill numeric missing values with median
        numeric_cols = ['Age', 'Billing_Amount', 'Room_Number']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)

        # Fill categorical missing values with mode
        categorical_cols = ['Gender', 'Blood_Type', 'Medical_Condition', 'Admission_Type', 'Medication']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # Fill text fields
        text_cols = ['Name', 'Doctor', 'Hospital', 'Insurance_Provider']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col].fillna('Unknown', inplace=True)

        logger.info(f"Remaining missing values: {self.df.isnull().sum().sum()}")
        return self.df

    def remove_duplicates(self):
        """Remove duplicate records"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_count - len(self.df)
        logger.info(f"Removed {removed} duplicate records")
        return self.df

    def validate_data(self):
        """Validate data ranges and types"""
        logger.info("Validating data...")

        # Age validation
        self.df = self.df[(self.df['Age'] >= 0) & (self.df['Age'] <= 120)]

        # Billing amount validation
        self.df = self.df[self.df['Billing_Amount'] >= 0]

        # Room number validation
        self.df = self.df[self.df['Room_Number'] > 0]

        logger.info(f"Valid records after validation: {len(self.df)}")
        return self.df

    def feature_engineering(self):
        """Create additional features"""
        logger.info("Engineering features...")

        # Length of stay
        self.df['Date_of_Admission'] = pd.to_datetime(self.df['Date_of_Admission'])
        self.df['Discharge_Date'] = pd.to_datetime(self.df['Discharge_Date'])
        self.df['Length_of_Stay'] = (self.df['Discharge_Date'] - self.df['Date_of_Admission']).dt.days
        self.df['Length_of_Stay'] = self.df['Length_of_Stay'].clip(lower=0)

        # Age groups
        self.df['Age_Group'] = pd.cut(self.df['Age'], 
                                       bins=[0, 18, 35, 50, 65, 120], 
                                       labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])

        # Billing categories
        self.df['Billing_Category'] = pd.cut(self.df['Billing_Amount'],
                                              bins=[0, 5000, 15000, 30000, float('inf')],
                                              labels=['Low', 'Medium', 'High', 'Very High'])

        logger.info("Feature engineering completed")
        return self.df

    def clean(self):
        """Run full cleaning pipeline"""
        self.load_data()
        self.handle_missing_values()
        self.remove_duplicates()
        self.validate_data()
        self.feature_engineering()
        self.cleaned_df = self.df.copy()
        logger.info("Data cleaning completed successfully")
        return self.cleaned_df

    def save_cleaned_data(self, output_path):
        """Save cleaned data to CSV"""
        if self.cleaned_df is not None:
            self.cleaned_df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
        else:
            logger.error("No cleaned data to save. Run clean() first.")


if __name__ == "__main__":
    cleaner = HealthcareDataCleaner("healthcare_dataset.csv")
    cleaned_data = cleaner.clean()
    cleaner.save_cleaned_data("cleaned_healthcare_data.csv")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")
