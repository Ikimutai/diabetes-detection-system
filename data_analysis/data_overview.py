import pandas as pd
import streamlit as st

def get_data_overview(df):
    """Get overview statistics of the dataset"""
    overview = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'diabetes_positive': int(df['diabetes'].sum()),
        'diabetes_negative': int(len(df) - df['diabetes'].sum()),
        'diabetes_percentage': round(df['diabetes'].mean() * 100, 2),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'feature_names': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict(),
        'basic_stats': df.describe(include='all').to_dict()
    }
    return overview

def get_sample_data(df, n=10):
    """Get sample rows from the dataset"""
    return df.head(n).to_dict('records')

def get_feature_description():
    """Get descriptions for each feature"""
    descriptions = {
        'gender': 'Gender of the patient (Female, Male)',
        'age': 'Age of the patient in years',
        'hypertension': 'Whether the patient has hypertension (0 = No, 1 = Yes)',
        'heart_disease': 'Whether the patient has heart disease (0 = No, 1 = Yes)',
        'smoking_history': 'Smoking history status (never, No Info, current, former, ever, not current)',
        'bmi': 'Body Mass Index (BMI) of the patient',
        'HbA1c_level': 'Glycated hemoglobin level (measure of average blood sugar over 3 months)',
        'blood_glucose_level': 'Current blood glucose level (mg/dL)',
        'diabetes': 'Target variable (0 = No diabetes, 1 = Diabetes)'
    }
    return descriptions