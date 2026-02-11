"""
Data preprocessing utilities
Handles data cleaning, validation, and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class DataValidator:
    """
    Validates patient input data
    """
    
    # Define valid ranges for each field
    VALIDATION_RULES = {
        'age': {'min': 1, 'max': 120, 'type': int},
        'resting_bp': {'min': 80, 'max': 250, 'type': int},
        'cholesterol': {'min': 100, 'max': 600, 'type': int},
        'max_hr': {'min': 60, 'max': 220, 'type': int},
        'oldpeak': {'min': 0.0, 'max': 10.0, 'type': float},
        'sex': {'valid_values': ['Male', 'Female'], 'type': str},
        'chest_pain_type': {'valid_values': ['ATA', 'NAP', 'ASY', 'TA'], 'type': str},
        'resting_ecg': {'valid_values': ['Normal', 'ST', 'LVH'], 'type': str},
        'fasting_bs': {'type': bool},
        'exercise_angina': {'type': bool}
    }
    
    @staticmethod
    def validate_input(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate patient input data
        
        Args:
            input_data: Dictionary containing patient data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for field, rules in DataValidator.VALIDATION_RULES.items():
            if field not in input_data:
                errors.append(f"Missing required field: {field}")
                continue
            
            value = input_data[field]
            
            # Check type
            if 'type' in rules:
                if not isinstance(value, rules['type']):
                    try:
                        # Try to convert
                        if rules['type'] == int:
                            value = int(value)
                        elif rules['type'] == float:
                            value = float(value)
                        elif rules['type'] == bool:
                            value = bool(value)
                        input_data[field] = value
                    except (ValueError, TypeError):
                        errors.append(f"{field} must be of type {rules['type'].__name__}")
                        continue
            
            # Check range for numeric fields
            if 'min' in rules and value < rules['min']:
                errors.append(f"{field} must be >= {rules['min']}")
            
            if 'max' in rules and value > rules['max']:
                errors.append(f"{field} must be <= {rules['max']}")
            
            # Check valid values for categorical fields
            if 'valid_values' in rules and value not in rules['valid_values']:
                errors.append(f"{field} must be one of {rules['valid_values']}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize input data
        
        Args:
            input_data: Dictionary containing patient data
            
        Returns:
            Cleaned dictionary
        """
        cleaned = input_data.copy()
        
        # Strip whitespace from string fields
        for key, value in cleaned.items():
            if isinstance(value, str):
                cleaned[key] = value.strip()
        
        # Ensure boolean fields are proper booleans
        bool_fields = ['fasting_bs', 'exercise_angina']
        for field in bool_fields:
            if field in cleaned:
                if isinstance(cleaned[field], str):
                    cleaned[field] = cleaned[field].lower() in ['yes', 'true', '1']
                else:
                    cleaned[field] = bool(cleaned[field])
        
        return cleaned


class DataPreprocessor:
    """
    Handles data preprocessing and feature engineering
    """
    
    @staticmethod
    def create_feature_vector(patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Create feature vector from patient data
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            DataFrame with one-hot encoded features
        """
        # Create base DataFrame
        df = pd.DataFrame({
            'Age': [patient_data['age']],
            'RestingBP': [patient_data['resting_bp']],
            'Cholesterol': [patient_data['cholesterol']],
            'FastingBS': [1 if patient_data['fasting_bs'] else 0],
            'MaxHR': [patient_data['max_hr']],
            'Oldpeak': [patient_data['oldpeak']]
        })
        
        # One-hot encode sex
        df['Sex_M'] = 1 if patient_data['sex'] == 'Male' else 0
        
        # One-hot encode chest pain type
        chest_pain_types = ['ATA', 'NAP', 'TA']
        for cpt in chest_pain_types:
            df[f'ChestPainType_{cpt}'] = 1 if patient_data['chest_pain_type'] == cpt else 0
        
        # One-hot encode resting ECG
        ecg_types = ['LVH', 'ST']
        for ecg in ecg_types:
            df[f'RestingECG_{ecg}'] = 1 if patient_data['resting_ecg'] == ecg else 0
        
        # One-hot encode exercise angina
        df['ExerciseAngina_Y'] = 1 if patient_data['exercise_angina'] else 0
        
        return df
    
    @staticmethod
    def get_risk_factors(patient_data: Dict[str, Any]) -> List[str]:
        """
        Identify risk factors present in patient data
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            List of identified risk factors
        """
        risk_factors = []
        
        # Age-related risk
        if patient_data['age'] > 55:
            risk_factors.append("Advanced age (>55)")
        
        # Blood pressure
        if patient_data['resting_bp'] > 140:
            risk_factors.append("High blood pressure (>140 mm Hg)")
        
        # Cholesterol
        if patient_data['cholesterol'] > 240:
            risk_factors.append("High cholesterol (>240 mg/dL)")
        
        # Blood sugar
        if patient_data['fasting_bs']:
            risk_factors.append("Elevated fasting blood sugar (>120 mg/dL)")
        
        # Heart rate
        if patient_data['max_hr'] < 100:
            risk_factors.append("Low maximum heart rate (<100 bpm)")
        
        # ST depression
        if patient_data['oldpeak'] > 2.0:
            risk_factors.append("Significant ST depression (>2.0)")
        
        # Chest pain
        if patient_data['chest_pain_type'] in ['ASY', 'TA']:
            risk_factors.append(f"Concerning chest pain type ({patient_data['chest_pain_type']})")
        
        # ECG abnormalities
        if patient_data['resting_ecg'] in ['ST', 'LVH']:
            risk_factors.append(f"ECG abnormality ({patient_data['resting_ecg']})")
        
        # Exercise-induced angina
        if patient_data['exercise_angina']:
            risk_factors.append("Exercise-induced angina")
        
        return risk_factors
    
    @staticmethod
    def get_recommendations(risk_level: str, risk_factors: List[str]) -> List[str]:
        """
        Generate recommendations based on risk assessment
        
        Args:
            risk_level: 'High' or 'Low'
            risk_factors: List of identified risk factors
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append("⚠️ Consult a cardiologist immediately for comprehensive evaluation")
            recommendations.append("🏥 Consider additional diagnostic tests (ECG, stress test, angiography)")
            recommendations.append("💊 Review current medications with your doctor")
            
            if len(risk_factors) > 3:
                recommendations.append("🎯 Multiple risk factors detected - intensive intervention may be needed")
        else:
            recommendations.append("✅ Continue regular health check-ups with your primary care physician")
            recommendations.append("💪 Maintain a healthy lifestyle with regular exercise")
            recommendations.append("🥗 Follow a heart-healthy diet (low sodium, healthy fats)")
        
        # Specific recommendations based on risk factors
        if any('blood pressure' in rf.lower() for rf in risk_factors):
            recommendations.append("🩺 Monitor blood pressure regularly")
        
        if any('cholesterol' in rf.lower() for rf in risk_factors):
            recommendations.append("🍎 Consider cholesterol-lowering interventions (diet, medication)")
        
        if any('blood sugar' in rf.lower() for rf in risk_factors):
            recommendations.append("🍬 Monitor blood glucose levels and consider diabetes screening")
        
        return recommendations
