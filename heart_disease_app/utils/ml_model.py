"""
Machine Learning Model utilities
Handles model loading, prediction, and evaluation
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from django.conf import settings


class MLModelManager:
    """
    Manages ML model loading and predictions
    """
    
    def __init__(self):
        self.models_loaded = False
        self.log_model = None
        self.rf_model = None
        self.scaler = None
        self.features = None
        self.numerical_features = None
        self.load_models()
    
    def load_models(self):
        """
        Load trained ML models from disk
        """
        try:
            models_dir = settings.MODELS_DIR
            
            self.log_model = joblib.load(models_dir / 'logistic_model.pkl')
            self.rf_model = joblib.load(models_dir / 'rf_model.pkl')
            self.scaler = joblib.load(models_dir / 'scaler.pkl')
            self.features = joblib.load(models_dir / 'features.pkl')
            self.numerical_features = joblib.load(models_dir / 'numerical_features.pkl')
            
            self.models_loaded = True
            print("✅ ML models loaded successfully")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("Please run train_model.py first to generate model files")
            self.models_loaded = False
        except Exception as e:
            print(f"❌ Unexpected error loading models: {e}")
            self.models_loaded = False
    
    def prepare_input(self, input_dict):
        """
        Prepare input data for prediction
        
        Args:
            input_dict: Dictionary containing patient data
            
        Returns:
            Prepared DataFrame ready for prediction
        """
        if not self.models_loaded:
            raise Exception("Models not loaded. Please run train_model.py first.")
        
        # Create DataFrame with one-hot encoded features
        input_df = pd.DataFrame({
            "Age": [input_dict['age']],
            "RestingBP": [input_dict['resting_bp']],
            "Cholesterol": [input_dict['cholesterol']],
            "FastingBS": [1 if input_dict['fasting_bs'] else 0],
            "MaxHR": [input_dict['max_hr']],
            "Oldpeak": [input_dict['oldpeak']],
            "Sex_M": [1 if input_dict['sex'] == 'Male' else 0],
            "ChestPainType_ATA": [1 if input_dict['chest_pain_type'] == 'ATA' else 0],
            "ChestPainType_NAP": [1 if input_dict['chest_pain_type'] == 'NAP' else 0],
            "ChestPainType_TA": [1 if input_dict['chest_pain_type'] == 'TA' else 0],
            "RestingECG_LVH": [1 if input_dict['resting_ecg'] == 'LVH' else 0],
            "RestingECG_ST": [1 if input_dict['resting_ecg'] == 'ST' else 0],
            "ExerciseAngina_Y": [1 if input_dict['exercise_angina'] else 0],
        })
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[self.features]
        
        # Scale numerical features
        input_df[self.numerical_features] = self.scaler.transform(
            input_df[self.numerical_features]
        )
        
        return input_df
    
    def predict(self, input_dict):
        """
        Make prediction using both models
        
        Args:
            input_dict: Dictionary containing patient data
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models_loaded:
            raise Exception("Models not loaded. Please run train_model.py first.")
        
        # Prepare input
        input_data = self.prepare_input(input_dict)
        
        # Logistic Regression prediction
        log_pred = self.log_model.predict(input_data)[0]
        log_prob = self.log_model.predict_proba(input_data)[0][1]
        
        # Random Forest prediction
        rf_pred = self.rf_model.predict(input_data)[0]
        rf_prob = self.rf_model.predict_proba(input_data)[0][1]
        
        # Consensus prediction
        consensus_prob = (log_prob + rf_prob) / 2
        consensus_pred = 1 if consensus_prob >= 0.5 else 0
        
        return {
            'logistic_prob': float(log_prob),
            'logistic_pred': int(log_pred),
            'rf_prob': float(rf_prob),
            'rf_pred': int(rf_pred),
            'consensus_prob': float(consensus_prob),
            'consensus_pred': int(consensus_pred),
            'risk_level': 'High' if consensus_pred == 1 else 'Low',
            'prediction_result': 'Heart Disease' if consensus_pred == 1 else 'No Heart Disease'
        }
    
    def get_feature_importance(self, model_type='logistic'):
        """
        Get feature importance from specified model
        
        Args:
            model_type: 'logistic' or 'random_forest'
            
        Returns:
            Dictionary of feature importances
        """
        if not self.models_loaded:
            raise Exception("Models not loaded")
        
        if model_type == 'logistic':
            importance = np.abs(self.log_model.coef_[0])
        elif model_type == 'random_forest':
            importance = self.rf_model.feature_importances_
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
        
        feature_importance = {
            feature: float(imp) 
            for feature, imp in zip(self.features, importance)
        }
        
        # Sort by importance
        return dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))


# Global instance
ml_manager = MLModelManager()
