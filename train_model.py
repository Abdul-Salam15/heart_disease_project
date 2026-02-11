import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# ========================================
# Configuration
# ========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = "data/heart_disease_combined.csv"
MODELS_DIR = "models_data"

# ========================================
# Load and Explore Dataset
# ========================================
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names and types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['HeartDisease'].value_counts()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ========================================
# Data Preprocessing
# ========================================
print("\n🔧 Preprocessing data...")

# Handle missing values (if any)
df = df.dropna()

# Define features
target = "HeartDisease"

# Identify numerical features (these will be scaled)
numerical_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# Map numerical codes to categorical labels for better encoding
df['Sex'] = df['Sex'].map({0: 'F', 1: 'M'})
df['ChestPainType'] = df['ChestPainType'].map({1: 'ATA', 2: 'NAP', 3: 'ASY', 4: 'TA'})
df['RestingECG'] = df['RestingECG'].map({0: 'Normal', 1: 'ST', 2: 'LVH'})
df['ExerciseAngina'] = df['ExerciseAngina'].map({0: 'N', 1: 'Y'})

# One-hot encode categorical features
categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

print(f"Features after encoding: {df.columns.tolist()}")

# Separate features and target
X = df.drop(target, axis=1)
y = df[target]

# Store feature names for later use
feature_names = X.columns.tolist()
print(f"\nTotal features: {len(feature_names)}")
print(f"Feature names: {feature_names}")

# ========================================
# Train-Test Split
# ========================================
print("\n📂 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set class distribution:\n{y_train.value_counts()}")
print(f"Test set class distribution:\n{y_test.value_counts()}")

# ========================================
# Feature Scaling
# ========================================
print("\n⚖️ Scaling numerical features...")
scaler = StandardScaler()

# Only scale numerical features
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

# ========================================
# Model Training
# ========================================
print("\n🤖 Training models...")

# Logistic Regression
print("\n1. Training Logistic Regression...")
log_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
log_model.fit(X_train_scaled, y_train)

# Cross-validation
log_cv_scores = cross_val_score(log_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"   CV ROC-AUC: {log_cv_scores.mean():.4f} (+/- {log_cv_scores.std():.4f})")

# Random Forest
print("\n2. Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"   CV ROC-AUC: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# ========================================
# Model Evaluation
# ========================================
print("\n📈 Evaluating models on test set...")

# Logistic Regression
log_pred = log_model.predict(X_test_scaled)
log_pred_proba = log_model.predict_proba(X_test_scaled)[:, 1]
log_accuracy = accuracy_score(y_test, log_pred)
log_roc_auc = roc_auc_score(y_test, log_pred_proba)

print("\n--- Logistic Regression Results ---")
print(f"Accuracy: {log_accuracy:.4f}")
print(f"ROC-AUC: {log_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, log_pred, target_names=['No Disease', 'Heart Disease']))

# Random Forest
rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_pred_proba)

print("\n--- Random Forest Results ---")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"ROC-AUC: {rf_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['No Disease', 'Heart Disease']))

# ========================================
# Save Models and Artifacts
# ========================================
print("\n💾 Saving models and artifacts...")
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(log_model, f"{MODELS_DIR}/logistic_model.pkl")
joblib.dump(rf_model, f"{MODELS_DIR}/rf_model.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
joblib.dump(feature_names, f"{MODELS_DIR}/features.pkl")
joblib.dump(numerical_features, f"{MODELS_DIR}/numerical_features.pkl")

# Save test data for app evaluation
joblib.dump(X_test_scaled, f"{MODELS_DIR}/X_test.pkl")
joblib.dump(y_test, f"{MODELS_DIR}/y_test.pkl")

print("\n✅ Training complete!")
print(f"   - Logistic Regression: {log_accuracy:.2%} accuracy, {log_roc_auc:.4f} ROC-AUC")
print(f"   - Random Forest: {rf_accuracy:.2%} accuracy, {rf_roc_auc:.4f} ROC-AUC")
print(f"\n📦 Models and data saved to '{MODELS_DIR}/' directory")
print("\n🚀 You can now run the Django server:")
print("   python manage.py runserver")
