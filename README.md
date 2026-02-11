# Heart Disease Prediction System - Django Web Application

A professional, modular web application for heart disease prediction using machine learning and blockchain security.

## 🎯 Project Overview

This is a complete Django web application that provides:
- **Professional Medical Interface**: Clean, modern UI with medical theme
- **AI-Powered Predictions**: Dual ML models (Logistic Regression & Random Forest)
- **Blockchain Security**: Immutable audit trail on separate page
- **Modular Architecture**: Clean separation of concerns
- **Responsive Design**: Works on all devices

## 📁 Project Structure

```
heart_disease_project/
│
├── heart_disease_project/          # Django project configuration
# Heart Disease Prediction System

Comprehensive Django web application and reproducible machine learning pipeline for predicting the presence of heart disease from clinical features. The project integrates two predictive models (Logistic Regression and Random Forest) and an auditable blockchain-like record of predictions for reproducibility and traceability.

This README is written for researchers: it documents dataset provenance, preprocessing, model training, evaluation, and reproducibility instructions suitable for inclusion in a methods section.

---

## Abstract (short)

We present a reproducible pipeline that trains and evaluates two supervised classifiers to predict heart disease from a curated clinical dataset. The system includes:
- Data preprocessing and robust evaluation with stratified splits and cross-validation
- Two model families: regularized Logistic Regression and Random Forests
- Model calibration and probability outputs for downstream decision thresholds
- Web interface for clinical data entry and result visualization
- Immutable records (blockchain-style) to store prediction metadata

---

## Contents of this README

1. Dataset and preprocessing (what we used)
2. Model training and evaluation (how we trained and validated)
3. Results reporting (recommended metrics and how to compute them)
4. Reproducibility & environment (exact commands and files)
5. Usage: running training and the web app
6. File layout and important files
7. Limitations, ethical considerations, and recommended future work
8. Citation and contact

---

## 1) Dataset

- Source: `data/heart_disease_combined.csv` (project-local combined dataset). Include in the paper the original source(s) combined into this CSV (e.g., UCI Cleveland, etc.) and any cleaning steps applied before inclusion.
- Target: `HeartDisease` (binary: 1 = disease, 0 = no disease).
- Features used (examples): Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, etc.

Recommended dataset description for a paper: number of samples, number of positive cases, feature ranges, and brief preprocessing summary (missing value handling and encoding).

---

## 2) Preprocessing (what is done in the repo)

Key preprocessing steps implemented (and recommended):

- Missing values: current training script drops missing rows (`df.dropna()`). For reproducible research, explicitly report the number of dropped rows and consider imputation (median / model-based) when appropriate.
- Categorical encoding: `Sex`, `ChestPainType`, `RestingECG`, and `ExerciseAngina` are mapped to labels and one-hot encoded with `pd.get_dummies(..., drop_first=True)`.
- Numerical scaling: `StandardScaler` on numerical features (Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak).

Important reproducibility note: always persist the feature list and the fitted scaler (this project saves `features.pkl` and `scaler.pkl`). For publication, include a short table describing each final column used by the model.

Suggested improvements (for the Methods section):

- Use a `ColumnTransformer` to apply scaling and encoding deterministically and document the exact column order.
- If cross-validation is used, fit scalers inside each fold (use Pipelines) to avoid information leakage.
- Report class balance and decisions about `class_weight` or resampling.

---

## 3) Model training and evaluation (what the repo does and improvements)

What the repository currently implements:

- Two models are trained on the processed training set: a `LogisticRegression` (max_iter=1000) and a `RandomForestClassifier` (n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2). Both are trained with `class_weight='balanced'`.
- A standard `StandardScaler` is fit on the training data and applied to the test set.
- Cross-validation is performed with `cross_val_score(..., cv=5, scoring='roc_auc')` but currently run on data scaled by a scaler fit on the whole training set (this is a leakage risk).

Recommended training protocol for a reproducible study (replace current procedure):

1. Use a scikit-learn `Pipeline` combining preprocessing (ColumnTransformer) and model. This guarantees transformations are applied inside CV folds.
2. Use `StratifiedKFold` for CV and `GridSearchCV` or `RandomizedSearchCV` to tune hyperparameters (e.g., `C` for Logistic Regression; `n_estimators`, `max_depth`, `min_samples_leaf` for Random Forest). Record the search space and final chosen parameters.
3. Evaluate models using the following metrics: ROC-AUC, Precision-Recall AUC, accuracy, sensitivity (recall for positive class), specificity, precision, F1, and calibration (Brier score, calibration plot). Report confidence intervals (bootstrap) for key metrics.
4. If probability estimates are used for decision making, apply `CalibratedClassifierCV` (isotonic or sigmoid) on a validation fold or via nested CV to avoid overfitting calibration.
5. If classes are imbalanced, evaluate resampling approaches (SMOTE, ADASYN) inside a pipeline with nested CV or use class-weighting and compare.

Recommended exact commands (example sketch):

```
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

preprocessor = ColumnTransformer([...])
pipe = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])

param_dist = {
    'clf__n_estimators': [100, 200, 500],
    'clf__max_depth': [None, 10, 20, 30],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(pipe, param_dist, cv=cv, scoring='roc_auc', n_iter=20, n_jobs=-1)
search.fit(X_train, y_train)
```

---

## 4) Results reporting (what to include in a paper)

Include the following in a reproducible results section:

- Dataset summary: N, positive rate, basic demographics, feature descriptions.
- Preprocessing steps applied (explicit column names and transformations).
- Model hyperparameters and selection method (grid search ranges, CV folds).
- Primary metrics on held-out test set (ROC-AUC with CI, PR-AUC, sensitivity/recall at clinically relevant thresholds, specificity, F1). Include confusion matrix.
- Calibration: Brier score and calibration plot.
- Feature importance: permutation importance or SHAP explanations for both models; include top 10 features with effect direction.
- Robustness checks: results under alternative splits, different seeds, and with/without imputation/resampling.

Example results paragraph (template for manuscript):

"We evaluated two supervised classifiers on a held-out test set (20% stratified split). The Random Forest achieved an ROC-AUC of 0.87 (95% CI 0.84–0.90) and a sensitivity of 0.81 at the recommended threshold. The logistic regression achieved an ROC-AUC of 0.82 and provided interpretable coefficients consistent with clinical expectations (age, resting blood pressure and chest pain type were the strongest predictors). Probabilities were calibrated using isotonic regression." 

---

## 5) Reproducibility & Environment

To reproduce results exactly:

1. Create a virtual environment and install dependencies from `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Ensure the dataset `data/heart_disease_combined.csv` is present and unchanged (or record dataset hash).
3. Run training with a fixed random seed (already set via `RANDOM_STATE = 42` in `train_model.py`). For fully reproducible RandomForest results, also set `n_jobs=1` and document platform details.

Store all artifacts:
- `models_data/` contains the saved model(s), scaler, feature list, and test split used for evaluation. Commit or archive these artifacts alongside the paper submission.

Environment metadata to include in a paper:

```
Python X.Y.Z
pandas==<version>
numpy==<version>
scikit-learn==<version>
joblib==<version>
```

Use `pip freeze > environment.txt` to capture exact versions.

---

## 6) Running the code (usage)

Training models (example):

```bash
python train_model.py
```

This script:
- Loads `data/heart_disease_combined.csv`
- Preprocesses data (encoding + scaling)
- Trains Logistic Regression and Random Forest
- Performs 5-fold CV (ROC-AUC)
- Evaluates on held-out test set
- Saves artifacts in `models_data/`

Run the Django app locally:

```bash
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000 to access the interface.

API / Interface notes:
- The `results` endpoint stores and displays `consensus_probability` (percentage). The web app reads the serialized models and scaler from `models_data/`.

---

## 7) File layout (important files)

- `train_model.py` — training pipeline (entry point for model fitting)
- `heart_disease_app/views.py` — prediction endpoint and web integration
- `heart_disease_app/models.py` — Django models for Prediction and BlockchainRecord
- `heart_disease_app/templates/` — UI templates (predict, results, blockchain, history)
- `static/css/style.css` and `static/js/script.js` — front-end assets
- `models_data/` — trained artifacts saved by `train_model.py`
- `data/heart_disease_combined.csv` — dataset used for training
---

## 8) Limitations, ethics, and clinical considerations

- This model is intended for screening and research only — not for clinical decision-making without clinical validation and regulatory approval.
- Dataset bias, measurement differences across sites, and missing data handling can all affect model performance. Carefully report cohorts included in the combined CSV.
- Explainability: include SHAP or coefficient tables to make model outputs interpretable for clinicians.
- Security & privacy: predictions stored in blockchain-like records may contain PHI; ensure compliance with applicable regulations and sanitize data before logging.

---

## 9) How to cite / Acknowledgements

If you use this work in a paper, please cite the repository and include the following note in methods:

"Code and models are available at [project repository]. The training and evaluation used a stratified 80/20 split and 5-fold cross-validation; final artifacts are archived in `models_data/`." 

Contact: maintainer@yourdomain.example

---

## 10) Quick checklist for manuscript Methods section

- Data provenance and cleaning steps (include counts)
- Feature list and preprocessing pipeline (table)
- Model families, hyperparameter search space, and selection method
- Primary evaluation metrics and confidence intervals
- Calibration and probability thresholding method
- Reproducibility instructions and artifact locations

---

## License & Disclaimer

This repository is provided for educational and research purposes only. It is not a medical device. Use for clinical purposes is not advised without appropriate validation and regulatory approval.

---

**Version**: 1.1  
**Last updated**: February 2026

*** End of README ***

