# Models Directory

## Purpose
This directory stores the trained machine learning model files.

## Required Files

After running `train_model.py`, this directory should contain:

```
models_data/
├── logistic_model.pkl          # Trained Logistic Regression model
├── rf_model.pkl                # Trained Random Forest model
├── scaler.pkl                  # StandardScaler for feature normalization
├── features.pkl                # List of feature names
├── numerical_features.pkl      # List of numerical feature names
├── X_test.pkl                  # Test features (for evaluation)
└── y_test.pkl                  # Test labels (for evaluation)
```

## How These Files Are Created

These files are automatically generated when you run:

```bash
python train_model.py
```

## Important Notes

⚠️ **These files MUST exist before running the Django application**

If you see errors like "Model files not found" when running the Django app, it means you need to:

1. Ensure your dataset is in `data/heart_disease_combined.csv`
2. Run `python train_model.py`
3. Verify files are created in this directory

## File Descriptions

### logistic_model.pkl
- Trained Logistic Regression classifier
- Used for linear prediction
- Provides interpretable coefficients

### rf_model.pkl
- Trained Random Forest classifier
- Ensemble of decision trees
- Handles non-linear relationships

### scaler.pkl
- StandardScaler object
- Normalizes numerical features
- Essential for consistent predictions

### features.pkl
- List of all feature names in correct order
- Used to ensure input data has correct structure

### numerical_features.pkl
- List of features that need scaling
- Subset of all features

### X_test.pkl & y_test.pkl
- Test dataset
- Used for model evaluation in the app
- Optional for predictions, but good for validation

## Verifying Files

After training, verify all files exist:

```bash
ls -la models_data/
```

You should see all 7 .pkl files listed above.

## File Sizes

Typical file sizes (approximate):
- logistic_model.pkl: ~5-10 KB
- rf_model.pkl: ~500 KB - 2 MB (depends on n_estimators)
- scaler.pkl: ~1-5 KB
- features.pkl: ~1 KB
- numerical_features.pkl: ~1 KB
- X_test.pkl: ~50-200 KB
- y_test.pkl: ~5-10 KB

## Troubleshooting

### Error: "No such file or directory: models_data/"
**Solution**: The directory should already exist, but if not:
```bash
mkdir models_data
```

### Error: "Model files not found"
**Solution**: Run train_model.py:
```bash
python train_model.py
```

### Files exist but app still fails
**Solution**: Check file permissions:
```bash
chmod 644 models_data/*.pkl
```
