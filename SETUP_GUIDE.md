# Complete Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Your dataset: `heart_disease_combined.csv`

## 🚀 Step-by-Step Setup

### Step 1: Extract the Project

```bash
# If you downloaded the ZIP file
unzip heart_disease_django_project.zip

# If you downloaded the TAR.GZ file
tar -xzf heart_disease_django_project.tar.gz

# Navigate to the project
cd heart_disease_project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- Django (web framework)
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning)
- joblib (model serialization)

### Step 4: Add Your Dataset

**IMPORTANT**: Copy your dataset to the data directory

```bash
# Copy your dataset
cp /path/to/your/heart_disease_combined.csv data/

# Verify it's there
ls -la data/heart_disease_combined.csv
```

The dataset file should be exactly at: `data/heart_disease_combined.csv`

### Step 5: Train the ML Models

```bash
python train_model.py
```

This will:
- ✅ Load your dataset
- ✅ Preprocess the data
- ✅ Train Logistic Regression model
- ✅ Train Random Forest model
- ✅ Evaluate models
- ✅ Save 7 .pkl files to `models_data/` directory

**Expected Output:**
```
📊 Loading dataset...
Dataset shape: (2944, 11)
...
✅ Training complete!
   - Logistic Regression: XX% accuracy, X.XXX ROC-AUC
   - Random Forest: XX% accuracy, X.XXX ROC-AUC

📦 Models and data saved to 'models_data/' directory
```

**Verify model files:**
```bash
ls -la models_data/
```

You should see:
- logistic_model.pkl
- rf_model.pkl
- scaler.pkl
- features.pkl
- numerical_features.pkl
- X_test.pkl
- y_test.pkl

### Step 6: Setup Django Database

```bash
# Create database migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate
```

This creates the SQLite database and tables for:
- Predictions
- Blockchain records
- Django admin

### Step 7: Create Admin User (Optional)

```bash
python manage.py createsuperuser
```

Follow prompts to create username and password for admin access.

### Step 8: Run the Development Server

```bash
python manage.py runserver
```

**Output:**
```
Starting development server at http://127.0.0.1:8000/
```

### Step 9: Open Your Browser

Visit: **http://127.0.0.1:8000**

You should see the professional homepage!

## 🎯 Verify Everything Works

### Test the Prediction Flow:

1. **Home Page** (`/`)
   - Should load with nice medical design
   - Click "Start Prediction"

2. **Prediction Page** (`/predict/`)
   - Fill in the form with patient data
   - Click "Predict Risk"

3. **Results Page** (`/results/<id>/`)
   - See risk assessment
   - View model predictions
   - See brief blockchain info

4. **Blockchain Page** (`/blockchain/`)
   - Click "Blockchain" in navigation
   - See full blockchain chain
   - View all hashes

5. **History Page** (`/history/`)
   - View table of all predictions

### Test Admin Panel:

1. Visit: http://127.0.0.1:8000/admin
2. Login with superuser credentials
3. Browse predictions and blockchain records

## 📁 Project Structure Overview

```
heart_disease_project/
│
├── data/                          # ← YOUR DATASET GOES HERE
│   ├── heart_disease_combined.csv # ← REQUIRED
│   └── README.md
│
├── models_data/                   # ← ML MODELS GO HERE (auto-generated)
│   ├── logistic_model.pkl         # ← Created by train_model.py
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   ├── numerical_features.pkl
│   ├── X_test.pkl
│   ├── y_test.pkl
│   └── README.md
│
├── heart_disease_app/             # Main Django app
│   ├── utils/                     # ← NEW! Utility modules
│   │   ├── __init__.py
│   │   ├── blockchain.py          # Blockchain logic
│   │   ├── ml_model.py            # ML model management
│   │   └── preprocessing.py       # Data validation
│   ├── templates/                 # HTML files
│   ├── static/                    # CSS & JavaScript
│   ├── models.py                  # Database models
│   ├── views.py                   # Business logic
│   ├── urls.py                    # URL routing
│   └── admin.py                   # Admin config
│
├── heart_disease_project/         # Django config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── manage.py                      # Django management
├── train_model.py                 # ML training script
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── QUICKSTART.md                  # Quick reference
└── SETUP_GUIDE.md                 # This file
```

## ⚠️ Common Issues & Solutions

### Issue 1: "No module named 'django'"
```bash
# Solution:
pip install -r requirements.txt
```

### Issue 2: "No such file: data/heart_disease_combined.csv"
```bash
# Solution:
cp /your/path/heart_disease_combined.csv data/
```

### Issue 3: "Model files not found"
```bash
# Solution:
python train_model.py
```

### Issue 4: "no such table: heart_disease_app_prediction"
```bash
# Solution:
python manage.py makemigrations
python manage.py migrate
```

### Issue 5: Static files not loading
```bash
# Solution:
python manage.py collectstatic --noinput
```

### Issue 6: Port already in use
```bash
# Solution: Use different port
python manage.py runserver 8001
```

## ✅ Checklist

Before running:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`requirements.txt`)
- [ ] Dataset copied to `data/heart_disease_combined.csv`
- [ ] Models trained (`python train_model.py`)
- [ ] Migrations applied (`python manage.py migrate`)
- [ ] Server running (`python manage.py runserver`)

You're all set! 🎉
