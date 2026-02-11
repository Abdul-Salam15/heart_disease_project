# Data Directory

## Purpose
This directory contains the dataset used for training and evaluation.

## Required File

You need to place your dataset here:

```
data/heart_disease_combined.csv
```

## Dataset Location

The dataset should be located at: `/path/to/data/heart_disease_combined.csv`

As you mentioned in your conversation, the file is at:
```
data/heart_disease_combined.csv
```

## Important

⚠️ **Before running the application:**

1. Copy your dataset to this directory:
   ```bash
   cp /path/to/heart_disease_combined.csv data/
   ```

2. Verify the file exists:
   ```bash
   ls -la data/heart_disease_combined.csv
   ```

3. The file should have the following columns:
   - Age
   - Sex (0 or 1)
   - ChestPainType (1, 2, 3, or 4)
   - RestingBP
   - Cholesterol
   - FastingBS (0 or 1)
   - RestingECG (0, 1, or 2)
   - MaxHR
   - ExerciseAngina (0 or 1)
   - Oldpeak
   - HeartDisease (target: 0 or 1)

## After Adding Dataset

Once the dataset is in place, run:

```bash
python train_model.py
```

This will:
- Load the dataset from `data/heart_disease_combined.csv`
- Train the ML models
- Save model files to `models_data/`
