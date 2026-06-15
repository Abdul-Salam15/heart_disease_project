# CardioCare AI — Presentation Script for Lecturer

A talking-points script you can read from (or adapt) when presenting this project. It's organized as: elevator pitch → problem & objectives → architecture → features deep-dive → live demo walkthrough → anticipated questions → limitations & future work.

---

## 1. Elevator Pitch (30 seconds)

> "My project is **CardioCare AI**, a web application built with Django that predicts a patient's risk of heart disease from basic clinical data. It uses **two machine learning models** — Logistic Regression and Random Forest — and combines their outputs into a single 'consensus' risk score. Every prediction is also recorded in a **blockchain-style audit log**, so the history of predictions is tamper-evident and verifiable. The whole interface is **mobile-first and responsive**, so it works on a phone, tablet, or desktop."

---

## 2. Problem Statement & Motivation

- Cardiovascular disease is one of the leading causes of death globally, and early risk screening can help people seek medical attention sooner.
- Many existing prediction demos are just a single model with no record-keeping — if you wanted to audit *who* got *what* prediction *when*, there's no trustworthy trail.
- This project's goal was to build an **end-to-end system**: data → trained ML models → a usable web interface → persisted results → an **immutable audit trail** of every prediction made.

**Objectives:**
1. Train and compare two different ML algorithms on a real heart-disease dataset.
2. Combine both models into a more robust "consensus" prediction.
3. Build a clean, mobile-friendly Django web app around the models.
4. Add a blockchain-inspired mechanism to make the prediction history tamper-evident.
5. Make it deployable (Docker + Render.com).

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | Django 4.2 (Python) |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest), pandas, NumPy |
| Visualization | Matplotlib, Seaborn (confusion matrices, ROC curves) |
| Database | SQLite (dev) / PostgreSQL-ready (production) |
| Frontend | Django templates, custom mobile-first CSS, vanilla JavaScript |
| Security / Audit | Custom blockchain module using SHA-256 hashing |
| Deployment | Docker, Gunicorn, Render.com |

---

## 4. System Architecture (high level)

Walk the lecturer through this flow (you can show the Mermaid diagrams in `README.md` if presenting on a screen):

1. **User** opens the site in a browser and fills in the **Predict** form (age, sex, blood pressure, cholesterol, chest pain type, etc.).
2. The **Django view** (`predict_view`) receives the form, and a **DataPreprocessor / DataValidator** cleans and validates the input.
3. The cleaned data is converted into the exact feature format the models expect (one-hot encoding + scaling), via `MLModelManager`.
4. **Both ML models** (Logistic Regression and Random Forest) run inference and each produce a probability of heart disease.
5. The two probabilities are averaged into a **consensus probability**, which determines the final **risk level (High/Low)**.
6. The result is saved as a `Prediction` row in the database **and** a new **block** is appended to the blockchain table (`BlockchainRecord`), hashing the prediction data together with the previous block's hash.
7. The user is redirected to a **Results** page showing both individual model outputs and the consensus result.
8. The **History** page lists all past predictions, and the **Blockchain** page lets you inspect/verify the full audit chain.

---

## 5. Feature Deep-Dive

### 5.1 Dual ML Models + Consensus

- **Logistic Regression**: simple, interpretable, gives a clear linear relationship between features and risk — good for explaining *why* a prediction was made.
- **Random Forest** (200 trees, max depth 10): captures non-linear relationships and feature interactions, generally more accurate but less interpretable.
- **Consensus score** = average of the two probabilities (`(log_prob + rf_prob) / 2`). If ≥ 0.5 → classified as **High Risk**, otherwise **Low Risk**.
- Why two models? It mirrors a "second opinion" approach — if both models agree, you can be more confident; if they disagree, that itself is useful information.

**Training pipeline** (`train_model.py`):
- Loads `data/heart_disease_combined.csv` (based on the UCI Heart Disease dataset).
- Cleans data, maps numeric codes to readable categories (Sex, ChestPainType, RestingECG, ExerciseAngina).
- One-hot encodes categorical features, scales numerical features with `StandardScaler`.
- 80/20 train/test split, stratified by the target so the class balance is preserved.
- Trains both models with `class_weight='balanced'` to handle any class imbalance.
- Evaluates with accuracy, precision, recall, F1, and ROC-AUC, plus 5-fold cross-validation.
- Saves the trained models, scaler, and feature lists with `joblib` into `models_data/`.

**Approximate results** (from README):
- Logistic Regression: ~85% accuracy, ~0.87 ROC-AUC.
- Random Forest: ~88% accuracy, ~0.90 ROC-AUC.

The app also pre-renders **confusion matrices** and **ROC curves** for both models so you can show these on a "Model Performance" page.

### 5.2 Blockchain-Style Audit Trail

- Each prediction is wrapped in a `Block` containing: index, timestamp, prediction data (as JSON), and the **hash of the previous block**.
- The block's own hash is `SHA-256(index + timestamp + data + previous_hash)`.
- Because each block depends on the previous block's hash, **changing any past record would break the chain** — this is the same core idea as a real blockchain, just without distributed consensus/mining.
- `BlockchainManager.verify_chain()` walks the entire chain, recomputing hashes and checking links, so you can prove (or disprove) that the history hasn't been tampered with.
- This gives the system a built-in **integrity check** for medical prediction records — useful framing for "trustworthy AI" / data-integrity discussions.

### 5.3 Mobile-First Responsive UI

- Built with custom CSS using three breakpoints: mobile (<768px), tablet (768–1023px), desktop (1024px+).
- Hamburger navigation on mobile, full nav bar on larger screens.
- 44px minimum touch targets (accessibility best practice).
- Same Django templates render correctly across devices — no separate mobile app needed.

### 5.4 History & Admin

- **History page**: lists every prediction made, filterable by risk level, with summary statistics (total predictions, high/low risk counts), and the ability to delete individual records.
- **Django Admin**: superuser can inspect `Prediction` and `BlockchainRecord` tables directly.

---

## 6. Live Demo Script

If you can run the app live, follow this order:

1. **Home page** — briefly show the landing page, explain the purpose of the app.
2. **Predict page** — fill in a sample patient (e.g., Age 58, Male, ChestPainType ASY, RestingBP 140, Cholesterol 240, FastingBS yes, RestingECG ST, MaxHR 120, ExerciseAngina yes, Oldpeak 1.5). Click **Predict Risk**.
3. **Results page** — point out:
   - Individual Logistic Regression probability
   - Individual Random Forest probability
   - Consensus probability and final risk level (High/Low)
   - The patient data summary
4. **History page** — show the new prediction appearing in the list, with the running stats.
5. **Blockchain page** — show the new block that was created for this prediction, point out its hash and the link to the previous block's hash. If there's a "verify chain" feature, demonstrate that the chain is valid.
6. *(Optional)* **Model Performance** — show the confusion matrices and ROC curves for both models, and explain what they mean.
7. *(Optional)* **Admin panel** — log in as superuser and show the raw `Prediction` / `BlockchainRecord` tables.

---

## 7. Anticipated Questions & Suggested Answers

**Q: Why not just use one model — isn't averaging two models overkill?**
A: Each algorithm has different strengths — Logistic Regression is interpretable and good with linear relationships, Random Forest captures non-linear patterns. Averaging gives a more robust estimate and surfaces disagreement between models as a signal of uncertainty.

**Q: Is this a "real" blockchain?**
A: No — it's a **blockchain-inspired data structure** (hash-linked, tamper-evident records), not a distributed ledger with consensus/mining/multiple nodes. The goal is data integrity and auditability for the prediction history, not decentralization.

**Q: How accurate is it, and is it safe to use clinically?**
A: The models reach ~85–88% accuracy and ~0.87–0.90 ROC-AUC on the test set, which is solid for an educational project, but this is explicitly **not a diagnostic tool** — it's a statistical screening aid. The README includes a disclaimer that it's for educational/research purposes only.

**Q: How was the model trained / what dataset was used?**
A: A combined heart-disease dataset (based on the UCI Heart Disease dataset) with features like age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, and ST depression (Oldpeak).

**Q: How would you scale this to production?**
A: Switch from SQLite to PostgreSQL (already supported via `DATABASE_URL`), run behind Gunicorn with multiple workers (already configured), and the app is containerized with Docker for deployment on Render or similar platforms.

**Q: What would you improve given more time?**
A: See the "Limitations & Future Work" section below.

---

## 8. Limitations & Future Work

- The blockchain is local/single-node — it provides tamper-evidence but not the decentralization guarantees of a real blockchain.
- The dataset is relatively small and from a specific population; a production system would need much larger, more diverse, and more recent clinical data.
- No user authentication/authorization for patient-specific records — anyone with access to the app can view all history.
- Could add more models (e.g., gradient boosting, neural networks) to the consensus ensemble.
- Could add explainability tools (e.g., SHAP values) to show *why* a specific prediction was made, beyond just feature importance.
- Could add proper user accounts so each patient only sees their own prediction history.

---

## 9. Closing Statement

> "In summary, this project demonstrates the full pipeline of an applied ML system: from data preprocessing and model training, through a production-style web application, to a record-keeping mechanism inspired by blockchain for integrity and auditability — all wrapped in a responsive, mobile-friendly UI. It's intended as an educational demonstration of how AI predictions can be made, served, and audited responsibly."
