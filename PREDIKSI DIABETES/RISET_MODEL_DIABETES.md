# Riset Pre-Trained Model: Prediksi Diabetes

**Dibuat:** 2026-04-27 | **Konteks:** Skripsi RPM System

---

## Dataset Utama

| # | Dataset | Sumber | Sampel | Fitur Utama |
|---|---|---|---|---|
| 1 | PIMA Indians Diabetes | UCI / Kaggle | 768 | Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age |
| 2 | CDC BRFSS 2015 | Kaggle (AlexTeboul) | 253.680 | 21 fitur lifestyle + klinis |
| 3 | Diabetes Health Indicators | Kaggle | 70.692 | BMI, BP, Chol, Smoke, PhysActivity, Age |
| 4 | Frankfurt Hospital Diabetes | Kaggle | 2.000 | Age, Gender, Glucose, HbA1c, BMI |

---

## 12 Kandidat Pre-Trained Model

### Model 1 — Logistic Regression
- **Dataset:** PIMA Indians | **Akurasi:** 76-79% | **AUC:** 0.82
- **Fitur:** Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age
- **Format:** .pkl | **Framework:** scikit-learn
- **GitHub:** https://github.com/MrKhan0747/Diabetes-Prediction
- **Catatan:** Cocok sebagai baseline, paling interpretable

### Model 2 — K-Nearest Neighbors (KNN)
- **Dataset:** PIMA Indians | **Akurasi:** 74-80% | **AUC:** 0.79
- **Fitur:** BMI, BloodPressure, Glucose, Insulin (4-8 fitur)
- **Format:** .pkl | **Framework:** scikit-learn
- **GitHub:** https://github.com/zikry009/Diabetes_Prediction_Portal
- **Catatan:** Sensitif terhadap k dan scaling data

### Model 3 — Decision Tree (CART)
- **Dataset:** PIMA Indians | **Akurasi:** 72-76% | **AUC:** 0.76
- **Fitur:** Semua 8 fitur PIMA
- **Format:** .pkl | **Framework:** scikit-learn
- **GitHub:** https://github.com/sondosaabed/Predicting-Diabetes-with-Decision-Trees
- **Catatan:** Visualizable, prone to overfitting

### Model 4 — Naive Bayes (Gaussian)
- **Dataset:** PIMA Indians | **Akurasi:** 75-77% | **AUC:** 0.81
- **Fitur:** Semua 8 fitur PIMA
- **Format:** .pkl | **Framework:** scikit-learn
- **GitHub:** https://github.com/MrKhan0747/Diabetes-Prediction
- **Catatan:** Sangat cepat, asumsi independensi fitur

### Model 5 — Support Vector Machine (SVM-RBF)
- **Dataset:** PIMA Indians | **Akurasi:** 77-82% | **AUC:** 0.84
- **Fitur:** Semua 8 fitur PIMA (wajib di-scale)
- **Format:** .pkl | **Framework:** scikit-learn
- **GitHub:** https://github.com/ditikrushna/End-to-End-Diabetes-Prediction-Application-Using-Machine-Learning
- **Catatan:** Efektif, butuh StandardScaler sebelum prediksi

### Model 6 — Random Forest Classifier
- **Dataset:** PIMA / CDC BRFSS | **Akurasi:** 80-86% | **AUC:** 0.87
- **Fitur:** 8-21 fitur
- **Format:** .pkl / .joblib | **Framework:** scikit-learn
- **GitHub:** https://github.com/MrKhan0747/Diabetes-Prediction
- **Catatan:** Robust, feature importance tersedia

### Model 7 — XGBoost Classifier ⭐ TERBAIK SINGLE MODEL
- **Dataset:** PIMA / CDC BRFSS | **Akurasi:** 82-88% | **AUC:** 0.90
- **Fitur:** 8-21 fitur + feature engineering
- **Format:** .pkl / .json (xgb native) | **Framework:** XGBoost
- **GitHub:** https://github.com/Aditya-Mankar/Diabetes-Prediction
- **Kaggle:** https://www.kaggle.com/code?search=diabetes+xgboost
- **Catatan:** Regularisasi bawaan, sangat direkomendasikan

### Model 8 — LightGBM
- **Dataset:** CDC BRFSS 2021 | **Akurasi:** 83-89% | **AUC:** 0.91
- **Fitur:** 21 fitur lifestyle klinis
- **Format:** .txt / .pkl | **Framework:** LightGBM
- **Catatan:** Lebih cepat dari XGBoost, hemat memori

### Model 9 — AdaBoost Classifier
- **Dataset:** PIMA Indians | **Akurasi:** 79-83% | **AUC:** 0.85
- **Fitur:** Semua 8 fitur PIMA
- **Format:** .pkl | **Framework:** scikit-learn
- **Catatan:** Mengurangi bias DT, sensitif terhadap outlier

### Model 10 — ANN / MLP (Neural Network)
- **Dataset:** PIMA / Frankfurt | **Akurasi:** 80-86% | **AUC:** 0.87
- **Fitur:** 8 fitur (PIMA)
- **Format:** .h5 / .keras | **Framework:** TensorFlow/Keras atau sklearn MLPClassifier
- **GitHub:** https://github.com/razuswe/PrimaIndianDiabetesprediction
- **Catatan:** Non-linear, black box, butuh lebih banyak data

### Model 11 — Stacking Ensemble ⭐ TERBAIK OVERALL
- **Dataset:** PIMA / CDC BRFSS | **Akurasi:** 85-91% | **AUC:** 0.93
- **Fitur:** 8-21 fitur
- **Format:** .pkl (sklearn Pipeline) | **Framework:** scikit-learn StackingClassifier
- **Catatan:** RF + XGB + LR sebagai meta-learner

### Model 12 — CatBoost Classifier
- **Dataset:** CDC BRFSS 2021 | **Akurasi:** 83-88% | **AUC:** 0.90
- **Fitur:** 21 fitur (termasuk categorical)
- **Format:** .cbm / .pkl | **Framework:** CatBoost
- **Catatan:** Handle categorical otomatis, fast inference

---

## Perbandingan Akurasi

| # | Model | Akurasi | AUC-ROC | Rekomendasi |
|---|---|---|---|---|
| 1 | Logistic Regression | 76-79% | 0.82 | Baseline |
| 2 | KNN | 74-80% | 0.79 | Pembanding |
| 3 | Decision Tree | 72-76% | 0.76 | Pembanding |
| 4 | Naive Bayes | 75-77% | 0.81 | Pembanding |
| 5 | SVM (RBF) | 77-82% | 0.84 | Pembanding |
| 6 | Random Forest | 80-86% | 0.87 | Utama |
| 7 | XGBoost | 82-88% | 0.90 | Utama ⭐ |
| 8 | LightGBM | 83-89% | 0.91 | Utama ⭐ |
| 9 | AdaBoost | 79-83% | 0.85 | Pembanding |
| 10 | ANN / MLP | 80-86% | 0.87 | Pembanding |
| 11 | Stacking Ensemble | 85-91% | 0.93 | Utama ⭐⭐ |
| 12 | CatBoost | 83-88% | 0.90 | Utama |

---

## Fitur Standar (PIMA-based)

```
Pregnancies, Glucose, BloodPressure, SkinThickness,
Insulin, BMI, DiabetesPedigreeFunction, Age
OUTPUT: Outcome (0=Tidak Diabetes, 1=Diabetes)
```

---

## Referensi GitHub Utama

1. ditikrushna/End-to-End-Diabetes-Prediction — Flask + SVM ⭐189
   https://github.com/ditikrushna/End-to-End-Diabetes-Prediction-Application-Using-Machine-Learning
2. MrKhan0747/Diabetes-Prediction — 6 algoritma + CV ⭐114
   https://github.com/MrKhan0747/Diabetes-Prediction
3. Aditya-Mankar/Diabetes-Prediction — Multiple ML + Flask ⭐166
   https://github.com/Aditya-Mankar/Diabetes-Prediction
4. razuswe/PrimaIndianDiabetesprediction — ANN+SVM+RF+LR+NB+DT ⭐37
   https://github.com/razuswe/PrimaIndianDiabetesprediction
5. kanchitank/Medibuddy-Smart-Disease-Predictor — Multi-disease ⭐100
   https://github.com/kanchitank/Medibuddy-Smart-Disease-Predictor

## Referensi Dataset Kaggle

- PIMA: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- CDC BRFSS: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- Frankfurt: https://www.kaggle.com/datasets/mathchi/diabetes-data-set

---

## Rekomendasi untuk Skripsi

| Prioritas | Model | Alasan |
|---|---|---|
| Utama | XGBoost atau LightGBM | Akurasi tinggi, fast inference, export .pkl mudah |
| Pembanding | Random Forest | Robust, feature importance jelas |
| Baseline | Logistic Regression | Interpretable, mudah dijelaskan ke pembimbing |
| Dataset | PIMA + CDC BRFSS | PIMA=benchmark standar; CDC=fitur lifestyle lebih lengkap |
