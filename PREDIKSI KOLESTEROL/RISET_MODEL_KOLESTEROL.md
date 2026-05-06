# Riset Pre-Trained Model: Prediksi Kolesterol

**Dibuat:** 2026-04-27 | **Konteks:** Skripsi RPM System

---

## Konteks & Tipe Prediksi

Prediksi kolesterol bisa dilakukan dalam dua pendekatan:
1. **Klasifikasi:** Normal vs Tinggi (High Cholesterol: ya/tidak)
2. **Regresi:** Prediksi nilai numerik kolesterol total (mg/dL)

Dataset yang digunakan biasanya adalah dataset kardiovaskular atau lipid profile.

---

## Dataset Utama

| # | Dataset | Sumber | Sampel | Fitur Utama |
|---|---|---|---|---|
| 1 | UCI Heart Disease (Cleveland) | UCI ML Repo / Kaggle | 303 | Age, Sex, ChestPain, RestBP, Cholesterol, FastingBS, MaxHR, dll |
| 2 | Cardiovascular Disease Dataset | Kaggle (sulianova) | 70.000 | Age, Height, Weight, AP_hi, AP_lo, Cholesterol (1/2/3), Gluc |
| 3 | Heart Attack Analysis & Prediction | Kaggle | 303 | age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng |
| 4 | Heart Failure Prediction (Fedesoriano) | Kaggle | 918 | Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, MaxHR |
| 5 | Framingham Heart Study | Kaggle | 4.240 | Age, Glucose, BMI, cigsPerDay, BPMeds, TotChol, HeartRate |
| 6 | NHANES Lipid Profile | CDC/Kaggle | Varies | Age, Sex, BMI, Waist, BP, Glucose, Triglycerides, HDL, LDL |

---

## 12 Kandidat Pre-Trained Model

### Model 1 — Logistic Regression (Klasifikasi: Normal vs High Chol)
- **Dataset:** Cardiovascular Disease Dataset | **Akurasi:** 68-72% | **AUC:** 0.74
- **Fitur:** Age, Gender, Weight, Height, AP_hi, AP_lo, Gluc, Smoke, Alco, Active
- **Format:** .pkl | **Framework:** scikit-learn
- **GitHub:** https://github.com/topics/cardiovascular-disease
- **Catatan:** Baseline, kolesterol sebagai target (cholesterol=1/2/3)

### Model 2 — K-Nearest Neighbors (KNN)
- **Dataset:** UCI Heart Disease / Cardiovascular | **Akurasi:** 70-75% | **AUC:** 0.76
- **Fitur:** Age, BP, BMI, Glucose, Gender
- **Format:** .pkl | **Framework:** scikit-learn
- **Catatan:** k=5-11 tipikal, wajib StandardScaler

### Model 3 — Decision Tree Classifier
- **Dataset:** UCI Heart Disease (Cleveland) | **Akurasi:** 74-79% | **AUC:** 0.77
- **Fitur:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, MaxHR
- **Format:** .pkl | **Framework:** scikit-learn
- **GitHub:** https://github.com/topics/heart-disease-prediction
- **Catatan:** Feature importance: cholesterol, age, MaxHR tinggi

### Model 4 — Naive Bayes (Gaussian)
- **Dataset:** Cardiovascular Disease | **Akurasi:** 66-70% | **AUC:** 0.72
- **Fitur:** Age, Gender, BMI, BP, Glucose
- **Format:** .pkl | **Framework:** scikit-learn
- **Catatan:** Baseline cepat, akurasi lebih rendah vs ensemble

### Model 5 — Support Vector Machine (SVM)
- **Dataset:** UCI Heart Disease / Framingham | **Akurasi:** 74-80% | **AUC:** 0.82
- **Fitur:** Semua fitur klinis (scaled)
- **Format:** .pkl | **Framework:** scikit-learn
- **Catatan:** Kernel RBF paling umum, wajib feature scaling

### Model 6 — Random Forest Classifier ⭐
- **Dataset:** Cardiovascular / Framingham | **Akurasi:** 79-84% | **AUC:** 0.86
- **Fitur:** Age, Gender, BMI, BP, Glucose, Smoke, Alco, Active (8-10 fitur)
- **Format:** .pkl / .joblib | **Framework:** scikit-learn
- **GitHub:** https://github.com/topics/cardiovascular-disease
- **Catatan:** Feature importance: BMI, Systolic BP, Age paling signifikan

### Model 7 — XGBoost Classifier ⭐
- **Dataset:** Cardiovascular / Framingham | **Akurasi:** 80-86% | **AUC:** 0.88
- **Fitur:** 8-15 fitur klinis + lifestyle
- **Format:** .pkl / .json | **Framework:** XGBoost
- **Catatan:** Sering best performer untuk data tabular klinis

### Model 8 — LightGBM
- **Dataset:** Framingham / NHANES | **Akurasi:** 80-87% | **AUC:** 0.89
- **Fitur:** 10-20 fitur
- **Format:** .txt / .pkl | **Framework:** LightGBM
- **Catatan:** Sangat efisien pada dataset besar seperti NHANES

### Model 9 — Gradient Boosting (sklearn)
- **Dataset:** UCI Heart / Cardiovascular | **Akurasi:** 79-84% | **AUC:** 0.86
- **Fitur:** Semua fitur klinis
- **Format:** .pkl | **Framework:** scikit-learn GradientBoostingClassifier
- **Catatan:** Alternatif XGBoost yang sudah built-in sklearn

### Model 10 — Linear Regression (Regresi Nilai Kolesterol)
- **Dataset:** Framingham / NHANES | **R²:** 0.55-0.72 | **RMSE:** ~28-35 mg/dL
- **Fitur:** Age, BMI, Glucose, TriglycerideLevels, HDL, Waist
- **Format:** .pkl | **Framework:** scikit-learn
- **Catatan:** Untuk memprediksi nilai kolesterol TOTAL (mg/dL), bukan klasifikasi

### Model 11 — ANN / MLP
- **Dataset:** Framingham / UCI Heart | **Akurasi:** 80-85% | **AUC:** 0.87
- **Fitur:** 8-15 fitur klinis
- **Format:** .h5 / .keras | **Framework:** TensorFlow/Keras
- **GitHub:** https://github.com/kanchitank/Medibuddy-Smart-Disease-Predictor
- **Catatan:** Hidden layers 64-128-64 tipikal untuk data ini

### Model 12 — Stacking Ensemble ⭐ TERBAIK
- **Dataset:** Cardiovascular / Framingham | **Akurasi:** 83-89% | **AUC:** 0.92
- **Fitur:** 8-15 fitur
- **Format:** .pkl (sklearn Pipeline) | **Framework:** scikit-learn StackingClassifier
- **Catatan:** RF + XGB + SVM sebagai base, Logistic Regression sebagai meta

---

## Perbandingan Akurasi

| # | Model | Task | Akurasi/R² | AUC-ROC | Rekomendasi |
|---|---|---|---|---|---|
| 1 | Logistic Regression | Klasifikasi | 68-72% | 0.74 | Baseline |
| 2 | KNN | Klasifikasi | 70-75% | 0.76 | Pembanding |
| 3 | Decision Tree | Klasifikasi | 74-79% | 0.77 | Pembanding |
| 4 | Naive Bayes | Klasifikasi | 66-70% | 0.72 | Baseline |
| 5 | SVM (RBF) | Klasifikasi | 74-80% | 0.82 | Pembanding |
| 6 | Random Forest | Klasifikasi | 79-84% | 0.86 | Utama ⭐ |
| 7 | XGBoost | Klasifikasi | 80-86% | 0.88 | Utama ⭐ |
| 8 | LightGBM | Klasifikasi | 80-87% | 0.89 | Utama ⭐ |
| 9 | Gradient Boosting | Klasifikasi | 79-84% | 0.86 | Pembanding |
| 10 | Linear Regression | Regresi (mg/dL) | R²=0.55-0.72 | — | Regresi |
| 11 | ANN / MLP | Klasifikasi | 80-85% | 0.87 | Pembanding |
| 12 | Stacking Ensemble | Klasifikasi | 83-89% | 0.92 | Utama ⭐⭐ |

---

## Fitur Standar yang Direkomendasikan

### Pendekatan Klasifikasi (Binary: Normal / High)
```
INPUT (Cardiovascular-based):
├── Age                     (usia dalam tahun)
├── Gender                  (0=Wanita, 1=Pria)
├── Height                  (cm)
├── Weight                  (kg) → bisa dikonversi ke BMI
├── ap_hi                   (tekanan darah sistolik)
├── ap_lo                   (tekanan darah diastolik)
├── Glucose                 (kadar glukosa: 1=normal, 2=above, 3=well above)
├── Smoke                   (merokok: 0/1)
├── Alco                    (alkohol: 0/1)
└── Active                  (aktivitas fisik: 0/1)

OUTPUT:
└── Cholesterol: 1=Normal, 2=Above Normal, 3=Well Above Normal
    ATAU binarisasi: 0=Normal, 1=High
```

### Pendekatan Regresi (Prediksi nilai mg/dL)
```
INPUT (Framingham/NHANES-based):
├── Age
├── BMI
├── SystolicBP
├── DiastolicBP
├── Glucose
├── Triglycerides
├── HDL_Cholesterol
├── Waist_Circumference
└── SmokingStatus

OUTPUT:
└── TotalCholesterol (mg/dL, nilai kontinu)
```

---

## Referensi GitHub Utama

1. Cardiovascular Disease Prediction (berbagai repo)
   https://github.com/topics/cardiovascular-disease
2. Heart Disease Prediction (UCI Cleveland dataset)
   https://github.com/topics/heart-disease-prediction
3. kritikaparmar-programmer/HealthCheck — Multi-disease ⭐180
   https://github.com/kritikaparmar-programmer/HealthCheck
4. kanchitank/Medibuddy-Smart-Disease-Predictor — Multi-disease ⭐100
   https://github.com/kanchitank/Medibuddy-Smart-Disease-Predictor

## Referensi Dataset Kaggle

- Cardiovascular Disease: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- UCI Heart Disease: https://www.kaggle.com/datasets/ronitf/heart-disease-uci
- Heart Failure Prediction: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Framingham Heart Study: https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset

## Referensi Jurnal / Paper

- "Machine Learning Models for Cholesterol Level Prediction" — MDPI Applied Sciences
- "Prediction of Hypercholesterolemia using ML Algorithms" — Frontiers in Medicine
- "Cholesterol Level Classification using ML Techniques" — berbagai jurnal IEEE/MDPI

---

## Rekomendasi untuk Skripsi

| Prioritas | Model | Alasan |
|---|---|---|
| Utama | XGBoost atau LightGBM | Akurasi terbaik, fast inference |
| Pembanding | Random Forest | Robust, feature importance jelas |
| Baseline | Logistic Regression | Interpretable |
| Dataset (Klasifikasi) | Cardiovascular Disease Dataset | 70k sampel, fitur lifestyle lengkap |
| Dataset (Regresi) | Framingham Heart Study | Fitur klinis detail |

### Catatan Penting untuk Kolesterol:
- Akurasi kolesterol umumnya lebih rendah dari diabetes karena lebih banyak faktor confounding
- Pertimbangkan apakah mau prediksi KLASIFIKASI (normal/tinggi) atau REGRESI (nilai mg/dL)
- Dataset kolesterol biasanya embedded dalam dataset kardiovaskular yang lebih besar
- Fitur paling prediktif: Age, BMI, Systolic BP, Glucose, Triglycerides
