# ML Projects Summary — ALGORITMA Folder
> **Tujuan:** Memilih algoritma terbaik untuk sistem **Remote Patient Monitoring (RPM)** prediksi hipertensi.
> **Total Proyek:** 10 proyek (6 original + 4 tambahan, termasuk pre-trained models)

---

## Model Comparison Table

| # | Project Name | Primary Objective | Algorithms Used | Dataset Source | Dataset Size | Key Variables (Features) | Target Variable | Preprocessing Techniques | Reported Accuracy / Performance | Deployment |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **ML-SVM-DecisionTree-Hypertension-Prediction** | Classify hypertension from clinical & demographic data | SVM (RBF kernel) · Decision Tree (entropy) | Kaggle: BRFSS 2015 | 4,063 rows · 18 features | Gender, Age, Education, Work, Country, Systolic_BP, Diastolic_BP, Height, Weight, Cholesterol (SI), Triglycerides (SI), Smoking_History, DM, Alcohol_Use, Dyslipidemia, DM_Type, DM_Treatment | Hypertension (Binary: 0/1) | KNN Imputation (k=5) · Manual outlier correction · Label Encoding · Train/Test 70/30 | Accuracy: ~84–86% (varies by feature subset) | Jupyter Notebooks · No saved model · [GitHub](https://github.com/Mostapha-El-Kaddaoui/ML-SVM-DecisionTree-Hypertension-Prediction) |
| **2** | **heart_disease_prediction** | Predict heart disease presence from clinical indicators | Logistic Regression (max_iter=1000) | Kaggle: BRFSS 2015 (mislabeled as hypertension) | 70,692 rows · 14 features | age, sex, cp (chest pain type), trestbps (resting BP), chol (cholesterol), fbs (fasting glucose), restecg (EKG), thalach (max HR), exang (exercise angina), oldpeak (ST depression), slope, ca (vessels), thal | Heart Disease (Binary: 0/1) | Train/Test 80/20 with stratification · No null values | **Test Accuracy: 86.11%** | **Streamlit Web App** (deployed) · No saved model · [GitHub](https://github.com/irgiys/heart_disease_prediction) |
| **3** | **Hypertension-Symptoms-Prediction-Using-ML** | Predict hypertension risk from lab & health markers (Mexico) | Logistic Regression · Ridge · SVC · Random Forest · XGBoost · AdaBoost · Gradient Boosting · Bagging · Decision Tree | Kaggle: ENSANUT México 2022 ("Hipertensión Arterial México") | 4,363 rows · 36 features | Age, Sex, BMI, Waist, Weight, Height, Hemoglobin, Uric_Acid, HDL/LDL/Total_Cholesterol, Creatinine, Glucose, Insulin, Triglycerides, Ferritin, Folate, Homocysteine, CRP, Transferrin, Vitamin B12, Vitamin D, Sleep_Hours, Blood_Pressure, Physical_Activity | riesgo_hipertension (Binary: 0/1) | SMOTE (class imbalance) · Label Encoding · Exploratory visualizations (histograms, boxplots, correlation heatmap) | **Gradient Boosting: 100%** ⚠️ · XGBoost: 99.7% · Bagging: 99.5% · ⚠️ Suspected data leakage | Jupyter Notebook · No saved model · [GitHub](https://github.com/BhaveshBhakta/Hypertension-Symptoms-Prediction-Using-ML) |
| **4** | **Blood-Pressure-Prediction-and-Personalized-Health-Behavior-Recommendation** | Continuous BP estimation using wearable PPG data | Regression (Linear/RF/NN) · Time-series analysis | Samsung Galaxy Watch (real-time via Samsung Health SDK + OAuth) + Omron Wellness | User-specific (continuous streaming) | Heart_Rate (daily/weekly), Sleep_Duration, Sleep_Stages (REM/deep/light), Step_Count, Calories_Burned, Exercise_Events, Sedentary_Time_Ratio, PPG_Signals, Time-of-day, Activity_type | Systolic BP · Diastolic BP (Continuous values) | 24-hour window aggregation · Interpolation for irregular sampling · Unix time conversion · Lag features (lag 1–3) · Sleep data processing | Not specified (ongoing project) | Streamlit Visualizations · OAuth API integration · No saved model · [GitHub](https://github.com/kwanmolee/Blood-Pressure-Prediction-and-Peronalized-Health-Behavior-Recommendation) |
| **5** | **High-Blood-Pressure-Predictor-Machine-Learning** | Predict high blood pressure in smoker population | Gaussian Naive Bayes · Logistic Regression · Random Forest · SVM | NHANES III (National Health and Nutrition Examination Survey) | Not specified | SEQN, HSAGEIR (age), HSSEX (sex), BMPWTLBS (weight), BMPHTIN (height), SMOKE (smoking status), TCP (total cholesterol), HBP (target) — smokers only | HBP – High Blood Pressure (Binary: 0/1) | Data munging · Domain-specific feature engineering | Not documented in README | Jupyter Notebook · No saved model · [GitHub](https://github.com/febielgiva/High-Blood-Pressure-Predictor-Machine-Learning) |
| **6** | **Hypertension-risk-model** | Predict 10-year hypertension risk from Framingham variables | Logistic Regression · SVM · Decision Tree · Random Forest | Framingham Heart Study | 4,240 raw → 3,751 after cleaning | male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose | Risk (Binary: 0/1) | Drop missing values (~12.74% missing) · Class weight handling · EDA | Not documented (imbalanced classes: 2581 no-risk vs 1170 at-risk) | Tkinter GUI Desktop App · ✅ Saved model: `model_joblib_hypertension` (5.3 MB, joblib) · [GitHub](https://github.com/amakaogbu/Hypertension-risk-model) |
| **7** | **Hypertension-Risk-Prediction-Using-Machine-Learning (NaijaDataProfessor)** | Classify hypertension risk from lifestyle factors | **XGBoost** (best) · Random Forest · Decision Tree · SVC · Logistic Regression · KNN | Synthetic lifestyle dataset | 1,985 rows · 10 features | **bmi** (importance: 0.37), **family_history** (0.13), **smoking_status** (0.12), **stress_score** (0.09), **bp_history** (0.07), age, salt_intake, sleep_duration, medication, exercise_level | has_hypertension (Binary: 0/1) | Random UnderSampling (class balance) · fillna Medication="None" · Target encoding (Yes/No→1/0) · Feature separation (numeric vs categorical) | **XGBoost: Accuracy 99%, Precision 99%, Recall 99%, F1 99%** · RF: 96% · DT: 95% · SVC: 88% · LR: 82% · KNN: 80% | **Streamlit Web App** ✅ (`app.py`) · ✅ Saved model: `hypertension_model_v2.pkl` (235 KB, joblib) · [GitHub](https://github.com/NaijaDataProfessor/Hypertension-Risk-Prediction-Using-Machine-Learning) |
| **8** | **PredictingHypertension (seunghahh)** | Predict hypertension probability from clinical features | Unknown (sklearn classifier — loaded from pkl) | Unknown clinical dataset | ~200 samples (100 normal + 100 hypertension) | 18 clinical features (exact names not documented — min-max normalized via `.mat` file) | Hypertension probability (Continuous: 0–100%) | Min-Max normalization using `hypertension_norm.mat` · NaN values replaced with min values | Not documented | CLI script (`TestingModel.py`) · ✅ Saved model: `model.pkl` (266 KB, pickle) + `hypertension_norm.mat` (325 B) · [GitHub](https://github.com/seunghahh/PredictingHypertension) |
| **9** | **bpc-prediction-lmics (SiliconBlast)** | Predict high blood pressure across 57 low- & middle-income countries | **XGBoost** · **Random Forest** · Logistic Regression · KNN | WHO STEPS Survey (57 LMICs) | **184,674 rows** · 48 variables | 11 demographic (age, sex, education, occupation, marital, urban/rural, wealth, insurance, alcohol, tobacco) · 24 behavioral (physical activity, diet, sedentary, sleep, stress) · 5 physical (height, weight, waist, hip, BP) · 8 biochemical (glucose, cholesterol, triglycerides, creatinine, HbA1c) | Blood pressure status (Binary: normal/high) | StandardScaler · One-hot encoding for categoricals · 10-fold cross-validation · LabelEncoder on target | Not documented in README (10-fold CV metrics saved to Excel per model) | Research only · ✅ Saved models (4 files): `XGB_global.pkl`, `RF_global.pkl`, `LR_global.pkl`, `KNN_global.pkl` in `src/models/global/` · [GitHub](https://github.com/SiliconBlast/bpc-prediction-lmics) |
| **10** | **ypd774/hypertension (HuggingFace)** | Predict hypertension from Framingham-style clinical data via REST API | Unknown sklearn classifier (loaded from pkl) | Framingham-derived | Not documented | **gender** (Male/Female→1/0), **age**, **cigsPerDay**, **BPMeds** (Yes/No→1/0), **totChol**, **sysBP**, **diaBP**, **BMI**, **heartRate** — 9 features total | Hypertension (Binary: 0/1) | StandardScaler normalization (`scaler_hyper.pkl`) · Categorical mapping (gender, BPMeds) | Not documented | **Flask REST API** ✅ (`hypertension_api.py` with `/predict` endpoint) · ✅ Saved model: `hyptertension_model1.pkl` (16.1 MB, pickle) + `scaler_hyper.pkl` (832 B) · [HuggingFace](https://huggingface.co/ypd774/hypertension) |

---

## 🩺 Input Feature Dictionary — Consolidated (Semua Proyek)

### Fitur yang Muncul di Banyak Proyek (Prioritas Tinggi)

| Fitur | Nama Asli per Proyek | Tersedia di RPM? | Prioritas |
|---|---|---|---|
| **Age** | age, Age, edad, HSAGEIR | ✅ Form intake | ⭐⭐⭐⭐⭐ |
| **Sex / Gender** | sex, Gender, sexo, HSSEX, gender | ✅ Form intake | ⭐⭐⭐⭐⭐ |
| **Systolic BP** | Systolic_BP, trestbps, sysBP, tension_arterial | ✅ Core RPM metric | ⭐⭐⭐⭐⭐ |
| **Diastolic BP** | Diastolic_BP, diaBP | ✅ Core RPM metric | ⭐⭐⭐⭐⭐ |
| **BMI** | BMI, bmi, masa_corporal, Body_Mass_Index | ✅ Calculated (weight/height²) | ⭐⭐⭐⭐⭐ |
| **Cholesterol (Total)** | Cholesterol_Value_SI_Units, chol, totChol, colesterol_total | ⚠️ Lab test | ⭐⭐⭐⭐ |
| **Blood Pressure History** | bp_history | ✅ Survey question | ⭐⭐⭐⭐ |
| **Family History** | family_history | ✅ Survey question | ⭐⭐⭐⭐ |
| **Smoking Status** | Smoking_History, smoking_status, currentSmoker | ✅ Survey question | ⭐⭐⭐⭐ |
| **Diabetes Status** | DM, diabetes, DM_Type, DM_Treatment | ✅ Form intake | ⭐⭐⭐⭐ |
| **Heart Rate** | Heart_Rate, heartRate, thalach | ✅ Wearable/RPM device | ⭐⭐⭐ |
| **Glucose** | Glucose_Result, resultado_glucosa, glucose, fbs | ⚠️ Lab test | ⭐⭐⭐ |
| **Stress Level** | stress_score | ✅ Survey/questionnaire | ⭐⭐⭐ |
| **Medication** | BPMeds, Medication | ✅ Medical history | ⭐⭐⭐ |
| **Alcohol Use** | Alcohol_Use | ✅ Survey question | ⭐⭐ |
| **Exercise Level** | actividad_total, Exercise_Level, exang | ✅ Wearable/survey | ⭐⭐ |
| **Sleep Duration** | Sleep_Duration, sueno_horas | ✅ Wearable/survey | ⭐⭐ |
| **Weight / Height** | Weight/Height, peso/estatura, BMPWTLBS/BMPHTIN | ✅ Form intake | ⭐⭐⭐ |
| **HDL Cholesterol** | HDL_Cholesterol, colesterol_hdl | ⚠️ Lab test | ⭐⭐ |
| **Triglycerides** | Triglycerides_Value_SI_Units, valor_trigliceridos | ⚠️ Lab test | ⭐⭐ |

---

## 📋 Detail per Proyek

### Proyek 1: ML-SVM-DecisionTree-Hypertension-Prediction
| Variable | Tipe | Deskripsi | Satuan |
|---|---|---|---|
| Gender | Categorical | 1=Male, 0=Female | Binary |
| Age | Numeric | Usia pasien | Tahun |
| Education | Categorical | 0-5 (None → Post-grad) | Ordinal |
| Work | Categorical | 0=No, 1=Full-time, 2=Part-time | Categorical |
| Country | Categorical | 0=Bahrain, 1=UAE, 2=Oman, 3=Kuwait | Categorical |
| Systolic_BP | Numeric | Tekanan darah sistolik | mmHg (70–370) |
| Diastolic_BP | Numeric | Tekanan darah diastolik | mmHg (40–150) |
| Height | Numeric | Tinggi badan | cm |
| Weight | Numeric | Berat badan | kg |
| Cholesterol_Value_SI_Units | Numeric | Kolesterol total | mmol/L |
| Triglycerides_Value_SI_Units | Numeric | Trigliserida | mmol/L |
| Smoking_History | Categorical | 0=Never, 1=Past>1yr, 2=Recent, 3=Current | Ordinal |
| DM | Binary | Diabetes mellitus | 0=No, 1=Yes |
| Alcohol_Use | Categorical | 0=None, 1=≤1/wk, 2=2-7/wk, 3=≥8/wk | Ordinal |
| Dyslipidemia | Binary | Dislipidemia | 0=No, 1=Yes |
| DM_Type | Categorical | Tipe diabetes | Categorical |
| DM_Treatment | Categorical | Pengobatan diabetes | Categorical |
| **Hypertension** | **Binary** | **Target variable** | **0=No, 1=Yes** |

---

### Proyek 7: NaijaDataProfessor — Hypertension-Risk-Prediction ⭐ TERBAIK UNTUK RPM
| Variable | Tipe | Deskripsi | Top Feature? |
|---|---|---|---|
| bmi | Numeric | Body Mass Index | ✅ #1 (0.37 importance) |
| family_history | Categorical | yes/no | ✅ #2 (0.13) |
| smoking_status | Categorical | Never/Former/Current | ✅ #3 (0.12) |
| stress_score | Numeric | 0–10 scale | ✅ #4 (0.09) |
| bp_history | Categorical | Normal/Elevated/Stage 1/Stage 2 | ✅ #5 (0.07) |
| age | Numeric | Usia | Included |
| salt_intake | Categorical | Low/Normal/High | Included |
| sleep_duration | Numeric | Jam tidur | Included |
| medication | Categorical | None/Drug name | Included |
| exercise_level | Categorical | Low/Moderate/High | Included |
| **has_hypertension** | **Binary** | **Target variable** | **0=No, 1=Yes** |

**Performance detail (XGBoost):**
```
              precision  recall  f1-score  support
           0       0.99    0.99      0.99      188
           1       0.99    0.99      0.99      194
    accuracy                         0.99      382
```

---

### Proyek 9: SiliconBlast / bpc-prediction-lmics ⭐ TERBAIK SKALA BESAR
| Feature Category | Variables | Count |
|---|---|---|
| Demographic | Country, age, sex, education, occupation, marital status, urban/rural, wealth quintile, health insurance, alcohol use, tobacco use | 11 |
| Behavioral | Physical activity, diet (salt/fruit/veg), alcohol type/frequency, tobacco type/frequency, sedentary time, sleep, stress | 24 |
| Physical Measurements | Height, weight, waist, hip, blood pressure (systolic/diastolic) | 5 |
| Biochemical | Glucose, cholesterol (total/HDL/LDL), triglycerides, creatinine, HbA1c | 8 |
| **Target** | **High BP (normal/high)** | **1** |

**Model Files:**
- `src/models/global/XGB_global.pkl` — XGBoost (best)
- `src/models/global/RF_global.pkl` — Random Forest
- `src/models/global/LR_global.pkl` — Logistic Regression
- `src/models/global/KNN_global.pkl` — K-Nearest Neighbors

---

### Proyek 10: ypd774/hypertension ⭐ SIAP API — COCOK UNTUK RPM BACKEND
| Variable | Tipe | Mapping |
|---|---|---|
| gender | Categorical | Male=1, Female=0 |
| age | Numeric | Tahun |
| cigsPerDay | Numeric | Jumlah rokok/hari |
| BPMeds | Categorical | Yes=1, No=0 |
| totChol | Numeric | Kolesterol total (mg/dL) |
| sysBP | Numeric | Tekanan darah sistolik (mmHg) |
| diaBP | Numeric | Tekanan darah diastolik (mmHg) |
| BMI | Numeric | Body Mass Index |
| heartRate | Numeric | Denyut jantung (bpm) |
| **Hypertension** | **Binary** | **0=No, 1=Yes** |

**API Endpoint (Flask):**
```python
POST /predict
Content-Type: application/json

{
  "gender": "Male",
  "age": 45,
  "cigsPerDay": 0,
  "BPMeds": "No",
  "totChol": 200,
  "sysBP": 140,
  "diaBP": 90,
  "BMI": 28.5,
  "heartRate": 75
}

# Response:
{ "prediction": 1 }
```

---

## 🏆 Rekomendasi untuk Sistem RPM

| Skenario | Proyek Terbaik | Alasan |
|---|---|---|
| **Demo/Prototype cepat** | Proyek 7 (NaijaDataProfessor) | Ada Streamlit app + model.pkl langsung pakai |
| **Backend API untuk RPM** | Proyek 10 (ypd774) | Sudah ada Flask REST API + StandardScaler |
| **Research/Skripsi (akurasi tertinggi)** | Proyek 9 (SiliconBlast) | Dataset terbesar (184K), WHO STEPS, 4 model .pkl |
| **Fitur wearable (PPG/HR)** | Proyek 4 (Blood-Pressure-Prediction) | Didesain untuk Samsung Watch + PPG signal |
| **Fitur paling sederhana (RPM form)** | Proyek 7 atau 10 | Hanya butuh BMI, riwayat BP, kebiasaan pasien |

### ✅ Fitur Minimal yang Dibutuhkan RPM (Intersection Semua Proyek)
Berdasarkan analisis semua proyek, fitur berikut **selalu muncul** dan **bisa dikumpulkan via form RPM**:

```
1. Usia (Age)
2. Jenis Kelamin (Sex/Gender)
3. BMI (bisa dihitung dari berat + tinggi)
4. Riwayat Tekanan Darah (Blood Pressure History)
5. Riwayat Keluarga Hipertensi (Family History)
6. Status Merokok (Smoking Status)
7. Status Diabetes (DM)
8. Tekanan Darah Sistolik & Diastolik (jika ada alat ukur di RPM)
```

---

## 💾 Model Files Summary

| File | Lokasi Lokal | Ukuran | Format | Cara Load |
|---|---|---|---|---|
| `hypertension_model_v2.pkl` | `.\Hypertension-Risk-Prediction-Using-Machine-Learning\...\` | 235 KB | sklearn `.pkl` | `joblib.load()` |
| `model.pkl` | `.\PredictingHypertension\...\` | 266 KB | sklearn `.pkl` | `pickle.load()` |
| `hypertension_norm.mat` | `.\PredictingHypertension\...\` | 325 B | MATLAB | `scipy.io.loadmat()` |
| `XGB_global.pkl` | `.\bpc-prediction-lmics\...\src\models\global\` | 712 B | XGBoost `.pkl` | `joblib.load()` |
| `RF_global.pkl` | `.\bpc-prediction-lmics\...\src\models\global\` | 632 B | sklearn `.pkl` | `joblib.load()` |
| `LR_global.pkl` | `.\bpc-prediction-lmics\...\src\models\global\` | 325 B | sklearn `.pkl` | `joblib.load()` |
| `KNN_global.pkl` | `.\bpc-prediction-lmics\...\src\models\global\` | 241 B | sklearn `.pkl` | `joblib.load()` |
| `hyptertension_model1.pkl` | `.\ypd774Hypertension\hypertension\` | 16.1 MB | sklearn `.pkl` | `pickle.load()` |
| `scaler_hyper.pkl` | `.\ypd774Hypertension\hypertension\` | 832 B | sklearn `.pkl` | `pickle.load()` |
| `model_joblib_hypertension` | `.\Hypertension-risk-model\...\` | 5.3 MB | joblib | `joblib.load()` |
