import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="CVD",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Custom CSS -- professional, clean styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Global ---- */
    .block-container {
        max-width: 960px;
        padding-top: 2rem;
    }

    /* ---- Header banner ---- */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .header-banner h1 {
        margin: 0 0 0.4rem 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        margin: 0;
        font-size: 0.92rem;
        color: #b0c4de;
        line-height: 1.5;
    }

    /* ---- Section cards ---- */
    .section-card {
        background: #f8f9fb;
        border: 1px solid #e2e6ed;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .section-card h3 {
        margin: 0 0 0.8rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 0.4rem;
        display: inline-block;
    }

    /* ---- Result cards ---- */
    .result-card {
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-top: 0.5rem;
    }
    .result-high {
        background: #fff5f5;
        border: 1px solid #e53e3e;
    }
    .result-high h3 { color: #c53030; }
    .result-low {
        background: #f0fff4;
        border: 1px solid #38a169;
    }
    .result-low h3 { color: #276749; }

    /* ---- Metric tiles ---- */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.75rem;
    }
    .metric-tile {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e6ed;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-tile .label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .metric-tile .value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    /* ---- Footer disclaimer ---- */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 0.82rem;
        color: #92400e;
        margin-top: 1.5rem;
        line-height: 1.5;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    model_files = {
        "XGBoost": "xgboost_cvd_model.pkl",
        "Random Forest": "random_forest_cvd_model.pkl",
        "KNN": "knn_cvd_model.pkl",
        "SVM": "svm_cvd_model.pkl"
    }
    model_path = model_files[model_name]
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <h1>Cardiovascular Disease Risk Predictor</h1>
    <p>
        Sistem prediksi risiko penyakit kardiovaskular berbasis machine learning.
        Masukkan data klinis pasien untuk mendapatkan estimasi risiko secara real-time.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------
model_info = {
    "XGBoost": {"accuracy": "73.59%", "desc": "Gradient boosting dengan decision tree sekuensial"},
    "Random Forest": {"accuracy": "73.44%", "desc": "Ensemble dari banyak decision tree independen"},
    "KNN": {"accuracy": "72.24%", "desc": "Klasifikasi berdasarkan k-tetangga terdekat"},
    "SVM": {"accuracy": "~72%", "desc": "Klasifikasi berbasis hyperplane optimal"}
}

col_model, col_info = st.columns([1, 2])

with col_model:
    selected_model = st.selectbox(
        "Model Prediksi",
        list(model_info.keys()),
        help="Pilih algoritma machine learning yang akan digunakan untuk prediksi."
    )

with col_info:
    info = model_info[selected_model]
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="label">Model</div>
            <div class="value">{selected_model}</div>
        </div>
        <div class="metric-tile">
            <div class="label">Cross-Validation Accuracy</div>
            <div class="value">{info['accuracy']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

model = load_model(selected_model)

if model is None:
    st.error(
        f"File model untuk {selected_model} tidak ditemukan. "
        "Jalankan `python train_model.py` terlebih dahulu untuk melatih model."
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="section-card"><h3>Data Personal</h3></div>
    """, unsafe_allow_html=True)
    age = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=50)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

with col2:
    st.markdown("""
    <div class="section-card"><h3>Data Klinis</h3></div>
    """, unsafe_allow_html=True)
    ap_hi = st.number_input("Tekanan Sistolik (mmHg)", min_value=60, max_value=250, value=120)
    ap_lo = st.number_input("Tekanan Diastolik (mmHg)", min_value=40, max_value=150, value=80)

    cholesterol_mapping = {"Normal": 1, "Di Atas Normal": 2, "Jauh Di Atas Normal": 3}
    cholesterol_input = st.selectbox("Kadar Kolesterol", list(cholesterol_mapping.keys()))

    glucose_mapping = {"Normal": 1, "Di Atas Normal": 2, "Jauh Di Atas Normal": 3}
    gluc_input = st.selectbox("Kadar Glukosa", list(glucose_mapping.keys()))

with col3:
    st.markdown("""
    <div class="section-card"><h3>Faktor Gaya Hidup</h3></div>
    """, unsafe_allow_html=True)
    smoke = st.checkbox("Merokok")
    alco = st.checkbox("Konsumsi alkohol secara rutin")
    active = st.checkbox("Aktif berolahraga", value=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.divider()
submit = st.button("Jalankan Prediksi", type="primary", use_container_width=True)

if submit:
    # -- Feature engineering (replicate training pipeline) --
    female = 1 if gender == "Perempuan" else 0
    male = 1 if gender == "Laki-laki" else 0
    bmi = round((weight / (height / 100) ** 2), 2)

    cholesterol = cholesterol_mapping[cholesterol_input]
    gluc = glucose_mapping[gluc_input]
    smoke_val = 1 if smoke else 0
    alco_val = 1 if alco else 0
    active_val = 1 if active else 0

    features = [
        'age', 'female', 'male', 'height', 'weight', 'bmi',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
    ]

    input_data = pd.DataFrame([[
        age, female, male, height, weight, bmi,
        ap_hi, ap_lo, cholesterol, gluc, smoke_val, alco_val, active_val
    ]], columns=features)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -- BMI category --
    if bmi < 18.5:
        bmi_cat = "Berat Badan Kurang"
    elif bmi <= 24.9:
        bmi_cat = "Normal"
    elif bmi <= 29.9:
        bmi_cat = "Berat Badan Berlebih"
    else:
        bmi_cat = "Obesitas"

    # -- Blood pressure category --
    if ap_hi <= 120 and ap_lo <= 80:
        bp_cat = "Normal"
    elif ap_hi <= 129 and ap_lo <= 80:
        bp_cat = "Elevated"
    elif ap_hi <= 139 or ap_lo <= 89:
        bp_cat = "Hipertensi Tahap 1"
    elif ap_hi <= 180 or ap_lo <= 120:
        bp_cat = "Hipertensi Tahap 2"
    else:
        bp_cat = "Krisis Hipertensi"

    # -- Display computed metrics --
    st.markdown("#### Ringkasan Data Pasien")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", f"{bmi}", bmi_cat)
    m2.metric("Tekanan Darah", f"{ap_hi}/{ap_lo}", bp_cat)
    m3.metric("Model", selected_model)
    m4.metric("Probabilitas CVD", f"{probability * 100:.1f}%")

    st.markdown("---")

    # -- Display prediction result --
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-high">
            <h3>RISIKO TINGGI -- Penyakit Kardiovaskular Terdeteksi</h3>
            <p style="font-size:0.95rem; color:#742a2a; margin:0.5rem 0 0 0;">
                Model <strong>{selected_model}</strong> mengestimasi probabilitas penyakit kardiovaskular sebesar
                <strong>{probability * 100:.1f}%</strong>. Disarankan untuk melakukan konsultasi lebih lanjut
                dengan tenaga medis profesional untuk evaluasi klinis menyeluruh.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-low">
            <h3>RISIKO RENDAH -- Tidak Terdeteksi Penyakit Kardiovaskular</h3>
            <p style="font-size:0.95rem; color:#22543d; margin:0.5rem 0 0 0;">
                Model <strong>{selected_model}</strong> mengestimasi probabilitas penyakit kardiovaskular sebesar
                <strong>{probability * 100:.1f}%</strong>. Tetap jaga pola hidup sehat dan lakukan pemeriksaan
                kesehatan secara berkala.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -- Input summary table --
    with st.expander("Detail Input yang Digunakan"):
        summary_df = pd.DataFrame({
            "Parameter": [
                "Usia", "Jenis Kelamin", "Tinggi Badan", "Berat Badan", "BMI",
                "Tekanan Sistolik", "Tekanan Diastolik", "Kolesterol",
                "Glukosa", "Merokok", "Alkohol", "Aktivitas Fisik"
            ],
            "Nilai": [
                f"{age} tahun", gender, f"{height} cm", f"{weight} kg", f"{bmi} ({bmi_cat})",
                f"{ap_hi} mmHg", f"{ap_lo} mmHg", cholesterol_input,
                gluc_input,
                "Ya" if smoke else "Tidak",
                "Ya" if alco else "Tidak",
                "Ya" if active else "Tidak"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer">
    <strong>Perhatian:</strong> Hasil prediksi ini merupakan estimasi statistik berdasarkan model machine learning
    dengan akurasi terbatas (~73%). Hasil ini <strong>bukan</strong> diagnosis medis dan tidak dapat menggantikan
    pemeriksaan serta konsultasi dengan tenaga medis profesional. Gunakan hasil ini sebagai referensi awal saja.
</div>
""", unsafe_allow_html=True)
