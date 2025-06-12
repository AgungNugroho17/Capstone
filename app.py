import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("rf_obesity_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Tingkat Obesitas")

# ========================
# 🧠 Manual Mapping Kategori
# ========================
gender_map = {"Male": 1, "Female": 0}
family_map = {"yes": 1, "no": 0}
favc_map = {"yes": 1, "no": 0}
caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
smoke_map = {"yes": 1, "no": 0}
scc_map = {"yes": 1, "no": 0}
calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {
    "Public_Transportation": 0,
    "Walking": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4
}

# ========================
# 📋 Form Input Pengguna
# ========================
gender = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (meter)", value=1.70, step=0.01)
weight = st.number_input("Berat Badan (kg)", value=70.0)
family_history = st.selectbox("Riwayat Keluarga Gemuk", list(family_map.keys()))
favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori", list(favc_map.keys()))
caec = st.selectbox("Ngemil?", list(caec_map.keys()))
ch2o = st.slider("Minum Air per Hari (liter)", 1.0, 3.0, 2.0)
faf = st.slider("Frekuensi Aktivitas Fisik", 0.0, 3.0, 1.0)
calc = st.selectbox("Konsumsi Alkohol", list(calc_map.keys()))


# ========================
# 🔄 Encoding dan Prediksi
# ========================
if st.button("Prediksi"):
    # Manual encoding
    cat_encoded = [
        gender_map[gender],
        family_map[family_history],
        favc_map[favc],
        caec_map[caec],
        smoke_map[smoke],
        scc_map[scc],
        calc_map[calc],
        mtrans_map[mtrans]
    ]

    # Numerik
    num_features = [age, height, weight, fcvc, ncp, ch2o, faf, tue]
    num_scaled = scaler.transform([num_features])[0]

    # Gabung fitur
    final_input = np.concatenate([cat_encoded, num_scaled])

    # Prediksi
    prediction = model.predict([final_input])[0]

    # Tampilkan hasil
    st.success(f"Tingkat Obesitas Anda diprediksi: **{prediction}**")
