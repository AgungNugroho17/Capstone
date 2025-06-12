import streamlit as st
import numpy as np
import joblib

# Load model dan preprocessing tools
model = joblib.load("rf_obesity_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Prediksi Tingkat Obesitas")

# Form input pengguna
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (meter)", value=1.70)
weight = st.number_input("Berat Badan (kg)", value=70.0)
family_history = st.selectbox("Riwayat Keluarga Gemuk", ["yes", "no"])
favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
fcvc = st.slider("Porsi Sayur Setiap Makan (1-3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah Makan Besar per Hari", 1.0, 4.0, 3.0)
caec = st.selectbox("Ngemil?", ["Sometimes", "Frequently", "Always", "no"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
ch2o = st.slider("Minum Air per Hari (liter)", 1.0, 3.0, 2.0)
scc = st.selectbox("Kontrol Kalori?", ["yes", "no"])
faf = st.slider("Frekuensi Aktivitas Fisik", 0.0, 3.0, 1.0)
tue = st.slider("Waktu di Depan Layar", 0.0, 3.0, 1.0)
calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Kategorikal fitur yang perlu encoding
cat_features = [gender, family_history, favc, caec, smoke, scc, calc, mtrans]
cat_encoded = [label_encoder.transform([val])[0] for val in cat_features]

# Fitur numerik
num_features = [age, height, weight, fcvc, ncp, ch2o, faf, tue]
num_scaled = scaler.transform([num_features])[0]

# Gabungkan semua fitur
final_features = np.concatenate([cat_encoded, num_scaled])

if st.button("Prediksi"):
    prediction = model.predict([final_features])[0]
    st.success(f"Tingkat Obesitas Anda: **{prediction}**")
