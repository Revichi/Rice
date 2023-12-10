import streamlit as st
import numpy as np
import aksi
import time
import webbrowser
import pandas as pd
import joblib
# Memuat model SVM dari file .pkl
svm_model_path = 'Model/svm_model.pkl'
svm_model = joblib.load(svm_model_path)

st.title("Rice (Cammeo and Osmancik)")
st.write("Harap isi data sesuai kolom, data tidak boleh kosong")

# Menggunakan st.number_input untuk memasukkan angka dengan format float
Area = st.number_input("Area", format="%.5f")  # Set step ke nilai pecahan
Perimeter = st.number_input("Perimeter", format="%.5f")
MajorAxisLength = st.number_input("Major Axis Length", format="%.5f")
MinorAxisLength = st.number_input("Minor Axis Length", format="%.5f")
Eccentricity = st.number_input("Eccentricity", format="%.5f")
ConvexArea = st.number_input("Convex Area", format="%.5f")

# Tombol Submit
submit = st.button("Submit")

if submit:
    # Melakukan prediksi dengan model SVM
    # Perlu diubah menjadi format array 2D karena model.predict membutuhkan input berupa array 2D
    input_data = np.array([[Area, Perimeter, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea]])
    prediction = svm_model.predict(input_data)

    # Menampilkan hasil prediksi
    st.header('Hasil Prediksi')
    st.write(f'Model SVM memprediksi kelas: {prediction[0]}')
