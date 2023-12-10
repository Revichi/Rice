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
st.title("""
Rice (Cammeo and Osmancik)
""")
st.write("""
Harap Isi Data Sesuai Kolom, Data Tidak Boleh Kosong
""")
Area = st.number_input("Area", min_value=0.0, max_value=100.0, step=0.00001, format="%.5f")  # Set step ke nilai pecahan
Perimeter = st.number_input("Perimeter", min_value=0.0, max_value=100.0, step=0.00001, format="%.5f")
MajorAxisLength = st.number_input("Major_Axis_Length", min_value=0.0, max_value=100.0, step=0.00001, format="%.5f")
MinorAxisLength = st.number_input("Minor_Axis_Length", min_value=0.0, max_value=100.0, step=0.00001, format="%.5f")
Eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=100.0, step=0.00001, format="%.5f")
ConvexArea = st.number_input("Convex_Area", min_value=0.0, max_value=100.0, step=0.00001, format="%.5f")

columns = st.columns((2, 0.6, 2))
submit = columns[1].button("Submit")


  
if submit:
    # Normalisasi data menggunakan fungsi normalisasi yang telah ditambahkan
    normalized_data = aksi.normalisasi([Area, Perimeter, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea])

    # Melakukan prediksi dengan model SVM
    prediction = svm_model.predict([normalized_data])

    # Menampilkan hasil prediksi
    st.header('Hasil Prediksi')
    st.write(f'Model SVM memprediksi kelas: {prediction[0]}')
