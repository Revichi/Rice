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

st.write("""
Harap Isi Data Sesuai Kolom, Data Tidak Boleh Kosong
""")
Area = st.number_input("Area", step=0.01)  # Set step ke nilai pecahan
Perimeter = st.number_input("Perimeter", step=0.01)
MajorAxisLength = st.number_input("Major_Axis_Length", step=0.01)
MinorAxisLength = st.number_input("Minor_Axis_Length", step=0.01)
Eccentricity = st.number_input("Eccentricity", step=0.01)
ConvexArea = st.number_input("Convex_Area", step=0.01)

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
