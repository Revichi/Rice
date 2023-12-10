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

with st.sidebar:
    kolom = st.columns((1, 1, 2.7))
    home = kolom[1].button('Home',type='primary')
    Tools = kolom[2].button('Tools')

if Tools==False and home==True:
   # Memuat file CSV
    file_path = 'Model/Rice.csv'
    df = pd.read_csv(file_path)

    # Set judul halaman
    st.title('Rice (Cammeo and Osmancik)')

    # Menampilkan tabel
    st.write('## Dataset Overview')
    st.dataframe(df)

    # Menampilkan penjelasan dataset
    st.write('## Penjelasan Dataset')
    st.markdown('1. **Area (Luas):** Mengembalikan jumlah piksel dalam batas biji beras. Ini mengukur luas daerah biji beras dalam piksel.')
    st.markdown('2. **Perimeter (Keliling):** Menghitung keliling dengan mengukur jarak antara piksel di sekitar batas biji beras. Ini memberikan panjang keliling biji beras dalam piksel.')
    st.markdown('3. **Major Axis Length (Panjang Sumbu Utama):** Garis terpanjang yang dapat digambar pada biji beras, yaitu jarak sumbu utama. Ini mengukur panjang garis terpanjang pada biji beras dalam piksel.')
    st.markdown('4. **Minor Axis Length (Panjang Sumbu Kecil):** Garis terpendek yang dapat digambar pada biji beras, yaitu jarak sumbu kecil. Ini mengukur panjang garis terpendek pada biji beras dalam piksel.')
    st.markdown('5. **Eccentricity (Eksentrisitas):** Mengukur seberapa bulat elips yang memiliki momen yang sama dengan biji beras. Ini memberikan informasi tentang seberapa bulat biji beras, di mana nilai mendekati 0 menandakan elips yang lebih bulat.')
    st.markdown('6. **Convex Area (Luas Cembung):** Mengembalikan jumlah piksel dari cangkang cembung terkecil dari wilayah yang dibentuk oleh biji beras. Ini mengukur luas area cembung dalam piksel.')
    st.markdown('7. **Extent (Ketertelusuran):** Mengembalikan rasio wilayah yang dibentuk oleh biji beras terhadap piksel kotak pembatas. Ini memberikan informasi tentang seberapa banyak area yang diisi oleh biji beras dalam kotak pembatasnya.')
    st.markdown('8. **Class (Kelas):** Jenis beras, misalnya, Cammeo dan Osmancik. Ini adalah label kategoris yang menunjukkan jenis atau kelas dari biji beras yang diamati.')


if home == False and Tools == False or home == False and Tools == True:
    st.title('Tools')
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


# with st.sidebar:
#     st.write('Link:')
#     link = '[GitHub](https://github.com/Revichi/appdataset)'
#     st.markdown(link, unsafe_allow_html=True)
#     link = '[Jupyter Book](https://revichi.github.io/datamining/App.html?highlight=penambangan)'
#     st.markdown(link, unsafe_allow_html=True)
