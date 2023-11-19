from sklearn.preprocessing import StandardScaler

def normalisasi(features):
    # Gunakan StandardScaler untuk normalisasi
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform([features])
    
    # Hasil dari fit_transform akan berupa array numpy, ubah kembali menjadi list
    normalized_data = normalized_data.tolist()[0]
    
    return normalized_data
