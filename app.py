import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    dataset = pd.read_excel('dataset sembako.xlsx')
    return dataset

# Preprocessing Data
def preprocess_data(df):
    # Fitur yang digunakan untuk klasifikasi
    features = df[['jumlah_terjual_per_minggu', 'frekuensi_penjualan_per_minggu',
                   'umur_stok_hari', 'margin_persen', 'waktu_pengiriman_hari']]
    # Label
    labels = df['label']
    return features, labels

# Membuat model KNN
def build_knn_model(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

# Prediksi dan evaluasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Streamlit Interface
def main():
    st.title("Klasifikasi Barang Fast/Slow Moving menggunakan KNN")
    st.write("""
    Aplikasi ini menggunakan algoritma K-Nearest Neighbors (KNN) untuk mengklasifikasikan barang
    sebagai Fast, Medium, atau Slow Moving berdasarkan data penjualan dan karakteristik produk.
    """)

    # Load data
    df = load_data()

    # Preprocess data
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN Model
    k = st.slider("Pilih jumlah k untuk KNN", 1, 20, 3)
    model = build_knn_model(X_train_scaled, y_train, k)

    # Evaluate the model
    accuracy, predictions = evaluate_model(model, X_test_scaled, y_test)

    # Show evaluation results
    st.write(f"Akurasi Model: {accuracy * 100:.2f}%")
    st.write("Prediksi untuk data uji:")
    result_df = pd.DataFrame({
        'id_barang': df['id_barang'][X_test.index],
        'Nama Barang': df['nama_barang'][X_test.index],
        'Prediksi': predictions
    })
    st.write(result_df)

    # Input untuk prediksi baru
    st.header("Prediksi Klasifikasi Baru")
    jumlah_terjual = st.number_input("Jumlah Terjual Per Minggu", min_value=0)
    frekuensi_penjualan = st.number_input("Frekuensi Penjualan Per Minggu", min_value=0)
    umur_stok = st.number_input("Umur Stok (hari)", min_value=0)
    margin = st.number_input("Margin (%)", min_value=0.0)
    waktu_pengiriman = st.number_input("Waktu Pengiriman (hari)", min_value=0)

    if st.button("Prediksi"):
        input_data = [[jumlah_terjual, frekuensi_penjualan, umur_stok, margin, waktu_pengiriman]]
        input_data_scaled = scaler.transform(input_data)
        prediksi = model.predict(input_data_scaled)
        st.write(f"Prediksi Klasifikasi Barang: {prediksi[0]}")

if __name__ == "__main__":
    main()
