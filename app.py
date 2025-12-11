import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Saham Telkom",
    page_icon="🎯",
    layout="wide"
)

# --- 2. Konfigurasi File ---
# Pastikan nama file ini sesuai dengan file yang ada di folder Anda
DATASET_FILE = "LSTM Saham Telkom Indonesia.csv"
MODEL_FILE = "best_telkom_model.pkl"
SCALER_FILE = "scaler.pkl"

# --- 3. Fungsi Load Resource ---
@st.cache_resource
def load_model_resources():
    """Memuat model dan scaler dari file pickle."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None, None
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None, None

@st.cache_data
def load_dataset():
    """Membaca file CSV dataset lokal."""
    if not os.path.exists(DATASET_FILE):
        return None
    try:
        df = pd.read_csv(DATASET_FILE)
        # Konversi kolom Date ke datetime jika ada
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date') # Urutkan dari lama ke baru
        return df
    except Exception as e:
        st.error(f"Error membaca CSV: {e}")
        return None

# Memuat data dan model saat aplikasi dijalankan
model_svr, scaler = load_model_resources()
df_historis = load_dataset()

# --- 4. Sidebar (Input Data) ---
st.sidebar.header("Parameter Prediksi")
st.sidebar.markdown("Masukkan data harian untuk memprediksi harga *Close* besok.")

# Default Value Logic (Ambil dari data terakhir jika ada)
if df_historis is not None and not df_historis.empty:
    last_row = df_historis.iloc[-1]
    # Gunakan try-except agar aman jika nama kolom di CSV berbeda
    try:
        def_open = float(last_row['Open'])
        def_high = float(last_row['High'])
        def_low = float(last_row['Low'])
        def_vol = float(last_row['Volume'])
        # st.sidebar.success(f"Data terisi otomatis dari tanggal: {last_row['Date'].date()}")
    except KeyError:
        def_open, def_high, def_low, def_vol = 4000.0, 4050.0, 3980.0, 75000000.0
else:
    def_open, def_high, def_low, def_vol = 4000.0, 4050.0, 3980.0, 75000000.0

# Form Input
with st.sidebar.form(key='pred_form'):
    inp_open = st.number_input("Open Price", value=def_open, min_value=0.0, step=10.0)
    inp_high = st.number_input("High Price", value=def_high, min_value=0.0, step=10.0)
    inp_low = st.number_input("Low Price", value=def_low, min_value=0.0, step=10.0)
    inp_vol = st.number_input("Volume", value=def_vol, min_value=0.0, step=10000.0)
    
    submit_btn = st.form_submit_button(label="Prediksi Harga")

# --- 5. Tampilan Utama ---
st.title("🎯Prediksi Harga Saham TLKM (Telkom)")
st.write("""
Aplikasi ini menggunakan model *Machine Learning* SVM
untuk memprediksi harga penutupan (Close) saham Telkom (TLKM)
untuk hari perdagangan berikutnya.
""")

# Pesan peringatan jika file tidak lengkap
if model_svr is None:
    st.error(f"⚠️ Model tidak ditemukan! Pastikan '{MODEL_FILE}' dan '{SCALER_FILE}' ada di folder.")
if df_historis is None:
    st.warning(f"⚠️ Dataset '{DATASET_FILE}' tidak ditemukan. Tab Analisis tidak akan menampilkan data.")

# Tabs untuk memisahkan Hasil Prediksi dan Analisis Data
tab_pred, tab_data = st.tabs(["Hasil Prediksi", "Analisis Data Historis"])

# === TAB 1: HASIL PREDIKSI ===
with tab_pred:
    if submit_btn:
        if model_svr is not None and scaler is not None:
            # Membuat DataFrame untuk input (Urutan kolom harus sama saat training)
            input_df = pd.DataFrame([[inp_open, inp_high, inp_low, inp_vol]], 
                                    columns=['Open', 'High', 'Low', 'Volume'])
            
            try:
                # Scaling data
                input_scaled = scaler.transform(input_df)
                # Prediksi
                prediction = model_svr.predict(input_scaled)[0]
                
                # Menampilkan Output
                st.subheader("Hasil Estimasi")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(label="Prediksi Close Besok", value=f"Rp {prediction:,.2f}")
                
                with col2:
                    st.info("Nilai ini adalah estimasi harga penutupan (Close) berdasarkan pola data Open, High, Low, dan Volume yang Anda masukkan.")
                    #  st.info(f"Model memprediksi harga penutupan besok akan berada di sekitar **Rp {prediction[0]:,.2f}**.")
                # Tampilkan data input pengguna
                st.write("---")
                st.caption("Detail Input Anda:")
                st.dataframe(input_df, hide_index=True)

                st.warning("""
                           **Disclaimer:** Ini adalah prediksi berbasis model dan bukan merupakan saran finansial.
                           Harga saham dipengaruhi oleh banyak faktor. Lakukan riset Anda sendiri sebelum berinvestasi.
                           """, icon="⚠️")
                
            except Exception as e:
                st.error(f"Terjadi error saat melakukan prediksi: {e}")
        else:
            st.error("Model belum siap. Cek file .pkl Anda.")
    else:
        st.info("👈 Masukkan data di sidebar dan klik tombol **Prediksi Harga**.")
       

# === TAB 2: ANALISIS DATA HISTORIS ===
with tab_data:
    if df_historis is not None and not df_historis.empty:
        st.subheader("Visualisasi Data Historis Dataset")
        
        # Validasi kolom yang dibutuhkan untuk chart
        req_cols = ['Date', 'Close', 'Volume']
        if all(col in df_historis.columns for col in req_cols):
            
            # 1. Line Chart Harga Close
            st.markdown("#### Grafik Harga Penutupan (Close)")
            chart_data = df_historis.set_index('Date')['Close']
            st.line_chart(chart_data, color="#0000FF") # Biru
            
            # 2. Bar Chart Volume
            st.markdown("#### Volume Perdagangan")
            vol_data = df_historis.set_index('Date')['Volume']
            st.bar_chart(vol_data, color="#0800FF") # Merah
            
            # 3. Statistik Ringkas
            st.markdown("#### Statistik Data")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Highest Price", f"Rp {df_historis['High'].max():,.0f}")
            c2.metric("Lowest Price", f"Rp {df_historis['Low'].min():,.0f}")
            c3.metric("Average Close", f"Rp {df_historis['Close'].mean():,.0f}")
            c4.metric("Total Data", f"{len(df_historis)} Hari")
            
            # 4. Tabel Data
            with st.expander("📂 Data Mentah"):
                st.dataframe(df_historis.sort_values('Date', ascending=False), use_container_width=True)
        else:
            st.error(f"Kolom pada CSV tidak lengkap. Pastikan ada: {req_cols}")
    else:
        st.info("Data historis tidak tersedia. Silakan upload file 'Dataset.csv'.")