📈 Prediksi Harga Saham TLKM (Telkom Indonesia)

Aplikasi berbasis web yang dibangun dengan Streamlit untuk menganalisis dan memprediksi harga saham PT Telkom Indonesia (TLKM) menggunakan algoritma Machine Learning Support Vector Regression (SVR).

🌟 Fitur Utama

Prediksi Harian: Memprediksi harga penutupan (Close Price) untuk hari berikutnya berdasarkan input User (Open, High, Low, Volume).

Analisis Historis: Visualisasi interaktif pergerakan harga saham dan volume perdagangan menggunakan dataset historis lokal.

Auto-Fill Data: Secara otomatis mengisi form prediksi dengan data pasar terakhir yang tersedia di dataset.

📂 Struktur File

Pastikan file-file berikut ada dalam satu direktori:

app.py: File utama aplikasi Streamlit.

best_telkom_model.pkl: Model Machine Learning (SVR) yang sudah dilatih.

scaler.pkl: Scaler untuk normalisasi data input.

TLKM.JK.csv: Dataset historis harga saham.

requirements.txt: Daftar library Python yang dibutuhkan.

🚀 Cara Menjalankan di Lokal

Clone Repository ini

git clone [https://github.com/username-anda/nama-repo.git](https://github.com/username-anda/nama-repo.git)
cd nama-repo


Install Library yang Dibutuhkan
Pastikan Anda sudah menginstall Python, lalu jalankan:

pip install -r requirements.txt


Jalankan Aplikasi

streamlit run app.py


🧠 Tentang Model

Model yang digunakan adalah Support Vector Regression (SVR). Model ini dilatih menggunakan data historis saham dengan fitur:

Open Price

High Price

Low Price

Volume

⚠️ Disclaimer

Aplikasi ini dibuat untuk tujuan edukasi dan demonstrasi teknologi Machine Learning. Hasil prediksi tidak menjamin akurasi 100% dan bukan merupakan saran investasi finansial.

Dibuat dengan ❤️ menggunakan Python & Streamlit
