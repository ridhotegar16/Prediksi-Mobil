
# 🚗 Prediksi Harga Mobil

Proyek ini merupakan implementasi Python untuk memprediksi harga mobil berdasarkan data spesifikasi dan fitur yang dimiliki. Model ini memanfaatkan preprocessing data, scraping, dan machine learning.

## 📦 Fitur

- ⚡ Menggunakan XGBoost Regressor untuk performa tinggi

- 🔍 Scraping data mobil dari situs (otomatis)
- 🧹 Preprocessing dan pembersihan data
- 📈 Model machine learning untuk prediksi harga
- 📊 Visualisasi dan evaluasi performa model
- ✅ Modular (setiap bagian dibagi dalam file terpisah)

## 🧠 Teknologi dan Library

- XGBoost Regressor untuk model prediksi harga yang lebih akurat

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- requests, BeautifulSoup (untuk scraping)

## 📁 Struktur Folder

```
PrediksiMobil/
├── app.py                # Main app / integrasi pipeline
├── data_prepro.py        # Preprocessing & cleaning
├── mobil_scraper.py      # Web scraping data mobil
├── modelling.py          # Model training & evaluation
├── requirements.txt      # Dependency list
└── venv/                 # Virtual environment (ignored in Git)
```

## 🚀 Cara Menjalankan

1. **Clone repo ini**:
   ```bash
   git clone https://github.com/ridhotegar16/Prediksi-Mobil.git
   cd Prediksi-Mobil
   ```

2. **Aktifkan virtual environment (opsional tapi disarankan)**:
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan proyek**:
   ```bash
   python app.py
   ```

## 📌 Catatan

- Folder `venv/` tidak di-upload agar repo tetap ringan dan sesuai best practice.
- Kamu bisa sesuaikan `mobil_scraper.py` jika ingin scraping dari website lain.

## 🙋‍♂️ Kontribusi

Pull request sangat diterima. Jangan lupa untuk membuat branch baru untuk setiap fitur atau perbaikan bug.

---

📬 **Author**: [Ridho Tegar](https://github.com/ridhotegar16)
