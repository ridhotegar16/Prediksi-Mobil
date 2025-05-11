import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import joblib

print("--- [INFO] Memulai Skrip Training Model (Versi Asli Disederhanakan) ---")

# === 1. Load dan Preprocessing Data ===
df = pd.read_csv("data\data_mobil_fitur_depresiasi_inflasi.csv") 
print(f"[INFO] Data dimuat. Baris awal: {len(df)}, Kolom awal: {len(df.columns)}")
print("Kolom awal di df:", df.columns.tolist())


# a. Outlier Removal untuk 'Harga'
if 'Harga' in df.columns and pd.api.types.is_numeric_dtype(df['Harga']):
    Q1 = df['Harga'].quantile(0.25)
    Q3 = df['Harga'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask_harga = (df['Harga'] >= lower_bound) & (df['Harga'] <= upper_bound)
    df = df[mask_harga].copy()
    print(f"\n[INFO] Baris setelah filter outlier Harga: {len(df)}")
else:
    print("[WARNING] Kolom 'Harga' tidak valid untuk outlier removal atau tidak ditemukan.")

if len(df) == 0: print("[ERROR] Tidak ada data setelah filter outlier Harga."); exit()

# b. Log Transform Target Variabel 'Harga'
df['Harga'] = np.log1p(df['Harga'])
print("[INFO] Kolom 'Harga' di-log transform.")

# c. One-Hot Encoding Fitur Kategorikal
categorical_cols_to_encode = ['Merek', 'Model_Detail', 'Lokasi']
categorical_cols_to_encode = [col for col in categorical_cols_to_encode if col in df.columns]

if categorical_cols_to_encode:
    df = pd.get_dummies(df, columns=categorical_cols_to_encode, prefix=categorical_cols_to_encode, drop_first=True)
    print(f"[INFO] One-Hot Encoding diterapkan pada: {categorical_cols_to_encode}")
else:
    print("[INFO] Tidak ada kolom kategorikal yang di One-Hot Encode dari daftar.")

print("\n[INFO] Kolom DataFrame SETELAH get_dummies:")
print(df.columns.tolist())

# d. Definisikan Fitur (X) dan Target (y)
cols_to_drop_for_X = ['Harga', 'DepresiasiRiilNormal_PersenPerThn', 'Judul',
                      'EstimasiHargaAwal_NilaiSaatIni', 'DepresiasiAbsolut_PerThn_NilaiSaatIni','HargaSekarang_DeflasiKeThnBuat','EstimasiHargaAwal_PadaThnBuat','DepresiasiRiilAbsolut_PerThn_PadaThnBuat','DepresiasiAbsolut_PerThn_NilaiSaatIni']
cols_to_drop_existing = [col for col in cols_to_drop_for_X if col in df.columns]

X = df.drop(columns=cols_to_drop_existing)
y = df['Harga']

X_columns_for_model = X.columns.tolist()
print(f"\n[INFO] Fitur (X) yang akan digunakan model ({len(X_columns_for_model)} kolom):")
print(X_columns_for_model[:5], "...", X_columns_for_model[-5:] if len(X_columns_for_model) > 10 else X_columns_for_model)


# e. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"[INFO] Data di-split. Ukuran X_train: {X_train.shape}, Ukuran X_test: {X_test.shape}")

# f. Scaling Fitur
scaler = RobustScaler()
X_train_proc = scaler.fit_transform(X_train)
X_test_proc = scaler.transform(X_test)      
print("[INFO] Scaling diterapkan pada X_train dan X_test.")

# === 2. Melatih Model XGBoost ===
print("\n--- [INFO] Memulai Pelatihan Model XGBoost ---")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train_proc, y_train)
print("--- [INFO] Pelatihan Model XGBoost Selesai ---")

# === 3. Simpan Model dan Komponennya ===
MODEL_SAVE_PATH = "xgboost_mobil_model_v3.pkl"
joblib.dump((model, scaler, X_columns_for_model), MODEL_SAVE_PATH)
print(f"\n[INFO] Model, scaler, dan X.columns (sebelum scaling) berhasil disimpan ke '{MODEL_SAVE_PATH}'")

# === 4. Prediksi dan Evaluasi pada Data Test ===
print("\n--- [INFO] Melakukan Prediksi pada Data Test ---")
y_pred_log = model.predict(X_test_proc)

y_test_inv = np.expm1(y_test)
y_pred_inv = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
print(f"\n📉 Evaluasi Model pada Data Test:")
print(f"   MAE (Rp): {mae:,.0f}")
print(f"   MAPE: {mape:.2f}%")

# === 5. Visualisasi Prediksi vs Aktual ===
print("\n--- [INFO] Membuat Visualisasi Prediksi vs Aktual ---")
plt.figure(figsize=(8,5))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.6)
plt.plot([min(y_test_inv.min(), y_pred_inv.min()), max(y_test_inv.max(), y_pred_inv.max())], # Perbaikan batas plot
         [min(y_test_inv.min(), y_pred_inv.min()), max(y_test_inv.max(), y_pred_inv.max())], 'r--')
plt.title('Prediksi vs Aktual (XGBoost)')
plt.xlabel('Harga Aktual (Rp)')
plt.ylabel('Harga Prediksi (Rp)')
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- [SUCCESS] Skrip Training Model (Versi Asli Disederhanakan) Selesai ---")