# data_processing_final.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import re
import openpyxl

print("--- [INFO] Memulai Skrip Pengolahan Data Mobil ---")

# --- 1. KONFIGURASI ---
INPUT_CSV_FILE = 'data\hasil_scrape_mobil123.csv'
CURRENT_YEAR = datetime.now().year


# --- 2. MUAT DATA ---
print(f"--- [INFO] Memuat data dari: {INPUT_CSV_FILE} ---")
try:
    df = pd.read_csv(INPUT_CSV_FILE)
    print(f"[INFO] Data berhasil dimuat. Baris awal: {len(df)}, Kolom awal: {len(df.columns)}")
except FileNotFoundError:
    print(f"[ERROR] File '{INPUT_CSV_FILE}' tidak ditemukan.")
    exit()
except Exception as e:
    print(f"[ERROR] Gagal memuat CSV: {e}")
    exit()

print(df.head())

print(df.info())

# --- 4. PEMBERSIHAN DATA DAN TRANSFORMASI AWAL ---
print("\n--- [INFO] Memulai Pembersihan Data Inti ---")

rows_before_dropna_critical = len(df)
df.dropna(inplace=True)
if len(df) == 0:
    print("[ERROR] Tidak ada data tersisa setelah dropna krusial. Program berhenti.")
    exit()
#b.Penghapusan data anomali
df = df[(df['Harga'] >= 1000000) & (df['Harga'] <= 10000000000)]


print(df['Tahun'].unique())
print(df['Tahun'].dtype)
print(CURRENT_YEAR)


# c. Buat Fitur 'UsiaMobil'
if 'Tahun' in df.columns:
    df['UsiaMobil'] = CURRENT_YEAR - df['Tahun']
    # Validasi UsiaMobil agar tidak negatif (ini sebaiknya tetap ada)
    df = df[df['UsiaMobil'] >= 0] # Filter usia negatif

    print(f"[INFO] Fitur 'UsiaMobil' dibuat. Rentang: {df['UsiaMobil'].min() if not df.empty else 'N/A'} - {df['UsiaMobil'].max() if not df.empty else 'N/A'} tahun.")
else:
    print("[ERROR] Kolom 'Tahun' tidak valid untuk membuat 'UsiaMobil'. Program berhenti.")
    exit()

if len(df) == 0: # Cek ini setelah filter usia negatif
    print("[ERROR] Tidak ada data tersisa setelah filter UsiaMobil negatif. Program berhenti.")
    exit()

# e. Hapus Duplikasi
rows_before_dedup = len(df)
df.drop_duplicates(inplace=True)
print(f"[INFO] Menghapus {rows_before_dedup - len(df)} baris duplikat.")

print(df.head())

# --- 6. VISUALISASI DATA (Fokus Depresiasi) ---
print("\n--- [INFO] Memulai Visualisasi Data (Tutup plot untuk melanjutkan) ---")
plt.style.use('seaborn-v0_8-whitegrid')

# a. Distribusi Harga
plt.figure(figsize=(10, 6)); sns.histplot(df['Harga'], kde=True, bins=50, color='skyblue');
plt.title('Distribusi Harga Mobil Final', fontsize=15); plt.xlabel('Harga (Rp)', fontsize=12); plt.ylabel('Frekuensi', fontsize=12);
plt.ticklabel_format(style='plain', axis='x'); plt.tight_layout(); plt.show()

# b. Distribusi UsiaMobil
plt.figure(figsize=(10, 6)); sns.histplot(df['UsiaMobil'], kde=False, bins=min(30, df['UsiaMobil'].nunique()), color='salmon');
plt.title('Distribusi Usia Mobil Final', fontsize=15); plt.xlabel('Usia Mobil (Tahun)', fontsize=12); plt.ylabel('Frekuensi', fontsize=12);
plt.tight_layout(); plt.show()

# c. Pola Depresiasi (Harga vs UsiaMobil)
plt.figure(figsize=(12, 7)); sns.scatterplot(data=df, x='UsiaMobil', y='Harga', alpha=0.4, color='green', edgecolor=None, s=30);
plt.title('Harga vs. Usia Mobil (Pola Depresiasi)', fontsize=15); plt.xlabel('Usia Mobil (Tahun)', fontsize=12); plt.ylabel('Harga (Rp)', fontsize=12);
plt.ticklabel_format(style='plain', axis='y'); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Gaya plot
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# 1. Distribusi Harga Mobil Bekas
plt.figure(figsize=(8, 5))
sns.histplot(df['Harga'], bins=30, kde=True, color='green')
plt.title("Distribusi Harga Mobil Bekas")
plt.xlabel("Harga (Rp)")
plt.ylabel("Frekuensi")
plt.show()

# 2. Harga vs Usia Mobil (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='UsiaMobil', y='Harga', data=df, alpha=0.7, color='darkgreen')
sns.regplot(x='UsiaMobil', y='Harga', data=df, scatter=False, color='red', ci=None)
plt.title("Harga vs Usia Mobil")
plt.xlabel("Usia Mobil (Tahun)")
plt.ylabel("Harga (Rp)")
plt.show()

# 3. Boxplot Harga Berdasarkan Merek
plt.figure(figsize=(12, 6))
sns.boxplot(x='Merek', y='Harga', data=df, palette='Set3')
plt.title("Distribusi Harga Berdasarkan Merek")
plt.xticks(rotation=45)
plt.xlabel("Merek")
plt.ylabel("Harga (Rp)")
plt.show()

# Membaca data inflasi
df_inflasi = pd.read_excel('inflasi indonesia.xlsx')

# Gabungkan kedua data berdasarkan kolom 'Tahun'
df= pd.merge(df, df_inflasi[['Tahun', 'Inflasi']], on='Tahun', how='left')
df.head()

# --- 6. MEMBUAT FITUR DEPRESIASI EKSPLISIT (DENGAN INFLASI DARI KOLOM 'Inflasi') ---
print("\n--- [INFO] Membuat Fitur Depresiasi Eksplisit dengan Inflasi dari Kolom 'Inflasi' ---")

RATA_RATA_DEPRESIASI_RIIL_TAHUNAN_ASUMSI = 0.10

if 'Harga' in df.columns and 'UsiaMobil' in df.columns and \
   'Tahun' in df.columns and 'Inflasi' in df.columns:

    # Konversi nilai inflasi ke desimal
    df['Inflasi'] = df['Inflasi'] / 100

    df_dep_calc = df[df['UsiaMobil'] > 0].copy()
    df_dep_new = df[df['UsiaMobil'] == 0].copy()

    if not df_dep_calc.empty:
        # 1. Hitung Harga Sekarang yang Dideflasi ke Nilai Uang Tahun Pembuatan
        df_dep_calc['HargaSekarang_DeflasiKeThnBuat'] = df_dep_calc['Harga'] / (1 + df_dep_calc['Inflasi'])

        # 2. Estimasi Harga Awal Mobil (Harga Baru) pada Tahun Pembuatannya
        faktor_dep_riil_kumulatif = (1 - RATA_RATA_DEPRESIASI_RIIL_TAHUNAN_ASUMSI) ** df_dep_calc['UsiaMobil']
        faktor_dep_riil_kumulatif = np.where(faktor_dep_riil_kumulatif < 1e-9, 1e-9, faktor_dep_riil_kumulatif)

        df_dep_calc['EstimasiHargaAwal_PadaThnBuat'] = df_dep_calc['HargaSekarang_DeflasiKeThnBuat'] / faktor_dep_riil_kumulatif

        MAX_RASIO_ESTAWAL_VS_HARGADEFLASI = 7
        df_dep_calc['EstimasiHargaAwal_PadaThnBuat'] = np.minimum(
            df_dep_calc['EstimasiHargaAwal_PadaThnBuat'],
            df_dep_calc['HargaSekarang_DeflasiKeThnBuat'] * MAX_RASIO_ESTAWAL_VS_HARGADEFLASI
        )
        df_dep_calc['EstimasiHargaAwal_PadaThnBuat'] = np.maximum(
            df_dep_calc['EstimasiHargaAwal_PadaThnBuat'],
            df_dep_calc['HargaSekarang_DeflasiKeThnBuat']
        )

        # 3. Estimasi Harga Awal dalam Nilai Uang SAAT INI
        df_dep_calc['EstimasiHargaAwal_NilaiSaatIni'] = df_dep_calc['EstimasiHargaAwal_PadaThnBuat'] * (1 + df_dep_calc['Inflasi'])

        # 4. Hitung Depresiasi Riil per Tahun (Absolut, dalam nilai uang tahun pembuatan)
        df_dep_calc['DepresiasiRiilAbsolut_PerThn_PadaThnBuat'] = (
            df_dep_calc['EstimasiHargaAwal_PadaThnBuat'] - df_dep_calc['HargaSekarang_DeflasiKeThnBuat']
        ) / df_dep_calc['UsiaMobil']
        df_dep_calc['DepresiasiRiilAbsolut_PerThn_PadaThnBuat'] = np.maximum(0, df_dep_calc['DepresiasiRiilAbsolut_PerThn_PadaThnBuat'])

        # 5. Hitung Depresiasi Riil Normal (Persentase Depresiasi Tahunan Aktual)
        df_dep_calc['DepresiasiRiilNormal_PersenPerThn'] = np.where(
            df_dep_calc['EstimasiHargaAwal_PadaThnBuat'] > 1e-9,
            (df_dep_calc['DepresiasiRiilAbsolut_PerThn_PadaThnBuat'] / df_dep_calc['EstimasiHargaAwal_PadaThnBuat']) * 100,
            0
        )
        df_dep_calc['DepresiasiRiilNormal_PersenPerThn'] = np.clip(df_dep_calc['DepresiasiRiilNormal_PersenPerThn'], 0, 100)

        # 6. Depresiasi Absolut per Tahun dalam Nilai Uang SAAT INI
        df_dep_calc['DepresiasiAbsolut_PerThn_NilaiSaatIni'] = (
            df_dep_calc['EstimasiHargaAwal_NilaiSaatIni'] - df_dep_calc['Harga']
        ) / df_dep_calc['UsiaMobil']
        df_dep_calc['DepresiasiAbsolut_PerThn_NilaiSaatIni'] = np.maximum(0, df_dep_calc['DepresiasiAbsolut_PerThn_NilaiSaatIni'])

        print("[INFO] Fitur depresiasi riil (disesuaikan inflasi dari kolom 'Inflasi') dihitung untuk mobil usia > 0.")
    else:
        print("[INFO] Tidak ada mobil dengan UsiaMobil > 0 untuk dihitung fitur depresiasi eksplisit.")



    if not df_dep_new.empty:
        harga_mobil_baru = df_dep_new['Harga']
        inflasi_mobil_baru = df_dep_new['Inflasi'] # Mengambil dari kolom 'Inflasi'

        df_dep_new['HargaSekarang_DeflasiKeThnBuat'] = harga_mobil_baru / inflasi_mobil_baru
        df_dep_new['EstimasiHargaAwal_PadaThnBuat'] = harga_mobil_baru / inflasi_mobil_baru # Harga baru pada tahun itu
        df_dep_new['EstimasiHargaAwal_NilaiSaatIni'] = (harga_mobil_baru / inflasi_mobil_baru) * inflasi_mobil_baru # Seharusnya sama dengan harga_mobil_baru

        df_dep_new['DepresiasiRiilAbsolut_PerThn_PadaThnBuat'] = 0.0
        df_dep_new['DepresiasiRiilNormal_PersenPerThn'] = 0.0
        df_dep_new['DepresiasiAbsolut_PerThn_NilaiSaatIni'] = 0.0
        print("[INFO] Fitur depresiasi riil di-set 0 untuk mobil UsiaMobil == 0.")

    kolom_depresiasi_final = [
        'HargaSekarang_DeflasiKeThnBuat', 
        'EstimasiHargaAwal_PadaThnBuat',   
        'EstimasiHargaAwal_NilaiSaatIni',
        'DepresiasiRiilAbsolut_PerThn_PadaThnBuat', 
        'DepresiasiRiilNormal_PersenPerThn',
        'DepresiasiAbsolut_PerThn_NilaiSaatIni'
    ]

    df_temp_concat = pd.concat([df_dep_calc, df_dep_new], ignore_index=False)

    for col in kolom_depresiasi_final:
        if col in df_temp_concat.columns:
            df[col] = df_temp_concat[col]
            df[col].fillna(0, inplace=True)
        else:
            # Jika kolom tidak ada di salah satu (misal df_dep_calc kosong), buat kolomnya
            df[col] = 0.0

    print("[INFO] Fitur depresiasi riil (disesuaikan inflasi) telah ditambahkan/diperbarui di DataFrame utama.")

    cols_to_print_dep = ['Harga', 'Tahun', 'UsiaMobil', 'Inflasi'] + \
                        [c for c in ['EstimasiHargaAwal_NilaiSaatIni', 'DepresiasiAbsolut_PerThn_NilaiSaatIni', 'DepresiasiRiilNormal_PersenPerThn'] if c in df.columns]
    if cols_to_print_dep:
        print(df[cols_to_print_dep].head())
else:
    print("[WARNING] Kolom krusial ('Harga', 'UsiaMobil', 'Tahun', atau 'Inflasi') tidak lengkap. Fitur depresiasi dengan inflasi tidak bisa dibuat.")
# --- 7. TAMPILKAN CONTOH DATA & STATISTIK (Setelah Fitur Depresiasi Inflasi) ---
print("\n--- [INFO] Contoh Data dengan Fitur Depresiasi (Disesuaikan Inflasi) ---")
kolom_tampilan_dep_inflasi = [
    'Merek', 'Model_Detail', 'Harga', 'Tahun', 'Kilometer', 'UsiaMobil', 'Lokasi',
    'FaktorInflasiKumulatif', # Tampilkan juga ini untuk cek
    'EstimasiHargaAwal_NilaiSaatIni',
    'DepresiasiAbsolut_PerThn_NilaiSaatIni',
    'DepresiasiRiilNormal_PersenPerThn'
]
kolom_tersedia_tampil_dep_inflasi = [col for col in kolom_tampilan_dep_inflasi if col in df.columns]

if not df.empty and kolom_tersedia_tampil_dep_inflasi:
    jumlah_baris_tampil = 10
    print(f"\nMenampilkan {jumlah_baris_tampil} baris pertama dengan kolom terpilih (disesuaikan inflasi):")
    try:
        pd.set_option('display.max_columns', None); pd.set_option('display.width', 1200); pd.set_option('display.colheader_justify', 'left')
        print(df[kolom_tersedia_tampil_dep_inflasi].head(jumlah_baris_tampil).to_string(index=True))
    except Exception as e:
        print(f"[WARNING] Gagal menampilkan tabel dengan format to_string: {e}")
        print(df[kolom_tersedia_tampil_dep_inflasi].head(jumlah_baris_tampil))

    print("\n--- [INFO] Statistik Deskriptif untuk Fitur Depresiasi (Disesuaikan Inflasi) ---")
    # Kolom depresiasi yang relevan untuk deskripsi (pilih yang paling informatif)
    desc_cols_dep_final = [
        'EstimasiHargaAwal_NilaiSaatIni',
        'DepresiasiAbsolut_PerThn_NilaiSaatIni',
        'DepresiasiRiilNormal_PersenPerThn'
    ]
    desc_cols_dep_final_existing = [col for col in desc_cols_dep_final if col in df.columns]
    if desc_cols_dep_final_existing:
        print(df[desc_cols_dep_final_existing].describe())
    else:
        print("[INFO] Tidak ada kolom fitur depresiasi (disesuaikan inflasi) untuk statistik deskriptif.")
else:
    print("[INFO] DataFrame kosong atau kolom yang diminta tidak tersedia untuk ditampilkan (setelah fitur depresiasi inflasi).")


# --- 8. SIMPAN CSV DENGAN FITUR DEPRESIASI (SEBELUM ENCODING/SCALING) ---
OUTPUT_CSV_INTERMEDIATE_WITH_INFLATION = 'data_mobil_fitur_depresiasi_inflasi.csv'
print(f"\n--- [INFO] Menyimpan DataFrame dengan Fitur Depresiasi (Disesuaikan Inflasi) ke '{OUTPUT_CSV_INTERMEDIATE_WITH_INFLATION}' ---")
if not df.empty:
    try:
        df.to_csv(OUTPUT_CSV_INTERMEDIATE_WITH_INFLATION, index=False, encoding='utf-8')
        print(f"[INFO] DataFrame berhasil disimpan.")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan DataFrame ke '{OUTPUT_CSV_INTERMEDIATE_WITH_INFLATION}': {e}")
else:
    print("[WARNING] DataFrame kosong, tidak ada yang disimpan pada tahap ini.")


# --- 9. VISUALISASI FITUR DEPRESIASI EKSPLISIT (Disesuaikan Inflasi) ---
print("\n--- [INFO] Memulai Visualisasi Fitur Depresiasi (Disesuaikan Inflasi) ---")
plt.style.use('seaborn-v0_8-whitegrid')

# Kolom-kolom fitur depresiasi yang akan divisualisasikan
dep_feature_cols_to_plot = [
    'EstimasiHargaAwal_NilaiSaatIni',
    'DepresiasiAbsolut_PerThn_NilaiSaatIni',
    'DepresiasiRiilNormal_PersenPerThn'
]
existing_dep_features_to_plot = [col for col in dep_feature_cols_to_plot if col in df.columns]

if not existing_dep_features_to_plot:
    print("[INFO] Tidak ada fitur depresiasi (disesuaikan inflasi) untuk divisualisasikan.")
else:
    for feature_to_plot in existing_dep_features_to_plot:
        plt.figure(figsize=(10, 6))
        # Hilangkan outlier ekstrim untuk plot distribusi agar lebih informatif
        data_to_plot = df[feature_to_plot].dropna()
        if pd.api.types.is_numeric_dtype(data_to_plot) and len(data_to_plot) > 1 :
            sns.histplot(data_to_plot, kde=True, bins=50)

            plt.title(f'Distribusi {feature_to_plot}', fontsize=15)
            plt.xlabel(f'{feature_to_plot}', fontsize=12)
            plt.ylabel('Frekuensi', fontsize=12)
            if 'Harga' in feature_to_plot or 'Absolut' in feature_to_plot : plt.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            plt.show()

            # Scatter plot terhadap UsiaMobil
            if 'UsiaMobil' in df.columns:
                plt.figure(figsize=(12, 7))
                sns.scatterplot(data=df, x='UsiaMobil', y=feature_to_plot, alpha=0.4, edgecolor=None, s=30)
                plt.title(f'{feature_to_plot} vs. Usia Mobil', fontsize=15)
                plt.xlabel('Usia Mobil (Tahun)', fontsize=12)
                plt.ylabel(f'{feature_to_plot}', fontsize=12)
                if 'Harga' in feature_to_plot or 'Absolut' in feature_to_plot : plt.ticklabel_format(style='plain', axis='y')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
        else:
            print(f"[INFO] Kolom '{feature_to_plot}' tidak numerik atau kosong, dilewati untuk plot distribusi.")

print("--- [INFO] Selesai Visualisasi Fitur Depresiasi (Disesuaikan Inflasi) ---")