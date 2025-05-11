import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import re 
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. SET PAGE CONFIG ---
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide", initial_sidebar_state="expanded")

# --- 1. KONFIGURASI DAN DEFINISI GLOBAL ---
MODEL_PATH = "venv\model\xgboost_mobil_model_v3.pkl"
DATA_ASLI_PATH = "venv\data\data_mobil_fitur_depresiasi_inflasi.csv"
CURRENT_YEAR = datetime.now().year
MAX_USIA_MOBIL_APP_FILTER = 40

# --- 2. Fungsi Load Model dan Komponen ---
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model_and_components(model_path):
    try:
        model, scaler, trained_feature_columns = joblib.load(model_path) # Memuat 3 item
        if not isinstance(trained_feature_columns, list):
            trained_feature_columns = trained_feature_columns.tolist()
        print(f"[LOAD_INFO] Model, scaler, dan {len(trained_feature_columns)} kolom training (X.columns) berhasil dimuat.")
        return model, scaler, trained_feature_columns
    except FileNotFoundError:
        print(f"ERROR_LOAD: File model '{model_path}' tidak ditemukan.")
        return None, None, None
    except ValueError as ve: 
        print(f"ERROR_LOAD: Gagal unpack model dari '{model_path}'. Error: {ve}.")
        return None, None, None
    except Exception as e:
        print(f"ERROR_LOAD: Terjadi kesalahan umum saat memuat model: {e}")
        return None, None, None

model, scaler, X_COLUMNS_TRAINED = load_model_and_components(MODEL_PATH)

# --- 3. Fungsi untuk Mendapatkan Opsi dari Data Asli ---
@st.cache_data 
def load_dropdown_options(data_path):
    try:
        df_ref = pd.read_csv(data_path)
        known_merek = sorted(df_ref['Merek'].dropna().unique().tolist()) if 'Merek' in df_ref.columns else []
        known_lokasi = sorted(df_ref['Lokasi'].dropna().unique().tolist()) if 'Lokasi' in df_ref.columns else []
        print("[LOAD_INFO] Opsi dropdown Merek dan Lokasi dimuat dari data asli.")
        return df_ref, known_merek, known_lokasi
    except FileNotFoundError:
        st.warning(f"File data asli '{data_path}' tidak ditemukan untuk opsi dropdown. Menggunakan daftar fallback.")
        return pd.DataFrame(), ['Toyota', 'Honda', 'Lainnya'], ['Jakarta', 'Bandung', 'Lainnya'] # Fallback
    except Exception as e:
        st.warning(f"Error saat memuat opsi dari data asli: {e}. Menggunakan daftar fallback.")
        return pd.DataFrame(), ['Toyota', 'Honda', 'Lainnya'], ['Jakarta', 'Bandung', 'Lainnya'] # Fallback

data_asli_df, KNOWN_MEREK_FROM_DATA, KNOWN_LOKASI_FROM_DATA = load_dropdown_options(DATA_ASLI_PATH)


# --- 4. Fungsi Preprocessing Input Pengguna ---
def preprocess_user_input_dynamic(user_input_dict, all_trained_feature_names, scaler_obj):
    print("\n--- [DEBUG] Memulai preprocess_user_input (scaler di-fit ke semua X) ---")
    input_df = pd.DataFrame(0, index=[0], columns=all_trained_feature_names)
    tahun_pembuatan_input = user_input_dict.get('Tahun_Input', CURRENT_YEAR)
    usia_mobil_val = CURRENT_YEAR - tahun_pembuatan_input
    
    if 'Tahun' in all_trained_feature_names: input_df.loc[0, 'Tahun'] = tahun_pembuatan_input
    if 'UsiaMobil' in all_trained_feature_names: input_df.loc[0, 'UsiaMobil'] = max(0, usia_mobil_val)
    if 'Kilometer' in all_trained_feature_names: input_df.loc[0, 'Kilometer'] = user_input_dict.get('Kilometer_Input', 0)
    # One-Hot Encoding
    merek_input_val = user_input_dict.get('Merek_Input')
    if merek_input_val and merek_input_val != "Pilih Merek":
        merek_col_name = f"Merek_{merek_input_val}" 
        if merek_col_name in input_df.columns: input_df.loc[0, merek_col_name] = 1
    
    lokasi_input_val = user_input_dict.get('Lokasi_Input')
    if lokasi_input_val and lokasi_input_val != "Pilih Lokasi":
        lokasi_col_name_cleaned = str(lokasi_input_val)
        lokasi_col_name = f"Lokasi_{lokasi_col_name_cleaned}"
        if lokasi_col_name in input_df.columns: input_df.loc[0, lokasi_col_name] = 1
            
    model_detail_input_val = user_input_dict.get('Model_Detail_Input')
    if model_detail_input_val:
        model_detail_cleaned_for_col = re.sub(r'[^-a-zA-Z0-9_]', '', str(model_detail_input_val).lower().replace(' ', '_')).strip('_')
        model_detail_col_name = f"Model_Detail_{model_detail_cleaned_for_col}"
        if model_detail_col_name in input_df.columns: input_df.loc[0, model_detail_col_name] = 1
    
    owner_input_val = user_input_dict.get('Owner_Input')
    if owner_input_val and owner_input_val != "Pilih Jumlah Pemilik":
        owner_col_name_cleaned = str(owner_input_val).split(' ')[0] 
        owner_col_name = f"owner_{owner_col_name_cleaned}"
        if owner_col_name in input_df.columns: input_df.loc[0, owner_col_name] = 1
    if scaler_obj:
        try:
            input_df_ordered = input_df[all_trained_feature_names]
            scaled_values = scaler_obj.transform(input_df_ordered)
            input_df_scaled = pd.DataFrame(scaled_values, columns=all_trained_feature_names, index=input_df_ordered.index)
            return input_df_scaled
        except ValueError as ve_scaler:
            raise # Timbulkan kembali error untuk ditangkap di blok pemanggil
    else:
        return input_df[all_trained_feature_names]


# --- 5. UI STREAMLIT ---
if not model or not scaler or not X_COLUMNS_TRAINED:
    st.error("Gagal memuat komponen model (model, scaler, atau daftar kolom training). Aplikasi tidak dapat berjalan. Periksa file .pkl dan path-nya, pastikan berisi 3 item.")
    st.stop()

st.markdown("<h1 style='text-align: center; color: #007BFF;'>Prediksi Harga Mobil Bekas 🚗</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    input_tahun_ui = st.number_input("Tahun Mobil", min_value=1980, max_value=CURRENT_YEAR + 1, step=1, value=2018)
    input_km_driven_ui = st.number_input("Kilometer Tempuh", min_value=0, step=1000, value=50000, format="%d")
    input_harga_beli_dulu_ui = st.number_input("Harga Baru Mobil Dulu (Rp)", min_value=0, step=1000000, value=0, format="%d", help="Opsional, untuk analisis depresiasi.")

with col2:
    input_merek_ui = st.selectbox("Merek Mobil", options=["Pilih Merek"] + KNOWN_MEREK_FROM_DATA)
    
    filtered_models_ui_options = ["Ketik Manual Model Detail"]
    if input_merek_ui != "Pilih Merek" and not data_asli_df.empty and 'Model_Detail' in data_asli_df.columns:
        models_for_brand = sorted(data_asli_df[data_asli_df['Merek'] == input_merek_ui]['Model_Detail'].dropna().unique().tolist())
        if models_for_brand:
            filtered_models_ui_options = ["Pilih Model"] + models_for_brand
    
    
    if len(filtered_models_ui_options) > 1 :
        input_model_detail_ui = st.selectbox("Model Detail Mobil", options=filtered_models_ui_options, index=0)
        if input_model_detail_ui == "Pilih Model": input_model_detail_ui = ""
    else:
        input_model_detail_ui = st.text_input("Model Detail Mobil", placeholder="Contoh: Avanza G 1.3 AT")
    
    input_lokasi_ui = st.selectbox("Lokasi", options=["Pilih Lokasi"] + KNOWN_LOKASI_FROM_DATA)
    input_owner_ui = st.selectbox("Jumlah Pemilik Sebelumnya", options=["Pilih Jumlah Pemilik", "First", "Second", "Third", "Fourth & Above"])

st.markdown("---")

if st.button("🔍 Prediksi Harga", type="primary", use_container_width=True):
    valid_input_ui = True
    if input_merek_ui == "Pilih Merek": st.warning("Pilih Merek Mobil."); valid_input_ui = False
    if not input_model_detail_ui.strip(): st.warning("Isi atau Pilih Model Detail Mobil."); valid_input_ui = False
    if input_lokasi_ui == "Pilih Lokasi": st.warning("Pilih Lokasi."); valid_input_ui = False
    
    usia_mobil_input_val = CURRENT_YEAR - input_tahun_ui
    if usia_mobil_input_val < 0: st.error("Tahun pembuatan tidak valid."); valid_input_ui = False

    if valid_input_ui:
        user_data_for_preprocessing_dict = {
            'Tahun_Input': input_tahun_ui,
            'Kilometer_Input': input_km_driven_ui,
            'Merek_Input': input_merek_ui,
            'Model_Detail_Input': input_model_detail_ui,
            'Lokasi_Input': input_lokasi_ui,
            'Owner_Input': input_owner_ui,
        }
        
        try:
            with st.spinner("Memproses dan memprediksi harga... ⏳"):
                print("--- DEBUGGING PREPROCESSING DI TOMBOL PREDIKSI (KONSOL SERVER) ---")
                print("Input ke Preprocess:", user_data_for_preprocessing_dict)
                print("Kolom yang Diharapkan Model (X_COLUMNS_TRAINED):", X_COLUMNS_TRAINED[:7], f"... (total {len(X_COLUMNS_TRAINED)})")
                
                processed_df_for_predict = preprocess_user_input_dynamic(
                    user_data_for_preprocessing_dict, 
                    X_COLUMNS_TRAINED, 
                    scaler
                )
                
                print("DataFrame Setelah Preprocessing (Siap untuk Prediksi - KONSOL SERVER):")
                print(processed_df_for_predict.loc[0, processed_df_for_predict.loc[0] != 0].to_dict())
                print("--- AKHIR DEBUGGING PREPROCESSING (KONSOL SERVER) ---")

                prediction_log = model.predict(processed_df_for_predict)
                predicted_price_rp = np.expm1(prediction_log[0])

            st.subheader("💰 Estimasi Harga Bekas Saat Ini (dari Model AI)")
            st.markdown(f"<h2 style='text-align: center; color: #28a745;'>Rp {predicted_price_rp:,.0f}</h2>", unsafe_allow_html=True)
            st.caption("Estimasi ini berdasarkan analisis data pasar dan model machine learning.")
            st.markdown("---")

            if input_harga_beli_dulu_ui > 0:
                st.subheader("📉 Analisis Depresiasi Nominal (Berdasarkan Input Anda)")
                harga_beli_dulu_rp_val = float(input_harga_beli_dulu_ui)
                total_dep_rp = 0.0; avg_dep_per_year_rp = 0.0; perc_dep_total = 0.0; avg_perc_dep_per_year = 0.0
                catatan_depresiasi = ""
                if usia_mobil_input_val == 0:
                    total_dep_rp = max(0, harga_beli_dulu_rp_val - predicted_price_rp)
                    avg_dep_per_year_rp = total_dep_rp 
                    if harga_beli_dulu_rp_val > 0: perc_dep_total = (total_dep_rp / harga_beli_dulu_rp_val) * 100
                    avg_perc_dep_per_year = perc_dep_total
                    catatan_depresiasi = "Depresiasi awal tahun pertama." if total_dep_rp > 0 else "Harga prediksi sama/lebih tinggi dari harga beli (mobil baru)."
                elif predicted_price_rp >= harga_beli_dulu_rp_val:
                     catatan_depresiasi = f"Harga prediksi (Rp {predicted_price_rp:,.0f}) lebih tinggi atau sama dengan harga beli (Rp {harga_beli_dulu_rp_val:,.0f})."
                     total_dep_rp = harga_beli_dulu_rp_val - predicted_price_rp
                     if usia_mobil_input_val > 0: avg_dep_per_year_rp = total_dep_rp / usia_mobil_input_val
                     if harga_beli_dulu_rp_val > 0:
                        perc_dep_total = (total_dep_rp / harga_beli_dulu_rp_val) * 100
                        if usia_mobil_input_val > 0: avg_perc_dep_per_year = perc_dep_total / usia_mobil_input_val
                else: 
                    total_dep_rp = harga_beli_dulu_rp_val - predicted_price_rp
                    if usia_mobil_input_val > 0:
                        avg_dep_per_year_rp = total_dep_rp / usia_mobil_input_val
                        if harga_beli_dulu_rp_val > 0:
                            perc_dep_total = (total_dep_rp / harga_beli_dulu_rp_val) * 100
                            avg_perc_dep_per_year = perc_dep_total / usia_mobil_input_val
                    catatan_depresiasi = f"Perhitungan depresiasi nominal selama {usia_mobil_input_val} tahun."

                col_dep1, col_dep2, col_dep3 = st.columns(3)
                with col_dep1: st.metric(label="Harga Beli Baru Dulu", value=f"Rp {harga_beli_dulu_rp_val:,.0f}")
                with col_dep2: st.metric(label="Total Depresiasi Nominal", value=f"Rp {total_dep_rp:,.0f}", delta=f"{perc_dep_total:.1f}%" if harga_beli_dulu_rp_val > 0 else None, delta_color="inverse" if total_dep_rp > 0 and predicted_price_rp < harga_beli_dulu_rp_val else ("normal" if total_dep_rp < 0 else "off"))
                with col_dep3:
                    if usia_mobil_input_val > 0 and total_dep_rp != 0 :
                        st.metric(label="Rata-rata Depresiasi/Tahun", value=f"Rp {avg_dep_per_year_rp:,.0f}", delta=f"{avg_perc_dep_per_year:.1f}%/thn" if harga_beli_dulu_rp_val > 0 else None, delta_color="inverse" if avg_dep_per_year_rp > 0 and predicted_price_rp < harga_beli_dulu_rp_val else ("normal" if avg_dep_per_year_rp < 0 else "off"))
                st.caption(catatan_depresiasi)
            else:
                st.info("Masukkan 'Harga Baru Mobil Dulu' di sidebar untuk melihat analisis depresiasi.")
        
        except ValueError as ve:
            st.error(f"Kesalahan Nilai saat Preprocessing atau Prediksi: {ve}")
            st.exception(ve)
        except Exception as e:
            st.error(f"Terjadi kesalahan umum: {e}")
            st.exception(e)
