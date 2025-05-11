# mobil_scraper.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import logging

# Konfigurasi logging dasar untuk melihat proses dan error
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- KONFIGURASI SCRAPER ---
# URL dasar pencarian (contoh: Toyota Avanza di Mobil123). SESUAIKAN JIKA PERLU!
BASE_URL = 'https://www.mobil123.com/mobil-bekas-dijual/indonesia'

# !!! GANTI DENGAN USER-AGENT DARI BROWSER CHROME ANDA !!!
USER_AGENT_STRING = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36' # Contoh: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT_STRING}

MAX_PAGES_TO_SCRAPE = 3425     # Batasi jumlah halaman untuk uji coba awal (misal: 3-5 halaman)
REQUEST_DELAY_SECONDS = 8   # Jeda antar request halaman (detik) untuk etika dan menghindari blokir
OUTPUT_FILENAME = 'hasil_scrape_mobil123.csv'

# --- FUNGSI HELPER ---

def fetch_page_content(url, headers):
    """Mengambil konten HTML dari URL yang diberikan."""
    logging.info(f"Mengambil konten dari: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=20) # Timeout 20 detik
        response.raise_for_status() # Memunculkan error jika status code bukan 2xx (sukses)
        return response.text
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error terjadi: {http_err} - URL: {url}")
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Error koneksi: {conn_err} - URL: {url}")
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error: {timeout_err} - URL: {url}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Error request umum: {req_err} - URL: {url}")
    return None

def parse_price_to_int(price_text):
    """Membersihkan teks harga (mis. "Rp 150.000.000") menjadi integer."""
    if price_text:
        cleaned_price = re.sub(r'\D', '', price_text) # Hapus semua karakter non-digit
        if cleaned_price:
            return int(cleaned_price)
    return None
# ... (bagian kode lain seperti import, fetch_page_content, parse_price_to_int, dll. TETAP SAMA) ...
# Pastikan parse_year_to_int dan parse_km_to_int bisa menangani input string angka murni.

def parse_year_to_int(year_text):
    """Mengambil tahun sebagai integer dari teks (mis. "2020" atau "Tahun 2020")."""
    if year_text:
        # Coba konversi langsung jika sudah angka
        if isinstance(year_text, (int, float)):
            return int(year_text)
        if isinstance(year_text, str) and year_text.isdigit():
            return int(year_text)
        # Jika masih string, cari 4 digit angka
        match = re.search(r'\b(\d{4})\b', str(year_text))
        if match:
            return int(match.group(1))
    return None

def parse_km_to_int(km_text):
    """Membersihkan teks kilometer (mis. "10.000 km" atau "10000") menjadi integer."""
    if km_text:
        # Coba konversi langsung jika sudah angka
        if isinstance(km_text, (int, float)):
            return int(km_text)
        if isinstance(km_text, str) and km_text.isdigit():
            return int(km_text)
        # Jika masih string, bersihkan
        cleaned_km = re.sub(r'[^\d]', '', str(km_text).lower().replace('km','')) # Hanya ambil digit
        if cleaned_km:
            return int(cleaned_km)
    return None


def extract_listings_from_soup(soup, source_url):
    """Mengekstrak semua listing mobil dari objek BeautifulSoup satu halaman."""
    mobil_data_list = []
    
    # Selector utama untuk container listing, ini seharusnya sudah benar
    listings = soup.find_all('article', class_='listing--card')
    
    logging.info(f"Menemukan {len(listings)} listing dengan selector 'article', class_='listing--card'.")

    for item_html in listings: # item_html adalah satu <article>
        try:
            # --- Ekstraksi Judul ---
            # Cara 1: Dari data-display-title di <article> (lebih disukai jika ada dan bersih)
            judul = item_html.get('data-display-title')
            if not judul: # Cara 2: Fallback ke h2 > a
                title_h2 = item_html.find('h2', class_='listing__title')
                if title_h2:
                    title_a = title_h2.find('a')
                    if title_a:
                        judul = title_a.text.strip()
            
            # --- Ekstraksi Harga ---
            price_div = item_html.find('div', class_='listing__price')
            harga_text = price_div.text.strip() if price_div else None
            harga = parse_price_to_int(harga_text)
            # Jika harga dari div kosong, bisa coba ambil dari data-title di article (perlu parsing ekstra)
            if not harga:
                data_title_price = item_html.get('data-title')
                if data_title_price:
                    price_match_in_title = re.search(r'\(Rp\s*([\d.,]+)\)', data_title_price)
                    if price_match_in_title:
                        harga = parse_price_to_int(price_match_in_title.group(1))


            # --- Ekstraksi Tahun ---
            # Dari atribut data-year di <article> (paling akurat)
            tahun_str = item_html.get('data-year')
            tahun = parse_year_to_int(tahun_str)

            # --- Ekstraksi Kilometer ---
            # Dari atribut data-mileage di <article> (paling akurat)
            km_str = item_html.get('data-mileage')
            kilometer = parse_km_to_int(km_str)
            
            # Jika tahun atau km masih kosong dari data-atribut, coba dari listing__specs
            if tahun is None or kilometer is None:
                specs_div = item_html.find('div', class_='listing__specs')
                if specs_div:
                    items_spec = specs_div.find_all('div', class_='item')
                    for spec_item in items_spec:
                        icon_element = spec_item.find('i', class_='icon')
                        if icon_element:
                            # Kilometer dari icon--meter
                            if 'icon--meter' in icon_element.get('class', []) and kilometer is None:
                                km_text_from_spec = spec_item.text.strip() # Ambil semua teks di div item ini
                                kilometer = parse_km_to_int(km_text_from_spec) # parse_km_to_int akan membersihkan
                            # Tahun bisa jadi tidak ada di sini, karena sudah jelas dari data-year
                            # Lokasi
                            # if 'icon--location' in icon_element.get('class', []) and lokasi is None:
                            #     lokasi = spec_item.text.strip().replace(icon_element.text, '').strip() # Ambil teks setelah icon

            # --- Ekstraksi Lokasi ---
            lokasi = None
            specs_div_for_loc = item_html.find('div', class_='listing__specs')
            if specs_div_for_loc:
                items_spec_for_loc = specs_div_for_loc.find_all('div', class_='item')
                for spec_item_loc in items_spec_for_loc:
                    icon_loc = spec_item_loc.find('i', class_='icon--location')
                    if icon_loc:
                        # Ambil teks dari parent div.item, lalu hapus teks dari icon jika perlu, atau ambil sibling text
                        # Cara lebih aman: ambil semua teks dari div.item dan bersihkan
                        lokasi_text_raw = spec_item_loc.text.strip()
                        # Hapus teks icon jika ada, atau cari cara lain untuk mendapatkan teks bersihnya
                        # Untuk kasus ini, icon tidak punya teks, jadi teks di div.item adalah lokasinya.
                        lokasi = lokasi_text_raw.strip() # Seharusnya "DKI Jakarta"
                        break # Sudah ketemu lokasi

            # --- Ekstraksi Merek & Model ---
            # Dari atribut data-make dan data-model di <article>
            merek = item_html.get('data-make')
            model_utama = item_html.get('data-model') # Misal "City"
            # model_detail bisa dari judul atau data-variant
            model_variant = item_html.get('data-variant') # Misal "RS Honda Sensing"
            
            model_detail = model_utama # Default
            if model_variant:
                model_detail = f"{model_utama} {model_variant}" if model_utama else model_variant
            elif judul and merek: # Jika tidak ada data-variant, coba dari judul
                # Hapus merek dari judul untuk mendapatkan detail model
                if judul.lower().startswith(merek.lower()):
                    model_detail_from_title = judul[len(merek):].strip()
                    if model_detail_from_title: # Jika ada sisa setelah merek
                         model_detail = model_detail_from_title


            if judul and harga:
                mobil_data_list.append({
                    'Judul': judul,
                    'Merek': merek,
                    'Model_Detail': model_detail,
                    'Harga': harga,
                    'Tahun': tahun,
                    'Kilometer': kilometer,
                    'Lokasi': lokasi,
                    'SumberURL': source_url
                })
            # else: # Untuk debugging jika item tidak masuk karena judul/harga kosong
            #     logging.debug(f"Item dilewati: Judul='{judul}', Harga='{harga}', HTML Cuplikan: {str(item_html)[:200]}")

        except AttributeError as e:
            logging.warning(f"AttributeError saat parsing listing (elemen tidak ditemukan?): {e}. Melewati item ini. Cek selector. HTML Cuplikan:\n{str(item_html)[:500]}")
            continue
        except Exception as e:
            logging.error(f"Terjadi error umum saat parsing listing: {e}. Melewati item ini.")
            continue
            
    return mobil_data_list

# --- FUNGSI UTAMA SCRAPING (main_scraper) ---
# Pastikan BASE_URL di main_scraper sudah benar
# BASE_URL = 'https://www.mobil123.com/mobil-bekas-dijual/indonesia'
# ... (sisa kode main_scraper dan pemanggilan if __name__ == '__main__': tetap sama) ...
# Pastikan juga level logging di set ke INFO atau DEBUG untuk melihat output
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- FUNGSI UTAMA SCRAPING ---
def main_scraper():
    """Fungsi utama untuk menjalankan proses scraping."""
    all_scraped_data = []
    
    # Pastikan User-Agent sudah diisi
    if USER_AGENT_STRING == 'YOUR_USER_AGENT_STRING_HERE' or not USER_AGENT_STRING:
        logging.error("USER_AGENT_STRING belum diatur. Harap isi dengan User-Agent dari browser Anda.")
        return

    logging.info(f"Memulai scraping dari {BASE_URL} untuk maksimal {MAX_PAGES_TO_SCRAPE} halaman.")

    for page_num in range(1, MAX_PAGES_TO_SCRAPE + 1):
        logging.info(f"--- Memproses Halaman {page_num} ---")
        
        if page_num == 1:
            current_url = BASE_URL
        else:
            # Logika URL untuk halaman berikutnya. Mobil123 menggunakan ?page=N
            # Cek jika BASE_URL sudah punya query parameter (misalnya dari filter)
            if '?' in BASE_URL:
                 current_url = f"{BASE_URL}&page={page_num}"
            else:
                 current_url = f"{BASE_URL}?page={page_num}"

        html_content = fetch_page_content(current_url, HEADERS)
        
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            data_from_page = extract_listings_from_soup(soup, current_url)
            
            if not data_from_page and page_num > 1 : # Jika tidak ada data di halaman ini (setelah halaman 1)
                logging.info(f"Tidak ada data lagi ditemukan di halaman {page_num}. Menghentikan pagination.")
                break # Keluar dari loop jika tidak ada data lagi
                
            all_scraped_data.extend(data_from_page)
            logging.info(f"Berhasil scrape {len(data_from_page)} item dari halaman {page_num}.")
        else:
            logging.warning(f"Gagal mengambil konten halaman {page_num}. Mungkin sudah halaman terakhir atau ada masalah jaringan.")
            # Jika gagal di halaman > 1, anggap sudah habis atau ada masalah persisten
            if page_num > 1:
                 break 
            
        # Beri jeda antar request halaman
        logging.info(f"Menunggu {REQUEST_DELAY_SECONDS} detik sebelum halaman berikutnya...")
        time.sleep(REQUEST_DELAY_SECONDS)

    # --- Menyimpan Data ke CSV ---
    if all_scraped_data:
        df = pd.DataFrame(all_scraped_data)
        
        # Urutkan kolom agar lebih rapi (opsional)
        kolom_urut = ['Judul', 'Merek', 'Model_Detail', 'Harga', 'Tahun', 'Kilometer', 'Lokasi', 'SumberURL']
        # Hanya ambil kolom yang ada di DataFrame, antisipasi jika ada kolom yang tidak ter-scrape
        kolom_urut_valid = [kolom for kolom in kolom_urut if kolom in df.columns]
        df = df[kolom_urut_valid]
        
        try:
            df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
            logging.info(f"\nScraping selesai. Data berhasil disimpan ke {OUTPUT_FILENAME}")
            logging.info(f"Total data terkumpul: {len(df)} item.")
            print("\nContoh 5 data pertama yang berhasil di-scrape:")
            print(df.head())
        except Exception as e:
            logging.error(f"Gagal menyimpan data ke CSV: {e}")
            
    else:
        logging.info("\nScraping selesai. Tidak ada data yang berhasil di-scrape.")

# Panggil fungsi utama saat skrip dijalankan
if __name__ == '__main__':
    main_scraper()