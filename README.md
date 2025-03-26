# Mobile OCR API

API untuk pengenalan teks dan navigasi konten yang dioptimalkan untuk aplikasi mobile.

## Fitur

- Pengenalan teks (OCR) dengan Tesseract
- Navigasi konten fleksibel (halaman, paragraf, kalimat, kolom)
- Analisis layout dokumen
- Ringkasan teks
- Optimasi untuk performa mobile
- Pemrosesan asinkron untuk file besar
- Pengolahan PDF dengan Poppler
- Cache hasil OCR untuk meningkatkan performa

## Persyaratan

- Python 3.7+
- Tesseract OCR (v4.0+)
- Poppler (untuk pemrosesan PDF)

## Cara Instalasi

### 1. Install Dependensi Sistem

#### Untuk Windows:
- Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
- Pastikan kedua program tersebut ditambahkan ke PATH sistem

#### Untuk Linux:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
sudo apt-get install -y poppler-utils
```

#### Untuk macOS:
```bash
brew install tesseract
brew install poppler
```

### 2. Clone Repository dan Install Dependensi Python

```bash
git clone https://github.com/yourusername/mobile-ocr-api.git
cd mobile-ocr-api
pip install -r requirements.txt
```

## Cara Penggunaan

1. Pastikan Tesseract dan Poppler terinstal dengan benar
2. Install dependensi: `pip install -r requirements.txt`
3. Jalankan server:
   - Versi dasar: `python app.py`
   - Versi optimasi (rekomendasi): `python app2.py`
4. Server berjalan di `http://localhost:5000`

## API Endpoints

### Versi Dasar (app.py)

- `/api/ocr/health` - Health check
- `/api/ocr/session/create` - Membuat session baru
- `/api/ocr/session/{id}/upload` - Upload dokumen
- `/api/ocr/session/{id}/analyze` - Analisis dokumen
- `/api/ocr/session/{id}/navigate` - Navigasi dalam dokumen
- `/api/ocr/session/{id}/summarize` - Ringkasan teks
- `/api/ocr/recognize` - OCR cepat (tanpa session)
- `/api/ocr/languages` - Daftar bahasa yang tersedia

### Versi Optimasi (app2.py)

- `/api/ocr/v2/health` - Health check
- `/api/ocr/v2/session/create` - Membuat session baru
- `/api/ocr/v2/session/{id}/upload` - Upload dokumen
- `/api/ocr/v2/session/{id}/process` - Proses dokumen (sinkron/asinkron)
- `/api/ocr/v2/session/{id}/result` - Dapatkan hasil pemrosesan asinkron
- `/api/ocr/v2/session/{id}/page/{page}` - Dapatkan halaman tertentu
- `/api/ocr/v2/session/{id}/summary` - Dapatkan ringkasan teks
- `/api/ocr/v2/recognize` - OCR cepat (tanpa session)

## Contoh Penggunaan

### Upload dan Proses Dokumen

```python
import requests

# Buat session baru
response = requests.post('http://localhost:5000/api/ocr/v2/session/create')
session_id = response.json()['session_id']

# Upload file
files = {'file': open('dokumen.pdf', 'rb')}
data = {'language': 'ind+eng', 'quality': 'medium'}
response = requests.post(f'http://localhost:5000/api/ocr/v2/session/{session_id}/upload', 
                        files=files, data=data)
file_id = response.json()['file_id']

# Proses dokumen (asinkron)
data = {'file_id': file_id, 'page': 0, 'async': True}
response = requests.post(f'http://localhost:5000/api/ocr/v2/session/{session_id}/process', 
                        json=data)
task_id = response.json()['task_id']

# Dapatkan hasil
response = requests.get(f'http://localhost:5000/api/ocr/v2/session/{session_id}/result?task_id={task_id}')
result = response.json()
```

## Pengembangan

### Struktur Kode

- `app.py` - Versi dasar API OCR
- `app2.py` - Versi yang dioptimasi untuk mobile dengan fitur tambahan

## Catatan

- Pastikan Tesseract dan Poppler terpasang dengan benar
- Untuk pemrosesan PDF yang lebih baik, gunakan versi terbaru Poppler
- Untuk dokumen besar, gunakan mode asinkron di app2.py

## Lisensi

[MIT License](LICENSE)