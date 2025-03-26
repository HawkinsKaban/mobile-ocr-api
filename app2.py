import os
import time
import uuid
import logging
import re
import json
import numpy as np
import cv2
import pytesseract
import base64
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
from flask_cors import CORS
from collections import defaultdict
import threading
import queue
import functools

# Konfigurasi logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Konfigurasi upload
UPLOAD_FOLDER = '/tmp/ocr_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max

# Pastikan direktori upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache untuk session
SESSION_CACHE = {}
SESSION_TIMEOUT = 30  # Menit

# Cache untuk hasil OCR untuk mengurangi pemrosesan berulang
OCR_CACHE = {}
OCR_CACHE_TIMEOUT = 60  # Menit

# Konfigurasi Tesseract
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
elif os.path.exists('/usr/local/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Konfigurasi app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Worker thread pool untuk pemrosesan asinkron
worker_queue = queue.Queue()
MAX_WORKERS = 4
workers = []

# ===== FUNGSI UTILITAS =====

def register_poppler_path():
    """Mendaftarkan path Poppler ke environment PATH"""
    poppler_paths = [
        r'C:\Program Files\poppler\Library\bin',
        r'C:\Program Files\poppler\bin',
        r'C:\Program Files (x86)\poppler\bin',
        r'C:\poppler\bin',
    ]
    
    for path in poppler_paths:
        if os.path.exists(path):
            if os.path.exists(os.path.join(path, 'pdftoppm.exe')):
                logger.info(f"Menemukan Poppler di: {path}")
                
                if path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                    logger.info(f"Menambahkan {path} ke PATH environment")
                    
                return path
    
    logger.warning("Tidak dapat menemukan Poppler, fungsi PDF mungkin terbatas")
    return None

def allowed_file(filename):
    """Memeriksa jika ekstensi file diizinkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_session():
    """Membuat session baru"""
    session_id = str(uuid.uuid4())
    SESSION_CACHE[session_id] = {
        'created_at': datetime.now(),
        'last_accessed': datetime.now(),
        'files': {},
        'results': {},
        'current': {
            'file_id': None,
            'page': 0
        }
    }
    return session_id

def get_session(session_id):
    """Mendapatkan data session dan memperbaharui waktu akses terakhir"""
    if session_id in SESSION_CACHE:
        SESSION_CACHE[session_id]['last_accessed'] = datetime.now()
        return SESSION_CACHE[session_id]
    return None

def cleanup_sessions():
    """Menghapus session yang sudah kadaluarsa"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in SESSION_CACHE.items():
        if current_time - session_data['last_accessed'] > timedelta(minutes=SESSION_TIMEOUT):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        # Hapus file yang terkait
        if 'files' in SESSION_CACHE[session_id]:
            for file_id in SESSION_CACHE[session_id]['files']:
                file_data = SESSION_CACHE[session_id]['files'][file_id]
                try:
                    if 'filepath' in file_data and os.path.exists(file_data['filepath']):
                        os.remove(file_data['filepath'])
                except Exception as e:
                    logger.error(f"Error menghapus file: {e}")
        
        # Hapus session
        del SESSION_CACHE[session_id]
        logger.info(f"Session kadaluarsa dihapus: {session_id}")

def cleanup_ocr_cache():
    """Membersihkan cache OCR yang sudah kadaluarsa"""
    current_time = datetime.now()
    expired_keys = []
    
    for key, data in OCR_CACHE.items():
        if current_time - data['timestamp'] > timedelta(minutes=OCR_CACHE_TIMEOUT):
            expired_keys.append(key)
    
    for key in expired_keys:
        del OCR_CACHE[key]
        logger.info(f"Cache OCR kadaluarsa dihapus: {key}")

def schedule_cleanup():
    """Menjadwalkan pembersihan cache"""
    cleanup_sessions()
    cleanup_ocr_cache()
    threading.Timer(300, schedule_cleanup).start()  # Setiap 5 menit

# Fungsi worker untuk pemrosesan asinkron
def worker_thread():
    while True:
        try:
            task, args, kwargs, result_key, session_id = worker_queue.get()
            try:
                result = task(*args, **kwargs)
                # Simpan hasil ke cache session
                if session_id in SESSION_CACHE and result_key:
                    SESSION_CACHE[session_id]['results'][result_key] = {
                        'status': 'success',
                        'result': result,
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                if session_id in SESSION_CACHE and result_key:
                    SESSION_CACHE[session_id]['results'][result_key] = {
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now()
                    }
            finally:
                worker_queue.task_done()
        except Exception as e:
            logger.error(f"Worker thread error: {str(e)}")

# Mulai worker threads
def start_workers():
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker_thread, daemon=True)
        t.start()
        workers.append(t)

# ===== FUNGSI OPTIMASI GAMBAR =====

def optimize_image_for_mobile(image_path, quality='medium'):
    """
    Mengoptimasi gambar untuk pemrosesan mobile
    
    Args:
        image_path: Path ke gambar
        quality: low, medium, atau high
    
    Returns:
        Path ke gambar yang dioptimasi
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Tentukan ukuran maksimum berdasarkan kualitas
            if quality == 'low':
                max_dim = 800
                jpeg_quality = 75
            elif quality == 'high':
                max_dim = 2000
                jpeg_quality = 90
            else:  # medium (default)
                max_dim = 1200
                jpeg_quality = 85
            
            # Hanya resize jika gambar lebih besar dari dimensi maksimum
            if width > max_dim or height > max_dim:
                # Hitung dimensi baru dengan mempertahankan aspek rasio
                if width > height:
                    new_width = max_dim
                    new_height = int(height * (max_dim / width))
                else:
                    new_height = max_dim
                    new_width = int(width * (max_dim / height))
                
                # Resize gambar
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Simpan gambar yang dioptimasi
                optimized_path = f"{image_path}_optimized.jpg"
                img.save(optimized_path, 'JPEG', quality=jpeg_quality)
                
                logger.info(f"Gambar dioptimasi dari {width}x{height} ke {new_width}x{new_height}")
                return optimized_path
            
            return image_path
    except Exception as e:
        logger.error(f"Error mengoptimasi gambar: {e}")
        return image_path

def preprocess_image(image_path, preprocess_type='balanced'):
    """
    Memproses gambar untuk meningkatkan akurasi OCR - dioptimasi untuk mobile
    
    Args:
        image_path: Path ke gambar
        preprocess_type: fast, balanced, quality
    
    Returns:
        Path ke gambar yang diproses
    """
    try:
        # Baca gambar
        image = cv2.imread(image_path)
        
        if image is None:
            # Coba dengan PIL jika OpenCV gagal
            try:
                pil_image = Image.open(image_path)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                temp_jpg = f"{image_path}_temp.jpg"
                pil_image.save(temp_jpg)
                image = cv2.imread(temp_jpg)
                os.remove(temp_jpg)
            except Exception as e:
                logger.error(f"Error mengkonversi gambar: {e}")
                return image_path

        if image is None:
            return image_path
            
        # Konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Terapkan preprocessing berdasarkan tipe
        if preprocess_type == 'fast':
            # Paling cepat - threshold dasar
            _, processed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        elif preprocess_type == 'quality':
            # Kualitas tinggi - lebih lambat
            processed = cv2.GaussianBlur(gray, (3, 3), 0)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
        else:  # balanced (default)
            # Seimbang antara kecepatan dan kualitas
            processed = cv2.GaussianBlur(gray, (3, 3), 0)
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2)
        
        # Simpan gambar yang diproses
        processed_path = f"{image_path}_processed.jpg"
        cv2.imwrite(processed_path, processed)
        
        return processed_path
    
    except Exception as e:
        logger.error(f"Error selama preprocessing: {e}")
        return image_path

# ===== FUNGSI OCR =====

def perform_ocr(image_path, language='eng', config='', mode='standard'):
    """
    Melakukan OCR yang dioptimasi untuk mobile
    
    Args:
        image_path: Path ke gambar
        language: Bahasa OCR (default: eng)
        config: Konfigurasi Tesseract tambahan
        mode: Mode OCR (fast, standard, accurate)
    
    Returns:
        Teks yang diekstrak, waktu pemrosesan, tingkat kepercayaan
    """
    # Cek cache
    cache_key = f"{image_path}_{language}_{mode}"
    if cache_key in OCR_CACHE:
        cache_data = OCR_CACHE[cache_key]
        # Pastikan cache tidak kadaluarsa
        if datetime.now() - cache_data['timestamp'] < timedelta(minutes=OCR_CACHE_TIMEOUT):
            logger.info(f"Menggunakan hasil OCR dari cache untuk {image_path}")
            return cache_data['text'], 0, cache_data['confidence']
    
    start_time = time.time()
    
    try:
        # Baca gambar
        image = Image.open(image_path)
        
        # Set config berdasarkan mode
        if mode == 'fast':
            # Mode cepat - kualitas lebih rendah
            custom_config = f'-l {language} --oem 0 --psm 6'
        elif mode == 'accurate':
            # Mode akurat - lebih lambat
            custom_config = f'-l {language} --oem 1 --psm 3 -c tessedit_do_invert=0'
        else:  # standard
            # Mode standar - seimbang
            custom_config = f'-l {language} --oem 1 --psm 3'
        
        # Tambahkan config kustom
        if config:
            custom_config += f' {config}'
            
        # Jalankan OCR
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Dapatkan tingkat kepercayaan
        try:
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            # Fallback jika kalkulasi kepercayaan gagal
            avg_confidence = 80.0
        
        # Hitung waktu pemrosesan
        processing_time = time.time() - start_time
        
        # Simpan ke cache
        OCR_CACHE[cache_key] = {
            'text': text,
            'confidence': avg_confidence,
            'timestamp': datetime.now()
        }
        
        return text, processing_time, avg_confidence
    
    except Exception as e:
        logger.error(f"Error selama OCR: {e}")
        return str(e), time.time() - start_time, 0

def extract_pdf_with_subprocess(pdf_path, page_num=None):
    """
    Ekstrak halaman PDF menggunakan subprocess langsung ke pdftoppm
    
    Args:
        pdf_path: Path ke file PDF
        page_num: Nomor halaman yang diinginkan (dimulai dari 0)
    
    Returns:
        Path ke file gambar hasil, jumlah halaman total
    """
    logger.info(f"Mengekstrak PDF dengan subprocess: {pdf_path}")
    
    # Cek jumlah halaman
    try:
        pdfinfo_exe = r'C:\Program Files\poppler\Library\bin\pdfinfo.exe'
        if not os.path.exists(pdfinfo_exe):
            pdfinfo_exe = 'pdfinfo'
            
        result = subprocess.run([pdfinfo_exe, pdf_path], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            # Fallback ke PyPDF2
            from PyPDF2 import PdfReader
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
        else:
            # Parse output dari pdfinfo
            output = result.stdout
            pages_line = [line for line in output.split('\n') if 'Pages:' in line]
            if pages_line:
                total_pages = int(pages_line[0].split('Pages:')[1].strip())
            else:
                total_pages = 0
    except Exception as e:
        logger.error(f"Error mendapatkan jumlah halaman: {e}")
        # Fallback ke PyPDF2
        try:
            from PyPDF2 import PdfReader
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
        except:
            total_pages = 0
    
    # Cek validitas nomor halaman
    if page_num is not None and (page_num < 0 or page_num >= total_pages):
        return None, total_pages
    
    # Buat direktori temporary
    temp_dir = tempfile.mkdtemp()
    output_prefix = os.path.join(temp_dir, "page")
    
    try:
        # Path ke pdftoppm
        pdftoppm_exe = r'C:\Program Files\poppler\Library\bin\pdftoppm.exe'
        if not os.path.exists(pdftoppm_exe):
            pdftoppm_exe = 'pdftoppm'
        
        # Jalankan pdftoppm untuk mengkonversi halaman PDF ke JPG
        cmd = [
            pdftoppm_exe,
            "-jpeg",
            "-r", "150"  # Resolusi lebih rendah untuk mobile
        ]
        
        # Tambahkan range halaman
        if page_num is not None:
            cmd.extend(["-f", str(page_num + 1), "-l", str(page_num + 1)])
        else:
            # Default hanya halaman pertama
            cmd.extend(["-f", "1", "-l", "1"])
            
        # Tambahkan input dan output
        cmd.extend([pdf_path, output_prefix])
        
        logger.info(f"Menjalankan command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error(f"pdftoppm error: {result.stderr}")
            return None, total_pages
        
        # Cari file output
        output_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                       if f.startswith(os.path.basename(output_prefix)) and f.endswith(".jpg")]
        
        if not output_files:
            logger.error(f"Tidak ada file output yang dihasilkan di {temp_dir}")
            return None, total_pages
            
        # Urutkan file
        output_files.sort()
        
        logger.info(f"Berhasil menghasilkan {len(output_files)} file")
        return output_files[0], total_pages
        
    except Exception as e:
        logger.error(f"Error saat mengkonversi PDF: {e}")
        return None, total_pages

def extract_text_from_pdf(pdf_path, language='eng', page_num=None, quality='medium'):
    """
    Ekstrak teks dari PDF untuk mobile
    
    Args:
        pdf_path: Path ke file PDF
        language: Bahasa OCR
        page_num: Nomor halaman (None untuk semua halaman)
        quality: Kualitas OCR (low, medium, high)
    
    Returns:
        Teks yang diekstrak, waktu pemrosesan, tingkat kepercayaan, jumlah halaman
    """
    start_time = time.time()
    
    # Cache key
    cache_key = f"{pdf_path}_{language}_{page_num}_{quality}"
    
    # Cek cache
    if cache_key in OCR_CACHE:
        cache_data = OCR_CACHE[cache_key]
        if datetime.now() - cache_data['timestamp'] < timedelta(minutes=OCR_CACHE_TIMEOUT):
            logger.info(f"Menggunakan hasil PDF dari cache")
            return cache_data['text'], 0, cache_data['confidence'], cache_data['total_pages']
    
    try:
        # Coba mendapatkan jumlah halaman dengan PyPDF2
        try:
            from PyPDF2 import PdfReader
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
            logger.info(f"PDF memiliki {total_pages} halaman")
        except Exception as e:
            logger.error(f"Error membaca PDF dengan PyPDF2: {e}")
            total_pages = 0
        
        # Coba ekstrak teks langsung dari PDF
        if total_pages > 0:
            try:
                # Ekstrak teks langsung dari PDF jika halaman tertentu diminta
                if page_num is not None and 0 <= page_num < total_pages:
                    page = pdf.pages[page_num]
                    direct_text = page.extract_text()
                    if direct_text and len(direct_text.strip()) > 50:
                        logger.info(f"Berhasil mengekstrak teks langsung dari PDF halaman {page_num}")
                        
                        # Simpan ke cache
                        OCR_CACHE[cache_key] = {
                            'text': direct_text,
                            'confidence': 90.0,
                            'total_pages': total_pages,
                            'timestamp': datetime.now()
                        }
                        
                        return direct_text, time.time() - start_time, 90.0, total_pages
            except Exception as e:
                logger.warning(f"Tidak dapat mengekstrak teks langsung dari PDF: {e}")
        
        # Ekstrak dengan subprocess (lebih cepat dan handal)
        image_path, total_pages = extract_pdf_with_subprocess(pdf_path, page_num)
        
        if image_path:
            # Mode OCR berdasarkan kualitas
            ocr_mode = 'fast' if quality == 'low' else ('accurate' if quality == 'high' else 'standard')
            
            # Optimasi gambar untuk mobile
            optimized_path = optimize_image_for_mobile(image_path, quality)
            
            # Preprocess gambar
            processed_path = preprocess_image(optimized_path, 
                                             'fast' if quality == 'low' else ('quality' if quality == 'high' else 'balanced'))
            
            # OCR halaman
            text, proc_time, confidence = perform_ocr(processed_path, language, mode=ocr_mode)
            
            # Bersihkan file temporary
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                if optimized_path != image_path and os.path.exists(optimized_path):
                    os.remove(optimized_path)
                if processed_path != optimized_path and os.path.exists(processed_path):
                    os.remove(processed_path)
                # Hapus direktori temp (parent dari image_path)
                temp_dir = os.path.dirname(image_path)
                if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Error membersihkan file temporary: {e}")
            
            # Hitung waktu pemrosesan total
            processing_time = time.time() - start_time
            
            # Simpan ke cache
            OCR_CACHE[cache_key] = {
                'text': text,
                'confidence': confidence,
                'total_pages': total_pages,
                'timestamp': datetime.now()
            }
            
            return text, processing_time, confidence, total_pages
        else:
            # Fallback ke PyPDF2 jika ekstraksi gambar gagal
            if total_pages > 0:
                try:
                    if page_num is not None and 0 <= page_num < total_pages:
                        text = pdf.pages[page_num].extract_text()
                    else:
                        # Ekstrak dari beberapa halaman pertama
                        text = ""
                        for i in range(min(total_pages, 3)):
                            text += pdf.pages[i].extract_text() + "\n\n"
                    
                    # Simpan ke cache
                    OCR_CACHE[cache_key] = {
                        'text': text,
                        'confidence': 70.0,
                        'total_pages': total_pages,
                        'timestamp': datetime.now()
                    }
                    
                    return text, time.time() - start_time, 70.0, total_pages
                except Exception as inner_e:
                    logger.error(f"Gagal mengekstrak teks dengan PyPDF2: {inner_e}")
            
            return "Tidak dapat memproses PDF.", time.time() - start_time, 0, total_pages
            
    except Exception as e:
        logger.error(f"Error selama ekstraksi PDF: {e}")
        return f"Error saat memproses PDF: {str(e)}", time.time() - start_time, 0, 0

# ===== RINGKAS TEKS =====

class TextSummarizer:
    """Kelas untuk meringkas teks - versi dioptimasi untuk mobile"""
    
    def __init__(self):
        # Daftar kata stopwords dasar untuk Bahasa Indonesia dan Inggris
        self.stop_words = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
                          "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
                          "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
                          "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
                          "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
                          "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
                          "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my",
                          "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
                          "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
                          "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                          "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
                          "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
                          "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
                          "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
                          "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                          "you're", "you've", "your", "yours", "yourself", "yourselves",
                          # Stopwords Bahasa Indonesia
                          "yang", "dan", "di", "ini", "itu", "dari", "dalam", "tidak", "dengan", "akan",
                          "pada", "juga", "saya", "ke", "bisa", "untuk", "adalah"}
    
    def summarize(self, text, ratio=0.3, max_sentences=None):
        """Ringkas teks dengan metode ekstraktif yang dioptimasi untuk mobile"""
        if not text or len(text.strip()) < 100:
            return text
            
        try:
            # Pecah teks menjadi kalimat
            sentences = self.split_into_sentences(text)
            
            # Jika kalimat terlalu sedikit, kembalikan teks asli
            if len(sentences) <= 3:
                return text
                
            # Hitung jumlah kalimat untuk ringkasan
            if max_sentences:
                num_sentences = min(max_sentences, len(sentences))
            else:
                num_sentences = max(1, int(len(sentences) * ratio))
            
            # Hitung skor kalimat
            scores = self.score_sentences(sentences)
            
            # Pilih kalimat terbaik
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
            ranked_indices.sort()  # Urutkan kembali berdasarkan posisi
            
            # Buat ringkasan
            summary = " ".join([sentences[i] for i in ranked_indices])
            
            return summary
        except Exception as e:
            logger.error(f"Error dalam summarization: {e}")
            return text

    def split_into_sentences(self, text):
        """Pecah teks menjadi kalimat menggunakan regex - optimasi performa"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def score_sentences(self, sentences):
        """Beri skor pada kalimat berdasarkan tingkat kepentingan"""
        # Dapatkan frekuensi kata
        word_freq = defaultdict(int)
        
        for sentence in sentences:
            for word in re.findall(r'\w+', sentence.lower()):
                if word not in self.stop_words:
                    word_freq[word] += 1
        
        # Normalisasi frekuensi
        max_freq = max(word_freq.values()) if word_freq else 1
        word_freq = {word: freq/max_freq for word, freq in word_freq.items()}
        
        # Beri skor kalimat
        scores = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\w+', sentence.lower())
            score = 0
            
            # Tambahkan skor kata
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
            
            # Normalisasi berdasarkan panjang kalimat
            if words:
                score = score / len(words)
                
            # Bonus posisi (kalimat awal/akhir lebih penting)
            if i < len(sentences) * 0.2 or i > len(sentences) * 0.8:
                score *= 1.25
                
            scores.append(score)
            
        return scores

# Inisialisasi summarizer
summarizer = TextSummarizer()

# ===== API ROUTES =====

@app.route('/')
def home():
    """API home dengan daftar endpoint"""
    return jsonify({
        "status": "success",
        "message": "Mobile OCR API v2.0 - Optimized for mobile",
        "endpoints": [
            "/api/ocr/v2/health - Health check",
            "/api/ocr/v2/session/create - Create session",
            "/api/ocr/v2/session/{id}/upload - Upload document",
            "/api/ocr/v2/session/{id}/process - Process document (sync or async)",
            "/api/ocr/v2/session/{id}/result - Get processing result",
            "/api/ocr/v2/session/{id}/page/{page} - Get specific page",
            "/api/ocr/v2/session/{id}/summary - Get text summary",
            "/api/ocr/v2/recognize - Quick OCR without session"
        ]
    })

@app.route('/api/ocr/v2/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Cek Tesseract
        version = pytesseract.get_tesseract_version()
        
        # Cek Poppler
        try:
            pdftoppm_path = r'C:\Program Files\poppler\Library\bin\pdftoppm.exe'
            if not os.path.exists(pdftoppm_path):
                pdftoppm_path = shutil.which('pdftoppm')
                
            if pdftoppm_path:
                poppler_version = subprocess.run([pdftoppm_path, '-v'], 
                                               capture_output=True, text=True).stderr.strip()
            else:
                poppler_version = "Not found"
        except Exception as e:
            poppler_version = f"Error checking: {str(e)}"
            
        return jsonify({
            "status": "success",
            "message": "OCR API v2 is healthy",
            "tesseract_version": str(version),
            "poppler_version": poppler_version,
            "active_sessions": len(SESSION_CACHE),
            "cached_results": len(OCR_CACHE),
            "workers": MAX_WORKERS
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }), 500

@app.route('/api/ocr/v2/session/create', methods=['POST'])
def create_session_endpoint():
    """Buat session baru"""
    try:
        session_id = create_session()
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "message": "Session created successfully",
            "expires_in": f"{SESSION_TIMEOUT} minutes"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to create session: {str(e)}"
        }), 500

@app.route('/api/ocr/v2/session/<session_id>/upload', methods=['POST'])
def upload_document(session_id):
    """
    Upload dokumen ke session
    
    Request (form-data):
        - file: Dokumen (gambar atau PDF)
        - language: Bahasa OCR (default: eng)
        - quality: Kualitas OCR (low, medium, high)
    
    Response:
        - file_id: ID file yang diupload
    """
    # Cek apakah session ada
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Cek apakah file ada dalam request
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file part in the request"
        }), 400
    
    file = request.files['file']
    
    # Cek apakah file dipilih
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No selected file"
        }), 400
    
    # Cek apakah file diizinkan
    if file and allowed_file(file.filename):
        try:
            # Parameter
            language = request.form.get('language', 'eng')
            quality = request.form.get('quality', 'medium')
            
            # Validasi quality
            if quality not in ['low', 'medium', 'high']:
                quality = 'medium'
            
            # Simpan file
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
            file.save(filepath)
            
            # Data file awal
            file_data = {
                "file_id": file_id,
                "filename": filename,
                "filepath": filepath,
                "language": language,
                "quality": quality,
                "upload_time": datetime.now(),
                "is_pdf": filename.lower().endswith('.pdf'),
                "processed": False
            }
            
            # Tambahkan file ke session
            session_data["files"][file_id] = file_data
            
            # Atur sebagai file saat ini
            session_data["current"]["file_id"] = file_id
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "message": "File uploaded successfully",
                "is_pdf": file_data["is_pdf"],
                "filename": filename
            })
        
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error uploading file: {str(e)}"
            }), 500
    
    return jsonify({
        "status": "error",
        "message": "File type not allowed"
    }), 400

@app.route('/api/ocr/v2/session/<session_id>/process', methods=['POST'])
def process_document(session_id):
    """
    Proses dokumen dalam session (sinkron atau asinkron)
    
    Request (JSON):
        - file_id: ID file (default: file saat ini)
        - page: Nomor halaman untuk PDF (default: 0)
        - async: Boolean untuk pemrosesan asinkron (default: false)
    
    Response:
        - Teks hasil ekstraksi (jika sinkron)
        - ID task (jika asinkron)
    """
    # Cek session
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Data request
    data = request.json or {}
    
    # File ID
    file_id = data.get('file_id', session_data["current"]["file_id"])
    if not file_id or file_id not in session_data["files"]:
        return jsonify({
            "status": "error",
            "message": "Invalid file ID"
        }), 400
    
    # Data file
    file_data = session_data["files"][file_id]
    
    # Parameter
    page = data.get('page', 0)
    async_process = data.get('async', False)
    
    try:
        # Untuk pemrosesan asinkron
        if async_process:
            task_id = f"{file_id}_{page}"
            
            if file_data["is_pdf"]:
                # Tambahkan task ke queue
                worker_queue.put((
                    extract_text_from_pdf,
                    (file_data["filepath"], file_data["language"], page, file_data["quality"]),
                    {},
                    task_id,
                    session_id
                ))
            else:
                # Image processing
                # Optimize dan preprocess dilakukan di worker
                worker_queue.put((
                    lambda filepath, language, quality: (lambda optimized, processed: perform_ocr(
                        processed, language, mode='fast' if quality == 'low' else ('accurate' if quality == 'high' else 'standard')
                    ))(
                        optimize_image_for_mobile(filepath, quality),
                        preprocess_image(optimize_image_for_mobile(filepath, quality), 
                                        'fast' if quality == 'low' else ('quality' if quality == 'high' else 'balanced'))
                    ),
                    (file_data["filepath"], file_data["language"], file_data["quality"]),
                    {},
                    task_id,
                    session_id
                ))
            
            # Simpan status di session
            session_data["results"][task_id] = {
                "status": "processing",
                "timestamp": datetime.now()
            }
            
            return jsonify({
                "status": "success",
                "message": "Document processing started",
                "task_id": task_id,
                "async": True
            })
            
        else:
            # Pemrosesan sinkron
            if file_data["is_pdf"]:
                # PDF processing
                text, processing_time, confidence, total_pages = extract_text_from_pdf(
                    file_data["filepath"], 
                    file_data["language"],
                    page,
                    file_data["quality"]
                )
                
                # Update session
                if "pages" not in file_data:
                    file_data["pages"] = {}
                    file_data["total_pages"] = total_pages
                    
                # Simpan teks halaman
                file_data["pages"][page] = {
                    "text": text,
                    "confidence": confidence
                }
                
                # Update posisi saat ini
                session_data["current"]["file_id"] = file_id
                session_data["current"]["page"] = page
                
                return jsonify({
                    "status": "success",
                    "file_id": file_id,
                    "page": page,
                    "total_pages": total_pages,
                    "text": text,
                    "confidence": confidence,
                    "processing_time_ms": processing_time * 1000
                })
                
            else:
                # Image processing
                # Optimize gambar
                optimized_path = optimize_image_for_mobile(file_data["filepath"], file_data["quality"])
                
                # Preprocess gambar
                preprocess_type = 'fast' if file_data["quality"] == 'low' else \
                                ('quality' if file_data["quality"] == 'high' else 'balanced')
                processed_path = preprocess_image(optimized_path, preprocess_type)
                
                # Mode OCR
                ocr_mode = 'fast' if file_data["quality"] == 'low' else \
                          ('accurate' if file_data["quality"] == 'high' else 'standard')
                
                # Lakukan OCR
                text, processing_time, confidence = perform_ocr(
                    processed_path, 
                    file_data["language"],
                    mode=ocr_mode
                )
                
                # Simpan hasil
                file_data["text"] = text
                file_data["confidence"] = confidence
                file_data["processed"] = True
                
                # Bersihkan file temporary
                try:
                    if optimized_path != file_data["filepath"]:
                        os.remove(optimized_path)
                    if processed_path != optimized_path:
                        os.remove(processed_path)
                except:
                    pass
                
                return jsonify({
                    "status": "success",
                    "file_id": file_id,
                    "text": text,
                    "confidence": confidence,
                    "processing_time_ms": processing_time * 1000
                })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing document: {str(e)}"
        }), 500

@app.route('/api/ocr/v2/session/<session_id>/result', methods=['GET'])
def get_result(session_id):
    """
    Dapatkan hasil pemrosesan dari task asinkron
    
    Query params:
        - task_id: ID dari task yang dijalankan secara asinkron
    
    Response:
        - Hasil pemrosesan jika sudah selesai
        - Status task jika masih diproses
    """
    # Cek session
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Dapatkan task_id
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({
            "status": "error",
            "message": "Missing task_id parameter"
        }), 400
    
    # Cek result
    if task_id not in session_data.get("results", {}):
        return jsonify({
            "status": "error",
            "message": "Invalid task ID"
        }), 404
    
    # Dapatkan status
    result_data = session_data["results"][task_id]
    
    # Jika masih diproses
    if result_data["status"] == "processing":
        return jsonify({
            "status": "processing",
            "message": "Document is still being processed",
            "task_id": task_id
        })
    
    # Jika error
    elif result_data["status"] == "error":
        return jsonify({
            "status": "error",
            "message": result_data.get("error", "Unknown error during processing")
        }), 500
    
    # Jika sukses
    elif result_data["status"] == "success":
        # Untuk PDF
        if len(task_id.split('_')) == 2:
            file_id, page = task_id.split('_')
            page = int(page)
            
            if file_id in session_data["files"] and session_data["files"][file_id]["is_pdf"]:
                # Simpan ke struktur file_data
                file_data = session_data["files"][file_id]
                if "pages" not in file_data:
                    file_data["pages"] = {}
                
                # Ambil hasil dari pdf text extraction
                text, _, confidence, total_pages = result_data["result"]
                
                # Simpan teks halaman
                file_data["pages"][page] = {
                    "text": text,
                    "confidence": confidence
                }
                file_data["total_pages"] = total_pages
                
                return jsonify({
                    "status": "success",
                    "file_id": file_id,
                    "page": page,
                    "total_pages": total_pages,
                    "text": text,
                    "confidence": confidence
                })
        
        # Untuk gambar
        else:
            file_id = task_id
            text, _, confidence = result_data["result"]
            
            # Simpan hasil ke file_data
            if file_id in session_data["files"]:
                file_data = session_data["files"][file_id]
                file_data["text"] = text
                file_data["confidence"] = confidence
                file_data["processed"] = True
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "text": text,
                "confidence": confidence
            })
    
    # Default
    return jsonify({
        "status": "unknown",
        "message": "Unknown task status"
    }), 500

@app.route('/api/ocr/v2/session/<session_id>/page/<int:page_num>', methods=['GET'])
def get_page(session_id, page_num):
    """
    Dapatkan halaman tertentu dari PDF
    
    Path params:
        - session_id: ID session
        - page_num: Nomor halaman
    
    Query params:
        - file_id: ID file (default: file saat ini)
        - quality: Kualitas OCR (low, medium, high)
    
    Response:
        - Teks dari halaman yang diminta
    """
    # Cek session
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # File ID
    file_id = request.args.get('file_id', session_data["current"]["file_id"])
    if not file_id or file_id not in session_data["files"]:
        return jsonify({
            "status": "error",
            "message": "Invalid file ID"
        }), 400
    
    # Data file
    file_data = session_data["files"][file_id]
    
    # Pastikan ini PDF
    if not file_data["is_pdf"]:
        return jsonify({
            "status": "error",
            "message": "Not a PDF file"
        }), 400
    
    # Parameter quality
    quality = request.args.get('quality', file_data.get("quality", "medium"))
    if quality not in ['low', 'medium', 'high']:
        quality = 'medium'
    
    try:
        # Cek apakah halaman sudah diproses
        if "pages" in file_data and page_num in file_data["pages"]:
            # Update posisi saat ini
            session_data["current"]["file_id"] = file_id
            session_data["current"]["page"] = page_num
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "page": page_num,
                "total_pages": file_data.get("total_pages", 0),
                "text": file_data["pages"][page_num]["text"],
                "confidence": file_data["pages"][page_num].get("confidence", 0),
                "from_cache": True
            })
        
        # Jika belum, proses halaman
        text, processing_time, confidence, total_pages = extract_text_from_pdf(
            file_data["filepath"], 
            file_data["language"],
            page_num,
            quality
        )
        
        # Update file data
        if "pages" not in file_data:
            file_data["pages"] = {}
            file_data["total_pages"] = total_pages
            
        # Simpan teks halaman
        file_data["pages"][page_num] = {
            "text": text,
            "confidence": confidence
        }
        
        # Update posisi saat ini
        session_data["current"]["file_id"] = file_id
        session_data["current"]["page"] = page_num
        
        return jsonify({
            "status": "success",
            "file_id": file_id,
            "page": page_num,
            "total_pages": total_pages,
            "text": text,
            "confidence": confidence,
            "processing_time_ms": processing_time * 1000
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing page: {str(e)}"
        }), 500

@app.route('/api/ocr/v2/session/<session_id>/summary', methods=['POST'])
def get_summary(session_id):
    """
    Dapatkan ringkasan teks dari dokumen
    
    Request (JSON):
        - file_id: ID file (default: file saat ini)
        - page: Nomor halaman untuk PDF (default: halaman saat ini)
        - ratio: Rasio ringkasan (default: 0.3)
        - max_sentences: Jumlah maksimum kalimat (opsional)
    
    Response:
        - Ringkasan teks
    """
    # Cek session
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Data request
    data = request.json or {}
    
    # File ID
    file_id = data.get('file_id', session_data["current"]["file_id"])
    if not file_id or file_id not in session_data["files"]:
        return jsonify({
            "status": "error",
            "message": "Invalid file ID"
        }), 400
    
    # Data file
    file_data = session_data["files"][file_id]
    
    # Parameter
    current = session_data["current"]
    page = data.get('page', current["page"]) if file_data["is_pdf"] else None
    ratio = float(data.get('ratio', 0.3))
    max_sentences = data.get('max_sentences')
    
    try:
        # Dapatkan teks
        if file_data["is_pdf"]:
            # Untuk PDF
            if "pages" not in file_data or page not in file_data["pages"]:
                return jsonify({
                    "status": "error",
                    "message": f"Page {page} not processed yet"
                }), 400
                
            text = file_data["pages"][page]["text"]
        else:
            # Untuk gambar
            if not file_data.get("processed"):
                return jsonify({
                    "status": "error",
                    "message": "Document not processed yet"
                }), 400
                
            text = file_data.get("text", "")
        
        # Cek apakah teks cukup panjang untuk diringkas
        if len(text.split()) < 30:
            return jsonify({
                "status": "success",
                "message": "Text too short for summarization",
                "original_text": text,
                "summary": text
            })
        
        # Buat ringkasan
        summary = summarizer.summarize(text, ratio, max_sentences)
        
        return jsonify({
            "status": "success",
            "original_text": text,
            "summary": summary,
            "compression_ratio": len(summary) / len(text) if text else 0
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error summarizing text: {str(e)}"
        }), 500

@app.route('/api/ocr/v2/recognize', methods=['POST'])
def quick_recognize():
    """
    OCR cepat tanpa session (untuk kasus sederhana)
    
    Request (form-data):
        - image: File gambar
        - language: Bahasa OCR (default: eng)
        - quality: Kualitas OCR (low, medium, high)
        - include_summary: Sertakan ringkasan teks (default: false)
        
    Response:
        - Teks hasil ekstraksi dan ringkasan opsional
    """
    # Cek apakah file ada dalam request
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image part in the request"
        }), 400
    
    file = request.files['image']
    
    # Cek apakah file dipilih
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No selected file"
        }), 400
    
    # Cek apakah file diizinkan
    if file and allowed_file(file.filename):
        try:
            # Parameter
            language = request.form.get('language', 'eng')
            quality = request.form.get('quality', 'medium')
            include_summary = request.form.get('include_summary', 'false').lower() == 'true'
            
            # Validasi quality
            if quality not in ['low', 'medium', 'high']:
                quality = 'medium'
            
            # Simpan file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"quick_{uuid.uuid4()}_{filename}")
            file.save(filepath)
            
            # Untuk PDF
            if filename.lower().endswith('.pdf'):
                # Ekstrak teks dari halaman pertama
                text, processing_time, confidence, total_pages = extract_text_from_pdf(
                    filepath, language, 0, quality
                )
            else:
                # Untuk gambar
                # Optimize gambar
                optimized_path = optimize_image_for_mobile(filepath, quality)
                
                # Preprocess gambar
                preprocess_type = 'fast' if quality == 'low' else ('quality' if quality == 'high' else 'balanced')
                processed_path = preprocess_image(optimized_path, preprocess_type)
                
                # Mode OCR
                ocr_mode = 'fast' if quality == 'low' else ('accurate' if quality == 'high' else 'standard')
                
                # Lakukan OCR
                text, processing_time, confidence = perform_ocr(processed_path, language, mode=ocr_mode)
                
                # Bersihkan file temporary
                try:
                    if optimized_path != filepath:
                        os.remove(optimized_path)
                    if processed_path != optimized_path:
                        os.remove(processed_path)
                except:
                    pass
            
            # Siapkan response
            response = {
                "status": "success",
                "text": text,
                "confidence": confidence,
                "processing_time_ms": processing_time * 1000
            }
            
            # Tambahkan ringkasan jika diminta
            if include_summary and len(text.split()) >= 30:
                summary = summarizer.summarize(text, 0.3, 5)  # Max 5 kalimat untuk mobile
                response["summary"] = summary
            
            # Bersihkan
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error performing OCR: {str(e)}"
            }), 500
    
    return jsonify({
        "status": "error",
        "message": "File type not allowed"
    }), 400

# ===== MAIN =====

# Jalankan aplikasi
if __name__ == '__main__':
    # Inisialisasi
    poppler_path = register_poppler_path()
    if poppler_path:
        logger.info(f"Poppler berhasil didaftarkan dari: {poppler_path}")
    
    # Mulai worker threads
    start_workers()
    logger.info(f"Started {MAX_WORKERS} worker threads")
    
    # Mulai scheduler pembersihan
    schedule_cleanup()
    
    # Bersihkan folder upload saat startup
    for file in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
        except:
            pass
    
    # Jalankan app
    logger.info("Starting Mobile OCR API v2.0 optimized for mobile")
    app.run(host='0.0.0.0', port=5000, debug=True)