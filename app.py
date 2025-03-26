import os
import sys
import time
import uuid
import math
import logging
import re
import json
import numpy as np
import cv2
import pytesseract
import base64
import io
import gc
import shutil
import subprocess
import glob
import tempfile
from datetime import datetime, timedelta
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from flask_cors import CORS
from collections import defaultdict, OrderedDict

# Fungsi untuk memeriksa instalasi Poppler
def check_poppler_installation():
    """Periksa instalasi Poppler dan tampilkan info berguna untuk debugging"""
    logger.info("===== CHECKING POPPLER INSTALLATION =====")
    
    # Cek apakah executable ada di PATH
    pdftoppm_path = shutil.which('pdftoppm')
    
    if pdftoppm_path:
        logger.info(f"pdftoppm found in PATH: {pdftoppm_path}")
        try:
            result = subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True)
            logger.info(f"pdftoppm version: {result.stderr.strip()}")
        except Exception as e:
            logger.error(f"Error running pdftoppm: {e}")
    else:
        logger.warning("pdftoppm not found in PATH")
        
    # Cek di lokasi spesifik
    poppler_locations = [
        r'C:\Program Files\poppler\Library\bin',
        r'C:\Program Files\poppler\bin',
        r'C:\poppler\bin',
        r'C:\Program Files (x86)\poppler\bin'
    ]
    
    for location in poppler_locations:
        logger.info(f"Checking {location}...")
        pdftoppm_exe = os.path.join(location, 'pdftoppm.exe')
        if os.path.exists(pdftoppm_exe):
            logger.info(f"pdftoppm found at: {pdftoppm_exe}")
            try:
                result = subprocess.run([pdftoppm_exe, '-v'], capture_output=True, text=True)
                logger.info(f"Version: {result.stderr.strip()}")
            except Exception as e:
                logger.error(f"Error running {pdftoppm_exe}: {e}")
        else:
            logger.warning(f"pdftoppm not found at {pdftoppm_exe}")
    
    # Periksa environment PATH
    env_path = os.environ.get('PATH', '')
    logger.info(f"Current PATH: {env_path}")
    
    # Periksa pdf2image
    try:
        import pdf2image
        logger.info(f"pdf2image version: {pdf2image.__version__}")
    except ImportError as e:
        logger.error(f"pdf2image import error: {e}")
    except Exception as e:
        logger.error(f"Error checking pdf2image: {e}")
    
    logger.info("===== POPPLER CHECK FINISHED =====")

# Fungsi untuk mendaftarkan Poppler ke environment PATH
def register_poppler_path():
    """
    Mendaftarkan path Poppler ke environment PATH
    """
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Path Poppler yang diketahui
    poppler_paths = [
        r'C:\Program Files\poppler\Library\bin',  # Lokasi instalasi saat ini
        r'C:\Program Files\poppler\bin',          # Lokasi standar
        r'C:\Program Files (x86)\poppler\bin',
        r'C:\poppler\bin',
        # Tambahkan path lain sesuai kebutuhan
    ]
    
    # Periksa path yang ada
    for path in poppler_paths:
        if os.path.exists(path):
            # Cek apakah pdftoppm ada di lokasi tersebut
            if os.path.exists(os.path.join(path, 'pdftoppm.exe')):
                logger.info(f"Menemukan Poppler di: {path}")
                
                # Tambahkan ke PATH environment
                if path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                    logger.info(f"Menambahkan {path} ke PATH environment")
                    
                return path
            else:
                logger.warning(f"Path {path} ada, tapi tidak berisi pdftoppm.exe")
    
    logger.warning("Tidak dapat menemukan Poppler di path yang diketahui!")
    return None

# Ekstrak halaman PDF menggunakan subprocess
def extract_pdf_page_with_subprocess(pdf_path, page_num):
    """
    Ekstrak halaman PDF menggunakan subprocess langsung ke pdftoppm.
    
    Args:
        pdf_path: Path ke file PDF
        page_num: Nomor halaman yang ingin diekstrak (dimulai dari 0)
    
    Returns:
        Path ke file gambar hasil, jumlah halaman total
    """
    logger.info(f"Mencoba ekstrak PDF dengan subprocess langsung")
    
    # Cek jumlah halaman dulu dengan pdfinfo
    try:
        pdfinfo_exe = r'C:\Program Files\poppler\Library\bin\pdfinfo.exe'
        if not os.path.exists(pdfinfo_exe):
            pdfinfo_exe = 'pdfinfo'  # coba gunakan dari PATH
            
        result = subprocess.run([pdfinfo_exe, pdf_path], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error(f"pdfinfo error: {result.stderr}")
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
                
        logger.info(f"PDF memiliki {total_pages} halaman (dari pdfinfo)")
    except Exception as e:
        logger.error(f"Error saat mendapatkan jumlah halaman: {e}")
        total_pages = 0
    
    # Cek validitas nomor halaman
    if page_num is not None and (page_num < 0 or page_num >= total_pages):
        return None, total_pages
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    output_prefix = os.path.join(temp_dir, "page")
    
    try:
        # Path ke pdftoppm
        pdftoppm_exe = r'C:\Program Files\poppler\Library\bin\pdftoppm.exe'
        if not os.path.exists(pdftoppm_exe):
            pdftoppm_exe = 'pdftoppm'  # coba gunakan dari PATH
        
        # Jalankan pdftoppm untuk mengkonversi halaman PDF ke JPG
        cmd = [
            pdftoppm_exe,
            "-jpeg",
            "-r", "200"  # 200 DPI untuk kualitas yang baik
        ]
        
        # Tambahkan range halaman jika ada
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
            logger.error(f"Tidak ada file yang dihasilkan di {temp_dir}")
            return None, total_pages
            
        # Urutkan file (meskipun seharusnya hanya ada satu)
        output_files.sort()
        
        logger.info(f"Berhasil menghasilkan {len(output_files)} file: {output_files}")
        return output_files[0], total_pages
        
    except Exception as e:
        logger.error(f"Error saat mengkonversi PDF: {e}")
        return None, total_pages

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)  # Enable CORS for mobile app access

# Upload configuration
UPLOAD_FOLDER = '/tmp/ocr_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Session cache for document processing
# Key: session_id, Value: document data
SESSION_CACHE = {}
SESSION_TIMEOUT = 30  # Minutes

# Configure Tesseract - Prioritize explicit path for performance
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif os.path.exists('/usr/bin/tesseract'):
    # Linux/Colab path
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
elif os.path.exists('/usr/local/bin/tesseract'):
    # macOS path with Homebrew
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
else:
    # Fallback
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Utility functions for session management
def create_session():
    """Create a new processing session"""
    session_id = str(uuid.uuid4())
    SESSION_CACHE[session_id] = {
        'created_at': datetime.now(),
        'last_accessed': datetime.now(),
        'files': {},
        'current': {
            'file_id': None,
            'page': 0,
            'paragraph': 0,
            'sentence': 0,
            'column': 0
        }
    }
    return session_id

def get_session(session_id):
    """Get session data and update last accessed time"""
    if session_id in SESSION_CACHE:
        SESSION_CACHE[session_id]['last_accessed'] = datetime.now()
        return SESSION_CACHE[session_id]
    return None

def cleanup_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in SESSION_CACHE.items():
        if current_time - session_data['last_accessed'] > timedelta(minutes=SESSION_TIMEOUT):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        # Clean up any associated files
        if 'files' in SESSION_CACHE[session_id]:
            for file_id in SESSION_CACHE[session_id]['files']:
                file_data = SESSION_CACHE[session_id]['files'][file_id]
                try:
                    if 'filepath' in file_data and os.path.exists(file_data['filepath']):
                        os.remove(file_data['filepath'])
                except Exception as e:
                    logger.error(f"Error removing file: {e}")
        
        # Remove session
        del SESSION_CACHE[session_id]
        logger.info(f"Removed expired session: {session_id}")

# Run cleanup periodically
def schedule_cleanup():
    cleanup_sessions()
    # Schedule next cleanup
    from threading import Timer
    Timer(300, schedule_cleanup).start()  # Every 5 minutes

# Start cleanup scheduler
schedule_cleanup()

# Utility function to check allowed files
def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Optimize images for mobile processing
def optimize_image_for_mobile(image_path, max_dimension=1500):
    """
    Resize image if too large for efficient mobile processing
    
    Args:
        image_path: Path to the image
        max_dimension: Maximum width or height
        
    Returns:
        Path to optimized image (may be the same if no resizing needed)
    """
    try:
        # Open image and check dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Only resize if image is larger than max_dimension
            if width > max_dimension or height > max_dimension:
                # Calculate new dimensions preserving aspect ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                # Resize image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save optimized image
                optimized_path = f"{image_path}_optimized.jpg"
                img.save(optimized_path, 'JPEG', quality=85)
                
                logger.info(f"Optimized image from {width}x{height} to {new_width}x{new_height}")
                return optimized_path
            
            return image_path
    except Exception as e:
        logger.error(f"Error optimizing image: {e}")
        return image_path

class Summarizer:
    """Class for summarizing text using extractive methods"""

    def __init__(self):
        self.stop_words = set()
        try:
            # Try to import NLTK and download resources
            import nltk
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
            # Add Indonesian stopwords if available
            try:
                self.stop_words.update(set(stopwords.words('indonesian')))
            except:
                pass
        except:
            # Fallback stop words if NLTK is not available
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
                              # Indonesian stopwords
                              "yang", "dan", "di", "ini", "itu", "dari", "dalam", "tidak", "dengan", "akan",
                              "pada", "juga", "saya", "ke", "bisa", "untuk", "adalah"}

    def summarize(self, text, ratio=0.3):
        """
        Generate a summary of the provided text
        
        Args:
            text: The text to summarize
            ratio: The ratio of the original text to keep (0.1-0.9)
            
        Returns:
            A summary of the text
        """
        if not text or len(text.strip()) < 100:
            return text
            
        try:
            # Split text into sentences
            sentences = self.split_into_sentences(text)
            
            # If few sentences, return original
            if len(sentences) <= 3:
                return text
                
            # Calculate number of sentences for summary
            num_sentences = max(1, int(len(sentences) * ratio))
            
            # Calculate sentence scores
            scores = self.score_sentences(sentences)
            
            # Select top sentences
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
            ranked_indices.sort()  # Re-sort by position
            
            # Create summary
            summary = " ".join([sentences[i] for i in ranked_indices])
            
            return summary
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return text

    def split_into_sentences(self, text):
        """Split text into sentences using regex for performance"""
        # Simple but effective sentence splitting for mobile performance
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def score_sentences(self, sentences):
        """Score sentences by importance"""
        # Get word frequencies
        word_freq = defaultdict(int)
        
        for sentence in sentences:
            for word in re.findall(r'\w+', sentence.lower()):
                if word not in self.stop_words:
                    word_freq[word] += 1
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        word_freq = {word: freq/max_freq for word, freq in word_freq.items()}
        
        # Score sentences
        scores = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\w+', sentence.lower())
            score = 0
            
            # Sum word scores
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
            
            # Normalize by sentence length
            if words:
                score = score / len(words)
                
            # Position bonus (first/last sentences more important)
            if i < len(sentences) * 0.2 or i > len(sentences) * 0.8:
                score *= 1.25
                
            scores.append(score)
            
        return scores

# Initialize summarizer
summarizer = Summarizer()

def preprocess_image(image_path, preprocess_type='default'):
    """
    Preprocess image to improve OCR accuracy - optimized for mobile
    
    Args:
        image_path: Path to image
        preprocess_type: Type of preprocessing
        
    Returns:
        Path to processed image
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            # Try with PIL if OpenCV fails
            try:
                pil_image = Image.open(image_path)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                # Save as temporary file
                temp_jpg = f"{image_path}_temp.jpg"
                pil_image.save(temp_jpg)
                image = cv2.imread(temp_jpg)
                os.remove(temp_jpg)
            except Exception as e:
                logger.error(f"Error converting image: {e}")
                return image_path

        if image is None:
            return image_path
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing based on type
        if preprocess_type == 'fast':
            # Fastest processing for mobile - basic threshold
            _, processed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        elif preprocess_type == 'balanced':
            # Balanced preprocessing - good results with reasonable speed
            processed = cv2.GaussianBlur(gray, (3, 3), 0)
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2)
        
        elif preprocess_type == 'quality':
            # High quality preprocessing - slower but better results
            # Use blur + Otsu's threshold
            processed = cv2.GaussianBlur(gray, (5, 5), 0)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to improve quality
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
        else:  # default
            # Standard preprocessing - good balance
            processed = cv2.GaussianBlur(gray, (3, 3), 0)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save processed image
        processed_path = f"{image_path}_processed.jpg"
        cv2.imwrite(processed_path, processed)
        
        return processed_path
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return image_path

def perform_ocr(image_path, language='eng', config=''):
    """
    Perform OCR optimized for mobile
    
    Args:
        image_path: Path to image
        language: OCR language (default: eng)
        config: Additional Tesseract configuration
    
    Returns:
        Extracted text, processing time, confidence
    """
    start_time = time.time()
    
    try:
        # Read image
        image = Image.open(image_path)
        
        # Set config for OCR - optimize for speed on mobile
        custom_config = f'-l {language} --oem 1 --psm 3'
        if config:
            custom_config += f' {config}'
            
        # Run OCR
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Get confidence level - only if needed (this is slower)
        try:
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            # Fallback if confidence calculation fails
            avg_confidence = 80.0  # Assume reasonable confidence
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return text, processing_time, avg_confidence
    
    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        return str(e), time.time() - start_time, 0

def extract_text_from_pdf(pdf_path, language='eng', page_num=None):
    """
    Extract text from PDF with page selection for mobile
    
    Args:
        pdf_path: Path to PDF file
        language: OCR language
        page_num: Specific page number to process (None for all pages)
    
    Returns:
        Extracted text, processing time, confidence, page count
    """
    start_time = time.time()
    
    # DEBUGGING: Tampilkan PATH saat ini
    logger.info(f"Current PATH: {os.environ.get('PATH', '')}")
    
    try:
        # Coba mendapatkan jumlah halaman dengan PyPDF2 terlebih dahulu
        try:
            from PyPDF2 import PdfReader
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
            logger.info(f"PDF memiliki {total_pages} halaman berdasarkan PyPDF2")
        except Exception as e:
            logger.error(f"Error membaca PDF dengan PyPDF2: {e}")
            total_pages = 0
        
        # Coba ekstrak teks langsung dari PDF jika memungkinkan
        extracted_text = ""
        if total_pages > 0:
            try:
                # Ekstrak teks langsung dari PDF jika halaman tertentu diminta
                if page_num is not None and 0 <= page_num < total_pages:
                    page = pdf.pages[page_num]
                    direct_text = page.extract_text()
                    if direct_text and len(direct_text.strip()) > 50:  # Jika teks cukup bermakna
                        logger.info(f"Berhasil mengekstrak teks langsung dari PDF halaman {page_num}")
                        return direct_text, time.time() - start_time, 90.0, total_pages
                else:
                    # Jika tidak ada halaman tertentu, coba ekstrak dari halaman pertama
                    page = pdf.pages[0]
                    direct_text = page.extract_text()
                    if direct_text and len(direct_text.strip()) > 50:
                        logger.info("Berhasil mengekstrak teks langsung dari PDF halaman pertama")
                        return direct_text, time.time() - start_time, 90.0, total_pages
            except Exception as e:
                logger.warning(f"Tidak dapat mengekstrak teks langsung dari PDF: {e}")
        
        # Coba konversi PDF ke gambar dengan pdf2image
        try:
            from pdf2image import convert_from_path
            
            # PENDEKATAN 1: Gunakan path langsung ke executable Poppler
            pdftoppm_path = r'C:\Program Files\poppler\Library\bin\pdftoppm.exe'
            pdfinfo_path = r'C:\Program Files\poppler\Library\bin\pdfinfo.exe'
            
            if os.path.exists(pdftoppm_path) and os.path.exists(pdfinfo_path):
                poppler_path = os.path.dirname(pdftoppm_path)
                logger.info(f"Menggunakan Poppler dari path langsung: {poppler_path}")
                
                # Tambahkan poppler ke PATH env khusus untuk proses ini
                if poppler_path not in os.environ['PATH']:
                    os.environ['PATH'] = poppler_path + os.pathsep + os.environ['PATH']
                    logger.info(f"PATH diupdate: {os.environ['PATH']}")
                
                # Tentukan halaman yang akan diproses
                if page_num is not None:
                    # Proses hanya halaman tertentu
                    if page_num < 0 or page_num >= total_pages:
                        return f"Nomor halaman tidak valid. PDF memiliki {total_pages} halaman.", 0, 0, total_pages
                        
                    # Coba dengan opsi lengkap untuk diagnostik
                    try:
                        logger.info(f"Mencoba konversi PDF dengan path poppler: {poppler_path}")
                        pages = convert_from_path(
                            pdf_path, 
                            200, 
                            first_page=page_num+1, 
                            last_page=page_num+1, 
                            poppler_path=poppler_path,
                            use_pdftocairo=False,  # Gunakan pdftoppm (lebih stabil di Windows)
                            strict=False           # Mode tidak strict
                        )
                        logger.info("Berhasil mengkonversi PDF ke gambar")
                    except Exception as conv_error:
                        logger.error(f"Error konversi dengan opsi lengkap: {conv_error}")
                        
                        # PENDEKATAN 2: Coba dengan subprocess langsung
                        try:
                            logger.info("Mencoba konversi dengan subprocess langsung")
                            import subprocess
                            import tempfile
                            
                            # Buat direktori temporary
                            temp_dir = tempfile.mkdtemp()
                            output_file = os.path.join(temp_dir, "page")
                            
                            # Jalankan pdftoppm langsung
                            cmd = [pdftoppm_path, "-jpeg", "-f", str(page_num+1), "-l", str(page_num+1), 
                                  "-r", "200", pdf_path, output_file]
                            logger.info(f"Menjalankan command: {' '.join(cmd)}")
                            
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode != 0:
                                logger.error(f"pdftoppm error: {result.stderr}")
                                raise Exception(f"pdftoppm failed: {result.stderr}")
                                
                            # Cari file output
                            output_files = [f for f in os.listdir(temp_dir) if f.startswith("page") and f.endswith(".jpg")]
                            if not output_files:
                                raise Exception("No output files generated")
                                
                            # Load gambar hasil
                            from PIL import Image
                            pages = [Image.open(os.path.join(temp_dir, f)) for f in output_files]
                            logger.info(f"Berhasil konversi dengan subprocess: {len(pages)} halaman")
                            
                        except Exception as sub_error:
                            logger.error(f"Error konversi dengan subprocess: {sub_error}")
                            
                            # PENDEKATAN 3: Coba tanpa menentukan poppler_path
                            try:
                                logger.info("Mencoba konversi tanpa poppler_path")
                                pages = convert_from_path(pdf_path, 200, first_page=page_num+1, last_page=page_num+1)
                                logger.info("Berhasil konversi tanpa poppler_path")
                            except Exception as e:
                                logger.error(f"Semua metode konversi gagal: {e}")
                                return f"Gagal mengkonversi PDF: {str(e)}. Coba pastikan Poppler terinstal dengan benar.", time.time() - start_time, 0, total_pages
                else:
                    # Untuk panggilan awal, proses hanya halaman pertama agar cepat
                    try:
                        logger.info(f"Mencoba konversi PDF halaman awal dengan path poppler: {poppler_path}")
                        pages = convert_from_path(
                            pdf_path, 
                            200, 
                            first_page=1, 
                            last_page=1, 
                            poppler_path=poppler_path,
                            use_pdftocairo=False
                        )
                    except Exception as e:
                        logger.error(f"Error konversi halaman awal: {e}")
                        pages = convert_from_path(pdf_path, 200, first_page=1, last_page=1)
                
                page_range = [page_num] if page_num is not None else [0]
            else:
                logger.warning(f"File pdftoppm tidak ditemukan di {pdftoppm_path}")
                pages = convert_from_path(pdf_path, 200, first_page=1, last_page=1)
                page_range = [0]
            
            # Proses setiap halaman
            texts = []
            total_confidence = 0
            confidence_count = 0
            
            for i, page in enumerate(pages):
                # Simpan halaman sebagai gambar
                page_path = f"{pdf_path}_page_{page_range[i]}.jpg"
                page.save(page_path, 'JPEG')
                
                # Optimasi gambar untuk mobile
                optimized_path = optimize_image_for_mobile(page_path)
                
                # Proses gambar
                processed_path = preprocess_image(optimized_path, 'balanced')
                
                # OCR halaman
                text, _, confidence = perform_ocr(processed_path, language)
                texts.append(text)
                
                # Tambahkan confidence
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1
                
                # Hapus file temporary
                try:
                    os.remove(page_path)
                    if optimized_path != page_path:
                        os.remove(optimized_path)
                    if processed_path != optimized_path:
                        os.remove(processed_path)
                except:
                    pass
            
            # Gabungkan hasil
            full_text = "\n\n".join(texts)
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            processing_time = time.time() - start_time
            
            return full_text, processing_time, avg_confidence, total_pages
            
        except ImportError as import_err:
            logger.error(f"pdf2image tidak tersedia: {import_err}")
            # Jika PDF berhasil dibaca dengan PyPDF2, ekstrak teks sebagai fallback
            if total_pages > 0:
                try:
                    if page_num is not None and 0 <= page_num < total_pages:
                        extracted_text = pdf.pages[page_num].extract_text()
                    else:
                        # Ekstrak dari semua halaman
                        extracted_text = ""
                        for i in range(min(total_pages, 5)):  # Batasi ke 5 halaman pertama
                            extracted_text += pdf.pages[i].extract_text() + "\n\n"
                    
                    return extracted_text, time.time() - start_time, 70.0, total_pages
                except Exception as inner_e:
                    logger.error(f"Gagal mengekstrak teks dengan PyPDF2: {inner_e}")
            
            return "Tidak dapat memproses PDF. Pastikan pdf2image dan poppler terinstal dengan benar.", time.time() - start_time, 0, total_pages
            
    except Exception as e:
        logger.error(f"Error selama ekstraksi PDF: {e}")
        # Coba tentukan jumlah halaman meski ekstraksi gagal
        try:
            from PyPDF2 import PdfReader
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
        except:
            total_pages = 0
            
        return f"Error saat memproses PDF: {str(e)}\nPastikan Poppler terinstal dan berfungsi dengan benar.", time.time() - start_time, 0, total_pages

def analyze_document_structure(image_path):
    """
    Analyze document structure for rich navigation
    
    Args:
        image_path: Path to image
    
    Returns:
        Document structure (pages, paragraphs, sentences, columns)
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            # Try with PIL
            try:
                pil_image = Image.open(image_path)
                img_np = np.array(pil_image.convert('RGB'))
                image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Error reading image: {e}")
                return {"error": "Cannot read image"}
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find text regions (paragraphs)
        # Use connected components for better performance on mobile
        output = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        
        # Filter regions by size
        paragraphs = []
        min_area = width * height * 0.001  # Minimum area relative to image size
        max_area = width * height * 0.8    # Maximum area relative to image size
        
        # Create a visualization image
        viz_image = image.copy()
        
        # Process regions (skip background which is label 0)
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by size and aspect ratio
            if min_area < area < max_area and 0.1 < w/h < 10:
                # Extract region
                roi = gray[y:y+h, x:x+w]
                
                # Convert to PIL for OCR
                pil_roi = Image.fromarray(roi)
                
                # Perform quick OCR to check if it contains text
                text = pytesseract.image_to_string(pil_roi, config='--psm 6')
                
                if text.strip():
                    # Create paragraph object
                    para = {
                        "id": len(paragraphs),
                        "bounds": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "text": text.strip(),
                        "sentences": []
                    }
                    
                    # Split into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    for j, sentence in enumerate(sentences):
                        if sentence.strip():
                            para["sentences"].append({
                                "id": j,
                                "text": sentence.strip()
                            })
                    
                    paragraphs.append(para)
                    
                    # Draw rectangle for visualization
                    cv2.rectangle(viz_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect columns (simple approach: group paragraphs by x-coordinate)
        if paragraphs:
            # Sort paragraphs by x-coordinate
            sorted_by_x = sorted(paragraphs, key=lambda p: p["bounds"]["x"])
            
            # Find column boundaries using gaps in x-coordinates
            x_positions = [p["bounds"]["x"] for p in sorted_by_x]
            column_boundaries = [x_positions[0]]
            
            for i in range(1, len(x_positions)):
                if x_positions[i] - x_positions[i-1] > width * 0.1:  # Gap > 10% of width
                    column_boundaries.append(x_positions[i])
            
            # Assign paragraphs to columns
            columns = []
            for i in range(len(column_boundaries)):
                columns.append([])
            
            for para in paragraphs:
                # Find closest column
                x = para["bounds"]["x"]
                distances = [abs(x - cb) for cb in column_boundaries]
                col_idx = distances.index(min(distances))
                
                # Update paragraph with column info
                para["column"] = col_idx
                
                # Add to column
                columns[col_idx].append(para["id"])
                
                # Draw column indicator
                cv2.line(viz_image, 
                         (column_boundaries[col_idx], 0), 
                         (column_boundaries[col_idx], height), 
                         (0, 0, 255), 1)
        else:
            columns = []
        
        # Save visualization
        viz_path = f"{image_path}_structure.jpg"
        cv2.imwrite(viz_path, viz_image)
        
        # Get base64 of visualization
        with open(viz_path, "rb") as img_file:
            viz_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
        # Clean up
        try:
            os.remove(viz_path)
        except:
            pass
        
        # Create document structure
        structure = {
            "paragraphs": paragraphs,
            "columns": [{
                "id": i,
                "paragraphs": col
            } for i, col in enumerate(columns)],
            "total_paragraphs": len(paragraphs),
            "total_columns": len(columns),
            "visualization": viz_base64
        }
        
        # Add suggested actions based on structure
        suggested_actions = []
        
        if len(paragraphs) > 1:
            suggested_actions.append({
                "type": "read_by_paragraph",
                "description": "Read paragraph by paragraph"
            })
            
        if len(columns) > 1:
            suggested_actions.append({
                "type": "read_by_column",
                "description": "Read column by column"
            })
            
        total_sentences = sum(len(p["sentences"]) for p in paragraphs)
        if total_sentences > 5:
            suggested_actions.append({
                "type": "summarize",
                "description": "Read summary of text"
            })
        
        suggested_actions.append({
            "type": "read_full",
            "description": "Read full text"
        })
        
        structure["suggested_actions"] = suggested_actions
        
        return structure
        
    except Exception as e:
        logger.error(f"Error analyzing document structure: {e}")
        return {"error": f"Error analyzing document: {str(e)}"}

# API Routes

@app.route('/')
def home():
    """API home with endpoints"""
    return jsonify({
        "status": "success",
        "message": "Mobile OCR API",
        "version": "1.1.0",
        "endpoints": [
            "/api/ocr/health - Health check",
            "/api/ocr/session/create - Create processing session",
            "/api/ocr/session/{id}/upload - Upload document",
            "/api/ocr/session/{id}/analyze - Analyze document",
            "/api/ocr/session/{id}/navigate - Navigate through document",
            "/api/ocr/session/{id}/summarize - Summarize current selection",
            "/api/ocr/recognize - Quick OCR (no session)",
            "/api/ocr/languages - Get available languages"
        ]
    })

@app.route('/api/ocr/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        version = pytesseract.get_tesseract_version()
        
        # Check Poppler
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
            "message": "OCR API is healthy",
            "tesseract_version": str(version),
            "poppler_version": poppler_version,
            "active_sessions": len(SESSION_CACHE)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }), 500

@app.route('/api/ocr/session/create', methods=['POST'])
def create_session_endpoint():
    """Create a new processing session"""
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

@app.route('/api/ocr/session/<session_id>/upload', methods=['POST'])
def upload_document(session_id):
    """
    Upload document to session
    
    Request (form-data):
        - file: Document file (image or PDF)
        - language: OCR language (default: eng)
        - optimize: Whether to optimize for mobile (default: true)
    
    Response:
        - file_id: ID of uploaded file
    """
    # Check if session exists
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Check if file exists in request
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file part in the request"
        }), 400
    
    file = request.files['file']
    
    # Check if no file selected
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No selected file"
        }), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        try:
            # Get parameters
            language = request.form.get('language', 'eng')
            optimize = request.form.get('optimize', 'true').lower() == 'true'
            
            # Save file
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
            file.save(filepath)
            
            # Initial file data
            file_data = {
                "file_id": file_id,
                "filename": filename,
                "filepath": filepath,
                "language": language,
                "upload_time": datetime.now(),
                "is_pdf": filename.lower().endswith('.pdf'),
                "processed": False,
                "optimized": False
            }
            
            # Optimize if requested and not a PDF
            if optimize and not file_data["is_pdf"]:
                optimized_path = optimize_image_for_mobile(filepath)
                if optimized_path != filepath:
                    file_data["filepath"] = optimized_path
                    file_data["optimized"] = True
            
            # Add file to session
            session_data["files"][file_id] = file_data
            
            # Set as current file
            session_data["current"]["file_id"] = file_id
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "message": "File uploaded successfully",
                "is_pdf": file_data["is_pdf"],
                "optimized": file_data["optimized"]
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

@app.route('/api/ocr/session/<session_id>/analyze', methods=['POST'])
def analyze_document(session_id):
    """
    Analyze document in session
    
    Request (JSON):
        - file_id: File ID to analyze (default: current file)
        - page: Page number for PDFs (default: 0)
        - full_analysis: Whether to perform full analysis (default: false)
    
    Response:
        - document structure and suggested actions
    """
    # Check if session exists
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Get request data
    data = request.json or {}
    
    # Get file_id (use current if not specified)
    file_id = data.get('file_id', session_data["current"]["file_id"])
    
    if not file_id or file_id not in session_data["files"]:
        return jsonify({
            "status": "error",
            "message": "Invalid file ID"
        }), 400
    
    # Get file data
    file_data = session_data["files"][file_id]
    
    # Get page number (for PDFs)
    page = data.get('page', 0)
    
    # Get analysis level
    full_analysis = data.get('full_analysis', False)
    
    try:
        # Process based on file type
        if file_data["is_pdf"]:
            # PDF processing - coba dengan dua metode
            try:
                # Coba dengan extract_text_from_pdf dulu
                text, processing_time, confidence, total_pages = extract_text_from_pdf(
                    file_data["filepath"], 
                    file_data["language"],
                    page
                )
                
                # Jika berhasil (tidak mengandung error message)
                if not text.startswith("Error") and not text.startswith("Tidak dapat") and not text.startswith("Gagal"):
                    logger.info("Berhasil ekstrak PDF dengan extract_text_from_pdf")
                else:
                    # Jika gagal, coba dengan subprocess
                    logger.info("Mencoba metode alternatif dengan subprocess")
                    image_path, total_pages = extract_pdf_page_with_subprocess(file_data["filepath"], page)
                    
                    if image_path:
                        # Berhasil mengkonversi ke gambar, lakukan OCR
                        processed_path = preprocess_image(image_path, 'balanced')
                        text, proc_time, confidence = perform_ocr(processed_path, file_data["language"])
                        processing_time = proc_time
                        
                        # Hapus file temporary
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                            if processed_path != image_path and os.path.exists(processed_path):
                                os.remove(processed_path)
                        except Exception as e:
                            logger.error(f"Error menghapus file temporary: {e}")
                    else:
                        # Kedua metode gagal, berikan pesan error
                        return jsonify({
                            "status": "error",
                            "message": "Gagal memproses PDF dengan kedua metode. Pastikan Poppler terinstal dengan benar."
                        }), 500
            except Exception as pdf_error:
                logger.error(f"Kedua metode ekstraksi PDF gagal: {pdf_error}")
                return jsonify({
                    "status": "error",
                    "message": f"Error saat memproses PDF: {str(pdf_error)}"
                }), 500
            
            # Update file data
            if "pages" not in file_data:
                file_data["pages"] = {}
                file_data["total_pages"] = total_pages
                
            # Save page text
            file_data["pages"][page] = {
                "text": text,
                "processing_time": processing_time,
                "confidence": confidence,
                "analyzed": False
            }
            
            # Perform structure analysis if requested
            if full_analysis:
                # For PDFs, we need to extract the page first
                try:
                    # Gunakan hasil konversi dari subprocess jika ada
                    if 'image_path' in locals() and image_path:
                        # Analyze structure pada gambar yang sudah dikonversi
                        structure = analyze_document_structure(image_path)
                    else:
                        # Coba ekstrak ulang dengan subprocess
                        image_path, _ = extract_pdf_page_with_subprocess(file_data["filepath"], page)
                        if image_path:
                            structure = analyze_document_structure(image_path)
                        else:
                            # Fallback jika gagal
                            structure = {
                                "suggested_actions": [
                                    {
                                        "type": "read_full",
                                        "description": "Read full text"
                                    },
                                    {
                                        "type": "summarize",
                                        "description": "Read summary of text"
                                    }
                                ]
                            }
                    
                    # Store analysis results
                    file_data["pages"][page]["structure"] = structure
                    file_data["pages"][page]["analyzed"] = True
                    
                    # Clean up
                    try:
                        if 'image_path' in locals() and image_path and os.path.exists(image_path):
                            os.remove(image_path)
                    except:
                        pass
                        
                    # Update response with structure
                    response = {
                        "status": "success",
                        "file_id": file_id,
                        "page": page,
                        "total_pages": total_pages,
                        "text": text,
                        "confidence": confidence,
                        "processing_time_ms": processing_time * 1000,
                        "structure": structure
                    }
                except Exception as struct_error:
                    logger.error(f"Error saat analisis struktur: {struct_error}")
                    response = {
                        "status": "success",
                        "file_id": file_id,
                        "page": page,
                        "total_pages": total_pages,
                        "text": text,
                        "confidence": confidence,
                        "processing_time_ms": processing_time * 1000,
                        "structure": {
                            "suggested_actions": [
                                {
                                    "type": "read_full",
                                    "description": "Read full text"
                                },
                                {
                                    "type": "summarize",
                                    "description": "Read summary of text"
                                }
                            ]
                        }
                    }
            else:
                # Basic response without structure
                response = {
                    "status": "success",
                    "file_id": file_id,
                    "page": page,
                    "total_pages": total_pages,
                    "text": text,
                    "confidence": confidence,
                    "processing_time_ms": processing_time * 1000
                }
            
            # Update current position
            session_data["current"]["file_id"] = file_id
            session_data["current"]["page"] = page
            session_data["current"]["paragraph"] = 0
            session_data["current"]["sentence"] = 0
            session_data["current"]["column"] = 0
            
        else:
            # Image processing (kode asli tidak berubah)
            if not file_data.get("processed"):
                # Preprocess image
                processed_path = preprocess_image(file_data["filepath"], 'balanced')
                
                # Perform OCR
                text, processing_time, confidence = perform_ocr(
                    processed_path, 
                    file_data["language"]
                )
                
                # Store results
                file_data["text"] = text
                file_data["processing_time"] = processing_time
                file_data["confidence"] = confidence
                file_data["processed"] = True
                
                # Clean up
                if processed_path != file_data["filepath"]:
                    try:
                        os.remove(processed_path)
                    except:
                        pass
            
            # Perform structure analysis if requested or not already done
            if full_analysis or not file_data.get("analyzed"):
                structure = analyze_document_structure(file_data["filepath"])
                file_data["structure"] = structure
                file_data["analyzed"] = True
                
                # Update response with structure
                response = {
                    "status": "success",
                    "file_id": file_id,
                    "text": file_data["text"],
                    "confidence": file_data["confidence"],
                    "processing_time_ms": file_data["processing_time"] * 1000,
                    "structure": structure
                }
            else:
                # Use cached structure
                response = {
                    "status": "success",
                    "file_id": file_id,
                    "text": file_data["text"],
                    "confidence": file_data["confidence"],
                    "processing_time_ms": file_data["processing_time"] * 1000,
                    "structure": file_data.get("structure", {})
                }
            
            # Update current position
            session_data["current"]["file_id"] = file_id
            session_data["current"]["page"] = 0
            session_data["current"]["paragraph"] = 0
            session_data["current"]["sentence"] = 0
            session_data["current"]["column"] = 0
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error analyzing document: {str(e)}"
        }), 500

@app.route('/api/ocr/session/<session_id>/navigate', methods=['POST'])
def navigate_document(session_id):
    """
    Navigate through document
    
    Request (JSON):
        - file_id: File ID (default: current file)
        - navigation_type: Type of navigation (page, paragraph, sentence, column)
        - action: Navigation action (next, previous, goto)
        - position: Position for goto action
    
    Response:
        - Current position and content
    """
    # Check if session exists
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Get request data
    data = request.json or {}
    
    # Get file_id (use current if not specified)
    file_id = data.get('file_id', session_data["current"]["file_id"])
    
    if not file_id or file_id not in session_data["files"]:
        return jsonify({
            "status": "error",
            "message": "Invalid file ID"
        }), 400
    
    # Get file data
    file_data = session_data["files"][file_id]
    
    # Get navigation parameters
    navigation_type = data.get('navigation_type', 'page')
    action = data.get('action', 'next')
    position = data.get('position', 0)
    
    # Get current position
    current = session_data["current"]
    
    try:
        # Handle different navigation types
        if navigation_type == 'page':
            # PDF page navigation
            if not file_data["is_pdf"]:
                return jsonify({
                    "status": "error",
                    "message": "Page navigation only applies to PDF files"
                }), 400
            
            # Get current and total pages
            current_page = current["page"]
            total_pages = file_data.get("total_pages", 1)
            
            # Calculate new page
            if action == 'next':
                new_page = min(current_page + 1, total_pages - 1)
            elif action == 'previous':
                new_page = max(current_page - 1, 0)
            else:  # goto
                new_page = max(0, min(position, total_pages - 1))
            
            # Check if page data exists
            if new_page != current_page or new_page not in file_data.get("pages", {}):
                # Need to process the page
                try:
                    # Coba dengan extract_text_from_pdf dulu
                    text, processing_time, confidence, _ = extract_text_from_pdf(
                        file_data["filepath"], 
                        file_data["language"],
                        new_page
                    )
                    
                    # Jika hasil tidak memuaskan, coba dengan subprocess
                    if text.startswith("Error") or text.startswith("Tidak dapat") or text.startswith("Gagal"):
                        image_path, _ = extract_pdf_page_with_subprocess(file_data["filepath"], new_page)
                        if image_path:
                            processed_path = preprocess_image(image_path, 'balanced')
                            text, processing_time, confidence = perform_ocr(processed_path, file_data["language"])
                            
                            # Hapus file temporary
                            try:
                                os.remove(image_path)
                                if processed_path != image_path:
                                    os.remove(processed_path)
                            except:
                                pass
                except Exception as e:
                    logger.error(f"Error saat memproses halaman baru: {e}")
                    text = f"Error memproses halaman {new_page}: {str(e)}"
                    processing_time = 0
                    confidence = 0
                
                # Save page data
                if "pages" not in file_data:
                    file_data["pages"] = {}
                    
                file_data["pages"][new_page] = {
                    "text": text,
                    "processing_time": processing_time,
                    "confidence": confidence,
                    "analyzed": False
                }
            
            # Get page text
            page_data = file_data["pages"][new_page]
            text = page_data["text"]
            
            # Update current position
            current["file_id"] = file_id
            current["page"] = new_page
            current["paragraph"] = 0
            current["sentence"] = 0
            current["column"] = 0
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "navigation_type": navigation_type,
                "current_page": new_page,
                "total_pages": total_pages,
                "text": text
            })
            
        elif navigation_type == 'paragraph':
            # Paragraph navigation
            
            # Get structure data
            if file_data["is_pdf"]:
                # For PDFs, get page data
                page = current["page"]
                if page not in file_data.get("pages", {}) or not file_data["pages"][page].get("analyzed"):
                    # Need to analyze page structure
                    try:
                        # Coba ekstrak halaman dengan subprocess
                        image_path, _ = extract_pdf_page_with_subprocess(file_data["filepath"], page)
                        
                        if image_path:
                            # Analyze structure pada gambar yang dikonversi
                            structure = analyze_document_structure(image_path)
                            
                            # Store analysis results
                            if "pages" not in file_data:
                                file_data["pages"] = {}
                                
                            if page not in file_data["pages"]:
                                file_data["pages"][page] = {}
                                
                            file_data["pages"][page]["structure"] = structure
                            file_data["pages"][page]["analyzed"] = True
                            
                            # Clean up
                            try:
                                os.remove(image_path)
                            except:
                                pass
                                
                            paragraphs = structure.get("paragraphs", [])
                        else:
                            # Fallback jika ekstraksi gagal
                            paragraphs = []
                    except Exception as e:
                        logger.error(f"Error saat ekstraksi halaman untuk navigasi paragraf: {e}")
                        paragraphs = []
                else:
                    # Use cached structure
                    structure = file_data["pages"][page].get("structure", {})
                    paragraphs = structure.get("paragraphs", [])
            else:
                # For images, get structure directly
                if not file_data.get("analyzed"):
                    # Need to analyze structure
                    structure = analyze_document_structure(file_data["filepath"])
                    file_data["structure"] = structure
                    file_data["analyzed"] = True
                    paragraphs = structure.get("paragraphs", [])
                else:
                    # Use cached structure
                    structure = file_data.get("structure", {})
                    paragraphs = structure.get("paragraphs", [])
            
            # Get current paragraph
            current_paragraph = current["paragraph"]
            total_paragraphs = len(paragraphs)
            
            if total_paragraphs == 0:
                return jsonify({
                    "status": "error",
                    "message": "No paragraphs found in document"
                }), 404
            
            # Calculate new paragraph
            if action == 'next':
                new_paragraph = min(current_paragraph + 1, total_paragraphs - 1)
            elif action == 'previous':
                new_paragraph = max(current_paragraph - 1, 0)
            else:  # goto
                new_paragraph = max(0, min(position, total_paragraphs - 1))
            
            # Get paragraph text
            paragraph = paragraphs[new_paragraph] if new_paragraph < len(paragraphs) else None
            
            if not paragraph:
                return jsonify({
                    "status": "error",
                    "message": f"Paragraph {new_paragraph} not found"
                }), 404
            
            # Update current position
            current["file_id"] = file_id
            current["paragraph"] = new_paragraph
            current["sentence"] = 0
            
            # Get column info if available
            if "column" in paragraph:
                current["column"] = paragraph["column"]
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "navigation_type": navigation_type,
                "current_paragraph": new_paragraph,
                "total_paragraphs": total_paragraphs,
                "paragraph": paragraph
            })
            
        elif navigation_type == 'sentence':
            # Sentence navigation
            
            # First get current paragraph
            if file_data["is_pdf"]:
                # For PDFs, get page data
                page = current["page"]
                if page in file_data.get("pages", {}) and file_data["pages"][page].get("analyzed"):
                    structure = file_data["pages"][page].get("structure", {})
                    paragraphs = structure.get("paragraphs", [])
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Page not analyzed"
                    }), 400
            else:
                # For images, get structure directly
                if file_data.get("analyzed"):
                    structure = file_data.get("structure", {})
                    paragraphs = structure.get("paragraphs", [])
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Document not analyzed"
                    }), 400
            
            # Get current paragraph
            current_paragraph = current["paragraph"]
            if current_paragraph >= len(paragraphs):
                return jsonify({
                    "status": "error",
                    "message": "Invalid paragraph index"
                }), 400
                
            paragraph = paragraphs[current_paragraph]
            
            # Get sentences
            sentences = paragraph.get("sentences", [])
            
            if not sentences:
                return jsonify({
                    "status": "error",
                    "message": "No sentences found in paragraph"
                }), 404
            
            # Get current sentence
            current_sentence = current["sentence"]
            total_sentences = len(sentences)
            
            # Calculate new sentence
            if action == 'next':
                new_sentence = min(current_sentence + 1, total_sentences - 1)
            elif action == 'previous':
                new_sentence = max(current_sentence - 1, 0)
            else:  # goto
                new_sentence = max(0, min(position, total_sentences - 1))
            
            # Get sentence text
            sentence = sentences[new_sentence] if new_sentence < len(sentences) else None
            
            if not sentence:
                return jsonify({
                    "status": "error",
                    "message": f"Sentence {new_sentence} not found"
                }), 404
            
            # Update current position
            current["file_id"] = file_id
            current["sentence"] = new_sentence
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "navigation_type": navigation_type,
                "current_paragraph": current_paragraph,
                "current_sentence": new_sentence,
                "total_sentences": total_sentences,
                "sentence": sentence
            })
            
        elif navigation_type == 'column':
            # Column navigation
            
            # Get structure data
            if file_data["is_pdf"]:
                # For PDFs, get page data
                page = current["page"]
                if page in file_data.get("pages", {}) and file_data["pages"][page].get("analyzed"):
                    structure = file_data["pages"][page].get("structure", {})
                    columns = structure.get("columns", [])
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Page not analyzed"
                    }), 400
            else:
                # For images, get structure directly
                if file_data.get("analyzed"):
                    structure = file_data.get("structure", {})
                    columns = structure.get("columns", [])
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Document not analyzed"
                    }), 400
            
            # Get total columns
            total_columns = len(columns)
            
            if total_columns == 0:
                return jsonify({
                    "status": "error",
                    "message": "No columns found in document"
                }), 404
            
            # Get current column
            current_column = current["column"]
            
            # Calculate new column
            if action == 'next':
                new_column = min(current_column + 1, total_columns - 1)
            elif action == 'previous':
                new_column = max(current_column - 1, 0)
            else:  # goto
                new_column = max(0, min(position, total_columns - 1))
            
            # Get column data
            column = columns[new_column] if new_column < len(columns) else None
            
            if not column:
                return jsonify({
                    "status": "error",
                    "message": f"Column {new_column} not found"
                }), 404
            
            # Get paragraphs in this column
            paragraphs = structure.get("paragraphs", [])
            column_paragraphs = []
            
            for para_id in column.get("paragraphs", []):
                if para_id < len(paragraphs):
                    column_paragraphs.append(paragraphs[para_id])
            
            # Update current position
            current["file_id"] = file_id
            current["column"] = new_column
            current["paragraph"] = 0
            current["sentence"] = 0
            
            return jsonify({
                "status": "success",
                "file_id": file_id,
                "navigation_type": navigation_type,
                "current_column": new_column,
                "total_columns": total_columns,
                "column": {
                    "id": column.get("id"),
                    "paragraphs": column_paragraphs
                }
            })
            
        else:
            return jsonify({
                "status": "error",
                "message": f"Invalid navigation type: {navigation_type}"
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error navigating document: {str(e)}"
        }), 500

@app.route('/api/ocr/session/<session_id>/summarize', methods=['POST'])
def summarize_document(session_id):
    """
    Summarize document or part of document
    
    Request (JSON):
        - file_id: File ID (default: current file)
        - scope: Summarization scope (full, page, paragraph, column)
        - page: Page number for PDFs (default: current page)
        - paragraph: Paragraph index (default: current paragraph)
        - column: Column index (default: current column)
        - ratio: Summary ratio (default: 0.3)
    
    Response:
        - summary and original text
    """
    # Check if session exists
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired session"
        }), 404
    
    # Get request data
    data = request.json or {}
    
    # Get file_id (use current if not specified)
    file_id = data.get('file_id', session_data["current"]["file_id"])
    
    if not file_id or file_id not in session_data["files"]:
        return jsonify({
            "status": "error",
            "message": "Invalid file ID"
        }), 400
    
    # Get file data
    file_data = session_data["files"][file_id]
    
    # Get current positions
    current = session_data["current"]
    
    # Get summarization parameters
    scope = data.get('scope', 'full')
    page = data.get('page', current["page"])
    paragraph_index = data.get('paragraph', current["paragraph"])
    column_index = data.get('column', current["column"])
    ratio = float(data.get('ratio', 0.3))
    
    try:
        # Get text based on scope
        if scope == 'full':
            # Full document
            if file_data["is_pdf"]:
                # For PDFs, combine all processed pages
                if "pages" not in file_data or not file_data["pages"]:
                    return jsonify({
                        "status": "error",
                        "message": "No pages processed yet"
                    }), 400
                    
                text = "\n\n".join([page_data.get("text", "") 
                                   for page_data in file_data["pages"].values()])
            else:
# For images, use full text
                if not file_data.get("processed"):
                    return jsonify({
                        "status": "error",
                        "message": "Document not processed yet"
                    }), 400
                    
                text = file_data.get("text", "")
        
        elif scope == 'page':
            # Single page
            if not file_data["is_pdf"]:
                return jsonify({
                    "status": "error",
                    "message": "Page scope only applies to PDF files"
                }), 400
                
            if "pages" not in file_data or page not in file_data["pages"]:
                return jsonify({
                    "status": "error",
                    "message": f"Page {page} not processed yet"
                }), 400
                
            text = file_data["pages"][page].get("text", "")
            
        elif scope == 'paragraph':
            # Single paragraph
            if file_data["is_pdf"]:
                # For PDFs, get from page structure
                if ("pages" not in file_data or page not in file_data["pages"] or
                    not file_data["pages"][page].get("analyzed")):
                    return jsonify({
                        "status": "error",
                        "message": "Page not analyzed yet"
                    }), 400
                    
                structure = file_data["pages"][page].get("structure", {})
                paragraphs = structure.get("paragraphs", [])
            else:
                # For images, get from structure
                if not file_data.get("analyzed"):
                    return jsonify({
                        "status": "error",
                        "message": "Document not analyzed yet"
                    }), 400
                    
                structure = file_data.get("structure", {})
                paragraphs = structure.get("paragraphs", [])
            
            if not paragraphs or paragraph_index >= len(paragraphs):
                return jsonify({
                    "status": "error",
                    "message": f"Paragraph {paragraph_index} not found"
                }), 404
                
            text = paragraphs[paragraph_index].get("text", "")
            
        elif scope == 'column':
            # Single column
            if file_data["is_pdf"]:
                # For PDFs, get from page structure
                if ("pages" not in file_data or page not in file_data["pages"] or
                    not file_data["pages"][page].get("analyzed")):
                    return jsonify({
                        "status": "error",
                        "message": "Page not analyzed yet"
                    }), 400
                    
                structure = file_data["pages"][page].get("structure", {})
            else:
                # For images, get from structure
                if not file_data.get("analyzed"):
                    return jsonify({
                        "status": "error",
                        "message": "Document not analyzed yet"
                    }), 400
                    
                structure = file_data.get("structure", {})
            
            columns = structure.get("columns", [])
            paragraphs = structure.get("paragraphs", [])
            
            if not columns or column_index >= len(columns):
                return jsonify({
                    "status": "error",
                    "message": f"Column {column_index} not found"
                }), 404
            
            # Get paragraphs in this column
            column = columns[column_index]
            column_paragraphs = []
            
            for para_id in column.get("paragraphs", []):
                if para_id < len(paragraphs):
                    column_paragraphs.append(paragraphs[para_id])
            
            # Combine text from all paragraphs
            text = "\n\n".join([para.get("text", "") for para in column_paragraphs])
            
        else:
            return jsonify({
                "status": "error",
                "message": f"Invalid scope: {scope}"
            }), 400
        
        # Check if text is long enough for summarization
        if len(text.split()) < 50:
            return jsonify({
                "status": "success",
                "message": "Text too short for summarization",
                "original_text": text,
                "summary": text,
                "scope": scope
            })
        
        # Create summary
        summary = summarizer.summarize(text, ratio)
        
        return jsonify({
            "status": "success",
            "scope": scope,
            "original_text": text,
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if text else 0
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error summarizing document: {str(e)}"
        }), 500

@app.route('/api/ocr/recognize', methods=['POST'])
def quick_recognize():
    """
    Quick OCR without session (for simple use cases)
    
    Request (form-data):
        - image: Image file
        - language: OCR language (default: eng)
        - preprocess: Preprocessing type (default, fast, balanced, quality)
        - include_summary: Include text summary (default: false)
        
    Response:
        - Extracted text and optional summary
    """
    # Check if file exists in request
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image part in the request"
        }), 400
    
    file = request.files['image']
    
    # Check if no file selected
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No selected file"
        }), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        try:
            # Get parameters
            language = request.form.get('language', 'eng')
            preprocess_type = request.form.get('preprocess', 'balanced')
            include_summary = request.form.get('include_summary', 'false').lower() == 'true'
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"quick_{uuid.uuid4()}_{filename}")
            file.save(filepath)
            
            # Optimize for mobile
            optimized_path = optimize_image_for_mobile(filepath)
            
            # Preprocess image
            processed_path = preprocess_image(optimized_path, preprocess_type)
            
            # Perform OCR
            text, processing_time, confidence = perform_ocr(processed_path, language)
            
            # Prepare response
            response = {
                "status": "success",
                "text": text,
                "confidence": confidence,
                "processing_time_ms": processing_time * 1000
            }
            
            # Add summary if requested
            if include_summary and len(text.split()) >= 50:
                summary = summarizer.summarize(text)
                response["summary"] = summary
            
            # Clean up
            try:
                os.remove(filepath)
                if optimized_path != filepath:
                    os.remove(optimized_path)
                if processed_path != optimized_path:
                    os.remove(processed_path)
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

@app.route('/api/ocr/languages', methods=['GET'])
def get_languages():
    """Get available OCR languages"""
    try:
        # Get language list from Tesseract
        result = os.popen(f"{pytesseract.pytesseract.tesseract_cmd} --list-langs").read()
        languages = result.strip().split('\n')
        
        # Remove first line (usually "List of available languages (x):")
        if len(languages) > 0 and not languages[0].endswith('.traineddata'):
            languages = languages[1:]
        
        return jsonify({
            "status": "success",
            "languages": languages,
            "default": "eng"
        })
    except Exception as e:
        logger.error(f"Error getting languages: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error getting languages: {str(e)}",
            "languages": ["eng"]  # Fallback to English
        }), 500

# Main execution
if __name__ == '__main__':
    # Periksa instalasi Poppler
    check_poppler_installation()
    
    # Daftarkan Poppler ke environment PATH
    poppler_path = register_poppler_path()
    if poppler_path:
        logger.info(f"Berhasil mendaftarkan Poppler dari: {poppler_path}")
    else:
        logger.warning("Tidak dapat menemukan Poppler. Fungsi PDF mungkin tidak berfungsi dengan baik.")
    
    # Clean up the upload folder at startup
    for file in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
        except:
            pass
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)