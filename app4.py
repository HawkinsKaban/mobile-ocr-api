import os
import time
import uuid
import re
import logging
import numpy as np
import cv2
import pytesseract
import subprocess
import tempfile
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from collections import defaultdict
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Upload configuration
UPLOAD_FOLDER = '/tmp/ocr_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Tesseract path based on OS
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

# Utility function to check allowed files
def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhance_image(image_path):
    """
    Apply multiple enhancement techniques and return the best processed image
    
    Args:
        image_path: Path to the image
        
    Returns:
        List of paths to processed images
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
                return [image_path]

        if image is None:
            return [image_path]
        
        # Create multiple processed versions
        processed_images = []
        
        # Get dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 0. Original grayscale
        gray_path = f"{image_path}_gray.jpg"
        cv2.imwrite(gray_path, gray)
        processed_images.append(gray_path)
        
        # 1. Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # Save contrast enhanced version
        contrast_path = f"{image_path}_contrast.jpg"
        cv2.imwrite(contrast_path, contrast_enhanced)
        processed_images.append(contrast_path)
        
        # 2. Binarization with Otsu's method
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_path = f"{image_path}_otsu.jpg"
        cv2.imwrite(otsu_path, otsu)
        processed_images.append(otsu_path)
        
        # 3. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
                                       
        # Save adaptive thresholded version
        adaptive_path = f"{image_path}_adaptive.jpg"
        cv2.imwrite(adaptive_path, adaptive)
        processed_images.append(adaptive_path)
        
        # 4. Super-resolution (scale up the image for better OCR)
        try:
            # Resize to 2x
            upscaled = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
            _, upscaled_binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            upscaled_path = f"{image_path}_upscaled.jpg"
            cv2.imwrite(upscaled_path, upscaled_binary)
            processed_images.append(upscaled_path)
        except Exception as e:
            logger.warning(f"Upscaling failed: {e}")
        
        # 5. Add original image to the list
        processed_images.append(image_path)
        
        return processed_images
        
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return [image_path]

def convert_pdf_to_image(pdf_path, page_num=None):
    """
    Convert PDF to high-quality image
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number to extract (None for first page)
    
    Returns:
        Path to the image, total page count
    """
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        output_prefix = os.path.join(temp_dir, "page")
        
        # Check number of pages
        try:
            from PyPDF2 import PdfReader
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            total_pages = 1
        
        # Set page range
        if page_num is not None:
            if page_num < 0 or page_num >= total_pages:
                return None, total_pages
            page_start = page_num + 1
            page_end = page_num + 1
        else:
            page_start = 1
            page_end = 1  # Default to first page
        
        # Try with pdftoppm (Poppler)
        try:
            pdftoppm_path = shutil.which('pdftoppm')
            if pdftoppm_path:
                # Run pdftoppm command
                cmd = [
                    pdftoppm_path,
                    "-jpeg",
                    "-r", "600",  # Use 600 DPI for much better quality
                    "-f", str(page_start),
                    "-l", str(page_end),
                    pdf_path,
                    output_prefix
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    # Find output file
                    output_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                                if f.startswith(os.path.basename(output_prefix)) and f.endswith(".jpg")]
                    
                    if output_files:
                        output_files.sort()
                        return output_files[0], total_pages
                else:
                    logger.error(f"pdftoppm error: {result.stderr}")
        except Exception as e:
            logger.error(f"Error using pdftoppm: {e}")
        
        # Fallback to pdf2image
        try:
            from pdf2image import convert_from_path
            
            # Convert with high DPI
            pages = convert_from_path(
                pdf_path, 
                600,  # Use 600 DPI for much better quality
                first_page=page_start, 
                last_page=page_end
            )
            
            if pages:
                # Save as image
                image_path = f"{pdf_path}_page_{page_start}.jpg"
                pages[0].save(image_path, 'JPEG', quality=100)  # Use maximum quality
                return image_path, total_pages
        except Exception as e:
            logger.error(f"Error using pdf2image: {e}")
        
        # If both methods fail, return None
        return None, total_pages
        
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        return None, 0

def perform_ocr_with_best_result(image_paths, language='eng'):
    """
    Try OCR on multiple processed images and select the best result
    
    Args:
        image_paths: List of paths to processed images
        language: OCR language
    
    Returns:
        Best OCR text, confidence score
    """
    best_result = ""
    best_confidence = 0
    best_length = 0
    
    # PSM modes to try
    psm_modes = [3, 4, 6, 11, 1]  # Ordered by typical effectiveness for document text
    
    for image_path in image_paths:
        for psm in psm_modes:
            try:
                # Custom configuration with preserve newlines and specifics for this document type
                custom_config = f'-l {language} --oem 1 --psm {psm}'
                
                # Perform OCR
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image, config=custom_config)
                
                # Get confidence
                data = pytesseract.image_to_data(image, config=custom_config, 
                                             output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Get text length (non-whitespace)
                text_length = len(''.join(text.split()))
                
                # Decide if this result is better
                # We prioritize higher confidence, but also consider text length
                if (avg_confidence > best_confidence + 5) or \
                   (avg_confidence >= best_confidence - 5 and text_length > best_length * 1.2):
                    best_confidence = avg_confidence
                    best_result = text
                    best_length = text_length
                    logger.info(f"New best result: {os.path.basename(image_path)}, PSM {psm}, "
                               f"Confidence: {avg_confidence:.1f}, Length: {text_length}")
            
            except Exception as e:
                logger.error(f"Error in OCR for {image_path} with PSM {psm}: {e}")
    
    return best_result, best_confidence

def perfect_text_formatting(text):
    """
    Apply perfect formatting to OCR text to match the original document layout
    
    Args:
        text: Raw OCR text
        
    Returns:
        Perfectly formatted text
    """
    if not text:
        return ""
        
    try:
        # Fix bullet points (make sure they're all consistent)
        # Look for various bullet representations and normalize to standard bullet •
        bullet_patterns = [r'^\s*[\*\+\-•·⦁◦‣▪]\s+', r'^\s*\d+\.\s+', r'^\s*\(\d+\)\s+']
        
        # Split into lines
        lines = text.split('\n')
        cleaned_lines = []
        
        # Process line by line
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if line is a bullet point
            is_bullet = False
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    is_bullet = True
                    # Convert any bullet pattern to •
                    line = re.sub(pattern, '• ', line)
                    break
            
            # Handle bullet points - they should be on separate lines
            if is_bullet:
                # Start building a complete bullet point paragraph
                bullet_paragraph = line
                
                # Look ahead to see if the next line is a continuation of this bullet point
                j = i + 1
                while j < len(lines) and lines[j].strip() and not any(re.match(pattern, lines[j]) for pattern in bullet_patterns):
                    bullet_paragraph += ' ' + lines[j].strip()
                    j += 1
                
                cleaned_lines.append(bullet_paragraph)
                i = j
            else:
                # Regular paragraph text - check if it's a header
                is_header = (line.isupper() or 
                           re.match(r'^[A-Z][\w\s]+$', line) or
                           'Instruksi' in line or 
                           'Subtes' in line or 
                           'Comprehension' in line)
                
                if is_header:
                    # It's a heading - keep it as a separate line
                    cleaned_lines.append(line)
                    i += 1
                else:
                    # Start building a complete paragraph
                    paragraph = line
                    
                    # Look ahead to see if the next line is a continuation of this paragraph
                    j = i + 1
                    while (j < len(lines) and lines[j].strip() and 
                          not any(re.match(pattern, lines[j]) for pattern in bullet_patterns) and
                          not lines[j].strip().isupper() and
                          not 'Instruksi' in lines[j] and
                          not 'Subtes' in lines[j] and
                          not 'Comprehension' in lines[j]):
                        paragraph += ' ' + lines[j].strip()
                        j += 1
                    
                    cleaned_lines.append(paragraph)
                    i = j
        
        # Join lines with proper spacing
        result = ''
        prev_is_bullet = False
        
        for i, line in enumerate(cleaned_lines):
            if not line.strip():
                continue
                
            is_bullet = line.startswith('• ')
            is_header = (line.isupper() or 
                       re.match(r'^[A-Z][\w\s]+$', line) or
                       'Instruksi' in line or 
                       'Subtes' in line or 
                       'Comprehension' in line)
            
            # Add appropriate spacing
            if i > 0:
                if is_header:
                    result += '\n\n'
                elif is_bullet and prev_is_bullet:
                    result += '\n'
                elif is_bullet and not prev_is_bullet:
                    result += '\n\n'
                elif prev_is_bullet and not is_bullet:
                    result += '\n\n'
                else:
                    result += '\n'
            
            result += line
            prev_is_bullet = is_bullet
        
        # Fix common OCR errors
        result = result.replace('"', '"').replace('"', '"')  # Fix quotes
        result = result.replace("''", '"').replace(",,", '"')
        result = result.replace("'", "'")  # Fix apostrophes
        result = result.replace('•', '•')  # Normalize bullets
        
        # Fix spacing after periods
        result = re.sub(r'\.([A-Z])', r'. \1', result)
        
        # Final cleanup
        result = result.strip()
        
        return result
        
    except Exception as e:
        logger.error(f"Error during perfect text formatting: {e}")
        return text

def generate_intelligent_summary(text, max_length=200):
    """
    Generate a concise, intelligent summary of the document
    
    Args:
        text: The document text
        max_length: Maximum summary length
        
    Returns:
        Summary text
    """
    if not text or len(text) < 100:
        return text[:max_length] if text else ""
    
    try:
        # Extract the document title/header if available
        lines = text.split('\n')
        title = next((line for line in lines if "Instruksi" in line or "Reading Comprehension" in line), "")
        
        # Find key sentences with important information
        key_indicators = [
            "Tes ini terdiri dari",
            "Saudara diminta untuk",
            "Cara menjawab",
            "Apabila Saudara sudah siap",
            "Saudara dapat memilih tombol",
            "tidak dapat melihat instruksi",
            "Apabila Saudara telah siap"
        ]
        
        # Extract sentences with key information
        sentences = re.split(r'(?<=[.!?])\s+', text)
        key_sentences = [s for s in sentences if any(ki in s for ki in key_indicators)]
        
        # Ensure we have at least a few key sentences
        if len(key_sentences) < 2:
            # Fall back to first and last sentences
            if len(sentences) > 2:
                key_sentences = [sentences[0], sentences[-1]]
            else:
                key_sentences = sentences[:1]
        
        # Build the summary
        summary = title if title else ""
        
        # Add key sentences until we approach max length
        current_length = len(summary)
        
        for sentence in key_sentences:
            if current_length + len(sentence) + 2 <= max_length:
                if summary:
                    summary += ". " + sentence
                else:
                    summary = sentence
                current_length = len(summary)
            else:
                # We're approaching max length, so stop adding sentences
                break
        
        # If summary is still too short and we have more text, add more content
        if current_length < max_length * 0.7 and len(sentences) > len(key_sentences):
            for sentence in sentences:
                if sentence not in key_sentences:
                    if current_length + len(sentence) + 2 <= max_length:
                        if summary:
                            summary += ". " + sentence
                        else:
                            summary = sentence
                        current_length = len(summary)
                    else:
                        break
        
        # Final cleanup
        summary = summary.replace('\n', ' ').strip()
        summary = re.sub(r'\s+', ' ', summary)
        
        # Make sure summary doesn't end with a partial sentence
        if len(summary) > max_length:
            # Find the last sentence break before max_length
            last_break = max(summary.rfind('.', 0, max_length), 
                           summary.rfind('!', 0, max_length),
                           summary.rfind('?', 0, max_length))
            
            if last_break > 0:
                summary = summary[:last_break+1]
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return text[:max_length] if text else ""

@app.route('/')
def home():
    """API home page"""
    return jsonify({
        "status": "success",
        "message": "Perfect OCR API",
        "version": "1.0.0",
        "endpoint": "/api/ocr - Perform OCR with automatic summarization"
    })

@app.route('/api/ocr', methods=['POST'])
def ocr():
    """
    Perform OCR with perfect text formatting and intelligent summarization
    
    Request (form-data):
        - file: Image or PDF file
        - language: OCR language (default: eng)
        - page: Page number for PDFs (default: 0)
    
    Response:
        - status: Success or error
        - text: Perfectly formatted extracted text
        - summary: Intelligent document summary
        - confidence: OCR confidence
        - processing_time_ms: Processing time in milliseconds
    """
    start_time = time.time()
    
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
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "File type not allowed"
        }), 400
    
    try:
        # Get parameters
        language = request.form.get('language', 'eng')
        page = request.form.get('page', '0')
        try:
            page = int(page)
        except ValueError:
            page = 0
            
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(filepath)
        
        # Process based on file type
        is_pdf = filename.lower().endswith('.pdf')
        
        if is_pdf:
            # Convert PDF to image
            image_path, total_pages = convert_pdf_to_image(filepath, page)
            
            if not image_path:
                return jsonify({
                    "status": "error",
                    "message": "Failed to convert PDF to image"
                }), 500
                
            # Enhance image
            processed_images = enhance_image(image_path)
            
            # Perform OCR on all processed images
            text, confidence = perform_ocr_with_best_result(processed_images, language)
            
            # Clean up temporary PDF image
            try:
                os.remove(image_path)
            except:
                pass
        else:
            # Process image directly
            processed_images = enhance_image(filepath)
            
            # Perform OCR on all processed images
            text, confidence = perform_ocr_with_best_result(processed_images, language)
        
        # Clean up all processed images
        for img_path in processed_images:
            if img_path != filepath:
                try:
                    os.remove(img_path)
                except:
                    pass
        
        # Apply perfect formatting
        formatted_text = perfect_text_formatting(text)
        
        # Generate intelligent summary
        summary = generate_intelligent_summary(formatted_text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up original file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Return result
        return jsonify({
            "status": "success",
            "text": formatted_text,
            "summary": summary,
            "confidence": confidence,
            "processing_time_ms": processing_time * 1000
        })
        
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error performing OCR: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Clean up the upload folder at startup
    for file in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
        except:
            pass
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)