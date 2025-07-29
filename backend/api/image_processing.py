from utils.quote_extraction import is_photo_image, extract_quote_fields
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance
import pytesseract
import base64
import io
import cv2
import numpy as np
import requests
import json
import logging
from pathlib import Path
import aiofiles
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava:latest"

# Supported image formats
SUPPORTED_FORMATS = {"jpeg", "jpg", "png", "bmp", "tiff", "webp"}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

def validate_image(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    # Check file extension
    if not file.filename:
        return False
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        return False
    
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        return False
    
    return True

def preprocess_image(image: Image.Image) -> Image.Image:
    """Fast image preprocessing for better OCR accuracy"""
    try:
        # Quick resize if too small
        width, height = image.size
        if width < 300 or height < 300:
            scale_factor = max(300 / width, 300 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return image  # Return original image if preprocessing fails

def extract_text_from_image(image: Image.Image) -> str:
    """Fast and efficient text extraction from image using OCR"""
    try:
        # Quick preprocessing for speed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (faster processing)
        width, height = image.size
        if width > 1200 or height > 1200:
            ratio = min(1200/width, 1200/height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Fast OCR with optimized settings
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;-()[]{}/"\'@#$%^&*+=<>|`~_'
        
        # Single OCR call for speed
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Quick text cleanup
        text = text.strip()
        if not text:
            return "No text detected in the image."
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return "Failed to extract text from image."

def image_to_base64(image: Image.Image) -> str:
    """Fast conversion of PIL Image to base64 string with compression"""
    try:
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Compress image for faster processing (reduce quality for speed)
        width, height = image.size
        
        # Resize if too large (speed optimization)
        if width > 800 or height > 800:
            ratio = min(800/width, 800/height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save image to bytes buffer with lower quality for speed
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=60, optimize=True)  # Reduced quality for speed
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Base64 conversion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image")

async def query_ollama_llava(base64_image: str, ocr_text: str) -> str:
    """Fast query to local Ollama LLaVA model with optimized settings"""
    try:
        # Simple, direct prompt for faster processing
        prompt = "Describe this image briefly."
        if ocr_text and ocr_text not in ["No text detected in the image.", "Failed to extract text from image."]:
            prompt = f"Describe this image briefly. Text found: '{ocr_text[:100]}...'" if len(ocr_text) > 100 else f"Describe this image briefly. Text found: '{ocr_text}'"
        
        # Optimized payload for speed
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower for faster, more focused responses
                "top_p": 0.8,
                "max_tokens": 150,   # Limit response length for speed
                "num_predict": 150   # Ollama-specific token limit
            }
        }
        
        logger.info("Sending fast request to Ollama...")
        
        # Single attempt with reasonable timeout
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=20  # Reduced timeout for speed
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result.get("response", "").strip()
            if analysis:
                logger.info("Ollama request completed successfully")
                return analysis
            else:
                return "Vision model processed the image but provided no description."
        else:
            logger.error(f"Ollama request failed: {response.status_code}")
            return f"Vision model temporarily unavailable (Error: {response.status_code})"
            
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        return "Vision analysis timed out. Please try again with a smaller image."
        
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama")
        return "Vision model is not available. Please ensure Ollama is running."
        
    except Exception as e:
        logger.error(f"Ollama query failed: {e}")
        return f"Vision analysis failed: {str(e)[:50]}..."

@router.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    Process uploaded image:
    1. Extract text using OCR (pytesseract)
    2. Analyze image using local LLaVA model via Ollama
    3. Return both OCR text and vision analysis
    """
    try:
        # Validate file
        if not validate_image(file):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail="File size too large. Maximum size is 10MB."
            )
        
        # Load image
        try:
            image = Image.open(io.BytesIO(file_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run OCR and vision analysis in parallel for speed
        logger.info("Processing image with parallel OCR and vision analysis...")

        # Create executor for CPU-bound tasks
        executor = ThreadPoolExecutor(max_workers=2)
        loop = asyncio.get_event_loop()
        ocr_task = loop.run_in_executor(executor, extract_text_from_image, image)
        base64_task = loop.run_in_executor(executor, image_to_base64, image)
        ocr_text, base64_image = await asyncio.gather(ocr_task, base64_task)
        vision_task = asyncio.create_task(query_ollama_llava(base64_image, ocr_text))
        vision_response = await vision_task

        # --- New logic: classify photo and extract quote ---
        is_photo = is_photo_image(image)
        quote = None
        if is_photo:
            quote = extract_quote_fields(ocr_text)

        result = {
            "success": True,
            "ocr_text": ocr_text,
            "vision_analysis": vision_response,
            "image_info": {
                "filename": file.filename,
                "size": f"{len(file_content) / 1024:.1f} KB",
                "dimensions": f"{image.width}x{image.height}",
                "format": image.format
            },
            "is_photo": is_photo,
            "quote": quote
        }

        logger.info(f"Successfully processed image: {file.filename}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@router.post("/analyze-image-with-question")
async def analyze_image_with_question(
    file: UploadFile = File(...),
    question: str = None
):
    """
    Process image with a specific question from user
    """
    try:
        # Validate file
        if not validate_image(file):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail="File size too large. Maximum size is 10MB."
            )
        
        # Load image
        try:
            image = Image.open(io.BytesIO(file_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract text using OCR
        logger.info("Extracting text from image using OCR...")
        ocr_text = extract_text_from_image(image)
        
        # Convert image to base64
        base64_image = image_to_base64(image)
        
        # Create custom prompt with user question
        try:
            prompt = question if question else "What can you see in this image?"
            if ocr_text and ocr_text != "No text detected in the image.":
                prompt += f" The image contains the following text: '{ocr_text}'."
            
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                vision_response = result.get("response", "No response from vision model.")
            else:
                vision_response = "Vision model is currently unavailable."
                
        except requests.exceptions.ConnectionError:
            vision_response = "Vision model is not available. Please ensure Ollama with LLaVA model is running."
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            vision_response = "Failed to analyze image."
        
        # Return combined response
        combined_response = vision_response
        if ocr_text and ocr_text != "No text detected in the image.":
            combined_response = f"Text in image: {ocr_text}\n\nImage analysis: {vision_response}"
        
        return JSONResponse(content={
            "success": True,
            "answer": combined_response,
            "ocr_text": ocr_text,
            "vision_analysis": vision_response
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")

@router.get("/health")
async def health_check():
    """Check if Ollama LLaVA model is available"""
    try:
        # Test connection to Ollama
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            llava_available = any(OLLAMA_MODEL in model.get("name", "") for model in models.get("models", []))
            
            return {
                "ollama_available": True,
                "llava_model_available": llava_available,
                "tesseract_available": True,  # If we reach here, pytesseract should work
                "status": "healthy" if llava_available else "ollama_running_but_llava_missing"
            }
        else:
            return {
                "ollama_available": False,
                "llava_model_available": False,
                "tesseract_available": True,
                "status": "ollama_not_responding"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "ollama_available": False,
            "llava_model_available": False,
            "tesseract_available": True,
            "status": "ollama_not_running"
        }
    except Exception as e:
        return {
            "ollama_available": False,
            "llava_model_available": False,
            "tesseract_available": False,
            "status": f"error: {str(e)}"
        }
