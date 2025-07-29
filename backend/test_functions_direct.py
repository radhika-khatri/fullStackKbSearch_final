#!/usr/bin/env python3
"""
Test OCR functionality directly with copied functions
"""
from PIL import Image, ImageDraw
import pytesseract
import io
import base64

def extract_text_from_image_simple(pil_image):
    """Simple OCR extraction using pytesseract"""
    try:
        text = pytesseract.image_to_string(pil_image, config='--psm 6')
        cleaned_text = text.strip()
        return cleaned_text if cleaned_text else "No text detected in the image."
    except Exception as e:
        print(f"OCR error: {e}")
        return "Failed to extract text from image."

def image_to_base64_simple(pil_image):
    """Simple base64 conversion for PIL images"""
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Compress image for better performance
        width, height = pil_image.size
        if width > 800 or height > 800:
            ratio = min(800/width, 800/height)
            new_size = (int(width * ratio), int(height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=60, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"Base64 conversion error: {e}")
        return ""

def test_functions():
    """Test both functions directly"""
    print("🚀 Testing OCR and base64 conversion directly...")
    
    # Create a test image with text
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "TEST OCR", fill='black')
    
    print("🖼️ Test image created")
    
    # Test OCR
    print("📝 Testing OCR...")
    ocr_text = extract_text_from_image_simple(img)
    print(f"✅ OCR result: '{ocr_text}'")
    
    # Test base64 conversion
    print("🔄 Testing base64 conversion...")
    base64_data = image_to_base64_simple(img)
    print(f"✅ Base64 result: {len(base64_data)} characters")
    print(f"   Preview: {base64_data[:50]}...")
    
    return True

if __name__ == "__main__":
    test_functions()
