#!/usr/bin/env python3
"""
Test OCR functionality directly
"""
import sys
sys.path.append('/home/ishaan/Documents/fishai/fullStackKbSearch/backend')

from PIL import Image, ImageDraw
import io
from api.qa import extract_text_from_image_simple, image_to_base64_simple

def test_ocr_direct():
    """Test OCR and base64 conversion directly"""
    print("🚀 Testing OCR and base64 conversion directly...")
    
    # Create a test image with text
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "TEST OCR", fill='black')
    
    print("🖼️ Test image created")
    
    # Test OCR
    print("📝 Testing OCR...")
    try:
        ocr_text = extract_text_from_image_simple(img)
        print(f"✅ OCR successful: '{ocr_text}'")
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return False
    
    # Test base64 conversion
    print("🔄 Testing base64 conversion...")
    try:
        base64_data = image_to_base64_simple(img)
        print(f"✅ Base64 successful: {len(base64_data)} characters")
        print(f"   Preview: {base64_data[:50]}...")
    except Exception as e:
        print(f"❌ Base64 failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_ocr_direct()
