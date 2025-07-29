#!/usr/bin/env python3
"""
Test just image upload and OCR without vision model
"""
import requests
import io
from PIL import Image, ImageDraw

def test_image_only():
    """Test endpoint with just OCR processing"""
    print("🚀 Testing image upload with OCR only...")
    
    # Create a test image with text
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "TEST OCR", fill='black')
    
    # Save image to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    image_bytes = buffer.getvalue()
    
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Test with just image, no question (should trigger OCR only path)
    files = {
        'image': ('test.jpg', image_bytes, 'image/jpeg')
    }
    
    print("📤 Sending image-only request...")
    try:
        response = requests.post(
            "http://localhost:8004/api/qa/chat-multimodal",
            files=files,
            timeout=30
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"💬 Response: {result.get('response', 'No response')}")
            print(f"🔍 OCR Text: {result.get('ocr_text', 'No OCR text')}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_image_only()
