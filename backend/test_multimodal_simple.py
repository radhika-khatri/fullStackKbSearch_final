#!/usr/bin/env python3
"""
Simple test for multimodal endpoint with optimized settings
"""
import requests
import io
import base64
from PIL import Image, ImageDraw

def test_multimodal_simple():
    """Test multimodal endpoint with a very simple image"""
    print("🚀 Testing multimodal endpoint with simple image...")
    
    # Create a very small, simple test image
    img = Image.new('RGB', (100, 50), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Hi", fill='black')
    
    # Save image to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    image_bytes = buffer.getvalue()
    
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Prepare form data
    files = {
        'image': ('test.jpg', image_bytes, 'image/jpeg')
    }
    data = {
        'question': 'What do you see in this image?'
    }
    
    print("📤 Sending request to multimodal endpoint...")
    try:
        response = requests.post(
            "http://localhost:8004/api/qa/chat-multimodal",
            data=data,
            files=files,
            timeout=60  # Increased timeout
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
        print("⏰ Request timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_multimodal_simple()
