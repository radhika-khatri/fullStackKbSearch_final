#!/usr/bin/env python3
"""
Test the minimal file upload endpoint
"""
import requests
import io
from PIL import Image, ImageDraw
import time

def test_minimal_upload():
    """Test the minimal upload endpoint"""
    print("🚀 Testing minimal file upload endpoint...")
    
    # Create a tiny test image
    img = Image.new('RGB', (30, 15), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), "Hi", fill='black')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=50)
    image_bytes = buffer.getvalue()
    
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    
    print("📤 Testing minimal upload endpoint...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8004/test-upload",
            files=files,
            timeout=10
        )
        elapsed = time.time() - start_time
        print(f"⏱️ Time taken: {elapsed:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS: {result}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"⏰ Timed out after {elapsed:.2f} seconds")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
        return False

if __name__ == "__main__":
    test_minimal_upload()
