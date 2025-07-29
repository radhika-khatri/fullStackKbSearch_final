#!/usr/bin/env python3
"""
Minimal test with detailed error tracking
"""
import requests
import io
from PIL import Image, ImageDraw
import time

def test_with_timing():
    """Test with detailed timing to find bottleneck"""
    print("🚀 Testing with detailed timing...")
    
    # Create a tiny test image
    img = Image.new('RGB', (50, 25), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "Hi", fill='black')
    
    # Save to minimal bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=50)
    image_bytes = buffer.getvalue()
    
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Prepare form data
    files = {
        'image': ('test.jpg', image_bytes, 'image/jpeg')
    }
    data = {
        'question': 'Hi'
    }
    
    print("📤 Sending request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8004/api/qa/chat-multimodal",
            data=data,
            files=files,
            timeout=15  # Shorter timeout to fail faster
        )
        
        end_time = time.time()
        print(f"⏱️ Request took: {end_time - start_time:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"💬 Response: {result.get('answer', 'No answer')}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        end_time = time.time()
        print(f"⏰ Request timed out after {end_time - start_time:.2f} seconds")
        return False
    except Exception as e:
        end_time = time.time()
        print(f"❌ Error after {end_time - start_time:.2f} seconds: {e}")
        return False

if __name__ == "__main__":
    test_with_timing()
