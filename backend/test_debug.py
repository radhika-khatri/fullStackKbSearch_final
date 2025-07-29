#!/usr/bin/env python3
"""
Test debug server
"""
import requests
import io
from PIL import Image, ImageDraw

def test_debug_server():
    """Test the debug server with image upload"""
    print("🚀 Testing debug server...")
    
    # Create a tiny test image
    img = Image.new('RGB', (50, 25), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "Hi", fill='black')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=50)
    image_bytes = buffer.getvalue()
    
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Test debug endpoint
    files = {
        'image': ('test.jpg', image_bytes, 'image/jpeg')
    }
    data = {
        'question': 'What do you see?'
    }
    
    print("📤 Sending to debug server...")
    try:
        response = requests.post(
            "http://localhost:8005/debug-multimodal",
            data=data,
            files=files,
            timeout=10
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"💬 Response: {result.get('answer', 'No answer')}")
            print(f"🖼️ Image info: {result.get('image_info', 'No info')}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Debug server timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_debug_server()
