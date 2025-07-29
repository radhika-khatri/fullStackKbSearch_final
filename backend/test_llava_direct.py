#!/usr/bin/env python3
"""
Test LLaVA model directly with base64 image
"""
import requests
import io
import base64
from PIL import Image, ImageDraw, ImageFont

def create_and_test_base64():
    """Create test image and test with LLaVA directly"""
    # Create a simple test image
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "TEST IMAGE", fill='black')
    
    # Convert to base64 using our exact same method
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Compress image (same as our backend)
    width, height = img.size
    if width > 800 or height > 800:
        ratio = min(800/width, 800/height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=60, optimize=True)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    print(f"Base64 length: {len(img_base64)}")
    print(f"Base64 preview: {img_base64[:50]}...")
    
    # Test with LLaVA
    payload = {
        "model": "llava:latest",
        "prompt": "What do you see?",
        "images": [img_base64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 50
        }
    }
    
    print("📤 Testing LLaVA with base64 image...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ LLaVA response:")
            print(f"  Response: {result.get('response', 'No response')}")
            print(f"  Done: {result.get('done', False)}")
            return True
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ LLaVA timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing LLaVA model directly with base64 image...\n")
    create_and_test_base64()
